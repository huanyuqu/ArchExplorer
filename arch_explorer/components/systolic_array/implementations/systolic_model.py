"""Systolic-array GEMM timing with SCALE-Sim v3 semantics.

GEMM convention: ``C[m, n] = A[m, k] @ B[k, n]``. In SCALE-Sim's GEMM mode a
topology row ``name, M, N, K`` is lowered by ``topologies.load_arrays_gemm``
to a convolution with an ``M x K`` ifmap, a ``1 x K`` filter window and ``N``
filters. Its operand matrices are therefore ifmap = A (``m x k``), filter = B
(``k x n``) and ofmap = C (``m x n``): SCALE-Sim's M is our ``m``, its N our
``n`` and its K the reduction ``k``.

Each dataflow keeps one operand resident in the ``rows x cols`` array and maps
two GEMM dimensions spatially (``Sr`` onto rows, ``Sc`` onto columns) while the
third is streamed over time (``T``); see ``calc_spatio_temporal_params`` in
``scalesim/topology_utils.py``:

========  ==========  ==========  ==========
dataflow  Sr (rows)   Sc (cols)   T (time)
========  ==========  ==========  ==========
``os``    m           n           k
``ws``    k           n           m
``is``    k           m           n
========  ==========  ==========  ==========

The problem is processed in ``ceil(Sr/rows) * ceil(Sc/cols)`` folds, one after
another. The per-fold length below is the row count of the demand matrices
built in ``scalesim/compute/systolic_compute_{os,ws,is}.py``
(``create_*_demand_mat``); each term has a dataflow meaning:

* ``os``: ``T`` reduction steps stream through, plus ``rows - 1`` cycles of
  skew for the operands to reach the last row and ``cols - 1`` cycles for the
  results to drain out of the bottom edge before the next fold starts
  (``T + rows + cols - 2``).
* ``ws``/``is``: ``rows`` cycles to load the stationary operand, ``T`` cycles
  of streaming, ``rows - 1`` cycles until the streamed operand reaches the
  last row and ``cols - 1`` cycles of skew across the columns
  (``2*rows + cols + T - 2``).

Partial folds are padded with idle requests, so they take as long as full
folds. SCALE-Sim's memory system services one demand row per cycle; with its
default stall-free bandwidth-estimation mode (``CALC``) the reported layer
"compute cycles" is the cycle index of the last ofmap demand row, i.e. the
total number of demand rows minus one
(``double_buffered_scratchpad_mem.service_memory_requests``). ``compute_cycles``
reproduces that reported number; it excludes DRAM traffic, which ArchExplorer
models with separate memory and transfer tasks.

SRAM access counts follow the counters in the same ``create_*_demand_mat``
functions, element by element. One SCALE-Sim 3.0.0 quirk is reproduced on
purpose: the ``os`` ofmap counter adds ``rows + cols`` per fold on top of the
``m * n`` outputs. Mapping efficiency is SCALE-Sim's fold average of
``used rows * used cols / (rows * cols)``.

Stationary-operand buffering (``buffering``). SCALE-Sim's ``ws``/``is``
folds are serial: each fold loads its stationary tile, then streams. An array
with a second (shadow) register per PE can load the next fold's stationary
tile while the current fold streams: the TPU v1 matrix unit double-buffers its
weights so that a new tile shifts in during the computation of the previous one
(Jouppi et al., ISCA'17, Sec. 2), and NeuPIMs' ``SystolicWS::issue_ex_inst``
overlaps a ``GEMM_PRELOAD`` with the previous instruction and charges the full
preload only when the pipeline is empty. With ``buffering="double"`` a fold
starts ``max(rows, T)`` cycles after the previous one (the shadow load takes
``rows`` cycles; the array accepts one streamed operand vector per cycle, so
the previous fold's ``T`` vectors must have entered), and only the first fold
pays the load and the last the drain::

    compute_cycles = (2*rows + cols + T - 2) - 1 + (folds - 1) * max(rows, T)

With one fold this equals the single-buffered value. ``os`` has no stationary
operand to preload and accepts ``single`` only. NeuPIMs differs in three
details that are simulator choices rather than array structure: a preload
starts ``rows`` cycles after the previous instruction even when that one
streams longer, the first preload costs ``2*rows - 1`` cycles, and an
instruction streams at least 4 cycles. SRAM access counts do not change.

``scalesim_gemm`` runs SCALE-Sim's own operand and compute modules (no report
files, no memory simulation). The memory simulation of SCALE-Sim 3.0.0 fails
with NumPy >= 2.5 (``int()`` of a one-element array); the compute modules do
not, and in ``CALC`` mode the memory system adds no stalls, so the reported
compute cycles equal the demand-row count minus one.
"""
from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass
import functools
import io
import math
import os
import tempfile

DATAFLOWS = ("os", "ws", "is")
BUFFERINGS = ("single", "double")
SCALESIM_INSTALL_HINT = "pip install 'arch-explorer[scalesim]' (or pip install scalesim)"


@dataclass(frozen=True)
class SystolicGemmResult:
    compute_cycles: int        # SCALE-Sim "compute cycles": stall-free pipeline cycles of all folds
    ifmap_sram_reads: int      # element reads of A, SCALE-Sim counter semantics
    filter_sram_reads: int     # element reads of B
    ofmap_sram_writes: int     # element writes of C (os includes SCALE-Sim's per-fold rows+cols term)
    row_folds: int
    col_folds: int
    mapping_efficiency: float  # fold average of used PEs / all PEs, as a fraction
    model: str                 # "native" or "scalesim-<version>"

    def to_dict(self):
        return dict(self.__dict__)


def _check(rows, cols, dataflow, m, k, n):
    for name, value in (("rows", rows), ("cols", cols), ("m", m), ("k", k), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if dataflow not in DATAFLOWS:
        raise ValueError(f"dataflow must be one of {DATAFLOWS}, got {dataflow!r}")


def spatio_temporal(dataflow, m, k, n):
    """``(Sr, Sc, T)`` for a GEMM under ``dataflow`` (SCALE-Sim definition)."""
    return {"os": (m, n, k), "ws": (k, n, m), "is": (k, m, n)}[dataflow]


def native_gemm(rows, cols, dataflow, m, k, n) -> SystolicGemmResult:
    """Closed-form SCALE-Sim v3 compute cycles and SRAM accesses for a dense GEMM."""
    _check(rows, cols, dataflow, m, k, n)
    sr, sc, t = spatio_temporal(dataflow, m, k, n)
    row_folds, col_folds = math.ceil(sr / rows), math.ceil(sc / cols)
    folds = row_folds * col_folds
    if dataflow == "os":
        fold_cycles = t + rows + cols - 2
        ifmap_reads = col_folds * t * sr       # every row fold re-reads its A rows for each column fold
        filter_reads = row_folds * t * sc      # every column fold re-reads its B columns for each row fold
        ofmap_writes = sr * sc + folds * (rows + cols)
    else:
        fold_cycles = 2 * rows + cols + t - 2
        if dataflow == "ws":                    # B resident: read once; A streamed once per column fold
            ifmap_reads = col_folds * t * sr
            filter_reads = sr * sc
        else:                                   # A resident: read once; B streamed once per column fold
            ifmap_reads = sr * sc
            filter_reads = col_folds * sr * t
        ofmap_writes = row_folds * t * sc       # partial sums written once per reduction fold
    return SystolicGemmResult(
        compute_cycles=folds * fold_cycles - 1,
        ifmap_sram_reads=ifmap_reads, filter_sram_reads=filter_reads, ofmap_sram_writes=ofmap_writes,
        row_folds=row_folds, col_folds=col_folds,
        mapping_efficiency=sr * sc / (rows * cols * folds), model="native")


def scalesim_available() -> bool:
    try:
        import scalesim  # noqa: F401
    except ImportError:
        return False
    return True


def _scalesim_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("scalesim")
    except PackageNotFoundError:
        return "unknown"


@functools.lru_cache(maxsize=4096)
def scalesim_gemm(rows, cols, dataflow, m, k, n) -> SystolicGemmResult:
    """Run SCALE-Sim's operand and compute modules for one GEMM layer."""
    _check(rows, cols, dataflow, m, k, n)
    try:
        from scalesim.compute.operand_matrix import operand_matrix
        from scalesim.compute.systolic_compute_is import systolic_compute_is
        from scalesim.compute.systolic_compute_os import systolic_compute_os
        from scalesim.compute.systolic_compute_ws import systolic_compute_ws
        from scalesim.layout_utils import layouts
        from scalesim.scale_config import scale_config
        from scalesim.topology_utils import topologies
    except ImportError as error:
        raise ImportError(f"SCALE-Sim is not installed; {SCALESIM_INSTALL_HINT}") from error
    compute_class = {"os": systolic_compute_os, "ws": systolic_compute_ws, "is": systolic_compute_is}[dataflow]
    sink = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="archexplorer-scalesim-") as directory, \
            contextlib.redirect_stdout(sink):
        topology_file = os.path.join(directory, "gemm.csv")
        with open(topology_file, "w") as handle:
            handle.write(f"Layer, M, N, K,\ngemm, {m}, {n}, {k},\n")
        config = scale_config()
        config.set_arr_dims(rows=rows, cols=cols)
        config.set_dataflow(dataflow)
        config.force_valid()
        topology = topologies()
        topology.load_arrays(topofile=topology_file, mnk_inputs=True)
        operands = operand_matrix()
        operands.set_params(config_obj=config, topoutil_obj=topology, layoututil_obj=layouts(), layer_id=0)
        _, ifmap = operands.get_ifmap_matrix()
        _, filters = operands.get_filter_matrix()
        _, ofmap = operands.get_ofmap_matrix()
        compute = compute_class()
        compute.set_params(config_obj=config, ifmap_op_mat=ifmap, filter_op_mat=filters, ofmap_op_mat=ofmap)
        compute.create_demand_matrices()
        demand_rows = compute.ofmap_demand_matrix.shape[0]
        return SystolicGemmResult(
            compute_cycles=int(demand_rows) - 1,
            ifmap_sram_reads=int(compute.get_ifmap_requests()),
            filter_sram_reads=int(compute.get_filter_requests()),
            ofmap_sram_writes=int(compute.get_ofmap_requests()),
            row_folds=int(compute.row_fold), col_folds=int(compute.col_fold),
            mapping_efficiency=float(compute.get_avg_mapping_efficiency()),
            model=f"scalesim-{_scalesim_version()}")


def fold_interval(rows, cols, dataflow, m, k, n, buffering="single") -> int:
    """Cycles between the starts of two successive folds."""
    _check(rows, cols, dataflow, m, k, n)
    sr, sc, t = spatio_temporal(dataflow, m, k, n)
    if buffering == "double":
        if dataflow == "os":
            raise ValueError("double buffering preloads a stationary operand; dataflow 'os' has none")
        return max(rows, t)
    if buffering != "single":
        raise ValueError(f"buffering must be one of {BUFFERINGS}, got {buffering!r}")
    return t + rows + cols - 2 if dataflow == "os" else 2 * rows + cols + t - 2


def double_buffered(result: SystolicGemmResult, rows, cols, dataflow, m, k, n) -> SystolicGemmResult:
    """``result`` (single-buffered, SCALE-Sim semantics) with the next stationary tile loaded during a fold."""
    single = fold_interval(rows, cols, dataflow, m, k, n)
    folds = result.row_folds * result.col_folds
    saved = (folds - 1) * (single - fold_interval(rows, cols, dataflow, m, k, n, "double"))
    return dataclasses.replace(result, compute_cycles=result.compute_cycles - saved,
                               model=f"{result.model}+double-buffered")


def gemm(rows, cols, dataflow, m, k, n, model="auto", buffering="single") -> SystolicGemmResult:
    """GEMM timing; ``model`` is ``auto`` (SCALE-Sim if installed), ``scalesim`` or ``native``."""
    if buffering not in BUFFERINGS:
        raise ValueError(f"buffering must be one of {BUFFERINGS}, got {buffering!r}")
    if model == "native":
        result = native_gemm(rows, cols, dataflow, m, k, n)
    elif model == "scalesim" or (model == "auto" and scalesim_available()):
        result = scalesim_gemm(rows, cols, dataflow, m, k, n)
    elif model == "auto":
        result = native_gemm(rows, cols, dataflow, m, k, n)
    else:
        raise ValueError(f"model must be auto, scalesim or native, got {model!r}")
    if buffering == "double":
        return double_buffered(result, rows, cols, dataflow, m, k, n)
    return result
