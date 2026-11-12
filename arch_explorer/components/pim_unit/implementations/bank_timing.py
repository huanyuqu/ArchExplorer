"""In-bank processing-in-memory: ``pim_unit`` and the DRAM command timing it runs on.

A ``pim_unit`` sits in one DRAM bank (``pim_transform`` puts one next to every
bank part ``dram_i_j``) and multiplies the matrix stored in that bank's rows
by a vector held in the channel's global buffer. All PIM units of a channel
work in lockstep: one command drives every bank of the channel (Newton,
MICRO'20; AiM, ISSCC'22; NeuPIMs, ASPLOS'24). Supported kernel: ``mvm``
(GEMV, ``(batch, m, k)``: ``batch`` matrices of ``m x k`` each times one
vector of ``k``). Decoder attention maps onto it as ``q . K^T`` (matrix
``K``, ``seq x head_dim``) and ``p . V`` (matrix ``V^T``, ``head_dim x
seq``). GEMM, softmax and other vector kernels are not supported: the bank
units have only multipliers and an adder tree.

The latency of one GEMV on a channel of ``B`` banks follows the command
sequence (NeuPIMs, Algorithm 1, with the latencies of its two steps derived
from the bank timing instead of being inputs):

* ``GWRITE``: copy one row of the vector into the global buffer; one bank,
  ``max(tRCD + n_col * tCCD, tRAS) + tRP``. A GEMV needs
  ``ceil(batch * k * e / row_bytes)`` of them.
* tile: one row in each of the ``B`` banks. Activate the banks
  ``act_banks`` at a time; commands are ``tRRD`` apart and at most four bank
  activations fall in any ``tFAW`` window (JEDEC four-activate rule). Then
  ``tRCD``, ``row_bytes / (c * e)`` MAC commands each taking
  ``max(tCCD, c / MACs per ns)``, the result readout
  ``tCL + r * ceil(B * e / column_bytes) * tCCD``, and the all-bank precharge
  ``tRP``, no earlier than ``tRAS`` after the last activation. A GEMV needs
  ``ceil(batch * m * k * e / (row_bytes * B))`` tiles (rows fully packed).

``c`` is ``command_elements``, the row elements one MAC command covers
(default one column, ``column_bytes / e``; NeuPIMs' ``pim_comp_coverage``).
``r`` is the number of dot products in one bank row, ``row_bytes / (k * e)``
when a row holds several whole ``k``-vectors (NeuPIMs reads one result per
head of the row, ``num_readres``), otherwise 1. ``n_col = row_bytes /
column_bytes``; ``e`` is the element size. The tile loop is serial: the next
activation waits for the precharge.

Refresh, when the bank gives ``trfc`` and ``trefi``: every ``tREFI`` the
channel is refreshed for ``tRFC`` and no command runs. A PIM operation (one
``GWRITE`` or one tile) accumulates over a whole open row and cannot be
interrupted, so the controller starts it only if it ends before the next
refresh; otherwise it waits for the refresh to finish (Newton, MICRO'20,
§IV "delay the Newton operation if a refresh request would mature within the
operation's latency"; NeuPIMs §5.2 ``PIM_HEADER``, which tells the controller
the GEMV size so that it can schedule the commands around refresh; NewtonSim
``CanMeetRefreshDeadline``). The operations of a GEMV (its ``GWRITE``\\ s, then
its tiles) are packed into the ``tREFI - tRFC`` windows between refreshes.
The phase of the refresh schedule at the GEMV's start is unknown to the
static task duration, so the GEMV time is the mean over evenly spaced start
phases (``REFRESH_PHASES``). Command-bus conflicts with normal accesses are not
modeled.

``row_buffer`` says whether normal (NPU/host) accesses to the bank proceed
during PIM work. ``dual``: the bank has a separate PIM row buffer (NeuPIMs),
so they do. ``shared`` (default; Newton, AiM): PIM work holds the bank, so
the template charges the bank port for the full PIM duration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel, Kernel, dtype_bytes

ROW_BUFFERS = ("shared", "dual")

# Optional ``dram`` parameters that ``pim_unit`` requires on its bank.
DRAM_TIMING = {
    "trcd": ParamSpec("ns", required=False, doc="ACT to column command (PIM: read-to-read tRCDRD)"),
    "trp": ParamSpec("ns", required=False, doc="precharge time"),
    "tras": ParamSpec("ns", required=False, doc="minimum ACT to PRE"),
    "trrd": ParamSpec("ns", required=False, doc="ACT to ACT of different banks (same bank group)"),
    "tfaw": ParamSpec("ns", required=False, doc="four-activate window"),
    "tccd": ParamSpec("ns", required=False, doc="column to column command, same bank group"),
    "tcl": ParamSpec("ns", required=False, doc="CAS (read) latency"),
    "row_bytes": ParamSpec("B", "int", required=False, doc="row (page) size of one bank"),
    "column_bytes": ParamSpec("B", "int", required=False, doc="bytes one column command reads from one bank"),
}

# Optional ``dram`` refresh parameters (both or neither); see ``refresh_of``.
DRAM_REFRESH = {
    "trfc": ParamSpec("ns", required=False, doc="refresh cycle time: no access during it, once per trefi"),
    "trefi": ParamSpec("ns", required=False, doc="average refresh interval"),
}

REFRESH_PHASES = 64


def refresh_of(dram):
    """``(trfc, trefi)`` of a ``dram`` model, or ``None`` when it declares no refresh."""
    given = [key for key in DRAM_REFRESH if key in dram.point.params]
    if not given:
        return None
    if len(given) != len(DRAM_REFRESH):
        raise ValueError(f"{dram.type_path}: give both trfc and trefi or neither")
    trfc, trefi = dram.param("trfc"), dram.param("trefi")
    for value in (trfc, trefi):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{dram.type_path}: trfc and trefi must be numbers")
    if not 0 < trfc < trefi:
        raise ValueError(f"{dram.type_path}: refresh needs 0 < trfc < trefi")
    return float(trfc), float(trefi)


@dataclass(frozen=True)
class BankTiming:
    trcd: float
    trp: float
    tras: float
    trrd: float
    tfaw: float
    tccd: float
    tcl: float
    row_bytes: int
    column_bytes: int
    refresh: tuple = None       # (trfc, trefi) or None

    @classmethod
    def of(cls, dram):
        """Timing of a ``dram`` model (bank part); every ``DRAM_TIMING`` field must be given."""
        missing = [key for key in DRAM_TIMING if key not in dram.point.params]
        if missing:
            raise ValueError(f"{dram.type_path}: a pim_unit bank needs DRAM timing parameters {missing}")
        values = {key: dram.param(key) for key in DRAM_TIMING}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{dram.type_path}.{key} must be a nonnegative number")
        if values["row_bytes"] <= 0 or values["column_bytes"] <= 0 or values["row_bytes"] % values["column_bytes"]:
            raise ValueError(f"{dram.type_path}: row_bytes must be a positive multiple of column_bytes")
        return cls(**values, refresh=refresh_of(dram))

    @property
    def columns(self) -> int:
        return self.row_bytes // self.column_bytes


def refresh_packed(runs, trfc: float, trefi: float, phase: float) -> float:
    """ns to run ``runs`` (``(count, ns)`` pairs of uninterruptible operations, in order) under refresh.

    The refresh cycle is a window of ``trefi - trfc`` followed by the refresh;
    ``phase`` is the position in that cycle at the start. An operation that
    would not end inside the current window waits for the next one.
    """
    window = trefi - trfc
    cycles, pos = (1, 0.0) if phase >= window else (0, phase)
    for count, duration in runs:
        if count <= 0:
            continue
        if duration > window:
            raise ValueError(f"a PIM operation of {duration} ns does not fit between refreshes ({window} ns)")
        fit = min(count, int((window - pos) / duration + 1e-9)) if duration > 0 else count
        pos += fit * duration
        count -= fit
        if count:
            per_window = int(window / duration + 1e-9)
            full, rest = divmod(count, per_window)
            cycles += 1 + full - (rest == 0)
            pos = (rest or per_window) * duration
    return cycles * trefi + pos - phase


def refresh_time(runs, trfc: float, trefi: float, phases: int = REFRESH_PHASES) -> float:
    """``refresh_packed`` averaged over ``phases`` evenly spaced start phases of the refresh cycle."""
    return sum(refresh_packed(runs, trfc, trefi, (i + 0.5) * trefi / phases) for i in range(phases)) / phases


def activation_times(commands: int, banks_per_command: int, trrd: float, tfaw: float) -> list:
    """Issue times of ``commands`` activate commands, each opening ``banks_per_command`` banks.

    Commands are at least ``trrd`` apart and any ``tfaw`` window holds at most
    four bank activations (a command opening more than four banks fills a
    window on its own).
    """
    times = []
    for i in range(commands):
        t = times[-1] + trrd if times else 0.0
        opened = banks_per_command
        for j in range(i - 1, -1, -1):
            opened += banks_per_command
            if opened > 4:
                t = max(t, times[j] + tfaw)
                break
        times.append(t)
    return times


def gwrite_time(timing: BankTiming) -> float:
    """One row of the input vector into the channel's global buffer."""
    return max(timing.trcd + timing.columns * timing.tccd, timing.tras) + timing.trp


def tile_time(timing: BankTiming, banks: int, act_banks: int, macs_per_ns: float, element_bytes: int,
              command_elements: int = None, readouts: int = 1, ranks: int = 1, tick: float = 1.) -> float:
    """One row processed in each of ``banks`` banks in lockstep (ACT ... MAC commands ... readout ... PRE).

    A MAC command covers ``command_elements`` row elements (default: one
    column); ``readouts`` results per bank are read out after the row.
    """
    acts = activation_times(math.ceil(banks / act_banks), act_banks, timing.trrd, timing.tfaw)
    if ranks != 1:
        from arch_explorer.components.pim_commands import rank_activations
        acts = [t for t, _ in rank_activations(banks, act_banks, ranks, timing.trrd, timing.tfaw, tick)]
    elements = command_elements or timing.column_bytes // element_bytes
    commands = math.ceil(timing.row_bytes / element_bytes / elements)
    command = max(timing.tccd, elements / macs_per_ns)
    readout = timing.tcl + readouts * math.ceil(banks * element_bytes / timing.column_bytes) * timing.tccd
    precharge = max(acts[-1] + timing.trcd + commands * command + readout, acts[-1] + timing.tras)
    return precharge + timing.trp


def row_results(kernel: Kernel, row_bytes: int) -> int:
    """Dot products one bank row holds: rows are packed with whole ``k``-vectors when ``k`` is shorter than a row."""
    k = kernel.shape[2]
    row_elements = row_bytes // dtype_bytes(kernel.precision)
    return max(1, row_elements // k)


def gemv_counts(kernel: Kernel, banks: int, row_bytes: int):
    """``(gwrites, tiles)`` of a GEMV ``(batch, m, k)`` on ``banks`` banks."""
    batch, m, k = kernel.shape
    e = dtype_bytes(kernel.precision)
    return math.ceil(batch * k * e / row_bytes), math.ceil(batch * m * k * e / (row_bytes * banks))


class PimUnitModel(ComputeModel):
    """The PIM unit of one bank; ``bank`` (the bank's DRAM model) is linked by the simulator builder."""

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.bank = None
        self.ops_per_cycle = self.positive("ops_per_cycle")
        self.clock = self.positive("clock_ghz")
        self.act_banks = self.positive("act_banks", 1)
        if not float(self.act_banks).is_integer():
            raise ValueError(f"{self.type_path}.act_banks must be an integer")
        self.row_buffer = self.param("row_buffer", "shared")
        if self.row_buffer not in ROW_BUFFERS:
            raise ValueError(f"{self.type_path}.row_buffer must be one of {ROW_BUFFERS}")
        self.command_elements = self.positive("command_elements", None)
        if self.command_elements is not None:
            if not float(self.command_elements).is_integer():
                raise ValueError(f"{self.type_path}.command_elements must be an integer")
            self.command_elements = int(self.command_elements)
        self.ranks = self.positive("ranks", 1)
        if not float(self.ranks).is_integer():
            raise ValueError("pim_unit.ranks must be an integer")
        self.ranks = int(self.ranks)
        for key, default, choices in (
                ("refresh_mode", "mean", ("mean", "rank_staggered")),
                ("completion_mode", "atomic", ("atomic", "commands")),
                ("attention_pipeline", "off", ("off", "heads")),
                ("write_response", "physical", ("physical", "early"))):
            value = self.param(key, default)
            if value not in choices:
                raise ValueError(f"pim_unit.{key} must be one of {choices}")
            setattr(self, key, value)
        self.tccd_s = self.positive("tccd_s", 1.)
        self.burst = self.positive("burst_ns", 2.)
        self.trtp = self.positive("trtp", 5.)

    @property
    def command_mapping(self):
        return (self.refresh_mode != "mean" or self.completion_mode != "atomic"
                or self.attention_pipeline != "off" or self.write_response != "physical")

    @property
    def macs_per_ns(self):
        return self.ops_per_cycle / 2 * self.clock

    def timing(self) -> BankTiming:
        if self.bank is None:
            raise ValueError(f"{self.path}: a pim_unit needs exactly one sibling dram (its bank)")
        return BankTiming.of(self.bank)

    def gemv_time(self, kernel: Kernel, banks: int) -> float:
        """ns for a GEMV run in lockstep on ``banks`` banks (this unit's channel), matrices already in the banks."""
        if kernel.op != "mvm":
            raise NotImplementedError(f"{self.type_path}: pim_unit runs mvm (GEMV) only, not {kernel.op}")
        self.precision(kernel)
        timing = self.timing()
        gwrites, tiles = gemv_counts(kernel, banks, timing.row_bytes)
        e = dtype_bytes(kernel.precision)
        tile = tile_time(timing, banks, int(self.act_banks), self.macs_per_ns, e, self.command_elements,
                         row_results(kernel, timing.row_bytes), self.ranks, 1/self.clock)
        runs = [(gwrites, gwrite_time(timing)), (tiles, tile)]
        if timing.refresh is None:
            return sum(count * duration for count, duration in runs)
        return refresh_time(runs, *timing.refresh)

    def kernel_time(self, kernel: Kernel) -> float:
        return self.gemv_time(kernel, 1)


class PimUnit(ComponentType):
    from arch_explorer.components.banked_memory import assemble_banked_memory
    assembly_hooks = (assemble_banked_memory,)
    execution_modes = ("demand", "stateful")
    execution_profiles = ("neupims-f299af3-dual-buffer",)
    name = "pim_unit"
    role = "compute"
    params = {
        "ops_per_cycle": ParamSpec("ops/cycle", doc="multiply-accumulate ops per cycle of one bank unit (MAC = 2)"),
        "clock_ghz": ParamSpec("GHz", doc="clock of the MAC datapath"),
        "precision": ParamSpec("1", "str", doc="operand precision"),
        "act_banks": ParamSpec("count", "int", required=False,
                               doc="banks one PIM activate command opens (default 1)"),
        "row_buffer": ParamSpec("1", "str", required=False, domain=list(ROW_BUFFERS),
                                doc="'dual': separate PIM row buffer, normal accesses proceed; default 'shared'"),
        "command_elements": ParamSpec("count", "int", required=False,
                                      doc="row elements one MAC command covers (default: one DRAM column)"),
        "ranks": ParamSpec("count", "int", required=False, doc="independent activation windows per channel; default 1"),
        "refresh_mode": ParamSpec("1", "str", required=False, domain=["mean", "rank_staggered"],
                                  doc="default mean; rank_staggered uses absolute simulation time in pim_offload"),
        "completion_mode": ParamSpec("1", "str", required=False, domain=["atomic", "commands"],
                                     doc="default atomic; commands separates callback and command-specific readiness"),
        "attention_pipeline": ParamSpec("1", "str", required=False, domain=["off", "heads"],
                                        doc="default off; heads overlaps softmax and models AV partial ADD"),
        "write_response": ParamSpec("1", "str", required=False, domain=["physical", "early"],
                                    doc="default physical; early is NewtonSim accept+1 tick, attention writes only"),
        "tccd_s": ParamSpec("ns", required=False, doc="short column delay for simulator GWRITE; default 1"),
        "burst_ns": ParamSpec("ns", required=False, doc="simulator read burst; default 2"),
        "trtp": ParamSpec("ns", required=False, doc="simulator read to PIM PRE; default 5"),
    }
    conventions = Conventions(time_base="ns", arbitration="fifo", pipelined=False, includes_startup=True)

    def build(self, point, context):
        return PimUnitModel(self, point, context)


__all__ = ["BankTiming", "DRAM_REFRESH", "DRAM_TIMING", "PimUnit", "activation_times", "gemv_counts", "gwrite_time",
           "refresh_of", "refresh_packed", "refresh_time", "row_results", "tile_time"]
