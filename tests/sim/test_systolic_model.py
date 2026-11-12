import itertools

import pytest

from arch_explorer.components.systolic_model import (DATAFLOWS, gemm, native_gemm, scalesim_available,
                                                     scalesim_gemm)

COUNTERS = ("compute_cycles", "ifmap_sram_reads", "filter_sram_reads", "ofmap_sram_writes",
            "row_folds", "col_folds")


def test_single_fold_closed_forms():
    # 4x4 array, 4x4x4 GEMM: one fold in every dataflow.
    os_ = native_gemm(4, 4, "os", 4, 4, 4)
    assert os_.compute_cycles == (4 + 4 + 4 - 2) - 1          # T + R + C - 2 rows, last index
    assert (os_.ifmap_sram_reads, os_.filter_sram_reads) == (16, 16)
    assert os_.ofmap_sram_writes == 16 + (4 + 4)              # SCALE-Sim's per-fold rows+cols term
    ws = native_gemm(4, 4, "ws", 4, 4, 4)
    assert ws.compute_cycles == (2 * 4 + 4 + 4 - 2) - 1       # load R, stream T, skew R-1 and C-1
    assert (ws.ifmap_sram_reads, ws.filter_sram_reads, ws.ofmap_sram_writes) == (16, 16, 16)
    assert native_gemm(4, 4, "is", 4, 4, 4).compute_cycles == ws.compute_cycles
    assert os_.mapping_efficiency == ws.mapping_efficiency == 1.0


def test_folds_partial_tiles_and_stationary_reuse():
    # os on 4x8 array: m=10 -> 3 row folds, n=13 -> 2 column folds, T=k=7.
    result = native_gemm(4, 8, "os", 10, 7, 13)
    assert (result.row_folds, result.col_folds) == (3, 2)
    assert result.compute_cycles == 6 * (7 + 4 + 8 - 2) - 1
    assert result.ifmap_sram_reads == 2 * 10 * 7               # A re-read once per column fold
    assert result.filter_sram_reads == 3 * 7 * 13              # B re-read once per row fold
    assert result.mapping_efficiency == pytest.approx(10 * 13 / (32 * 6))
    ws = native_gemm(4, 8, "ws", 10, 7, 13)                   # B (7x13) resident: read exactly once
    assert ws.filter_sram_reads == 7 * 13 and ws.ofmap_sram_writes == 2 * 10 * 13
    is_ = native_gemm(4, 8, "is", 10, 7, 13)                  # A (10x7) resident: read exactly once
    assert is_.ifmap_sram_reads == 10 * 7 and is_.filter_sram_reads == 2 * 7 * 13


def test_longer_streams_take_proportionally_longer():
    base = native_gemm(16, 16, "os", 16, 64, 16)
    assert native_gemm(16, 16, "os", 16, 128, 16).compute_cycles - base.compute_cycles == 64
    assert native_gemm(16, 16, "ws", 128, 16, 16).compute_cycles > native_gemm(16, 16, "ws", 64, 16, 16).compute_cycles


def test_input_validation_and_model_selection():
    with pytest.raises(ValueError, match="dataflow"):
        native_gemm(4, 4, "rs", 4, 4, 4)
    with pytest.raises(ValueError, match="positive"):
        native_gemm(4, 4, "os", 0, 4, 4)
    with pytest.raises(ValueError, match="model"):
        gemm(4, 4, "os", 4, 4, 4, model="other")
    assert gemm(4, 4, "os", 4, 4, 4, model="native").model == "native"


def test_double_buffering_overlaps_the_next_stationary_load():
    for dataflow, (m, k, n) in (("ws", (2, 40, 20)), ("ws", (30, 40, 20)), ("is", (40, 20, 2))):
        single = native_gemm(8, 8, dataflow, m, k, n)
        double = gemm(8, 8, dataflow, m, k, n, model="native", buffering="double")
        t = m if dataflow == "ws" else n
        folds = single.row_folds * single.col_folds
        assert folds == 15
        # First fold as SCALE-Sim (load 8, stream T, skew 7 + 7), then one fold every max(rows, T) cycles.
        assert double.compute_cycles == (2 * 8 + 8 + t - 2) - 1 + (folds - 1) * max(8, t)
        assert double.compute_cycles < single.compute_cycles
        assert (double.ifmap_sram_reads, double.filter_sram_reads, double.ofmap_sram_writes) == (
            single.ifmap_sram_reads, single.filter_sram_reads, single.ofmap_sram_writes)
        assert double.model == "native+double-buffered"
    # One fold: nothing to overlap.
    assert gemm(8, 8, "ws", 4, 8, 8, model="native", buffering="double").compute_cycles == \
        native_gemm(8, 8, "ws", 4, 8, 8).compute_cycles
    # The default is SCALE-Sim's single buffering.
    assert gemm(8, 8, "ws", 2, 40, 20, model="native") == native_gemm(8, 8, "ws", 2, 40, 20)
    with pytest.raises(ValueError, match="stationary"):
        gemm(8, 8, "os", 2, 40, 20, model="native", buffering="double")
    with pytest.raises(ValueError, match="buffering"):
        gemm(8, 8, "ws", 2, 40, 20, model="native", buffering="triple")


def test_systolic_array_buffering_parameter_and_batched_chain():
    from arch_explorer.adf import Param, SpacePoint
    from arch_explorer.adf.registry import COMPONENTS
    from arch_explorer.components.base import BuildContext, Kernel, Provenance

    def array(dataflow="ws", **extra):
        params = {"rows": 8, "cols": 8, "clock_ghz": 1.0, "dataflow": dataflow, "precision": "fp16", **extra}
        point = SpacePoint("pe", "systolic_array", {k: Param(v, "1", "test") for k, v in params.items()})
        context = BuildContext("pe", "pe", (), Provenance(), options={"systolic_model": "native"})
        return COMPONENTS.get("systolic_array").build(point, context)

    kernel = Kernel("gemm", (3, 2, 40, 20), "fp16")
    single, double = array(), array(stationary_buffering="double")
    assert single.kernel_time(kernel) == 3 * native_gemm(8, 8, "ws", 2, 40, 20).compute_cycles
    # Three GEMMs of 15 folds each form one chain of 45 folds, 8 cycles apart after the first.
    assert double.kernel_time(kernel) == (2 * 8 + 8 + 2 - 2) - 1 + 44 * 8
    with pytest.raises(ValueError, match="ws or is"):
        array("os", stationary_buffering="double")
    with pytest.raises(ValueError, match="stationary_buffering"):
        array(stationary_buffering="shadow")


@pytest.mark.skipif(not scalesim_available(), reason="SCALE-Sim (pip install scalesim) not installed")
def test_native_model_equals_scalesim_exactly():
    shapes = [(1, 1, 1), (4, 4, 4), (10, 7, 13), (3, 5, 9), (1, 9, 17), (17, 1, 3), (33, 20, 15)]
    for dataflow, (rows, cols), (m, k, n) in itertools.product(DATAFLOWS, [(4, 4), (4, 8), (8, 4), (3, 5)], shapes):
        native, reference = native_gemm(rows, cols, dataflow, m, k, n), scalesim_gemm(rows, cols, dataflow, m, k, n)
        for counter in COUNTERS:
            assert getattr(native, counter) == getattr(reference, counter), (dataflow, rows, cols, m, k, n, counter)
        assert native.mapping_efficiency == pytest.approx(reference.mapping_efficiency, abs=1e-12)
    assert reference.model.startswith("scalesim-")
    assert gemm(4, 4, "os", 4, 4, 4).model.startswith("scalesim-")
