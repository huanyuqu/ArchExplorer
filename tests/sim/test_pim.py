"""In-bank PIM (``pim_unit``), DRAM groups, ``pim_offload``, hybrid-bonded DRAM and the PIM presets."""
import dataclasses
import math
import re

import pytest

from arch_explorer import devices
from arch_explorer.adf import MappingSpec, Param, WorkloadSpec, evaluate
from arch_explorer.adf.model import find, walk
from arch_explorer.archs.adf_seeds import dmc_adf, point
from arch_explorer.components.base import Kernel
from arch_explorer.components.pim import (BankTiming, activation_times, gemv_counts, gwrite_time, refresh_packed,
                                          refresh_time, row_results, tile_time)
from arch_explorer.devices.pim import BONDED_MAX_LAYERS, bonded_dram, pim_structure
from arch_explorer.mapping.base import adf_devices
from arch_explorer.simulator.events import build_simulator
from arch_explorer.structure.transforms import bond_memory, pim_transform

S = "unit test"
NEUPIMS_TIMING = BankTiming(trcd=14, trp=14, tras=34, trrd=6, tfaw=30, tccd=2, tcl=14, row_bytes=1024,
                            column_bytes=64)
TIMING_PARAMS = {"trcd": (14.0, "ns"), "trp": (14.0, "ns"), "tras": (34.0, "ns"), "trrd": (6.0, "ns"),
                 "tfaw": (30.0, "ns"), "tccd": (2.0, "ns"), "tcl": (14.0, "ns"), "row_bytes": (1024, "B"),
                 "column_bytes": (64, "B")}
TINY = {"name": "tiny", "hidden": 256, "heads": 4, "kv_heads": 4, "head_dim": 64, "intermediate": 512, "layers": 1,
        "vocab": 1000, "gated": False, "norm": "layernorm", "activation": "gelu", "source": S}


def pim_spec(**extra):
    values = {"ops_per_cycle": (32, "ops/cycle"), "clock_ghz": (1.0, "GHz"), "precision": ("fp16", "1"),
              "act_banks": (4, "count"), "row_buffer": ("dual", "1"), **extra}
    return point("pim", "pim_unit", values, source=S)


def pim_dmc(channels=2, banks=4, cores=2, **pim):
    """A DMC seed whose DRAM (with bank timing) is turned into DRAM-PIM by ``pim_transform``."""
    adf = dmc_adf(cores)
    dram = find(adf, "chip/dram")
    for key, (value, unit) in TIMING_PARAMS.items():
        dram.params[key] = Param(value, unit, S)
    return pim_transform(adf, "chip/dram", ["channel", "bank"], [channels, banks], pim_spec(**pim).to_dict()).adf


def decode(batch=2, seq=256, model=TINY, **extra):
    return WorkloadSpec("transformer", {"model": model, "phase": "decode", "batch": batch, "seq_len": seq, **extra})


# -- command timing ------------------------------------------------------------------------
def test_activation_windows_follow_trrd_and_tfaw():
    # One bank per command: tRRD apart, and the fifth activation waits for the tFAW window.
    assert activation_times(6, 1, 6, 30) == [0, 6, 12, 18, 30, 36]
    # Four banks per command (Newton G_ACT, NeuPIMs PIM_ACTIVATION): one command per tFAW.
    assert activation_times(3, 4, 6, 30) == [0, 30, 60]
    # All sixteen banks at once (AiM ACT16): one command, no window wait.
    assert activation_times(1, 16, 5.5, 21) == [0.0]


def test_tile_and_gwrite_latency_from_neupims_timing():
    # 32 banks, 4 per activation: last ACT at 7 x tFAW = 210 ns; 16 columns of 64 B at tCCD_L = 2 ns (16 MAC/ns
    # consume 32 fp16 in 2 ns as well); readout CL 14 + one 64 B burst; PRE at 272 ns; + tRP.
    assert tile_time(NEUPIMS_TIMING, 32, 4, 16.0, 2) == 210 + 14 + 32 + 16 + 14
    assert gwrite_time(NEUPIMS_TIMING) == 14 + 16 * 2 + 14
    # A slower datapath (8 MAC/ns) makes each column 4 ns.
    assert tile_time(NEUPIMS_TIMING, 32, 4, 8.0, 2) == 210 + 14 + 64 + 16 + 14
    # tRAS binds when the row work is short: one bank, one column.
    short = BankTiming(10, 10, 100, 1, 1, 1, 0, 64, 64)
    assert tile_time(short, 1, 1, 64.0, 2) == 100 + 10
    # NeuPIMs' COMP covers 16 elements (half a 64 B column): 32 commands, tCCD apart; four heads in a 1 KB row
    # are four dot products, read out one burst each.
    assert tile_time(NEUPIMS_TIMING, 32, 4, 16.0, 2, command_elements=16) == 210 + 14 + 32 * 2 + 16 + 14
    assert tile_time(NEUPIMS_TIMING, 32, 4, 16.0, 2, 16, readouts=4) == 210 + 14 + 32 * 2 + 14 + 4 * 2 + 14
    assert row_results(Kernel("mvm", (8, 1536, 128), "fp16"), 1024) == 4      # q . K^T: k = head_dim
    assert row_results(Kernel("mvm", (8, 128, 1536), "fp16"), 1024) == 1      # p . V: k = seq spans rows


def test_refresh_packs_uninterruptible_operations_between_refreshes():
    # Window 3640 ns (tREFI 3900 - tRFC 260): twelve 286 ns tiles fit (3432 ns), the 13th waits for the refresh.
    assert refresh_packed([(12, 286.0)], 260, 3900, 0.0) == 12 * 286
    assert refresh_packed([(13, 286.0)], 260, 3900, 0.0) == 3900 + 286
    assert refresh_packed([(36, 286.0)], 260, 3900, 0.0) == 2 * 3900 + 12 * 286
    # Started inside the refresh: wait for it to end. Started late in a window: the first operation waits too.
    assert refresh_packed([(1, 286.0)], 260, 3900, 3700.0) == 200 + 286
    assert refresh_packed([(1, 286.0)], 260, 3900, 3500.0) == 400 + 286
    # Runs share windows: two 60 ns GWRITEs, then tiles fill what is left (3520 ns -> 12 tiles).
    assert refresh_packed([(2, 60.0), (13, 286.0)], 260, 3900, 0.0) == 3900 + 286
    with pytest.raises(ValueError, match="does not fit"):
        refresh_packed([(1, 4000.0)], 260, 3900, 0.0)
    # Over many windows the phase mean approaches 12 tiles per tREFI, i.e. 325 ns per 286 ns tile (refresh plus
    # the 208 ns at each window's end that no tile fits into).
    mean = refresh_time([(1200, 286.0)], 260, 3900)
    assert 100 * 3900 - 3900 < mean < 100 * 3900 + 3900
    assert refresh_time([(1, 60.0)], 260, 3900) > 60


def test_pim_gemv_and_host_port_see_refresh():
    sim = build_simulator(pim_dmc())
    pim = sim.points["chip/dram/channel_0/dram_0/bank_0/pim"]
    kernel = Kernel("mvm", (8, 1024, 512), "fp16")
    plain = pim.gemv_time(kernel, 4)
    gwrites, tiles = gemv_counts(kernel, 4, 1024)
    adf = pim_dmc()
    banks = [node for _, node in walk(adf) if getattr(node, "component", None) == "dram" and "trcd" in node.params]
    for node in banks:
        node.params["trfc"], node.params["trefi"] = Param(260.0, "ns", S), Param(3900.0, "ns", S)
    refreshed = build_simulator(adf)
    pim_r = refreshed.points["chip/dram/channel_0/dram_0/bank_0/pim"]
    runs = [(gwrites, gwrite_time(NEUPIMS_TIMING)), (tiles, tile_time(NEUPIMS_TIMING, 4, 4, 16.0, 2))]
    assert pim_r.gemv_time(kernel, 4) == pytest.approx(refresh_time(runs, 260, 3900)) and pim_r.gemv_time(kernel, 4) > plain
    bank = "chip/dram/channel_0/dram_0/bank_0/dram_0_0"
    assert refreshed.points[bank].port.capacity == pytest.approx(sim.points[bank].port.capacity * (1 - 260 / 3900))
    del banks[0].params["trefi"]
    with pytest.raises(ValueError, match="both trfc and trefi"):
        build_simulator(adf)


def test_gemv_counts_match_neupims_algorithm_1():
    heads, seq, head_dim, banks, page = 8, 1536, 128, 32, 512        # page = 1 KB / 2 B
    embedding = heads * head_dim
    logits = gemv_counts(Kernel("mvm", (heads, seq, head_dim), "fp16"), banks, 1024)
    attend = gemv_counts(Kernel("mvm", (heads, head_dim, seq), "fp16"), banks, 1024)
    assert logits == (embedding // page, (seq // banks) * (embedding // page))
    assert attend == ((seq // page) * heads, (head_dim // banks) * (seq // page) * heads)


def test_pim_unit_runs_gemv_only_and_needs_bank_timing():
    sim = build_simulator(pim_dmc())
    pim = sim.points["chip/dram/channel_0/dram_0/bank_0/pim"]
    assert pim.bank.path == "chip/dram/channel_0/dram_0/bank_0/dram_0_0"
    kernel = Kernel("mvm", (1, 64, 512), "fp16")
    expected = gwrite_time(NEUPIMS_TIMING) + math.ceil(64 * 512 * 2 / (1024 * 4)) * tile_time(NEUPIMS_TIMING, 4, 4,
                                                                                                16.0, 2)
    assert pim.gemv_time(kernel, 4) == pytest.approx(expected)
    with pytest.raises(NotImplementedError):
        pim.gemv_time(Kernel("gemm", (1, 4, 4, 4), "fp16"), 4)
    with pytest.raises(NotImplementedError):
        pim.gemv_time(Kernel("mvm", (1, 4, 4), "bf16"), 4)
    untimed = pim_transform(dmc_adf(2), "chip/dram", "bank", 2, pim_spec().to_dict()).adf
    with pytest.raises(ValueError, match="DRAM timing"):
        build_simulator(untimed).points["chip/dram/bank_0/pim"].gemv_time(kernel, 2)


# -- DRAM groups and pim_transform -------------------------------------------------------------
def test_pim_transform_output_is_one_striped_main_memory():
    adf = pim_dmc(channels=2, banks=4)
    sim = build_simulator(adf)
    assert list(sim.memory_groups) == ["chip/dram"]
    group = sim.memory_groups["chip/dram"]
    assert len(group.parts) == 8 and len(group.pims) == 8
    assert group.capacity == 2**30
    assert adf_devices(adf) == [("chip", 1)]
    # Host streaming of a memory-bound GEMV sees the bandwidth of all banks together, as before the split.
    work = WorkloadSpec("operator", {"op": "mvm", "m": 2048, "k": 2048})
    for template in ("gsm_tiled", "dmc_partition"):
        params = {"weights": "stream"} if template == "dmc_partition" else {}
        split = evaluate(adf, work, MappingSpec(template, params)).latency_ns
        whole = evaluate(dmc_adf(2), work, MappingSpec(template, params)).latency_ns
        assert split == pytest.approx(whole, rel=0.02)


def test_direct_pim_structure_equals_pim_transform():
    adf = dmc_adf(2)
    spec = {"compute": pim_spec().to_dict(), "network": point("bank_bus", "bus", {"bandwidth": (32.0, "B/ns"),
                                                                                    "latency": (0.0, "ns")}).to_dict()}
    for levels, counts in ((["channel", "bank"], [3, 2]), ("bank", 4)):
        transformed = pim_transform(adf, "chip/dram", levels, counts, spec).adf
        direct = pim_structure(find(adf, "chip/dram"), levels if isinstance(levels, list) else [levels],
                               counts if isinstance(counts, list) else [counts], pim_spec(),
                               find(transformed, f"chip/dram/{'channel_0/dram_0/' if counts == [3, 2] else ''}"
                                                 "bank_0/bank_bus"))
        assert direct.to_dict() == find(transformed, "chip/dram").to_dict()


def test_pim_offload_places_attention_on_the_banks_end_to_end():
    adf = pim_dmc(channels=2, banks=4)
    for inner in ("gsm_tiled", "dmc_partition"):
        result = evaluate(adf, decode(batch=3), MappingSpec("pim_offload", {"inner": inner}))
        labels = {row["label"] for row in result.breakdown}
        assert {"LayerNorm+QKV", "PIM attention", "Projection+FFN"} <= labels
        pim_rows = [row for row in result.breakdown if row["task_type"] == "compute"
                    and row["operator"].endswith(("attn_qk", "attn_av"))]
        assert pim_rows and all(row["resource"] == "pim_unit" for row in pim_rows)
        softmax = [row for row in result.breakdown if row["operator"].endswith("softmax") and row["task_type"] == "compute"]
        assert all("pim_unit" not in row["resource"] for row in softmax)
    # Requests go round-robin to the channels: 3 requests on 2 channels -> channel 0 runs two of them.
    sim = build_simulator(adf)
    pim = sim.points["chip/dram/channel_0/dram_0/bank_0/pim"]
    one = pim.gemv_time(Kernel("mvm", (4, 256, 64), "fp16"), 4)
    two = pim.gemv_time(Kernel("mvm", (8, 256, 64), "fp16"), 4)
    result = evaluate(adf, decode(batch=3), MappingSpec("pim_offload", {}))
    qk = [row for row in result.breakdown if row["operator"] == "l0.attn_qk" and row["task_type"] == "compute"]
    assert qk[0]["tasks"] == 2                      # one task per busy channel
    assert qk[0]["end_ns"] - qk[0]["start_ns"] == pytest.approx(two)
    assert one < two


def test_host_attention_and_sub_batch_interleaving():
    adf = pim_dmc(channels=2, banks=4)
    host = evaluate(adf, decode(batch=4), MappingSpec("pim_offload", {"attention": "inner"}))
    assert "Attention" in {row["label"] for row in host.breakdown}
    assert all("pim_unit" not in row["resource"] for row in host.breakdown)
    split = evaluate(adf, decode(batch=4), MappingSpec("pim_offload", {"sub_batches": 2}))
    operators = [row["operator"] for row in split.breakdown]
    assert any(o.startswith("sb0.") for o in operators) and any(o.startswith("sb1.") for o in operators)
    # Weights are shared: the QKV weight is stored once although both halves read it.
    sim_peak = split.details["stats"]["memory_peak"]["chip/dram"]
    whole = evaluate(adf, decode(batch=4), MappingSpec("pim_offload", {}))
    assert sim_peak <= whole.details["stats"]["memory_peak"]["chip/dram"] + 1
    # The PIM attention of half 0 overlaps host work of half 1.
    spans = {}
    for row in split.breakdown:
        spans.setdefault(row["operator"], []).append(row)
    qk0 = min(r["start_ns"] for r in spans["sb0.l0.attn_qk"])
    qkv1_end = max(r["end_ns"] for r in spans["sb1.l0.qkv"])
    assert qk0 < qkv1_end


def test_pim_offload_is_only_for_pim_devices_and_decode():
    from arch_explorer.adf.registry import MAPPING_TEMPLATES
    template = MAPPING_TEMPLATES.get("pim_offload")
    assert not template.applicable(dmc_adf(2), decode())
    assert not template.applicable(pim_dmc(), WorkloadSpec("transformer", {"model": TINY, "phase": "prefill"}))
    assert template.applicable(pim_dmc(), decode())


# -- hybrid-bonded DRAM ------------------------------------------------------------------------------
def test_bonded_dram_density_and_layers():
    shard = bonded_dram(2.0, layers=3)
    assert shard.value("capacity") == int(25.6 * 2**20 * 2.0 * 3)
    assert shard.value("bandwidth") == pytest.approx(40.96)
    assert shard.value("technology") == "bonded"
    with pytest.raises(ValueError):
        bonded_dram(1.0, layers=BONDED_MAX_LAYERS + 1)
    with pytest.raises(ValueError):
        bonded_dram(0)


def test_bond_memory_gives_every_core_a_shard_and_dmc_streams_from_it():
    adf = bond_memory(dmc_adf(4), "chip/core", bonded_dram(1.0).to_dict()).adf
    sim = build_simulator(adf)
    shards = [p.path for p in sim.find_points(component="dram") if p.technology == "bonded"]
    assert shards == [f"chip/core[{i}]/bonded" for i in range(4)]
    assert adf_devices(adf) == [("chip", 1)] and not sim.memory_groups
    work = WorkloadSpec("operator", {"op": "mvm", "m": 1024, "k": 1024})
    stream = evaluate(adf, work, MappingSpec("dmc_partition", {"weights": "stream"}))
    bonded = evaluate(adf, work, MappingSpec("dmc_partition", {"weights": "bonded"}))
    assert bonded.latency_ns < stream.latency_ns
    assert all(bonded.details["stats"]["memory_peak"].get(shard, 0) > 0 for shard in shards)
    with pytest.raises(ValueError):
        evaluate(dmc_adf(4), work, MappingSpec("dmc_partition", {"weights": "bonded"}))


# -- presets -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,options", [("neupims", {"channels": 4, "banks": 8}), ("hsndi", {"cores": 4, "channels": 2})])
def test_pim_presets_have_tagged_sources(name, options):
    adf = devices.get(name, **options)
    from arch_explorer.adf.model import walk
    for path, node in walk(adf):
        for key, param in getattr(node, "params", {}).items():
            source = re.sub(r"^(split\(\d+\) of )+", "", param.source)    # bank parts keep the DRAM's source
            assert source.startswith("[") or source.startswith("workload"), (path, key, param.source)


def test_neupims_preset_values_and_unit_choice():
    adf = devices.neupims(128, channels=4, banks=8)
    sim = build_simulator(adf)
    bank = sim.points["neupims/hbm/channel_0/hbm_0/bank_0/hbm_0_0"]
    assert BankTiming.of(bank) == dataclasses.replace(NEUPIMS_TIMING, refresh=(260.0, 3900.0))
    assert bank.port.capacity == pytest.approx(4 * 32.0 / 32 * (1 - 260 / 3900))
    arrays = [model for model in sim.points.values() if model.component.name == "systolic_array"]
    assert len(arrays) == 8 and all(model.buffering == "double" for model in arrays)
    assert sim.points["neupims/hbm/channel_0/hbm_0/bank_0/pim"].command_elements == 16
    single = devices.neupims(128, channels=4, banks=8)
    del find(single, "neupims/sa").params["stationary_buffering"]
    # Synthetic generic variant: restore the default vector matrix capability
    # to test fastest-unit selection independently of the NeuPIMs restriction.
    del find(single, "neupims/vu").params["matrix_kernels"]
    restricted_single = devices.neupims(128, channels=4, banks=8)
    del find(restricted_single, "neupims/sa").params["stationary_buffering"]
    for arch, batch, gemm_unit in ((adf, 12, "systolic_array"), (adf, 2, "systolic_array"),
                                   (restricted_single, 2, "systolic_array"),
                                   (single, 2, "vector_unit")):
        # Select the fastest eligible unit. Only the synthetic variant allows
        # vector GEMM, which wins against single-buffered arrays at batch 2.
        # The preset forbids vector GEMM even without double buffering.
        result = evaluate(arch, decode(batch=batch, seq=512, model="gpt3-6.7b"), MappingSpec("pim_offload", {"tp": 4}))
        resources = {row["operator"]: row["resource"] for row in result.breakdown if row["task_type"] == "compute"}
        assert gemm_unit in resources["l0.qkv"] and "vector_unit" in resources["l0.softmax"]
        assert resources["l0.attn_qk"] == "pim_unit"
        assert resources["l0.attn_av"] == "pim_unit"


def test_hsndi_preset_runs_bf16_decode_with_dmc_cores():
    adf = devices.hsndi(cores=4, channels=2)
    result = evaluate(adf, decode(batch=1, seq=256, dtype="bf16"),
                      MappingSpec("pim_offload", {"inner": "dmc_partition", "inner.weights": "stream"}))
    assert {"LayerNorm+QKV", "PIM attention", "Projection+FFN"} <= {row["label"] for row in result.breakdown}
    with pytest.raises(ValueError):
        devices.hsndi(cores=0)
