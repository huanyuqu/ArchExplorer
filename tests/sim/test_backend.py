from non_gpu_fixture import shared_adf as _shared_adf

def shared_adf(num_cores=4):
    return _shared_adf(num_cores, dram_capacity_bytes=2**40)

"""The ``events`` backend end to end: device presets, templates, workloads and mapping search."""
import json

import pytest

from arch_explorer import devices
from arch_explorer.adf import MappingSpec, Param, SpaceMatrix, SpacePoint, WorkloadSpec, evaluate
from arch_explorer.mapping import best_mapping, mapping_space
from arch_explorer.simulator.events import CapacityError, label_latency
from arch_explorer.workload.llm import MODELS, transformer_workload

S = "unit test"


def matmul(batch, m, k, n, **extra):
    return WorkloadSpec("operator", {"op": "matmul", "batch": batch, "m": m, "k": k, "n": n, **extra})


def dmc_chip(cores=4, sram_bytes=8 * 2**20):
    p = lambda value, unit: Param(value, unit, S)
    core = SpaceMatrix("core", [
        SpacePoint("pe", "systolic_array", {"rows": p(16, "count"), "cols": p(16, "count"), "clock_ghz": p(1.0, "GHz"),
                                            "dataflow": p("ws", "1"), "precision": p("fp16", "1")}),
        SpacePoint("vpu", "vector_unit", {"lanes": p(32, "count"), "sfu_lanes": p(8, "count"),
                                          "clock_ghz": p(1.0, "GHz"), "precision": p("fp16", "1")}),
        SpacePoint("sram", "sram", {"capacity": p(sram_bytes, "B"), "bandwidth": p(128.0, "B/ns"),
                                    "latency": p(1.0, "ns")})], repeat=cores)
    dram = SpacePoint("dram", "dram", {"technology": p("ddr", "1"), "capacity": p(2**32, "B"),
                                       "bandwidth": p(64.0, "B/ns"), "latency": p(80.0, "ns")})
    noc = SpacePoint("noc", "mesh2d", {"rows": p(1, "count"), "cols": p(cores + 1, "count"),
                                       "link_bandwidth": p(32.0, "B/ns"), "hop_latency": p(1.0, "ns")})
    return SpaceMatrix("chip", [core, dram], [noc])





def test_softmax_is_bound_by_explicit_memory_bandwidth():
    adf = shared_adf()
    next(n for n in adf.elements if n.name == "dram").params["bandwidth"] = Param(1.0, "B/ns", "synthetic slow memory")
    result = evaluate(adf, WorkloadSpec("operator", {"op": "softmax", "rows": 1024, "cols": 256}),
                      MappingSpec("gsm_tiled", {"rows_per_cta": 4}))
    effective = 2 * 1024 * 256 * 2 / result.latency_ns
    assert 0.9 < effective <= 1.0
    assert result.utilization["chip/dram:port"] > 0.9



@pytest.mark.parametrize("kind", ["shared", "distributed"])
def test_non_gpu_architectures_have_complete_provenance(kind):
    adf = shared_adf() if kind == "shared" else dmc_chip()
    mapping = MappingSpec("gsm_tiled" if kind == "shared" else "dmc_partition", {})
    result = evaluate(adf, matmul(2, 16, 32, 32), mapping)
    assert result.provenance and all(row["source"] for row in result.provenance)
    consumed = {(row["path"].split("/")[-1], row["param"]) for row in result.provenance}
    assert ("dram", "bandwidth") in consumed
    assert any(param in ("lanes", "rows") for _, param in consumed)



def test_four_non_gpu_ranks_prefill_breakdown_and_collectives():
    chip = shared_adf(1)
    chip.repeat = 4
    link = SpacePoint("sw", "crossbar", {"port_bandwidth": Param(50.0, "B/ns", S), "latency": Param(2.0, "ns", S)})
    adf = SpaceMatrix("server", [chip], [link])
    model = dict(hidden=64, heads=4, kv_heads=2, head_dim=16, intermediate=128, layers=1,
                 vocab=64, gated=True, norm="rmsnorm", activation="silu")
    workload = WorkloadSpec("transformer", {"model": model, "phase": "prefill", "batch": 1, "seq_len": 4})
    results = {algorithm: evaluate(adf, workload, MappingSpec("tensor_parallel",
               {"tp": 4, "collective": algorithm, "inner": "gsm_tiled"}))
               for algorithm in ("ring", "multiring", "tree")}
    groups = label_latency(results["ring"])
    for label in ("QKV", "Attention", "O proj", "AllReduce", "FFN up", "FFN down"):
        assert groups[label] > 0, label
    assert label_latency(results["multiring"])["AllReduce"] < groups["AllReduce"]
    assert any("sw" in name for name in results["ring"].utilization)
    single = evaluate(adf, workload, MappingSpec("tensor_parallel", {"tp": 1, "inner": "gsm_tiled"}))
    assert "AllReduce" not in label_latency(single)
    with pytest.raises(ValueError, match="not in its domain"):
        evaluate(adf, workload, MappingSpec("tensor_parallel", {"tp": 3, "inner": "gsm_tiled"}))



def test_dmc_partition_placements_use_the_mesh_and_memories():
    adf = dmc_chip()
    workload = matmul(1, 64, 256, 256, shared_b=True)
    local = evaluate(adf, workload, MappingSpec("dmc_partition", {"split": "n", "weights": "local"}))
    stream = evaluate(adf, workload, MappingSpec("dmc_partition", {"split": "n", "weights": "stream"}))
    assert local.utilization["chip/dram:port"] < stream.utilization["chip/dram:port"] * stream.latency_ns / local.latency_ns
    assert any(":link" in name for name in local.utilization)
    layer = WorkloadSpec("transformer", {"model": "gpt3-6.7b", "phase": "prefill", "batch": 1, "seq_len": 32})
    result = evaluate(adf, layer, MappingSpec("dmc_partition", {"weights": "stream", "activations": "local"}))
    assert {"QKV", "Attention", "FFN up", "FFN down", "Norm"} <= set(label_latency(result))
    with pytest.raises(CapacityError):
        evaluate(adf, layer, MappingSpec("dmc_partition", {"weights": "local"}))
    warned = evaluate(adf, layer, MappingSpec("dmc_partition", {"weights": "local"}), check_capacity=False)
    assert any("capacity exceeded" in w for w in warned.warnings)


def test_retained_scheduler_options_and_iterations():
    adf = shared_adf()
    workload = WorkloadSpec("transformer", {"model": "llama2-7b", "phase": "decode", "batch": 4, "seq_len": 256,
                                            "iterations": 2})
    consistent = evaluate(adf, workload, MappingSpec("gsm_tiled", {}))
    conservative = evaluate(adf, workload, MappingSpec("gsm_tiled", {}), scheduler="conservative")
    single = evaluate(adf, workload, MappingSpec("gsm_tiled", {}), iterations=1)
    assert consistent.latency_ns > single.latency_ns
    assert conservative.latency_ns == pytest.approx(consistent.latency_ns)
    for unknown in ("legacy", "optimistic"):
        with pytest.raises(ValueError, match="scheduler"):
            evaluate(adf, workload, MappingSpec("gsm_tiled", {}), scheduler=unknown)


def test_mapping_search_enumerates_and_reports_budget():
    adf, workload = shared_adf(), matmul(1, 256, 1024, 1024)
    space = mapping_space(adf, workload, "gsm_tiled", {"rows_per_cta": 4})
    assert len(space) == 16  # non-GPU tile_m x tile_n; no GPU K/warp choices
    result = best_mapping(adf, workload, ["gsm_tiled"], fixed={"gsm_tiled": {"rows_per_cta": 4}}, limit=10)
    assert result.skipped == len(space) - 10 and len(result.evaluated) == 10
    assert result.evaluation.latency_ns == min(latency for _, latency in result.evaluated)


def test_models_and_tensor_parallel_sharding():
    assert set(MODELS) == {"llama2-7b", "llama2-70b", "llama3-70b", "qwen-72b", "gpt3-6.7b"}
    assert all(config.source for config in MODELS.values())
    assert MODELS["qwen-72b"].intermediate == 24576 and MODELS["llama2-70b"].kv_heads == 8
    workload = transformer_workload("llama2-70b", "prefill", batch=1, seq_len=128, tp=4)
    qkv = next(op for op in workload.operators if op.label == "QKV")
    assert qkv.attrs["n"] == (64 // 4 + 2 * 8 // 4) * 128
    assert [op.op for op in workload.operators].count("allreduce") == 2
    with pytest.raises(ValueError, match="tp=3"):
        transformer_workload("llama2-70b", tp=3)
    decode = transformer_workload("llama2-7b", "decode", batch=2, seq_len=64)
    attn = next(op for op in decode.operators if op.name.endswith("attn_qk"))
    assert attn.attrs["m"] == 1 and attn.attrs["n"] == 64


def _graph(adf, workload, mapping):
    from arch_explorer.simulator.events.backend import run_mapping
    sim, trace = run_mapping(adf, workload, mapping)
    return sim.mapping.graph, trace


def test_outputs_are_resident_together_with_inputs():
    adf = shared_adf()
    for params in ({"op": "matmul", "batch": 1, "m": 1024, "k": 1024, "n": 1024}, {"op": "add", "rows": 1024, "cols": 1024}):
        result = evaluate(adf, WorkloadSpec("operator", params), MappingSpec("gsm_tiled", {"tile_m": 128, "tile_n": 128}))
        assert result.details["stats"]["memory_peak"]["chip/dram"] == 3 * 2**21     # A + B + C (or x + y + out)


def test_details_carry_stats_and_trace():
    adf = shared_adf()
    workload = WorkloadSpec("operator", {"op": "softmax", "rows": 512, "cols": 512})
    plain = evaluate(adf, workload, MappingSpec("gsm_tiled", {}))
    assert "stats" in plain.details and "trace" not in plain.details and not hasattr(plain, "stats")
    traced = evaluate(adf, workload, MappingSpec("gsm_tiled", {}), trace=True)
    data = json.loads(json.dumps(traced.to_dict()))
    stats = data["details"]["stats"]
    assert stats["scheduler"] == "consistent" and stats["instances"] == len(data["details"]["trace"]) > 0
    assert max(record["end_ns"] for record in data["details"]["trace"]) == pytest.approx(traced.latency_ns)


def test_allreduce_buffers_live_until_the_collective_ends():
    chip = shared_adf(1)
    chip.repeat = 4
    adf = SpaceMatrix("system", [chip], [SpacePoint("sw", "crossbar",
        {"port_bandwidth": Param(50., "B/ns", S), "latency": Param(2., "ns", S)})])
    model = dict(hidden=64, heads=4, kv_heads=2, head_dim=16, intermediate=128, layers=1,
                 vocab=64, gated=True, norm="rmsnorm", activation="silu")
    workload = WorkloadSpec("transformer", {"model": model, "phase": "prefill", "batch": 1, "seq_len": 4})
    graph, trace = _graph(adf, workload, MappingSpec("tensor_parallel", {"tp": 4, "inner": "gsm_tiled"}))
    end = {record.task: record.end_ns for record in trace.records}
    storage = next(t for t in graph.tasks if t.kind == "storage" and t.name.endswith("chip[0]:l0.attn_out"))
    done = next(t for t in graph.tasks if t.name == "l0.allreduce1.done0")
    assert max(end[s] for s, _ in graph.succs[storage.id]) >= end[done.id]


def test_traffic_follows_tensor_sizes_gqa_and_write_direction():
    gpu = shared_adf()
    graph, _ = _graph(gpu, WorkloadSpec("transformer", {"model": "llama2-70b", "phase": "decode", "batch": 1,
                                                        "seq_len": 4096}), MappingSpec("gsm_tiled", {}))
    port = "chip/dram:port"
    qk = sum(t.demand.get(port, 0) for t in graph.tasks if t.operator == "l0.attn_qk" and t.kind == "compute")
    k_cache = 4096 * 8 * 128 * 2
    assert k_cache <= qk < 1.2 * k_cache                  # 64 query heads share 8 K heads
    graph, _ = _graph(shared_adf(), WorkloadSpec("transformer", {"model": "llama2-7b", "phase": "prefill",
                                                                             "batch": 1, "seq_len": 128}),
                      MappingSpec("gsm_tiled", {}))
    act = sum(t.demand.get("chip/dram:port", 0) for t in graph.tasks if t.operator == "l0.act"
              and t.kind == "compute")
    assert act == pytest.approx(128 * 11008 * 2 * 3)     # read gate and up, write the product
    graph, _ = _graph(dmc_chip(sram_bytes=2**30), WorkloadSpec("operator", {"op": "add", "rows": 64, "cols": 256}),
                      MappingSpec("dmc_partition", {"activations": "dram"}))
    task = next(t for t in graph.tasks if t.kind == "compute" and t.name.endswith("@1"))
    links = {name for name in task.demand if ":link" in name}
    assert links == {"chip/noc:link(0, 1)->(0, 2)", "chip/noc:link(0, 2)->(0, 3)", "chip/noc:link(0, 3)->(0, 4)"}


def test_tensor_parallel_over_dmc_chips_reduces_on_vector_units():
    p = lambda value, unit: Param(value, unit, S)
    chip = dmc_chip(sram_bytes=2**30)
    chip.repeat = 2
    server = SpaceMatrix("server", [chip], [SpacePoint("sw", "crossbar", {"port_bandwidth": p(50.0, "B/ns"),
                                                                          "latency": p(0.0, "ns")})])
    workload = WorkloadSpec("transformer", {"model": "gpt3-6.7b", "phase": "prefill", "batch": 1, "seq_len": 32})
    result = evaluate(server, workload, MappingSpec("tensor_parallel", {"tp": 2, "inner": "dmc_partition"}))
    assert label_latency(result)["AllReduce"] > 0
