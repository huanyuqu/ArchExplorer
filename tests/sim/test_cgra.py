"""CGRA contracts, graph semantics and independently hand-counted timing."""
from dataclasses import replace

import pytest

from arch_explorer.adf import COMPONENTS, MappingSpec, WorkloadSpec, evaluate
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.components import Kernel
from arch_explorer.devices import get
from arch_explorer.mapping.cgra import CgraSpatial
from arch_explorer.simulator.events.backend import run_mapping
from arch_explorer.simulator.events.builder import build_simulator


def workload(op="matmul", m=4, k=4, n=4):
    return WorkloadSpec("operator", {"op": op, "m": m, "k": k, "n": n, "dtype": "uint32"})


def run(adf=None, w=None, **params):
    return run_mapping(adf or get("vectorcgra_4x4", words_per_bank=128), w or workload(),
                       MappingSpec("cgra_spatial", params))


def test_cgra_component_scalar_contract_and_sources():
    assert {k: p.unit for k, p in COMPONENTS.get("cgra_pe").params.items()} == {
        "clock_ghz": "GHz", "precision": "1"}
    sim = build_simulator(get("vectorcgra_4x4"))
    pe = sim.find_points(component="cgra_pe")[0]
    for op in ("mul", "add", "load", "store"):
        assert pe.kernel_time(Kernel(op, (1,), "uint32")) == 1
    with pytest.raises(NotImplementedError):
        pe.kernel_time(Kernel("gemm", (1, 4, 4, 4), "uint32"))
    with pytest.raises(NotImplementedError):
        pe.kernel_time(Kernel("mul", (1,), "fp32"))
    assert "unknown" in pe.point.params["clock_ghz"].source
    assert all(row["source"] for row in sim.provenance.as_list())


@pytest.mark.parametrize("diagonal,hops", [(False, 6), (True, 3)])
def test_mesh_corner_routes_bidirectional_and_no_wrap(diagonal, hops):
    sim = build_simulator(get("vectorcgra_4x4", diagonal=diagonal))
    mesh = sim.container("vectorcgra/array").networks[0]
    forward, reverse = mesh.route(0, 15), mesh.route(15, 0)
    assert len(forward.resources) == len(reverse.resources) == hops
    assert forward.latency_ns == hops
    assert not set(forward.resources) & set(reverse.resources)
    assert len(mesh.route(0, 3).resources) == 3
    with pytest.raises(IndexError):
        mesh.route(0, 16)


def test_hand_counted_scalar_with_nearest_memory_port():
    # 2 preload words; 16*(17 control words+2 count packets+launch)=320
    # injections, last packet to tile15 takes 1 hop. Two reads, MUL, ADD,
    # one write at tile0 = 5 ticks. Return injection+one hop = 2 ticks.
    sim, trace = run(w=workload(m=1, k=1, n=1), memory_port="nearest")
    cycles = sim.mapping.cycle_report(trace)["cycles"]
    assert cycles == {"data_load": 2, "config": 321, "compute": 5, "completion": 2, "end_to_end": 330}
    assert trace.latency_ns == cycles["end_to_end"]


@pytest.mark.parametrize("op,m,k,n", [("matmul", 3, 4, 5), ("mvm", 1, 8, 8), ("linear2", 2, 4, [8, 4])])
@pytest.mark.parametrize("diagonal", [False, True])
def test_dfg_values_addresses_and_fu_serialization(op, m, k, n, diagonal):
    sim, trace = run(get("vectorcgra_4x4", diagonal=diagonal, words_per_bank=128), workload(op, m, k, n))
    meta = sim.mapping.cycle_report(trace)["cgra"]
    graph = sim.mapping.graph
    memory, values = {}, {}
    expected = None
    for li, buffer in enumerate(meta["buffers"]):
        rows, inner, cols = buffer["shape"]
        a = expected if li else [[(i*inner+t+1) * 1234567 for t in range(inner)] for i in range(rows)]
        b = [[t*cols+j+3 for j in range(cols)] for t in range(inner)]
        if not li:
            memory.update({buffer["a"]+i*inner+t: a[i][t] for i in range(rows) for t in range(inner)})
        memory.update({buffer["b"]+t*cols+j: b[t][j] for t in range(inner) for j in range(cols)})
        expected = [[sum(a[i][t]*b[t][j] for t in range(inner)) & 0xffffffff for j in range(cols)]
                    for i in range(rows)]
    # Execute the emitted graph's value dependencies, independently of timing.
    for ident, task in enumerate(graph.tasks):
        info = task.meta
        if info.get("op") == "load":
            assert info["tile"] in (0, 1, 2, 3, 4, 8, 12)
            values[ident] = memory[info["address"]]
        elif "operands" in info:
            a, b = [values[x] if x is not None else 0 for x in info["operands"]]
            values[ident] = (a*b if info["op"] == "mul" else a+b) & 0xffffffff
        elif "value_task" in info:
            values[ident] = values[info["value_task"]]
            if info.get("op") == "store":
                assert info["tile"] in (0, 1, 2, 3, 4, 8, 12)
                memory[info["address"]] = values[ident]
    last = meta["buffers"][-1]
    assert [memory[last["c"]+i] for i in range(m*last["shape"][2])] == sum(expected, [])
    macs = sum(a*b*c for a,b,c in [row["shape"] for row in meta["buffers"]])
    assert meta["operations"]["mul"] == meta["operations"]["add"] == macs
    by_tile = {}
    for record in trace.records:
        info = graph.tasks[record.task].meta
        if "tile" in info:
            by_tile.setdefault(info["tile"], []).append((record.start_ns, record.end_ns))
    for intervals in by_tile.values():
        intervals.sort()
        assert all(a[1] <= b[0] for a,b in zip(intervals, intervals[1:]))


def test_diagonal_reduces_routed_dfg_hops_and_clock_rescaling_preserves_cycles():
    ordinary, trace = run()
    diagonal, dtrace = run(get("vectorcgra_4x4_diag", words_per_bank=128))
    fast, ftrace = run(get("vectorcgra_4x4_diag", words_per_bank=128, clock_ghz=2))
    a, b, c = [sim.mapping.cycle_report(t) for sim,t in [(ordinary,trace),(diagonal,dtrace),(fast,ftrace)]]
    assert b["cgra"]["operations"]["link_hops"] < a["cgra"]["operations"]["link_hops"]
    assert b["cycles"]["compute"] < a["cycles"]["compute"]
    assert b["cycles"] == c["cycles"]
    assert ftrace.latency_ns * 2 == dtrace.latency_ns


def test_mapping_choices_legality_and_capacity():
    assert {"ii", "placement", "routing"} <= CgraSpatial().space(None, None).keys()
    with pytest.raises(ValueError, match="resident buffers"):
        run(get("vectorcgra_4x4"), workload(m=6, k=6, n=6))
    with pytest.raises(ValueError):
        run(ii=True)
    with pytest.raises(ValueError):
        run(routing="magic")
    with pytest.raises(ValueError):
        get("vectorcgra_4x4", words_per_bank=33)
    with pytest.raises(ValueError):
        get("vectorcgra_4x4", clock_ghz=float("nan"))
    bad = get("vectorcgra_4x4")
    bad.child("array").child("pe").repeat = 15
    with pytest.raises(ValueError, match="16 PEs"):
        run(bad)
    bad = get("vectorcgra_4x4")
    p = bad.child("array").child("pe").params
    p["precision"] = replace(p["precision"], value="fp32")
    with pytest.raises(ValueError, match="uint32"):
        run(bad)


def test_issue_spacing_public_mvm_and_invocation_boundary():
    w = WorkloadSpec("operator", {"op": "mvm", "m": 3, "k": 2, "dtype": "uint32"})
    sim, trace = run(w=w, ii=4, routing="yx", placement="column_major")
    assert sim.mapping.cycle_report(trace)["cgra"]["operations"]["mul"] == 6
    starts = {}
    for record in trace.records:
        info = sim.mapping.graph.tasks[record.task].meta
        if "tile" in info:
            starts.setdefault(info["tile"], []).append(record.start_ns)
    for times in starts.values():
        times.sort()
        assert all(b-a >= 4 for a,b in zip(times,times[1:]))
    with pytest.raises(ValueError, match="iterations"):
        run_mapping(get("vectorcgra_4x4"), workload(m=1,k=1,n=1), MappingSpec("cgra_spatial"), iterations=2)
    for values in ({"batch": True}, {"m": 0}, {"dtype": "fp32"}):
        invalid = WorkloadSpec("operator", {**workload().params, **values})
        assert not CgraSpatial().applicable(get("vectorcgra_4x4"), invalid)


@pytest.mark.parametrize("preset", ["vectorcgra_4x4", "vectorcgra_4x4_diag"])
def test_adf_and_agent_evaluate_presets(preset, tmp_path):
    a, w = get(preset), workload(m=2, k=2, n=2)
    result = evaluate(a, w, MappingSpec("cgra_spatial"))
    assert result.details["cycles"]["end_to_end"] > 0
    assert result.area_mm2 is None  # no fabricated CGRA area constants
    env = ExplorationEnv({"seed_adf": a, "workloads": [w], "max_evaluations": 2,
                          "mapping_templates": ["cgra_spatial"], "mapping_samples": 2,
                          "cost_penalty": 0, "area_penalty": 0,
                          "trace_path": str(tmp_path / "trace.jsonl")})
    response = env.evaluate(optimize="mapping")
    assert "error" not in response, response
    assert response["result"]["score"]["feasible"]
    assert response["result"]["workloads"][0]["evaluation"]["details"]["cycles"]["compute"] > 0
