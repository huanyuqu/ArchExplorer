"""Real builtin events execution of a workload-owned external definition."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from arch_explorer.adf import MappingSpec, SpaceMatrix, WorkloadSpec
from arch_explorer.archs.adf_seeds import memory, mesh, point
from arch_explorer.components.api import Kernel
from arch_explorer.simulator.api import assess_request, evaluate_request, prepare_request
from arch_explorer.workload.api import (
    OperatorDefinition, Workload, register_operator, register_workload,
)


def validate_offset(op, tensors):
    if set(op.attrs) != {"rows", "cols", "offset"} or type(op.attrs["offset"]) is not float:
        raise ValueError("explicit fractional offset required")
    for name in (*op.inputs, *op.outputs):
        if tensors[name].shape != (op.attrs["rows"], op.attrs["cols"]):
            raise ValueError("offset preserves the complete input shape")


def lower_offset(op, dtype, rows=None, tile=None):
    if dtype != "fp32" or tile is not None:
        raise NotImplementedError("FP32 row slices only")
    return Kernel("vector", (op.attrs["rows"] if rows is None else rows, op.attrs["cols"]),
                  dtype, simt=1)


@pytest.fixture
def external_spec():
    name = "qa_offset_" + uuid4().hex
    register_operator(OperatorDefinition(name, validate_offset, lower_offset))
    graph = Workload("offset_branch", dtype="fp32")
    for tensor in ("x", "left", "right", "out"):
        graph.tensor(tensor, (2, 8))
    for side, offset in (("left", -0.125), ("right", 0.25)):
        graph.add(side, name, ["x"], [side], rows=2, cols=8, offset=offset)
    graph.add("join", "add", ["left", "right"], ["out"], rows=2, cols=8)
    return WorkloadSpec("graph", graph.to_dict())


def hardware():
    vector = point("vector", "vector_unit", {
        "lanes": (16, "count"), "sfu_lanes": (4, "count"),
        "clock_ghz": (1.0, "GHz"), "precision": ("fp32", "1"),
    }, source="synthetic QA execution fixture")
    return SpaceMatrix("chip", [SpaceMatrix("core", [vector], repeat=2),
        memory("l2", capacity=2**20), memory("dram", technology="ddr", capacity=2**24,
                                               bandwidth=64.0, latency=10.0)], [mesh(cols=4)])


def test_registered_operator_runs_on_builtin_mapping_and_replay_never_executes(
        external_spec, tmp_path, monkeypatch):
    request = prepare_request(hardware(), external_spec,
        MappingSpec("gsm_tiled", {"tile_m": 64, "tile_n": 64}),
        provider="events", output_dir=tmp_path / "run", options={"trace": True})
    assert assess_request(request)["supported"]
    result = evaluate_request(request)
    trace = result["evaluation"]["details"]["trace"]
    assert {record["operator"] for record in trace if record["kind"] == "compute"} == {
        "left", "right", "join"}
    assert result["evaluation"]["latency_ns"] > 0
    assert any("test_extension_execution" in key for key in request["payload"]["sources"])
    raw = tmp_path / "run" / "RAW.json"
    original = raw.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("replay attempted execution")

    monkeypatch.setattr("arch_explorer.components.events_provider.EventsProvider._execute", forbidden)
    monkeypatch.setattr("arch_explorer.simulator.events.engine.simulate", forbidden)
    assert evaluate_request(result) == result
    assert raw.read_bytes() == original


@pytest.mark.parametrize("bad", ["mapping", "params", "tensor"])
def test_invalid_external_graph_request_has_no_execution_artifacts(external_spec, tmp_path, bad):
    data = deepcopy(external_spec.params)
    if bad == "tensor":
        data["operators"][0]["inputs"] = ["undeclared"]
    mapping = MappingSpec("qa_unknown_mapping" if bad == "mapping" else "gsm_tiled",
                          {"unknown_parameter": 1} if bad == "params" else {})
    directory = tmp_path / "rejected"
    with pytest.raises((ValueError, TypeError, KeyError, NotImplementedError)):
        prepare_request(hardware(), WorkloadSpec("graph", data), mapping,
                        provider="events", output_dir=directory)
    assert not directory.exists()


@pytest.mark.parametrize("failure_stage", ["load", "register"])
def test_discovery_rolls_back_both_workload_tables_even_when_module_load_fails(
        monkeypatch, failure_stage):
    from arch_explorer.components import extensions
    from arch_explorer.workload.api import workload_from_spec
    extensions.discover_components()
    op_name, kind = ("qa_import_op_" + uuid4().hex, "qa_import_kind_" + uuid4().hex)

    def registrations_then_fail():
        register_operator(OperatorDefinition(op_name, validate_offset, lower_offset))
        register_workload(kind, lambda params, *, tp=1: Workload("must_not_survive"))
        raise RuntimeError("QA failure during " + failure_stage)

    def load():
        if failure_stage == "load":
            registrations_then_fail()
        return registrations_then_fail

    entry = SimpleNamespace(name="qa_broken", value="qa_broken:register", load=load,
                            dist=SimpleNamespace(name="qa_broken_distribution"))
    monkeypatch.setattr(extensions, "_DISCOVERED", False)
    monkeypatch.setattr(extensions, "_LOADED", set(extensions._LOADED))
    monkeypatch.setattr(extensions, "_FAILURES", dict(extensions._FAILURES))
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda **kwargs: [entry])
    report = extensions.discover_components()
    assert any("QA failure during " + failure_stage in error for error in report["failures"].values())
    graph = Workload("after_failed_discovery", dtype="fp32")
    graph.tensor("x", (1, 1))
    graph.tensor("out", (1, 1))
    with pytest.raises(ValueError, match="unknown operator"):
        graph.add("ghost", op_name, ["x"], ["out"], rows=1, cols=1, offset=0.125)
    with pytest.raises(ValueError, match="unknown workload kind"):
        workload_from_spec(WorkloadSpec(kind, {}))
    assert workload_from_spec(WorkloadSpec("operator", {
        "op": "softmax", "rows": 1, "cols": 8, "dtype": "fp32"})).operators[0].kernel("fp32").sfu == 1


def test_native_source_closure_includes_new_workload_dispatch_without_running_native():
    from arch_explorer.components.accelsim.program import _sources, _check_binding
    sources = _sources()
    assert {"workload/api.py", "workload/builtins.py", "workload/extensions.py",
            "workload/graph.py", "workload/operators.py", "workload/execution.py",
            "workload/reference_api.py", "components/accelsim/program_sources.py"} <= sources.keys()
    # The active program reader owns the current extended closure. The legacy
    # finite-migration classifier alone no longer defines current equality.
    _check_binding({"sources": sources}, {"sources": sources})
    for key in ("workload/extensions.py", "workload/execution.py", "workload/reference_api.py"):
        incomplete = {name: value for name, value in sources.items() if name != key}
        changed = dict(sources, **{key: "0" * 64})
        for producer in (incomplete, changed):
            with pytest.raises(ValueError):
                _check_binding({"sources": producer}, {"sources": sources})
