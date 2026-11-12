"""Frozen VAL48 workload contracts, executed only against an actual delivery.

The fixture intentionally imports the new API at execution time: collecting
these prepared tests on the pre-feature baseline must not imply they passed.
There are no compatibility fallbacks, skips, or expected-failure markers.
"""
import importlib
import json
import math
from uuid import uuid4

import pytest

from arch_explorer.adf import WorkloadSpec
from arch_explorer.components.api import Kernel


REJECT = (ValueError, TypeError, KeyError, NotImplementedError)


@pytest.fixture
def api():
    return importlib.import_module("arch_explorer.workload.api")


def unique(prefix):
    return "qa_" + prefix + "_" + uuid4().hex


def branched(api):
    graph = api.Workload("qa_existing_operator_branch", dtype="fp32")
    for name in ("x", "left", "right", "out"):
        graph.tensor(name, (2, 8))
    graph.add("left_branch", "softmax", ["x"], ["left"], rows=2, cols=8)
    graph.add("right_branch", "gelu", ["x"], ["right"], rows=2, cols=8)
    graph.add("join", "add", ["left", "right"], ["out"], rows=2, cols=8)
    return graph


def roundtrip(api, graph):
    return api.Workload.from_dict(graph.to_dict())


def register_affine(api):
    name = unique("affine")
    seen = {"validate": [], "lower": []}

    def validate(operator, tensors):
        attrs = operator.attrs
        if set(attrs) != {"rows", "cols", "gain", "offset", "mode"}:
            raise ValueError("affine requires explicit dimensions and scalar semantics")
        if any(type(attrs[key]) is not int or attrs[key] < 1 for key in ("rows", "cols")):
            raise ValueError("positive integer dimensions required")
        if any(type(attrs[key]) not in (int, float) or not math.isfinite(attrs[key]) for key in ("gain", "offset")):
            raise ValueError("finite numeric gain/offset required; bool is not a scalar")
        if attrs["mode"] != "affine":
            raise ValueError("only the explicit affine semantic is supported")
        if len(operator.inputs) != 1 or len(operator.outputs) != 1:
            raise ValueError("affine has one input and output")
        shape = (attrs["rows"], attrs["cols"])
        for tensor in (*operator.inputs, *operator.outputs):
            if tensors[tensor].shape != shape:
                raise ValueError("affine tensor shape differs from semantic dimensions")
        seen["validate"].append(operator.name)

    def lower(operator, dtype, rows=None, tile=None):
        if dtype != "fp32" or tile is not None:
            raise NotImplementedError("QA affine only supports untiled FP32 vector execution")
        seen["lower"].append(dict(name=operator.name, gain=operator.attrs["gain"], offset=operator.attrs["offset"]))
        # One multiply and one addition for y = gain*x + offset. This is an
        # explicit use of the existing vector capability, not a new device ISA.
        return Kernel("vector", (operator.attrs["rows"] if rows is None else rows,
                                  operator.attrs["cols"]), dtype, simt=2)

    definition = api.OperatorDefinition(name, validate, lower)
    api.register_operator(definition)
    return definition, seen


def affine_graph(api, definition, **changes):
    graph = api.Workload("qa_affine_graph", dtype="fp32")
    graph.tensor("x", (2, 8))
    graph.tensor("out", (2, 8))
    attrs = dict(rows=2, cols=8, gain=-0.5, offset=0.125, mode="affine")
    attrs.update(changes)
    operator = graph.add("transform", definition.name, ["x"], ["out"], **attrs)
    return graph, operator


def test_existing_operators_form_a_new_branched_graph_without_a_model_kind(api):
    graph = branched(api)
    data = graph.to_dict()
    decoded = json.loads(json.dumps(data, allow_nan=False))
    assert api.Workload.from_dict(decoded).to_dict() == data
    loaded = api.workload_from_spec(WorkloadSpec("graph", decoded))
    assert loaded.to_dict() == data
    assert {op.name: (op.inputs, op.outputs) for op in loaded.operators} == {
        "left_branch": (("x",), ("left",)), "right_branch": (("x",), ("right",)),
        "join": (("left", "right"), ("out",))}


def test_registered_operator_validates_non_integer_attrs_and_really_lowers(api):
    definition, seen = register_affine(api)
    graph, operator = affine_graph(api, definition)
    assert seen["validate"], "registered validation must actually be called"
    again = roundtrip(api, graph)
    assert again.to_dict() == graph.to_dict()
    kernel = operator.kernel(dtype="fp32", rows=1)
    assert kernel.op == "vector" and kernel.shape == (1, 8) and kernel.simt == 2
    assert seen["lower"] == [dict(name="transform", gain=-0.5, offset=0.125)]
    with pytest.raises(NotImplementedError, match="FP32"):
        operator.kernel(dtype="bf16")


@pytest.mark.parametrize("change", [dict(gain=True), dict(gain=float("nan")),
    dict(offset=float("inf")), dict(offset=-float("inf")), dict(mode="unknown"),
    dict(rows=True), dict(cols=7)])
def test_registered_validation_rejects_invalid_semantics(api, change):
    definition, _ = register_affine(api)
    with pytest.raises(REJECT):
        graph, _ = affine_graph(api, definition, **change)
        roundtrip(api, graph)


def test_registration_is_idempotent_for_same_definition_and_rejects_conflicts(api):
    definition, _ = register_affine(api)
    api.register_operator(definition)
    with pytest.raises(REJECT):
        api.register_operator(api.OperatorDefinition(definition.name, lambda op, ts: None,
            lambda op, dtype, rows=None, tile=None: Kernel("vector", (1, 1), dtype, simt=99)))
    _, operator = affine_graph(api, definition)
    assert operator.kernel(dtype="fp32").simt == 2


def test_builtin_semantics_cannot_be_replaced_by_an_extension(api):
    graph = branched(api)
    with pytest.raises(REJECT):
        api.register_operator(api.OperatorDefinition("softmax", lambda op, ts: None,
            lambda op, dtype, rows=None, tile=None: Kernel("vector", (1, 1), dtype, simt=99)))
    softmax = next(op for op in graph.operators if op.op == "softmax")
    kernel = softmax.kernel(dtype="fp32")
    assert kernel.simt == 4 and kernel.sfu == 1 and kernel.row_sfu == 1


def test_custom_workload_builder_receives_params_and_explicit_tp(api):
    kind = unique("builder")
    calls = []
    def builder(params, *, tp=1):
        calls.append((dict(params), tp))
        graph = branched(api)
        graph.meta["qa_tp"] = tp
        return graph
    api.register_workload(kind, builder)
    api.register_workload(kind, builder)
    value = api.workload_from_spec(WorkloadSpec(kind, {"tag": "external"}), tp=2)
    assert calls == [({"tag": "external"}, 2)] and value.meta["qa_tp"] == 2
    with pytest.raises(REJECT):
        api.register_workload(kind, lambda params, *, tp=1: api.Workload("conflicting"))


def test_builder_unsupported_tp_is_not_silently_changed_to_one(api):
    kind = unique("single_device")
    calls = []
    def builder(params, *, tp=1):
        calls.append(tp)
        if tp != 1:
            raise NotImplementedError("this external builder has no TP lowering")
        return branched(api)
    api.register_workload(kind, builder)
    with pytest.raises(NotImplementedError, match="no TP lowering"):
        api.workload_from_spec(WorkloadSpec(kind, {}), tp=2)
    assert calls == [2]


def test_builtin_workload_builder_cannot_be_replaced(api):
    spec = WorkloadSpec("operator", {"op": "softmax", "rows": 2, "cols": 8, "dtype": "fp32"})
    original = api.workload_from_spec(spec).to_dict()
    with pytest.raises(REJECT):
        api.register_workload("operator", lambda params, *, tp=1: api.Workload("wrong_semantics"))
    assert api.workload_from_spec(spec).to_dict() == original


@pytest.mark.parametrize("shape", [(0, 8), (-1, 8), (True, 8), (1.5, 8), (float("nan"), 8), (float("inf"), 8)])
def test_invalid_tensor_shapes_are_not_coerced_into_valid_graphs(api, shape):
    with pytest.raises(REJECT):
        graph = api.Workload("invalid_shape", dtype="fp32")
        graph.tensor("x", shape)
        graph.tensor("out", (2, 8))
        graph.add("use", "softmax", ["x"], ["out"], rows=2, cols=8)
        roundtrip(api, graph)


@pytest.mark.parametrize("reference", ["input", "output"])
def test_undefined_tensor_references_are_rejected(api, reference):
    with pytest.raises(REJECT):
        graph = api.Workload("undefined_reference", dtype="fp32")
        graph.tensor("x", (2, 8))
        graph.tensor("out", (2, 8))
        graph.add("use", "softmax", ["missing" if reference == "input" else "x"],
            ["missing" if reference == "output" else "out"], rows=2, cols=8)
        roundtrip(api, graph)


def test_duplicate_tensor_producers_are_rejected(api):
    with pytest.raises(REJECT):
        graph = branched(api)
        graph.add("second_producer", "add", ["left", "right"], ["out"], rows=2, cols=8)
        roundtrip(api, graph)


def test_dependency_cycles_are_rejected_instead_of_becoming_external_inputs(api):
    with pytest.raises(REJECT):
        graph = api.Workload("cycle", dtype="fp32")
        graph.tensor("a", (2, 8))
        graph.tensor("b", (2, 8))
        graph.add("produce_a", "softmax", ["b"], ["a"], rows=2, cols=8)
        graph.add("produce_b", "gelu", ["a"], ["b"], rows=2, cols=8)
        roundtrip(api, graph)


def test_shape_incompatibility_in_builtin_graph_is_rejected(api):
    with pytest.raises(REJECT):
        graph = api.Workload("incompatible", dtype="fp32")
        graph.tensor("x", (2, 8))
        graph.tensor("out", (2, 7))
        graph.add("softmax", "softmax", ["x"], ["out"], rows=2, cols=8)
        roundtrip(api, graph)


def test_unknown_tensor_dtype_is_rejected(api):
    with pytest.raises(REJECT):
        graph = api.Workload("unknown_dtype", dtype="fp32")
        graph.tensor("x", (2, 8), dtype="qa_invalid_dtype")
        graph.tensor("out", (2, 8))
        graph.add("softmax", "softmax", ["x"], ["out"], rows=2, cols=8)
        roundtrip(api, graph)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_graph_serialization_does_not_accept_nonfinite_metadata(api, value):
    graph = branched(api)
    graph.meta["invalid_number"] = value
    with pytest.raises(REJECT):
        roundtrip(api, graph)


def test_unknown_operator_and_workload_kind_fail_explicitly(api):
    graph = api.Workload("unknown", dtype="fp32")
    graph.tensor("x", (2, 8))
    graph.tensor("out", (2, 8))
    with pytest.raises(REJECT):
        graph.add("use", unique("unknown_operator"), ["x"], ["out"], rows=2, cols=8)
    with pytest.raises(REJECT):
        api.workload_from_spec(WorkloadSpec(unique("unknown_workload"), {}))
