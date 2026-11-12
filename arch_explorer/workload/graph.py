"""Finite JSON graph interchange and strict ordered-DAG validation.

All declared tensors without a producer are external inputs/weights. An input
not declared in tensors is dangling. Operator order is preserved and must be
topological; forward edges, cycles and in-place writes are refused.
"""
import math
from .resource_budget import graph_budget

from arch_explorer.components.base import dtype_bytes

SCHEMA = "arch-explorer.workload-graph.v2"
LEGACY_SCHEMA = "arch-explorer.workload-graph.v1"
_FIELDS = {"schema", "name", "dtype", "iterations", "meta", "tensors", "operators"}
_TENSOR = {"name", "shape", "dtype", "kind", "alias_of", "strides", "offset"}
_OPERATOR = {"name", "op", "inputs", "outputs", "attrs", "label", "type"}


def finite_json(value, *, tuples=False, _active=None):
    """Copy JSON values without coercing keys, numbers or foreign objects."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("workload graph requires finite JSON numbers")
        return value
    if type(value) in (list, dict) or (tuples and type(value) is tuple):
        active = set() if _active is None else _active
        if id(value) in active:
            raise ValueError("cyclic Python containers are not JSON graph values")
        active.add(id(value))
        try:
            if type(value) is dict:
                if any(type(key) is not str for key in value):
                    raise TypeError("workload JSON object keys must be strings")
                return {key: finite_json(v, tuples=tuples, _active=active) for key, v in value.items()}
            return [finite_json(v, tuples=tuples, _active=active) for v in value]
        finally:
            active.remove(id(value))
    raise TypeError(f"workload requires JSON values, got {type(value).__name__}")


def name(value, field="name"):
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{field} must be a nonempty, unpadded string")


def shape(value, field="shape"):
    if (not isinstance(value, (tuple, list)) or not value
            or any(type(n) is not int or n < 1 for n in value)):
        raise ValueError(f"{field} must contain positive integer dimensions")
    return tuple(value)


def validate_tensor(tensor):
    name(tensor.name, "tensor name")
    shape(tensor.shape)
    dtype_bytes(tensor.dtype)
    if tensor.kind not in ("activation", "weight", "kv"):
        raise ValueError("tensor kind must be activation, weight or kv")
    if type(tensor.offset) is not int or tensor.offset < 0 or (tensor.alias_of is None and tensor.offset):
        raise ValueError("tensor offset requires an alias and nonnegative integer elements")
    if tensor.alias_of is not None:
        name(tensor.alias_of, "alias_of")
        if tensor.alias_of == tensor.name:
            raise ValueError("tensor cannot alias itself")
    if tensor.strides is not None:
        if (not isinstance(tensor.strides, (tuple, list)) or len(tensor.strides) != len(tensor.shape)
                or any(type(n) is not int or n < 0 for n in tensor.strides)):
            raise ValueError("strides must contain one nonnegative integer per dimension")
        if tensor.alias_of is None:
            raise ValueError("strides require an explicit alias_of storage source")


def validate_operator(op, tensors):
    from .operators import Operator
    from .full_primitives import FullOperator
    from .extensions import validate_operator as validate_registered, _record_source, _record_operator
    from .builtins import validate_full
    if type(op) not in (Operator, FullOperator):
        raise TypeError("operators must use Operator plus a registered definition, or builtin FullOperator")
    name(op.name, "operator name")
    name(op.op, "operator kind")
    if type(op.label) is not str:
        raise TypeError("operator label must be a string")
    if not isinstance(op.inputs, (tuple, list)) or not isinstance(op.outputs, (tuple, list)) or not op.outputs:
        raise TypeError("operator inputs/outputs must be sequences; outputs must be nonempty")
    for tensor in (*op.inputs, *op.outputs):
        name(tensor, "tensor reference")
        if tensor not in tensors:
            raise ValueError(f"{op.name}: dangling tensor {tensor!r}")
    if len(set(op.outputs)) != len(op.outputs):
        raise ValueError(f"{op.name}: duplicate output tensor")
    if set(op.inputs) & set(op.outputs):
        raise ValueError(f"{op.name}: in-place/cyclic tensor writes are unsupported")
    if type(op.attrs) is not dict:
        raise TypeError("operator attrs must be a JSON object")
    finite_json(op.attrs, tuples=True)
    if type(op) is FullOperator:
        _record_operator(op)
        _record_source(validate_full)
        validate_full(op, tensors)
    else:
        validate_registered(op, tensors)


@graph_budget
def validate_graph(workload):
    from .operators import Tensor
    name(workload.name, "workload name")
    dtype_bytes(workload.dtype)
    if type(workload.iterations) is not int or workload.iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if type(workload.meta) is not dict:
        raise TypeError("workload meta must be a JSON object")
    finite_json(workload.meta, tuples=True)
    if type(workload.tensors) is not dict or type(workload.operators) is not list:
        raise TypeError("workload tensors/operators must be dict/list")
    if not workload.operators:
        raise ValueError("workload DAG must contain at least one operator")
    from .resource_budget import check_graph_budget
    check_graph_budget(workload)
    for key, tensor in workload.tensors.items():
        if type(tensor) is not Tensor or key != tensor.name:
            raise ValueError("tensor dictionary key must match its Tensor.name")
        validate_tensor(tensor)
    producers, op_names = {}, set()
    for index, op in enumerate(workload.operators):
        validate_operator(op, workload.tensors)
        if op.name in op_names:
            raise ValueError(f"duplicate operator name {op.name!r}")
        op_names.add(op.name)
        for output in op.outputs:
            if output in producers:
                raise ValueError(f"duplicate producer for tensor {output!r}")
            producers[output] = index
    for index, op in enumerate(workload.operators):
        for input_name in op.inputs:
            if producers.get(input_name, -1) >= index:
                raise ValueError("only ordered DAGs are supported: forward edge or dependency cycle")
    for tensor in workload.tensors.values():
        seen = {tensor.name}
        alias = tensor.alias_of
        while alias is not None:
            if alias not in workload.tensors:
                raise ValueError(f"dangling tensor alias {alias!r}")
            if alias in seen:
                raise ValueError("tensor alias cycle")
            seen.add(alias)
            alias = workload.tensors[alias].alias_of
        if tensor.alias_of is not None:
            index = producers.get(tensor.name)
            if index is None or workload.operators[index].op not in ("view","storage.view"):
                raise ValueError("an alias must be produced by an explicit view operator")
            from .storage import resolve_layout
            resolve_layout(workload.tensors,tensor.name)
    graph_outputs(workload)
    return workload


def graph_outputs(workload):
    """Explicit retained outputs, or the legacy set of unconsumed produced tensors."""
    produced={n for op in workload.operators for n in op.outputs}
    if 'outputs' in workload.meta:
        outputs=workload.meta['outputs']
        if (type(outputs) is not list or not outputs or any(type(n) is not str for n in outputs)
                or len(set(outputs))!=len(outputs) or set(outputs)-produced):
            raise ValueError('graph meta.outputs must list unique produced tensors')
        return list(outputs)
    consumed={n for op in workload.operators for n in op.inputs}
    return [n for op in workload.operators for n in op.outputs if n not in consumed]


def to_dict(workload):
    from .full_primitives import FullOperator
    validate_graph(workload)
    return finite_json(dict(schema=SCHEMA, name=workload.name, dtype=workload.dtype,
        iterations=workload.iterations, meta=workload.meta,
        tensors=[dict(name=t.name, shape=t.shape, dtype=t.dtype, kind=t.kind,
                      alias_of=t.alias_of, strides=t.strides, offset=t.offset) for t in workload.tensors.values()],
        operators=[dict(name=op.name, op=op.op, inputs=op.inputs, outputs=op.outputs,
                        attrs=op.attrs, label=op.label,
                        type="full" if type(op) is FullOperator else "operator")
                   for op in workload.operators]), tuples=True)


def from_dict(data):
    from .operators import Tensor, Operator, Workload
    from .full_primitives import FullOperator
    data = finite_json(data)
    if type(data) is not dict or set(data) != _FIELDS or data["schema"] not in (SCHEMA,LEGACY_SCHEMA):
        raise ValueError("invalid workload graph fields/schema")
    if type(data["tensors"]) is not list or type(data["operators"]) is not list:
        raise TypeError("graph tensors/operators must be JSON arrays")
    tensors = {}
    for item in data["tensors"]:
        fields=_TENSOR-({'offset'} if data['schema']==LEGACY_SCHEMA else set())
        if type(item) is not dict or set(item) != fields:
            raise ValueError("invalid tensor fields")
        name(item["name"], "tensor name")
        if item["name"] in tensors:
            raise ValueError(f"duplicate tensor {item['name']!r}")
        tensor = Tensor(item["name"], shape(item["shape"]), item["dtype"], item["kind"],
                        item["alias_of"], None if item["strides"] is None else tuple(item["strides"]),item.get("offset",0))
        tensors[tensor.name] = tensor
    operators = []
    for item in data["operators"]:
        if type(item) is not dict or set(item) != _OPERATOR or item["type"] not in ("operator", "full"):
            raise ValueError("invalid operator fields/type")
        if type(item["inputs"]) is not list or type(item["outputs"]) is not list:
            raise TypeError("operator inputs/outputs must be JSON arrays")
        cls = FullOperator if item["type"] == "full" else Operator
        operators.append(cls(item["name"], item["op"], tuple(item["inputs"]), tuple(item["outputs"]),
                             item["attrs"], item["label"]))
    return validate_graph(Workload(data["name"], tensors, operators, data["dtype"],
                                   data["iterations"], data["meta"]))
