"""Operator-level workload IR consumed by the mapping templates (section III-C).

A :class:`Workload` lists tensors and operators in topological order. Mapping
templates tile operators into tasks, place them on SpacePoints and add the
storage and communication tasks the placement implies.

Vector operators carry per-element instruction counts of a straightforward
implementation (FP32 arithmetic on FP16 data). They define what the operator
computes, not how fast a device runs it:

============  ===================================================  ======  =====  ==========
operator      per element                                          SIMT    SFU    per row
============  ===================================================  ======  =====  ==========
softmax       max, subtract, exp, sum, scale by 1/sum              4       1      1 rcp
layernorm     sum, subtract mean, square-accumulate, scale, affine 5       0      1 rsqrt
rmsnorm       square-accumulate, scale, weight                     3       0      1 rsqrt
silu          exp(-x), add 1, reciprocal, multiply                 2       2
swiglu        silu(gate) * up                                      3       2
gelu          tanh form: x^3, FMA, scale, tanh, add, 2 multiplies  6       1
add           one add                                              1       0
============  ===================================================  ======  =====  ==========
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from arch_explorer.components.base import Kernel, dtype_bytes

VECTOR_COSTS = {
    "scale": dict(simt=1),
    "softmax": dict(simt=4, sfu=1, row_sfu=1),
    "layernorm": dict(simt=5, row_sfu=1),
    "rmsnorm": dict(simt=3, row_sfu=1),
    "silu": dict(simt=2, sfu=2),
    "swiglu": dict(simt=3, sfu=2),
    "gelu": dict(simt=6, sfu=1),
    "add": dict(simt=1),
}
MATRIX_OPS = ("matmul", "mvm")
COLLECTIVES = ("allreduce",)
OPERATORS = MATRIX_OPS + tuple(VECTOR_COSTS) + COLLECTIVES


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    dtype: str = "fp16"
    kind: str = "activation"   # activation | weight | kv
    alias_of: str | None = None
    strides: tuple | None = None   # element strides; None means contiguous
    offset: int = 0               # elements relative to alias_of's origin

    def __post_init__(self):
        from .graph import shape, validate_tensor
        object.__setattr__(self, "shape", shape(self.shape))
        if self.strides is not None:
            if not isinstance(self.strides, (tuple, list)):
                raise TypeError("strides must be a sequence")
            object.__setattr__(self, "strides", tuple(self.strides))
        validate_tensor(self)

    @property
    def bytes(self) -> int:
        return math.prod(self.shape) * dtype_bytes(self.dtype)


@dataclass(frozen=True)
class Operator:
    """``matmul``: attrs batch, m, k, n (A[b,m,k] @ B[b|1,k,n]); ``shared_b`` uses one B for
    every batch entry, ``kv_group`` = g lets g consecutive entries share one B (GQA); ``mvm``: batch, m, k;
    vector ops: rows, cols; ``allreduce``: bytes per participant."""
    name: str
    op: str
    inputs: tuple
    outputs: tuple
    attrs: dict = field(default_factory=dict)
    label: str = ""

    def kernel(self, dtype="fp16", rows=None, tile=None) -> Kernel:
        from .extensions import get_operator, _record_source, _record_operator
        dtype_bytes(dtype)
        definition = get_operator(self.op)
        _record_operator(self)
        if definition.lower is None:
            raise NotImplementedError(f"{self.name}: {self.op} has no legacy Kernel lower; select a target implementation")
        _record_source(definition.lower)
        kernel = definition.lower(self, dtype, rows=rows, tile=tile)
        from .execution import validate_kernel
        return validate_kernel(kernel)


@dataclass
class Workload:
    name: str
    tensors: dict = field(default_factory=dict)
    operators: list = field(default_factory=list)
    dtype: str = "fp16"
    iterations: int = 1
    meta: dict = field(default_factory=dict)

    def tensor(self, name, shape, kind="activation", dtype=None) -> str:
        if name in self.tensors:
            raise ValueError(f"Duplicate tensor {name}")
        self.tensors[name] = Tensor(name, shape, self.dtype if dtype is None else dtype, kind)
        return name

    def add(self, name, op, inputs, outputs, label="", **attrs) -> Operator:
        if not isinstance(inputs, (tuple, list)) or not isinstance(outputs, (tuple, list)):
            raise TypeError("operator inputs/outputs must be sequences")
        if type(label) is not str:
            raise TypeError("operator label must be a string")
        operator = Operator(name, op, tuple(inputs), tuple(outputs), attrs, label or op)
        return self._append_operator(operator)

    def _append_operator(self, operator, *, names=None, produced=None):
        """Common validation; private append-only builders supply their indexes.

        Public add always rescans its mutable sequence. Graph builders own
        their indexes and still pass full validate_graph at the public boundary.
        """
        from .graph import validate_operator
        validate_operator(operator, self.tensors)
        name=operator.name
        if (name in names if names is not None else any(previous.name == name for previous in self.operators)):
            raise ValueError(f"duplicate operator name {name!r}")
        known=self.producers() if produced is None else produced
        if any(tensor in known for tensor in operator.outputs):
            raise ValueError(f"{name}: duplicate output producer")
        self.operators.append(operator)
        if names is not None:names.add(name)
        if produced is not None:produced.update(operator.outputs)
        return operator

    def validate(self):
        from .graph import validate_graph
        return validate_graph(self)

    def to_dict(self):
        from .graph import to_dict
        return to_dict(self)

    @classmethod
    def from_dict(cls, data):
        from .graph import from_dict
        return from_dict(data)

    def producers(self):
        result = {}
        for operator in self.operators:
            for tensor in operator.outputs:
                if tensor in result:
                    raise ValueError(f"duplicate producer for tensor {tensor!r}")
                result[tensor] = operator
        return result


def operator_workload(params: dict) -> Workload:
    """A single-operator workload from ``WorkloadSpec(kind="operator").params``."""
    if params.get("op") == "attention":
        from arch_explorer.workload.attention import decode_attention
        return decode_attention(params)
    params = dict(params)
    op = params.pop("op", None)
    dtype = params.pop("dtype", "fp16")
    workload = Workload(f"{op}", dtype=dtype, iterations=params.pop("iterations", 1))
    if op == "matmul":
        batch, m, k, n = (params.pop("batch", 1), params.pop("m"), params.pop("k"), params.pop("n"))
        shared = params.pop("shared_b", False)
        workload.tensor("A", (batch, m, k))
        workload.tensor("B", (k, n) if shared else (batch, k, n), kind="weight")
        workload.tensor("C", (batch, m, n))
        workload.add("matmul", "matmul", ["A", "B"], ["C"], "Matmul", batch=batch, m=m, k=k, n=n, shared_b=shared)
    elif op == "mvm":
        batch, m, k = params.pop("batch", 1), params.pop("m"), params.pop("k")
        workload.tensor("A", (m, k), kind="weight")
        workload.tensor("x", (batch, k))
        workload.tensor("y", (batch, m))
        workload.add("mvm", "mvm", ["A", "x"], ["y"], "MVM", batch=batch, m=m, k=k)
    elif op in VECTOR_COSTS:
        rows, cols = params.pop("rows"), params.pop("cols")
        inputs = ["x", "y"] if op in ("add", "swiglu") else ["x"]
        for name in inputs:
            workload.tensor(name, (rows, cols))
        workload.tensor("out", (rows, cols))
        workload.add(op, op, inputs, ["out"], op.capitalize(), rows=rows, cols=cols)
    elif op == "allreduce":
        size = params.pop("bytes")
        elements = max(1, size // dtype_bytes(dtype))
        workload.tensor("x", (elements,))
        workload.tensor("out", (elements,))
        workload.add("allreduce", "allreduce", ["x"], ["out"], "AllReduce", bytes=elements * dtype_bytes(dtype))
    else:
        raise ValueError(f"Unknown operator workload {op!r}; supported: {OPERATORS}")
    if params:
        raise ValueError(f"Unknown {op} parameters: {sorted(params)}")
    return workload
