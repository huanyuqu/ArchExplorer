"""The extension owns the meaning of its FP32 y = x*x operator."""
from arch_explorer.workload.api import OperatorDefinition

KIND = "example.square"


def validate(op, tensors):
    if len(op.inputs) != 1 or len(op.outputs) != 1 or set(op.attrs) != {"shape"}:
        raise ValueError("example.square requires one input/output and exact shape attrs")
    shape = op.attrs["shape"]
    if not isinstance(shape, (tuple, list)) or not 1 <= len(shape) <= 4 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("example.square supports positive rank1..4 shapes")
    for name in (*op.inputs, *op.outputs):
        tensor = tensors[name]
        if tensor.dtype != "fp32" or tensor.shape != tuple(shape) or tensor.alias_of is not None or tensor.strides is not None:
            raise ValueError("example.square requires matching contiguous, materialized FP32 tensors")


def reference(op, inputs):
    import numpy as np
    if len(inputs) != 1:
        raise ValueError("example.square reference requires one array")
    value = np.asarray(inputs[0])
    if value.dtype != np.dtype("float32") or value.shape != tuple(op.attrs["shape"]) or not np.isfinite(value).all():
        raise ValueError("example.square reference input must match its finite FP32 tensor")
    with np.errstate(over="raise", invalid="raise"):
        return np.multiply(value, value, dtype=np.float32)


SQUARE = OperatorDefinition(KIND, validate, version="1", reference=reference)
