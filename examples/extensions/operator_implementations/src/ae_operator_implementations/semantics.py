"""Operator mathematics; no hardware, scheduling, or lowering knowledge."""
import math

import numpy as np

OPERATOR = "example_shifted_square"


def dimensions(rows, cols):
    if any(type(n) is not int or n < 1 for n in (rows, cols)):
        raise ValueError("rows and cols must be positive integers, not bool")
    return rows, cols


def coefficient(value):
    if type(value) not in (int, float) or abs(value) > float(np.finfo(np.float32).max) or not math.isfinite(value):
        raise ValueError("shift must be a finite FP32-representable scalar, not bool")
    return np.float32(value)


def validate(op, tensors):
    if set(op.attrs) != {"rows", "cols", "shift"}:
        raise ValueError("example_shifted_square requires rows, cols and shift")
    shape = dimensions(op.attrs["rows"], op.attrs["cols"])
    coefficient(op.attrs["shift"])
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        raise ValueError("example_shifted_square requires one input and one output")
    for name in (*op.inputs, *op.outputs):
        tensor = tensors[name]
        if tuple(tensor.shape) != shape or tensor.dtype != "fp32":
            raise ValueError("example_shifted_square requires matching FP32 [rows, cols] tensors")


def reference(op, inputs):
    if len(inputs) != 1:
        raise ValueError("example_shifted_square reference requires one input")
    value = np.asarray(inputs[0])
    if value.dtype != np.dtype("float32") or value.shape != dimensions(op.attrs["rows"], op.attrs["cols"]):
        raise ValueError("reference input must match the declared FP32 shape")
    if not np.isfinite(value).all():
        raise ValueError("example reference accepts finite input values only")
    with np.errstate(over="raise", invalid="raise"):
        shifted = np.add(value, coefficient(op.attrs["shift"]), dtype=np.float32)
        return np.multiply(shifted, shifted, dtype=np.float32)
