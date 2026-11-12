"""Elementwise y = scale*x + bias, optionally followed by ReLU.

The synthetic implementation uses separate multiply/add instructions, not a
fused instruction. Coefficients describe semantics, never hardware speed.
"""
import math

from arch_explorer.components.api import Kernel
from arch_explorer.workload.api import OperatorDefinition


def validate_affine(operator, tensors):
    attrs = operator.attrs
    if (set(attrs) != {"rows", "cols", "scale", "bias", "activation"}
            or any(type(attrs[key]) is not int or attrs[key] < 1 for key in ("rows", "cols"))):
        raise ValueError("tutorial_affine requires positive integer rows/cols and scale/bias/activation")
    for key in ("scale", "bias"):
        if type(attrs[key]) not in (int, float) or not math.isfinite(attrs[key]):
            raise ValueError(f"tutorial_affine {key} must be a finite number, not bool")
    if attrs["activation"] not in ("identity", "relu"):
        raise ValueError("tutorial_affine activation must be identity or relu")
    if len(operator.inputs) != 1 or len(operator.outputs) != 1:
        raise ValueError("tutorial_affine has one input and one output")
    shape = (attrs["rows"], attrs["cols"])
    for name in (*operator.inputs, *operator.outputs):
        tensor = tensors[name]
        if tensor.shape != shape or tensor.dtype != "fp32":
            raise ValueError("tutorial_affine requires matching FP32 [rows, cols] tensors")


def lower_affine(operator, dtype, rows=None, tile=None):
    if dtype != "fp32" or tile is not None:
        raise NotImplementedError("tutorial_affine lowers FP32 row slices, not matrix tiles or other dtypes")
    rows = operator.attrs["rows"] if rows is None else rows
    if type(rows) is not int or not 1 <= rows <= operator.attrs["rows"]:
        raise ValueError("tutorial_affine row slice must be a positive integer within the tensor")
    return Kernel("vector", (rows, operator.attrs["cols"]), "fp32",
                  simt=2 + int(operator.attrs["activation"] == "relu"))


AFFINE = OperatorDefinition("tutorial_affine", validate_affine, lower_affine)
