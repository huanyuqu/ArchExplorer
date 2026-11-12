"""One new branching graph, independent of execution target and component names."""
from fractions import Fraction

import numpy as np

from arch_explorer.workload.api import Workload
from .semantics import OPERATOR, dimensions

NODES = ("left_square", "right_square", "final_square")


def branch_graph(rows=8, cols=16, *, name="external-square-fork"):
    dimensions(rows, cols)
    graph = Workload(name, dtype="fp32")
    for tensor in ("x", "left", "right", "merged", "out"):
        graph.tensor(tensor, (rows, cols))
    for node, output, shift in (("left_square", "left", 1.25), ("right_square", "right", -0.5)):
        graph.add(node, OPERATOR, ["x"], [output], rows=rows, cols=cols, shift=shift)
    graph.add("merge", "add", ["left", "right"], ["merged"], rows=rows, cols=cols)
    graph.add("final_square", OPERATOR, ["merged"], ["out"], rows=rows, cols=cols, shift=0.25)
    return graph


def sample_inputs(rows=8, cols=16):
    dimensions(rows, cols)
    if rows * cols > 4096:
        raise ValueError("the teaching input generator is bounded to 4096 elements")
    values = (np.arange(rows * cols, dtype=np.int32) % 9 - 4).astype(np.float32) / np.float32(4)
    return {"x": values.reshape(rows, cols)}


def exact_expected(inputs):
    """Independent rational evaluation for this graph's quarter-integer inputs."""
    values = np.asarray(inputs["x"])
    if values.dtype != np.dtype("float32") or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expected values require finite two-dimensional FP32 inputs")
    outputs = {name: np.empty_like(values) for name in ("left", "right", "merged", "out")}
    for index in np.ndindex(values.shape):
        x = Fraction(float(values[index]))
        if x * 4 != int(x * 4) or not -1 <= x <= 1:
            raise ValueError("exact oracle is limited to the documented quarter-integer sample")
        left, right = (x + Fraction(5, 4)) ** 2, (x - Fraction(1, 2)) ** 2
        expected = {"left": left, "right": right, "merged": left + right,
                    "out": (left + right + Fraction(1, 4)) ** 2}
        for name, value in expected.items():
            outputs[name][index] = float(value)
    return outputs
