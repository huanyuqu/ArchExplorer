"""User graphs: tensor semantics only, no component or scheduler knowledge."""
from arch_explorer.workload.api import Workload


def _dimensions(rows, cols):
    if any(type(value) is not int or not 1 <= value <= 4096 for value in (rows, cols)):
        raise ValueError("Example rows/cols must be integers in [1, 4096]")


def affine_graph(rows=32, cols=16):
    _dimensions(rows, cols)
    graph = Workload("affine-branches", dtype="fp32")
    for name in ("x", "left", "right", "merged", "out"):
        graph.tensor(name, (rows, cols))
    graph.add("left_affine", "tutorial_affine", ["x"], ["left"], rows=rows, cols=cols,
              scale=1.25, bias=-0.5, activation="identity")
    graph.add("right_affine", "tutorial_affine", ["x"], ["right"], rows=rows, cols=cols,
              scale=-0.5, bias=0.75, activation="relu")
    graph.add("merge", "add", ["left", "right"], ["merged"], rows=rows, cols=cols)
    graph.add("project", "tutorial_affine", ["merged"], ["out"], rows=rows, cols=cols,
              scale=0.5, bias=0.0, activation="identity")
    return graph


def builtin_graph(rows=32, cols=16):
    """A new branching DAG reusing only existing scale/add definitions."""
    _dimensions(rows, cols)
    graph = Workload("builtin-branches", dtype="fp32")
    for name in ("x", "left", "right", "out"):
        graph.tensor(name, (rows, cols))
    graph.add("left_scale", "scale", ["x"], ["left"], rows=rows, cols=cols)
    graph.add("right_scale", "scale", ["x"], ["right"], rows=rows, cols=cols)
    graph.add("merge", "add", ["left", "right"], ["out"], rows=rows, cols=cols)
    return graph


def build_workload(params, *, tp=1):
    """Optional named builder; the same graph also uses WorkloadSpec('graph')."""
    if type(tp) is not int or tp != 1:
        raise NotImplementedError("tutorial_branches explicitly supports tp=1 only")
    if not isinstance(params, dict) or set(params) - {"rows", "cols"}:
        raise ValueError("tutorial_branches accepts rows and cols only")
    return affine_graph(**params)
