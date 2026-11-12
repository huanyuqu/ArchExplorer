"""Public semantic-to-Kernel conversion; compute service stays in model.py."""
from dataclasses import asdict
import json

from arch_explorer.components.api import Kernel
from arch_explorer.workload.api import (
    AdaptedGroup, ExecutionContext, TaskAdapter,
    lower_implementation, select_implementation,
)


SEMANTICS = ("compiled.matmul", "compiled.assemble")


def lower(context):
    """Reuse registered events operator lowerings, preserving every group task."""
    graph = context.program
    selected = set(context.tasks)
    tasks = {}
    for op in graph.operators:
        if op.name not in selected:
            continue
        definition = select_implementation(
            op, "events", dtype=graph.dtype, tensors=graph.tensors,
            implementation=context.implementations.get(op.op),
        )
        product = lower_implementation(
            definition, op, ExecutionContext("events", graph.dtype, graph.tensors),
        )
        if product.tensors or len(product.steps) != 1:
            raise NotImplementedError("serial adapter requires one materialized step without hidden tensors")
        step = product.steps[0]
        if step.inputs != tuple(op.inputs) or step.outputs != tuple(op.outputs):
            raise NotImplementedError("serial adapter requires exact ordered semantic inputs and outputs")
        # JSON serialization normalizes tuples and rejects NaN/Infinity. No
        # component duration or memory price appears in the adapted work.
        work = json.loads(json.dumps(asdict(step.kernel), allow_nan=False))
        tasks[op.name] = {"work": work, "traffic": "external"}
    if set(tasks) != selected:
        raise ValueError("serial adapter group tasks must identify actual public operators")
    return AdaptedGroup({"tasks": tasks}, scope={
        "input": "events.kernel.v1", "memory_traffic": "external_only",
        "conversion": "public registered events semantic lowerings",
    })


def restore(work):
    """Restore saved component input without inspecting operators or timing."""
    fields = {"op", "shape", "precision", "simt", "sfu", "row_simt", "row_sfu"}
    if type(work) is not dict or set(work) != fields:
        raise ValueError("serial adapter requires an exact serialized Kernel")
    if type(work["shape"]) is not list:
        raise TypeError("serialized Kernel.shape must be an array")
    return Kernel(**{**work, "shape": tuple(work["shape"])})


ADAPTER = TaskAdapter(
    id="example.serial.events", version="1", execution_domain="events",
    semantics=SEMANTICS, lower=lower, restore=restore,
    parameters={}, grouping="component",
)
