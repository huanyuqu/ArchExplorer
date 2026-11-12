"""Alternative implementations, lowered through the public events IR."""
from arch_explorer.components.api import Kernel
from arch_explorer.workload.api import ExecutionProgram, ExecutionStep, ImplementationDefinition, Tensor

from .semantics import OPERATOR
from .constraints import materialized

SINGLE_ID = "example.shifted_square.single"
STEPS_ID = "example.shifted_square.steps"


def single(op, context):
    shape = context.tensors[op.outputs[0]].shape
    return Kernel("vector", shape, context.dtype, simt=2)


def separate(op, context):
    shape = context.tensors[op.outputs[0]].shape
    return ExecutionProgram(
        tensors=(Tensor("shifted", shape, context.dtype),),
        steps=(
            ExecutionStep("shift", Kernel("vector", shape, context.dtype, simt=1), op.inputs, ("shifted",)),
            ExecutionStep("square", Kernel("vector", shape, context.dtype, simt=1),
                          ("shifted",), op.outputs, ("shift",)),
        ),
    )


IMPLEMENTATIONS = (
    ImplementationDefinition(SINGLE_ID, "1", OPERATOR, "events", single,
                             requirements=("kernel:vector",), dtypes=("fp32",), supports=materialized),
    ImplementationDefinition(STEPS_ID, "1", OPERATOR, "events", separate,
                             requirements=("kernel:vector",), dtypes=("fp32",), supports=materialized),
)
