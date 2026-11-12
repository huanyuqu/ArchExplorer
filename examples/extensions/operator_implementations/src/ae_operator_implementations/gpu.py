"""New target implementations using only the public evaluation-local emitter."""
from arch_explorer.workload.api import ImplementationDefinition

from .constraints import materialized
from .semantics import OPERATOR

SINGLE_ID = "example.shifted_square.scalar"
STEPS_ID = "example.shifted_square.scalar_steps"


def single(op, context):
    emitter = context.options["emitter"]
    output = op.outputs[0]
    kernel = emitter.kernel(output)
    kernel.load(8, op.inputs[0], kernel.index)
    kernel.constant(9, op.attrs["shift"])
    kernel.arithmetic("FADD", 8, 8, 9)
    kernel.arithmetic("FMUL", 8, 8, 8)
    kernel.finish(output, 8)
    return emitter.finish_operator()


def separate(op, context):
    emitter = context.options["emitter"]
    output = op.outputs[0]
    temporary = emitter.temporary("shifted", context.tensors[output].shape)
    first = emitter.kernel(temporary)
    first.load(8, op.inputs[0], first.index)
    first.constant(9, op.attrs["shift"])
    first.arithmetic("FADD", 8, 8, 9)
    first.finish(temporary, 8)
    second = emitter.kernel(output)
    second.load(8, temporary, second.index)
    second.arithmetic("FMUL", 8, 8, 8)
    second.finish(output, 8)
    return emitter.finish_operator()


IMPLEMENTATIONS = tuple(ImplementationDefinition(
    identity, "1", OPERATOR, "accelsim", lower,
    requirements=("scalar-fp32", "global-memory", "sm90"), dtypes=("fp32",), supports=materialized)
    for identity, lower in ((SINGLE_ID, single), (STEPS_ID, separate)))
