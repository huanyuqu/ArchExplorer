"""Materialized FP32 tensor algorithms using the existing scalar SM90 emitter.

Tensor/attribute semantics belong to their registered OperatorDefinition. This
module owns target eligibility and instructions, including real copy kernels
for reshape and same-precision cast. Lower precision storage is unsupported.
"""
import math

KINDS = ('add', 'sub', 'mul', 'div', 'scale', 'exp', 'transpose',
         'broadcast', 'reduce_max', 'reduce_sum', 'reshape', 'cast',
         'affine_norm', 'gelu_tanh')
_DEFINITIONS = None
_REUSED = frozenset({'add', 'scale', 'exp', 'transpose', 'affine_norm'})


def supports(op, context):
    """Reject true tensor dtype/layout gaps before any emitter allocation."""
    from ..graph import validate_operator
    from arch_explorer.components.sm.implementations.accelsim.lowering_limits import LIMITS
    if (context.target != 'accelsim' or context.dtype != 'fp32'
            or op.op not in KINDS or len(op.outputs) != 1):
        return False
    for name in (*op.inputs, *op.outputs):
        tensor = context.tensors[name]
        if (tensor.dtype != 'fp32' or tensor.alias_of is not None
                or tensor.strides is not None or not 1 <= len(tensor.shape) <= 4
                or math.prod(tensor.shape) > LIMITS['tensor_elements']):
            return False
    validate_operator(op, context.tensors)
    return True


def _emitter(op, context):
    import numpy as np
    from arch_explorer.components.sm.implementations.accelsim.emitter import ProgramEmitter
    if not supports(op, context):
        raise ValueError('materialized native lowering requires contiguous FP32 rank 1..4 tensors within scalar bounds')
    emitter = context.options.get('emitter')
    if (not isinstance(emitter, ProgramEmitter) or emitter.operator is not op
            or any(emitter.graph.tensors[name] != context.tensors[name]
                   for name in (*op.inputs, *op.outputs))):
        raise ValueError('materialized native lowering requires its own evaluation-local emitter')
    if any(name not in emitter.memory.addresses
           or emitter.memory.dtypes[name] != np.dtype('float32')
           or emitter.memory.addresses[name].shape != context.tensors[name].shape
           for name in op.inputs):
        raise ValueError('materialized native operand memory differs from its FP32 tensor declaration')
    return emitter


def _gelu(kernel, source, output):
    """x*sigmoid(2*z), z=sqrt(2/pi)*(x+.044715*x**3).

    sigmoid(t)=exp(min(t,0))/(1+exp(-abs(t))) avoids exp overflow
    for the negative tail. Every value is formed by actual emitter operations;
    no host-computed activation or fictional TANH instruction is introduced.
    """
    kernel.load(8, source, kernel.index)
    kernel.arithmetic('FMUL', 10, 8, 8)
    kernel.arithmetic('FMUL', 11, 10, 8)
    kernel.constant(9, .044715)
    kernel.arithmetic('FMUL', 11, 11, 9)
    kernel.arithmetic('FADD', 11, 8, 11)
    kernel.constant(9, 2 * math.sqrt(2 / math.pi))
    kernel.arithmetic('FMUL', 11, 11, 9)
    kernel.constant(12, 0)
    kernel.arithmetic('FMNMX', 12, 11, 12)
    kernel.constant(13, -1)
    kernel.arithmetic('FMUL', 12, 12, 13)
    kernel.arithmetic('FADD', 12, 11, 12)
    kernel.arithmetic('FMUL', 14, 11, 13)
    kernel.arithmetic('FMNMX', 14, 11, 14)
    kernel.arithmetic('FMUL', 14, 14, 13)
    kernel.exp(15, 12)
    kernel.exp(16, 14)
    kernel.constant(17, 1)
    kernel.arithmetic('FADD', 16, 16, 17)
    kernel.arithmetic('MUFU.RCP', 16, 16)
    kernel.arithmetic('FMUL', 16, 16, 15)
    kernel.arithmetic('FMUL', 16, 16, 8)
    kernel.finish(output, 16)


def lower_operator(op, context):
    """Emit one real materialized output, checking its declaration first."""
    import numpy as np
    emitter = _emitter(op, context)
    output = op.outputs[0]
    if op.op in _REUSED:
        from arch_explorer.components.sm.implementations.accelsim.builtin_emitters import emit_builtin
        from ..extensions import _record_source
        _record_source(emit_builtin)
        return emit_builtin(op, context)
    kernel = emitter.kernel(output)
    index = kernel.index
    if op.op == 'gelu_tanh':
        _gelu(kernel, op.inputs[0], output)
        return emitter.finish_operator()
    if op.op in ('sub', 'mul', 'div'):
        kernel.load(8, op.inputs[0], index)
        kernel.load(9, op.inputs[1], index)
        if op.op == 'sub':
            kernel.constant(10, -1)
            kernel.arithmetic('FMUL', 9, 9, 10)
            kernel.arithmetic('FADD', 8, 8, 9)
        else:
            if op.op == 'div':
                kernel.arithmetic('MUFU.RCP', 9, 9)
            kernel.arithmetic('FMUL', 8, 8, 9)
    elif op.op in ('reshape', 'cast'):
        # Same element order, distinct storage and a real LDG/STG sequence.
        # A cast reaches here only when *both* tensors are declared FP32.
        kernel.load(8, op.inputs[0], index)
    elif op.op == 'broadcast':
        from arch_explorer.components.sm.implementations.accelsim.builtin_emitters import _broadcast
        from ..extensions import _record_source
        _record_source(_broadcast)
        kernel.load(8, op.inputs[0], _broadcast(index, context.tensors[output].shape,
                                               context.tensors[op.inputs[0]].shape))
    elif op.op in ('reduce_max', 'reduce_sum'):
        width = context.tensors[op.inputs[0]].shape[-1]
        kernel.constant(8, -np.inf if op.op == 'reduce_max' else 0)
        instruction = 'FMNMX' if op.op == 'reduce_max' else 'FADD'
        for column in range(width):
            kernel.load(9, op.inputs[0], index * width + column)
            kernel.arithmetic(instruction, 8, 8, 9)
    else:
        raise ValueError('unimplemented materialized native operator: ' + op.op)
    kernel.finish(output, 8)
    return emitter.finish_operator()


def install():
    from ..execution import ImplementationDefinition, register_implementation
    from arch_explorer.components.sm.implementations.accelsim.lowering_limits import CAPABILITIES
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS = tuple(ImplementationDefinition(
            'compiled.accelsim.' + kind, '1', kind, 'accelsim', lower_operator,
            requirements=CAPABILITIES, dtypes=('fp32',), supports=supports)
            for kind in KINDS)
    for definition in _DEFINITIONS:
        register_implementation(definition)
