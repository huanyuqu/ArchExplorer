"""Builtin implementations registered through the public target selector.

Only this implementation family owns primitive-specific scalar algorithms.
The program assembler and providers consume Emission and know no operator IDs.
"""
import math
import numpy as np
from arch_explorer.components.accelsim.lowering_limits import CAPABILITIES, LIMITS

_NAMES = ('matmul','view','patchify','split_heads','merge_heads','transpose',
          'cls_concat','lookup','select','argmax','causal_mask','bias','add','scale',
          'mask','softmax','affine_norm','gelu_exact','quick_gelu','l2_norm','exp')
_DEFINITIONS = None


def supported(op, context):
    if len(op.outputs) != 1 or any(context.tensors[n].dtype not in ('fp32','int32')
                                   for n in (*op.inputs,*op.outputs)):
        return False
    if op.op == 'matmul':
        a = op.attrs
        return (a.get('kv_group',1) == 1
                and math.prod(a[k] for k in ('batch','m','k','n')) <= LIMITS['matrix_fmas']
                and 'input_elements' not in a)
    return True


def install():
    from arch_explorer.workload.execution import ImplementationDefinition, register_implementation
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS = tuple(ImplementationDefinition(
            id='builtin.accelsim.scalar.'+name, version='1', operator=name,
            target='accelsim', lower=emit_builtin, requirements=CAPABILITIES,
            dtypes=('fp32','int32'), supports=supported) for name in _NAMES)
    for definition in _DEFINITIONS:
        register_implementation(definition)


def _broadcast(indices, output_shape, input_shape):
    coordinates = np.unravel_index(indices, output_shape)
    pad = len(output_shape)-len(input_shape)
    return np.ravel_multi_index(tuple(np.zeros_like(coordinates[pad+j]) if n == 1 else coordinates[pad+j]
        for j, n in enumerate(input_shape)), input_shape)



def emit_builtin(op, context):
    emitter = context.options['emitter']
    graph, memory = emitter.graph, emitter.memory
    out = op.outputs[0]
    tensor = graph.tensors[out]
    shape = tuple(tensor.shape)
    if op.op == 'view':
        emitter.alias(out, op.inputs[0], op.attrs.get('axes'))
        return emitter.finish_operator()
    k = emitter.kernel(out)
    i, attrs = k.index, op.attrs
    inputs = [graph.tensors[x] for x in op.inputs]
    coord = np.unravel_index(i, shape)
    if op.op == 'matmul':
        b, m, inner, width = (attrs[x] for x in ('batch','m','k','n'))
        batch, row, col = i//(m*width), i//width % m, i % width
        k.constant(8, 0)
        for j in range(inner):
            k.load(9, op.inputs[0], (batch*m+row)*inner+j)
            k.load(10, op.inputs[1], ((0 if attrs.get('shared_b') else batch)*inner+j)*width+col)
            k.arithmetic('FFMA', 8, 9, 10, 8)
        register = 8
    elif op.op in ('bias','add','mask'):
        for r, t in zip((8,9), inputs): k.load(r, t.name, _broadcast(i, shape, t.shape))
        k.arithmetic('FADD', 8, 8, 9); register = 8
    elif op.op == 'scale':
        k.load(8, op.inputs[0], i)
        if len(inputs) == 2: k.load(9, op.inputs[1], _broadcast(i, shape, inputs[1].shape))
        else: k.constant(9, attrs['factor'])
        k.arithmetic('FMUL', 8, 8, 9); register = 8
    elif op.op in ('affine_norm','l2_norm','softmax'):
        width = shape[-1]; row = i//width; col = i % width
        if op.op == 'softmax':
            k.constant(8, -np.inf)
            for j in range(width):
                k.load(10, op.inputs[0], row*width+j)
                k.arithmetic('FMNMX', 8, 8, 10)
            k.constant(20, -1); k.arithmetic('FMUL', 8, 8, 20)
            k.constant(9, 0)
            for j in range(width):
                k.load(10, op.inputs[0], row*width+j)
                k.arithmetic('FADD', 11, 10, 8); k.exp(12, 11)
                k.arithmetic('FADD', 9, 9, 12)
            k.arithmetic('MUFU.RCP', 9, 9)
            k.load(10, op.inputs[0], i); k.arithmetic('FADD', 11, 10, 8); k.exp(12, 11)
            k.arithmetic('FMUL', 16, 12, 9)
        elif op.op == 'l2_norm':
            k.constant(8, 0)
            for j in range(width):
                k.load(10, op.inputs[0], row*width+j); k.arithmetic('FFMA', 8, 10, 10, 8)
            k.arithmetic('MUFU.RSQ', 8, 8); k.load(10, op.inputs[0], i)
            k.arithmetic('FMUL', 16, 10, 8)
        else:
            k.constant(8, 0); k.constant(21, 1/width); k.constant(20, -1)
            for j in range(width):
                k.load(10, op.inputs[0], row*width+j); k.arithmetic('FADD', 8, 8, 10)
            k.arithmetic('FMUL', 9, 8, 21); k.arithmetic('FMUL', 9, 9, 20)
            k.constant(12, 0)
            for j in range(width):
                k.load(10, op.inputs[0], row*width+j); k.arithmetic('FADD', 11, 10, 9)
                k.arithmetic('FFMA', 12, 11, 11, 12)
            k.constant(22, attrs['epsilon']); k.arithmetic('FFMA', 12, 12, 21, 22)
            k.arithmetic('MUFU.RSQ', 13, 12)
            k.load(10, op.inputs[0], i); k.arithmetic('FADD', 11, 10, 9)
            k.arithmetic('FMUL', 11, 11, 13)
            k.load(14, op.inputs[1], col); k.load(15, op.inputs[2], col)
            k.arithmetic('FFMA', 16, 11, 14, 15)
        register = 16
    elif op.op in ('quick_gelu','gelu_exact','exp'):
        k.load(8, op.inputs[0], i)
        if op.op == 'exp': k.exp(16, 8)
        elif op.op == 'quick_gelu':
            k.constant(9, -1.702); k.arithmetic('FMUL', 10, 8, 9); k.exp(11, 10)
            k.constant(12, 1); k.arithmetic('FADD', 11, 11, 12)
            k.arithmetic('MUFU.RCP', 11, 11); k.arithmetic('FMUL', 16, 8, 11)
        else:
            # Five-coefficient erf approximation, checked against the exact
            # numerical GELU at every element. No fictional single erf ISA.
            k.constant(9, 1/math.sqrt(2)); k.arithmetic('FMUL', 10, 8, 9)
            k.arithmetic('FMUL', 11, 10, 10); k.constant(12, 1e-30)
            k.arithmetic('FADD', 12, 11, 12); k.arithmetic('MUFU.RSQ', 12, 12)
            k.arithmetic('FMUL', 13, 11, 12); k.arithmetic('FMUL', 14, 10, 12)
            k.constant(15, .3275911); k.constant(17, 1)
            k.arithmetic('FFMA', 15, 13, 15, 17); k.arithmetic('MUFU.RCP', 15, 15)
            k.constant(19, 1.061405429)
            for c in (-1.453152027, 1.421413741, -.284496736, .254829592):
                k.constant(20, c); k.arithmetic('FFMA', 19, 19, 15, 20)
            k.arithmetic('FMUL', 19, 19, 15)
            k.constant(20, -1); k.arithmetic('FMUL', 11, 11, 20); k.exp(21, 11)
            k.arithmetic('FMUL', 19, 19, 21); k.arithmetic('FFMA', 19, 19, 20, 17)
            k.arithmetic('FMUL', 19, 19, 14); k.arithmetic('FADD', 19, 19, 17)
            k.constant(20, .5); k.arithmetic('FMUL', 19, 19, 20)
            k.arithmetic('FMUL', 16, 8, 19)
        register = 16
    elif op.op == 'lookup':
        width = shape[-1]; tokens, weights = op.inputs
        k.load(8, tokens, i//width, np.uint32)
        base = int(memory.addresses[weights].reshape(-1)[0])
        k.pointer(9, base+4*(i % width).astype(np.uint32)); k.integer(10, 4*width)
        k.imad(11, 8, 10, 9); k.ldg(16, k.reg[11], np.float32, [11]); register = 16
    elif op.op == 'select':
        width = shape[-1]; source = inputs[0]; seq = source.shape[1]
        if len(inputs) == 2:
            k.load(8, inputs[1].name, i//width, np.uint32)
            base = int(memory.addresses[source.name].reshape(-1)[0])
            k.pointer(9, base+4*((i//width)*seq*width+i % width).astype(np.uint32))
            k.integer(10, width*4); k.imad(11, 8, 10, 9)
            k.ldg(16, k.reg[11], np.float32, [11])
        else: k.load(16, source.name, (i//width)*seq*width+i % width)
        register = 16
    elif op.op == 'argmax':
        width = inputs[0].shape[-1]
        k.integer(8, 0); k.integer(9, width+1)
        for j in range(width):
            k.load(10, op.inputs[0], i*width+j, np.uint32); k.integer(11, width-j)
            k.imad(12, 10, 9, 11); k.arithmetic('IMNMX.U32', 8, 8, 12)
        if np.any(k.reg[8] >= 2**20): raise ValueError('token argmax encoding exceeds exact conversion bound')
        k.arithmetic('I2F.U32', 13, 8); k.constant(14, 1/(width+1))
        k.arithmetic('FMUL', 13, 13, 14); k.arithmetic('F2I.U32.TRUNC', 13, 13)
        k.integer(14, 0xffffffff); k.integer(15, 0); k.imad(13, 13, 9, 15)
        k.imad(13, 13, 14, 8); k.integer(15, width); k.imad(16, 13, 14, 15)
        register = 16
    elif op.op == 'causal_mask':
        values = np.where(coord[0] < coord[1], -np.inf, 0).astype(np.float32)
        literal = emitter.literal('causal_mask',values.reshape(shape))
        k.load(16, literal, i); register = 16
    else:
        source = inputs[0]; xshape = tuple(source.shape)
        if op.op == 'patchify':
            patch = attrs['patch']; _, _, side, _ = xshape; patches = side//patch
            b, token, entry = coord
            channel, dy, dx = entry//(patch*patch), entry//patch % patch, entry % patch
            index = ((b*3+channel)*side+(token//patches)*patch+dy)*side+(token % patches)*patch+dx
        elif op.op == 'split_heads':
            heads = attrs['heads']; bhead, middle, last = coord; batch, head = bhead//heads, bhead % heads
            seq, dim = xshape[1], xshape[2]//heads
            pos, feature = (last, middle) if attrs['key'] else (middle, last)
            index = (batch*seq+pos)*xshape[2]+head*dim+feature
        elif op.op == 'merge_heads':
            b, pos, feature = coord; heads = attrs['heads']; dim = shape[-1]//heads
            index = ((b*heads+feature//dim)*xshape[1]+pos)*dim+feature % dim
        elif op.op == 'transpose':
            axes = attrs['axes']; inverse = np.argsort(axes)
            index = np.ravel_multi_index(tuple(coord[j] for j in inverse), xshape)
        elif op.op == 'cls_concat':
            b, pos, feature = coord; width = shape[-1]
            xaddr = memory.addresses[source.name].reshape(-1)[(b*(shape[1]-1)+np.maximum(pos-1,0))*width+feature]
            clsaddr = memory.addresses[inputs[1].name].reshape(-1)[feature]
            k.gather(16, np.where(pos == 0, clsaddr, xaddr)); index = None
        else: raise ValueError('unsupported complete lowering '+op.op)
        if index is not None: k.load(16, source.name, index)
        register = 16

    dtype = np.float32 if tensor.dtype == 'fp32' else np.int32
    k.finish(out, register, dtype)
    return emitter.finish_operator()

