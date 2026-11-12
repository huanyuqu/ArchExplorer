"""Materialized matrix regions shared by the compilation target adapters.

These operators retain their source matrix dimensions, exact output regions
and ordered reduction loops. They own numerical semantics and data-access
counts; placement, allocation, lifetime and scheduling remain mapping facts.
Target products are registered by their adapters, independently of semantics.
"""
from __future__ import annotations

import math
from .resource_budget import limit

# Bounds are checked before allocating masks, reference outputs or trace code.
# Native has its additional, smaller limits in lowering_limits.LIMITS.
LIMITS = {'dimension': 4096, 'tensor_elements': 1_000_000, 'rank': 32,
          'matrix_fmas': 100_000_000, 'tiles': 4096}
_MATMUL = {'batch', 'm', 'k', 'n', 'shared_b', 'row', 'col',
           'rows', 'cols', 'k_tile'}
_ASSEMBLE = {'batch', 'm', 'n', 'tiles', 'output_shape'}
_REGION = {'row', 'col', 'rows', 'cols'}
_DEFINITIONS = None


def _integer(value, field, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f'compiled {field} must be a ' +
                         ('nonnegative' if zero else 'positive') + ' integer')
    if value > limit('max_dimension',LIMITS['dimension']):
        raise ValueError(f'compiled {field} exceeds dimension bound')
    return value


def _region(value, m, n):
    if type(value) is not dict or set(value) != _REGION:
        raise ValueError('compiled tile requires exactly row/col/rows/cols')
    for key in _REGION:
        _integer(value[key], key, zero=key in ('row', 'col'))
    if value['row'] + value['rows'] > m or value['col'] + value['cols'] > n:
        raise ValueError('compiled tile exceeds the original matrix region')
    return value


def _attributes(op):
    from .operators import Operator
    if type(op) is not Operator or type(op.attrs) is not dict:
        raise TypeError('compiled semantics require an Operator with JSON attributes')
    a = op.attrs
    if op.op == 'compiled.matmul':
        if set(a) != _MATMUL:
            raise ValueError('compiled.matmul requires exactly ' + ','.join(sorted(_MATMUL)))
        for key in ('batch', 'm', 'k', 'n', 'k_tile'):
            _integer(a[key], key)
        if type(a['shared_b']) is not bool:
            raise TypeError('compiled shared_b must be bool')
        _region({key: a[key] for key in _REGION}, a['m'], a['n'])
        sizes = (a['batch'] * a['m'] * a['k'],
                 (1 if a['shared_b'] else a['batch']) * a['k'] * a['n'],
                 a['batch'] * a['rows'] * a['cols'])
        if max(sizes) > limit('max_tensor_elements',LIMITS['tensor_elements']):
            raise ValueError('compiled.matmul tensor expansion bound exceeded')
        if sizes[-1] * a['k'] > limit('max_matrix_fmas',LIMITS['matrix_fmas']):
            raise ValueError('compiled.matmul reduction expansion bound exceeded')
    elif op.op == 'compiled.assemble':
        if set(a) != _ASSEMBLE:
            raise ValueError('compiled.assemble requires exactly batch/m/n/tiles/output_shape')
        for key in ('batch', 'm', 'n'):
            _integer(a[key], key)
        if a['batch'] * a['m'] * a['n'] > limit('max_tensor_elements',LIMITS['tensor_elements']):
            raise ValueError('compiled.assemble tensor expansion bound exceeded')
        shape = a['output_shape']
        if (type(shape) is not list or not 1 <= len(shape) <= LIMITS['rank']
                or any(type(n) is not int or n < 1 for n in shape)
                or math.prod(shape) != a['batch'] * a['m'] * a['n']):
            raise ValueError('compiled.assemble output_shape must exactly reshape batch/m/n within the rank bound')
        tiles = a['tiles']
        if type(tiles) not in (list, tuple) or not 1 <= len(tiles) <= limit('max_tiles',LIMITS['tiles']):
            raise ValueError('compiled.assemble requires a bounded nonempty tile sequence')
        for tile in tiles:
            _region(tile, a['m'], a['n'])
        # Disjoint in-bounds rectangles whose areas sum to the full matrix
        # prove exact coverage without allocating a matrix-sized bitmap.
        if sum(t['rows'] * t['cols'] for t in tiles) != a['m'] * a['n']:
            raise ValueError('compiled.assemble tiles do not cover the full matrix')
        active = []
        for tile in sorted(tiles, key=lambda t: t['row']):
            active = [t for t in active if t['row'] + t['rows'] > tile['row']]
            if any(t['col'] < tile['col'] + tile['cols'] and
                   tile['col'] < t['col'] + t['cols'] for t in active):
                raise ValueError('compiled.assemble tiles overlap')
            active.append(tile)
    else:
        raise ValueError('unsupported compiled semantic ' + repr(op.op))
    return a


def _validate(op, tensors):
    from .operators import Tensor
    from .graph import name, validate_tensor
    a = _attributes(op)
    name(op.name, 'compiled operator name')
    if (type(op.inputs) not in (list, tuple) or type(op.outputs) not in (list, tuple)
            or len(op.outputs) != 1):
        raise ValueError('compiled operator requires input names and one output')
    expected = 2 if op.op == 'compiled.matmul' else len(a['tiles'])
    if len(op.inputs) != expected:
        raise ValueError('compiled operand count differs from its semantics')
    if set(op.inputs) & set(op.outputs):
        raise ValueError('compiled operators cannot overwrite an input')
    for key in (*op.inputs, *op.outputs):
        name(key, 'compiled tensor name')
        if key not in tensors or type(tensors[key]) is not Tensor or tensors[key].name != key:
            raise ValueError('compiled tensor identity differs: ' + key)
        tensor = tensors[key]
        validate_tensor(tensor)
        if (tensor.dtype not in ('fp32','fp16','bf16') or tensor.kind == 'kv'
                or (key in op.outputs and (tensor.alias_of is not None or tensor.strides is not None))):
            raise ValueError('compiled outputs require independent floating non-KV storage; inputs may be readonly views')
        if math.prod(tensor.shape) > limit('max_tensor_elements',LIMITS['tensor_elements']):
            raise ValueError('compiled tensor expansion bound exceeded')
    output = tensors[op.outputs[0]]
    if output.kind != 'activation':
        raise ValueError('compiled output must be an activation tensor')
    if op.op == 'compiled.matmul':
        if tensors[op.inputs[0]].dtype != tensors[op.inputs[1]].dtype:
            raise ValueError('compiled matrix operands require equal storage dtype')
        sizes = (a['batch'] * a['m'] * a['k'],
                 (1 if a['shared_b'] else a['batch']) * a['k'] * a['n'])
        if any(math.prod(tensors[n].shape) != size for n, size in zip(op.inputs, sizes)):
            raise ValueError('compiled.matmul operands must be the complete original matrices')
        if output.shape != (a['batch'], a['rows'], a['cols']):
            raise ValueError('compiled.matmul output must have exact batch/rows/cols shape')
    else:
        if any(tensors[n].dtype != output.dtype for n in op.inputs):
            raise ValueError('compiled assembly must preserve tile storage dtype')
        if any(tensors[n].shape != (a['batch'], t['rows'], t['cols'])
               for n, t in zip(op.inputs, a['tiles'])):
            raise ValueError('compiled.assemble input shape differs from its tile')
        if output.shape != tuple(a['output_shape']):
            raise ValueError('compiled.assemble output shape differs from output_shape')
    return a


def validate_matmul(op, tensors):
    if op.op != 'compiled.matmul':
        raise ValueError('expected compiled.matmul')
    _validate(op, tensors)


def validate_assemble(op, tensors):
    if op.op != 'compiled.assemble':
        raise ValueError('expected compiled.assemble')
    _validate(op, tensors)


def reduction_blocks(op):
    """Ordered half-open K intervals; assembly has no reduction blocks."""
    a = _attributes(op)
    if op.op != 'compiled.matmul':
        return []
    return [{'start': start, 'stop': min(start + a['k_tile'], a['k'])}
            for start in range(0, a['k'], a['k_tile'])]


def access_counts(op, tensors=None):
    """Scalar FP32 operand traffic at component_io, independent of timing.

    This counts each actual A/B read and output store, including repeated
    operands. Native address-table metadata and instruction fetch traffic are
    separate from tensor data and remain present in the emitted trace.
    """
    a = _attributes(op)
    from arch_explorer.components.base import dtype_bytes
    dtypes = ['fp32'] * len(op.inputs) if tensors is None else [tensors[n].dtype for n in op.inputs]
    dtype = 'fp32' if tensors is None else tensors[op.outputs[0]].dtype
    widths = [dtype_bytes(d) for d in dtypes]
    out_width = dtype_bytes(dtype)
    if op.op == 'compiled.matmul':
        elements = a['batch'] * a['rows'] * a['cols']
        fmas = elements * a['k']
        conversions = fmas*sum(d != 'fp32' for d in dtypes) + elements*(dtype != 'fp32')
        reads={}
        for n,width in zip(op.inputs,widths):reads[n]=reads.get(n,0)+width*fmas
        return dict(operations=2 * fmas + conversions, read_bytes=sum(widths) * fmas,
                    read_by_tensor=reads,
                    write_bytes=out_width * elements, boundary='component_io',
                    output_elements=elements, reduction_fmas=fmas,
                    conversion_instructions=conversions,input_dtypes=dtypes,output_dtype=dtype,accumulation='fp32',
                    data_access='scalar operands; one A and B read per FMA',
                    address_metadata='excluded from component_io; emitted natively')
    elements = a['batch'] * a['m'] * a['n']
    reads={n:out_width*a['batch']*t['rows']*t['cols'] for n,t in zip(op.inputs,a['tiles'])}
    return dict(operations=0, read_bytes=out_width * elements, write_bytes=out_width * elements,
                read_by_tensor=reads,
                boundary='component_io', output_elements=elements, reduction_fmas=0,
                conversion_instructions=0,input_dtypes=dtypes,output_dtype=dtype,accumulation='copy',
                data_access='materialized tile read and assembled output store',
                address_metadata='excluded from component_io; emitted natively')


def _arrays(inputs, shapes, *, exact=False):
    import numpy as np
    if type(inputs) not in (list, tuple) or len(inputs) != len(shapes):
        raise ValueError('compiled reference operand count differs')
    arrays = []
    for value, shape in zip(inputs, shapes):
        value = np.asarray(value)
        if (value.dtype not in (np.dtype('float32'),np.dtype('float16')) or value.size != math.prod(shape)
                or (exact and value.shape != shape)
                or not np.isfinite(value).all()):
            raise ValueError('compiled reference requires finite contiguous floating operands of declared size')
        arrays.append(value.astype(np.float32).reshape(shape))
    return arrays


def reference_matmul(op, inputs):
    """Original ordered FP32 FMA reduction over a precise source matrix region."""
    import numpy as np
    a = _attributes(op)
    if op.op != 'compiled.matmul':
        raise ValueError('expected compiled.matmul')
    left, right = _arrays(inputs, [(a['batch'], a['m'], a['k']),
                                  (1 if a['shared_b'] else a['batch'], a['k'], a['n'])])
    left = left[:, a['row']:a['row'] + a['rows'], :]
    right = right[:, :, a['col']:a['col'] + a['cols']]
    from .precision import matmul_fp32
    return matmul_fp32(left, right, reduction_blocks(op))


def reference_assemble(op, inputs):
    """Materialize the exact nonoverlapping output regions in row-major order."""
    import numpy as np
    a = _attributes(op)
    if op.op != 'compiled.assemble':
        raise ValueError('expected compiled.assemble')
    arrays = _arrays(inputs, [(a['batch'], t['rows'], t['cols']) for t in a['tiles']], exact=True)
    result = np.empty((a['batch'], a['m'], a['n']), dtype=np.float32)
    for tile, value in zip(a['tiles'], arrays):
        result[:, tile['row']:tile['row'] + tile['rows'],
               tile['col']:tile['col'] + tile['cols']] = value
    return result.reshape(a['output_shape'])


def install():
    """Install only numerical semantics, without importing target capabilities."""
    from .extensions import OperatorDefinition, register_operator
    global _DEFINITIONS
    if _DEFINITIONS is None:
        operators = (
            OperatorDefinition('compiled.matmul', validate_matmul, version='1', reference=reference_matmul),
            OperatorDefinition('compiled.assemble', validate_assemble, version='1', reference=reference_assemble))
        _DEFINITIONS = operators
    for operator in _DEFINITIONS:
        register_operator(operator)
