"""Typed gathers, selections and exact integer arithmetic for ordinary graphs.

Validation, numerical domains and abstract issue/data counts live here.
Mapping and task adapters consume these contracts without deriving formulas.
FP32 Kernel precision denotes an abstract SIMT carrier, not a data conversion.
"""
import math

INTEGER_KINDS = ('integer_add', 'integer_sub', 'integer_mul', 'integer_div')
COMPARISON_KINDS = ('equal', 'greater', 'less')
KINDS = ('gather', 'gather_elements', 'argmax_axis', 'where',
         *INTEGER_KINDS, *COMPARISON_KINDS)
STORAGE_DTYPES = ('fp32', 'fp16', 'bf16', 'int32', 'int64', 'bool')
_DEFINITIONS = None


class ArraySemanticError(ValueError):
    """Stable semantic refusal; ``code`` is independent of node names."""

    def __init__(self, code, detail):
        self.code = code
        super().__init__('array.' + code + ': ' + detail)


def _fail(code, detail):
    raise ArraySemanticError(code, detail)


def _shape(value):
    if (not isinstance(value, (tuple, list)) or not value
            or any(type(n) is not int or n < 1 for n in value)):
        _fail('shape', 'shape must contain positive integer dimensions')
    return tuple(value)


def _attributes(op):
    if op.op not in KINDS:
        _fail('kind', 'unsupported array operator ' + str(op.op))
    required = {'shape'}
    optional = set()
    if op.op in ('gather', 'gather_elements', 'argmax_axis'):
        required.add('axis')
    if op.op == 'gather':
        optional.add('indices_scalar')
    if op.op == 'argmax_axis':
        required.add('keepdims')
        optional.add('select_last_index')
    if (type(op.attrs) is not dict or not required <= set(op.attrs)
            or set(op.attrs) - required - optional):
        _fail('attributes', op.op + ' attribute fields differ')
    for key in ('keepdims', 'select_last_index', 'indices_scalar'):
        if key in op.attrs and type(op.attrs[key]) is not bool:
            _fail('attributes', key + ' must be bool')
    return _shape(op.attrs['shape'])


def _axis(op, rank):
    value = op.attrs['axis']
    if type(value) is not int or not -rank <= value < rank:
        _fail('axis', 'axis must be an integer in [-rank, rank)')
    return value % rank


def _contract(op, shapes, dtypes, output_shape, output_dtype):
    """Shared shape/type contract for Tensor metadata and actual arrays."""
    from .resource_budget import limit
    declared = _attributes(op)
    count = 1 if op.op == 'argmax_axis' else 3 if op.op == 'where' else 2
    if len(shapes) != count or len(dtypes) != count:
        _fail('arity', op.op + ' input count differs')
    all_shapes = [*shapes, output_shape]
    if any(not 1 <= len(s) <= limit('max_rank', 4) for s in all_shapes):
        _fail('rank', 'array rank exceeds the resource budget')
    for value in all_shapes:
        _shape(value)
    if declared != output_shape:
        _fail('shape', 'declared shape differs from output')
    if any(d not in STORAGE_DTYPES for d in (*dtypes, output_dtype)):
        _fail('dtype', 'unsupported array storage dtype')
    source = shapes[0]
    axis = None
    if op.op in ('gather', 'gather_elements', 'argmax_axis'):
        axis = _axis(op, len(source))
    if op.op in ('gather', 'gather_elements'):
        if dtypes[1] not in ('int32', 'int64') or output_dtype != dtypes[0]:
            _fail('dtype', 'gather requires int32/int64 indices and unchanged data storage')
        if op.op == 'gather':
            scalar = op.attrs.get('indices_scalar', False)
            if scalar and shapes[1] != (1,):
                _fail('shape', 'logical scalar indices require physical shape (1,)')
            expected = source[:axis] + (() if scalar else shapes[1]) + source[axis+1:]
            if output_shape != (expected or (1,)):
                _fail('shape', 'gather output must replace the selected axis with index dimensions')
        elif (len(source) != len(shapes[1]) or output_shape != shapes[1]
                or any(n > source[i] for i, n in enumerate(shapes[1]) if i != axis)):
            _fail('shape', 'gather_elements requires equal ranks and fitting non-axis dimensions')
    elif op.op == 'argmax_axis':
        if dtypes[0] == 'bool' or output_dtype != 'int64':
            _fail('dtype', 'argmax_axis requires numeric input and int64 output')
        expected = source[:axis] + ((1,) if op.attrs['keepdims'] else ()) + source[axis+1:]
        if output_shape != (expected or (1,)):
            _fail('shape', 'argmax_axis output differs from axis/keepdims')
    else:
        if any(s != output_shape for s in shapes):
            _fail('shape', 'elementwise operands require equal shapes; broadcast explicitly')
        if op.op == 'where':
            if dtypes[0] != 'bool' or dtypes[1] != dtypes[2] or output_dtype != dtypes[1]:
                _fail('dtype', 'where requires bool condition and matching branch/output storage')
        elif op.op in INTEGER_KINDS:
            if dtypes[0] not in ('int32', 'int64') or any(d != dtypes[0] for d in (*dtypes, output_dtype)):
                _fail('dtype', 'integer arithmetic requires matching int32/int64 storage')
        elif dtypes[0] != dtypes[1] or output_dtype != 'bool':
            _fail('dtype', 'comparison requires matching operand storage and bool output')
    return axis


def validate(op, tensors):
    """Check metadata; dynamic index/arithmetic domains are checked by reference."""
    _attributes(op)
    if len(op.outputs) != 1:
        _fail('arity', 'array operators require exactly one output')
    try:
        inputs = [tensors[n] for n in op.inputs]
        output = tensors[op.outputs[0]]
    except KeyError as exc:
        _fail('arity', 'dangling tensor ' + str(exc))
    if any(t.kind == 'kv' for t in (*inputs, output)):
        _fail('storage', 'array inputs/outputs cannot use append-KV storage')
    if output.alias_of is not None or output.strides is not None:
        _fail('storage', 'array output must be independently materialized')
    _contract(op, [t.shape for t in inputs], [t.dtype for t in inputs], output.shape, output.dtype)


def _runtime_arrays(inputs):
    import numpy as np
    arrays = [np.asarray(v) for v in inputs]
    names = {'float32': 'fp32', 'float16': 'fp16', 'int32': 'int32', 'int64': 'int64', 'bool': 'bool'}
    dtypes = []
    for value in arrays:
        dtype = names.get(value.dtype.name)
        if dtype is None:
            _fail('reference_values', 'reference requires declared storage arrays, not coerced/object values')
        if value.dtype.kind == 'f' and not np.all(np.isfinite(value)):
            _fail('reference_values', 'floating array values must be finite')
        dtypes.append(dtype)
    return arrays, dtypes


def _index_array(value, tensor=None):
    """Decode audit JSON exactly or require actual signed integer arrays."""
    import numpy as np
    if isinstance(value, np.ndarray):
        if value.dtype.kind != 'i' or value.dtype.itemsize not in (4, 8):
            _fail('index_values', 'indices must be actual int32/int64 values')
        if tensor is not None and value.dtype.name != tensor.dtype:
            _fail('index_values', 'index array dtype differs from declared Tensor')
        result = value
    else:
        from .integer import exact_storage
        from .extensions import _record_source
        _record_source(exact_storage)
        try:
            result = exact_storage(value, tensor.dtype if tensor is not None else 'int64')
        except (TypeError, ValueError) as exc:
            _fail('index_values', str(exc))
    if tensor is not None and result.shape != tensor.shape:
        _fail('index_values', 'index values must have the complete declared shape')
    return result


def _normalized_indices(indices, dimension):
    import numpy as np
    if np.any(indices < -dimension) or np.any(indices >= dimension):
        _fail('index_oob', 'index must be in [-dimension, dimension)')
    # Widen signed indices exactly; never route addresses through floating point.
    result = indices.astype(np.int64, copy=True)
    negative = result < 0
    result[negative] += dimension
    return result


def _integer_reference(kind, lhs, rhs):
    import numpy as np
    bounds = np.iinfo(lhs.dtype)
    lower, upper = int(bounds.min), int(bounds.max)
    result = np.empty(lhs.shape, dtype=lhs.dtype)
    for i, (left, right) in enumerate(zip(lhs.flat, rhs.flat)):
        a, b = int(left), int(right)
        if kind == 'integer_add':
            value = a + b
        elif kind == 'integer_sub':
            value = a - b
        elif kind == 'integer_mul':
            value = a * b
        else:
            if b == 0:
                _fail('integer_div_zero', 'integer divisor must be nonzero')
            quotient = abs(a) // abs(b)
            value = -quotient if (a < 0) != (b < 0) else quotient
        if not lower <= value <= upper:
            _fail('integer_overflow', 'integer result exceeds declared signed storage')
        result.flat[i] = value
    return result


def reference(op, inputs):
    """Execute exact integer/selection semantics on the supplied storage values."""
    import numpy as np
    output_shape = _attributes(op)
    arrays, dtypes = _runtime_arrays(inputs)
    output_dtype = ('int64' if op.op == 'argmax_axis' else 'bool' if op.op in COMPARISON_KINDS
                    else dtypes[1] if op.op == 'where' and len(dtypes) > 1 else dtypes[0] if dtypes else '')
    axis = _contract(op, [a.shape for a in arrays], dtypes, output_shape, output_dtype)
    if op.op in ('gather', 'gather_elements'):
        data = arrays[0]
        indices = _normalized_indices(_index_array(arrays[1]), data.shape[axis])
        if op.op == 'gather':
            if op.attrs.get('indices_scalar', False):
                indices = indices.reshape(())
            result = np.take(data, indices, axis=axis)
        else:
            region = tuple(slice(None) if i == axis else slice(0, indices.shape[i])
                           for i in range(data.ndim))
            result = np.take_along_axis(data[region], indices, axis=axis)
    elif op.op == 'argmax_axis':
        data = arrays[0]
        if op.attrs.get('select_last_index', False):
            result = data.shape[axis] - 1 - np.argmax(np.flip(data, axis=axis), axis=axis)
        else:
            result = np.argmax(data, axis=axis)
        result = np.asarray(result, dtype=np.int64)
        if op.attrs['keepdims']:
            result = np.expand_dims(result, axis=axis)
    elif op.op == 'where':
        result = np.where(*arrays)
    elif op.op in INTEGER_KINDS:
        result = _integer_reference(op.op, *arrays)
    else:
        function = {'equal': np.equal, 'greater': np.greater, 'less': np.less}[op.op]
        result = function(*arrays)
    return np.asarray(result).reshape(output_shape).copy()


def access_counts(op, tensors):
    """Return typed component-I/O bytes and synthetic scalar issue counts."""
    from arch_explorer.components.base import dtype_bytes
    validate(op, tensors)
    output = tensors[op.outputs[0]]
    n = math.prod(output.shape)
    sizes = [n] * len(op.inputs)
    simt = n
    reuse = 'each input slot is read once per output element; repeated names accumulate'
    if op.op == 'gather':
        indices = math.prod(tensors[op.inputs[1]].shape)
        sizes = [n, indices]
        simt = 3 * indices + n
        reuse = 'each index is read once; repeated indices repeat data reads, no hidden cache'
    elif op.op == 'gather_elements':
        simt = 4 * n
        reuse = 'one index and one selected data element per output, including repeated indices'
    elif op.op == 'argmax_axis':
        source = tensors[op.inputs[0]].shape
        dimension = source[_axis(op, len(source))]
        sizes = [math.prod(source)]
        simt = 2 * n * (dimension - 1)
        reuse = 'each input element once; compare/index-select for each later candidate'
    elif op.op == 'where':
        reuse = 'eager condition and both branches; no value-dependent traffic reduction'
    reads = {}
    for name, count in zip(op.inputs, sizes):
        reads[name] = reads.get(name, 0) + count * dtype_bytes(tensors[name].dtype)
    writes = {op.outputs[0]: output.bytes}
    return dict(operations=simt, read_bytes=sum(reads.values()), write_bytes=output.bytes,
                boundary='component_io', read_by_tensor=reads, write_by_tensor=writes,
                simt_instructions=simt, sfu_instructions=0, conversion_instructions=0,
                input_dtypes=[tensors[n].dtype for n in op.inputs], output_dtype=output.dtype,
                accumulation='exact signed integer / direct storage comparison / bit-preserving selection',
                cost='synthetic scalar issues: gather two bounds + normalization + address; argmax compare/index-select; one eager select, integer arithmetic or comparison per output; no device cycle calibration',
                index_reuse=reuse)


def access_indices(op, tensors, values):
    """Yield only logical reads; the storage auditor owns output writes."""
    import numpy as np
    validate(op, tensors)
    output = tensors[op.outputs[0]]
    if op.op in ('gather', 'gather_elements'):
        index_name = op.inputs[1]
        if type(values) is not dict or index_name not in values:
            _fail('missing_values', 'logical gather audit requires complete index values for ' + index_name)
        source = tensors[op.inputs[0]].shape
        axis = _axis(op, len(source))
        indices = _normalized_indices(_index_array(values[index_name], tensors[index_name]), source[axis])
        if op.op == 'gather':
            for position in np.ndindex(indices.shape):
                yield index_name, position
            for prefix in np.ndindex(source[:axis]):
                for position in np.ndindex(indices.shape):
                    selected = int(indices[position])
                    for suffix in np.ndindex(source[axis+1:]):
                        yield op.inputs[0], prefix + (selected,) + suffix
        else:
            for position in np.ndindex(indices.shape):
                yield index_name, position
                selected = list(position)
                selected[axis] = int(indices[position])
                yield op.inputs[0], tuple(selected)
    elif op.op == 'argmax_axis':
        source = tensors[op.inputs[0]].shape
        axis = _axis(op, len(source))
        for remaining in np.ndindex(source[:axis] + source[axis+1:]):
            for selected in range(source[axis]):
                yield op.inputs[0], remaining[:axis] + (selected,) + remaining[axis:]
    else:
        for position in np.ndindex(output.shape):
            for name in op.inputs:
                yield name, position


def lower_kernel(op, context):
    """Project the same abstract counts into the standard events Kernel."""
    from arch_explorer.components.base import Kernel
    if context.dtype != 'fp32':
        _fail('dtype', 'array Kernel requires the abstract FP32 issue carrier')
    counts = access_counts(op, context.tensors)
    supplied = context.options.get('access')
    if supplied is not None:
        from arch_explorer.mapping.rules import validate_access
        supplied = validate_access(supplied)
        for key in ('operations', 'read_bytes', 'write_bytes', 'read_by_tensor', 'write_by_tensor',
                    'simt_instructions', 'sfu_instructions', 'conversion_instructions'):
            if supplied.get(key) != counts[key]:
                _fail('access', 'adapter access differs from semantic counts: ' + key)
    return Kernel('vector', (1, 1), 'fp32', row_simt=counts['simt_instructions'])


def install():
    """Register cached semantic definitions; target adapters own implementations."""
    from .extensions import OperatorDefinition, register_operator
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS = tuple(OperatorDefinition(k, validate, reference=reference) for k in KINDS)
    for definition in _DEFINITIONS:
        register_operator(definition)
