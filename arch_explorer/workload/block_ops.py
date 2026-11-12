"""Physical FP32 blocks with explicit storage, reads and ordered accumulation.

Partitioning changes the input placement boundary. It does not allocate a hidden
canonical cache, and it does not claim that distributing an existing device
buffer is free. ``block.load`` performs every later copy as an ordinary task.
"""
import hashlib
import itertools
import math
from dataclasses import asdict

KINDS = ('block.load', 'block.zero', 'block.matmul_accumulate')
_DEFINITIONS = None


def _record(function):
    from .extensions import _record_source
    _record_source(function)


def _slice(starts, shape):
    return tuple(slice(a, a+b) for a, b in zip(starts, shape))


def _digest(array):
    from .precision import storage_bytes
    return hashlib.sha256(storage_bytes(array, 'fp32')).hexdigest()


def partition_input(tensor, values, *, block_shape):
    """Return independent rectangular roots and an invertible source manifest.

    Blocks are C-order, clipped at the boundary, enumerated lexicographically
    by their starts. Every original element occurs exactly once. The returned
    JSON can be saved independently and supplies the entire physical root set.
    """
    from .operators import Tensor
    from .reference_api import _input
    _record(partition_input); _record(_input)
    if (type(tensor) is not Tensor or tensor.dtype != 'fp32' or tensor.alias_of is not None
            or tensor.kind == 'kv' or not 1 <= len(tensor.shape) <= 4):
        raise ValueError('block.partition: requires independent FP32 source')
    shape = tensor.shape
    if (not isinstance(block_shape, (list, tuple)) or len(block_shape) != len(shape)
            or any(type(n) is not int or n < 1 for n in block_shape)):
        raise ValueError('block.partition: positive block_shape must match source rank')
    if math.prod(math.ceil(a/b) for a, b in zip(shape, block_shape)) > 4096:
        raise ValueError('block.partition: at most 4096 physical pieces')
    array = _input(values, tensor)
    pieces, tensors, inputs = [], {}, {}
    for i, start in enumerate(itertools.product(*(range(0, n, b) for n, b in zip(shape, block_shape)))):
        extent = tuple(min(b, n-a) for n, a, b in zip(shape, start, block_shape))
        name = tensor.name + '.block.' + str(i)
        child = Tensor(name, extent, 'fp32', tensor.kind)
        tensors[name] = {**asdict(child),'shape':list(child.shape)}
        inputs[name] = array[_slice(start, extent)].tolist()
        pieces.append(dict(tensor=name, starts=list(start), shape=list(extent), bytes=child.bytes))
    manifest = dict(schema='arch-explorer.physical-partition.v1', source={**asdict(tensor),'shape':list(shape)},
                    source_sha256=_digest(array), blocks=pieces,
                    boundary='partitioned roots preloaded once; later copies are explicit block.load tasks')
    return dict(partition=manifest, tensors=tensors, inputs=inputs)


def restore_partition(partition, inputs):
    """Validate exact coverage, payloads and source hash, then restore one root."""
    import numpy as np
    from .operators import Tensor
    from .reference_api import _input
    _record(restore_partition); _record(_input)
    if (type(partition) is not dict or set(partition) != {'schema','source','source_sha256','blocks','boundary'}
            or partition['schema'] != 'arch-explorer.physical-partition.v1'):
        raise ValueError('block.partition: manifest fields/version differ')
    source = Tensor(**partition['source'])
    if source.dtype != 'fp32' or source.alias_of is not None or source.kind == 'kv' or not 1 <= len(source.shape) <= 4:
        raise ValueError('block.partition: source must be independent FP32')
    pieces = partition['blocks']
    if type(pieces) is not list or not 1 <= len(pieces) <= 4096:
        raise ValueError('block.partition: missing or excessive pieces')
    names, boxes, volume = set(), [], 0
    for piece in pieces:
        if type(piece) is not dict or set(piece) != {'tensor','starts','shape','bytes'}:
            raise ValueError('block.partition: piece fields differ')
        start, extent, name = piece['starts'], piece['shape'], piece['tensor']
        if (type(name) is not str or name in names or type(start) is not list or type(extent) is not list
                or len(start) != len(source.shape) or len(extent) != len(start)
                or any(type(a) is not int or type(b) is not int or a < 0 or b < 1 or a+b > n
                       for a,b,n in zip(start,extent,source.shape))):
            raise ValueError('block.partition: duplicate or out-of-range piece')
        if type(piece['bytes']) is not int or piece['bytes'] != math.prod(extent)*4:
            raise ValueError('block.partition: piece bytes differ')
        for other, size in boxes:
            if all(a < c+d and c < a+b for a,b,c,d in zip(start,extent,other,size)):
                raise ValueError('block.partition: overlapping pieces')
        names.add(name); boxes.append((start,extent)); volume += math.prod(extent)
    if volume != math.prod(source.shape):
        raise ValueError('block.partition: incomplete source coverage')
    if type(inputs) is not dict or set(inputs) != names:
        raise ValueError('block.partition: exact physical input set required')
    array = np.empty(source.shape, dtype=np.float32)
    for piece in pieces:
        t = Tensor(piece['tensor'], tuple(piece['shape']), 'fp32', source.kind)
        array[_slice(piece['starts'],piece['shape'])] = _input(inputs[t.name], t)
    if _digest(array) != partition['source_sha256']:
        raise ValueError('block.partition: source digest differs')
    array.flags.writeable = False
    return array


def validate(op, tensors):
    ins, outs = [tensors[n] for n in op.inputs], [tensors[n] for n in op.outputs]
    if (len(outs) != 1 or any(t.dtype != 'fp32' or t.kind == 'kv' or not 1<=len(t.shape)<=4 for t in (*ins,*outs))
            or outs[0].alias_of is not None or outs[0].strides is not None
            or op.attrs.get('shape') != list(outs[0].shape)):
        raise ValueError('block.contract: independent FP32 output and exact shape required')
    if op.op == 'block.load':
        if not ins or set(op.attrs) != {'shape','starts'}:
            raise ValueError('block.load: source and exact shape/starts required')
        start, shape = op.attrs['starts'], outs[0].shape
        if (type(start) is not list or len(start) != len(ins[0].shape) or len(shape) != len(start)
                or any(type(a) is not int or a < 0 or a+b > n for a,b,n in zip(start,shape,ins[0].shape))):
            raise ValueError('block.load: region outside source')
    elif op.op == 'block.zero':
        if ins or set(op.attrs) != {'shape'}:
            raise ValueError('block.zero: no inputs and exact shape required')
    elif op.op == 'block.matmul_accumulate':
        if (len(ins) != 3 or set(op.attrs) != {'shape'} or any(len(t.shape) != 3 for t in ins)
                or len(outs[0].shape) != 3):
            raise ValueError('block.matmul_accumulate: three rank-three operands required')
        a,b,c = (t.shape for t in ins)
        if a[0] != b[0] or a[2] != b[1] or c != (a[0],a[1],b[2]) or c != outs[0].shape:
            raise ValueError('block.matmul_accumulate: batched matrix dimensions differ')
        from .resource_budget import limit
        if a[0]*a[1]*a[2]*b[2] > limit('max_matrix_fmas',100_000_000):
            raise ValueError('block.matmul_accumulate: matrix FMA budget exceeded')
    else:
        raise ValueError('block.contract: unknown operator')


def reference(op, inputs):
    import numpy as np
    _record(reference)
    if op.op == 'block.load':
        return inputs[0][_slice(op.attrs['starts'],op.attrs['shape'])].copy()
    if op.op == 'block.zero':
        return np.zeros(op.attrs['shape'], dtype=np.float32)
    if op.op == 'block.matmul_accumulate':
        a,b,c = inputs
        result = c.copy()
        with np.errstate(over='raise',invalid='raise'):
            for j in range(a.shape[2]):
                result = np.float32(a[:,:,j,None].astype(np.float64) * b[:,None,j,:].astype(np.float64)
                                    + result.astype(np.float64))
        return result
    raise ValueError('block.contract: unknown reference')


def access_indices(op, tensors, values):
    """Yield every logical input read; the common storage API owns writes.

    Addresses of these operators do not depend on payload values. Seed C is
    read once, then A/B once per FFMA in increasing K order. Load guards are
    complete reads, including repeated input names; no reads are deduplicated.
    """
    _record(access_indices); validate(op,tensors)
    if op.op == 'block.load':
        for index in itertools.product(*(range(n) for n in op.attrs['shape'])):
            yield op.inputs[0],tuple(a+i for a,i in zip(op.attrs['starts'],index))
        for name in op.inputs[1:]:
            for index in itertools.product(*(range(n) for n in tensors[name].shape)):
                yield name,index
    elif op.op == 'block.matmul_accumulate':
        a,b,c=(tensors[n].shape for n in op.inputs)
        for index in itertools.product(*(range(n) for n in c)):
            yield op.inputs[2],index
        for k in range(a[2]):
            for batch,row,col in itertools.product(range(a[0]),range(a[1]),range(b[2])):
                yield op.inputs[0],(batch,row,k)
                yield op.inputs[1],(batch,k,col)


def install():
    global _DEFINITIONS
    from .extensions import OperatorDefinition, register_operator
    if _DEFINITIONS is None:
        _DEFINITIONS = tuple(OperatorDefinition(n,validate,reference=reference) for n in KINDS)
    for definition in _DEFINITIONS:
        register_operator(definition)
