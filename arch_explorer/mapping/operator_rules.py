"""Builtin mapping policies; all operator-specific regions and data counts.

Matrix tiling retains the original scalar reduction and materialized assembly.
Tensor rules retain existing registered numerical semantics and materialize
every declared output, including broadcast and transpose.
"""
import json
import math

from .rules import MappingExpansion, OperatorMapping, register_operator_mapping
from arch_explorer.workload.operators import Tensor, Operator

_DEFINITIONS = None


def matrix_options(options):
    if type(options) is not dict or set(options) != {'tiles'}:
        raise ValueError('matmul.tiled requires explicit tiles options')
    tiles = options['tiles']
    if (type(tiles) is not dict or set(tiles) != {'m', 'k', 'n'}
            or any(type(v) is not int or not 1 <= v <= 4096 for v in tiles.values())):
        raise ValueError('matmul.tiled tiles require integer m/k/n in 1..4096')
    return dict(tiles=dict(tiles))


def _local(op, kind, row=0, col=0):
    return '@compile/'+json.dumps([op, kind, row, col], separators=(',', ':'), ensure_ascii=True)


def expand_matrix(op, tensors, options):
    from .compilation import MAX_STEPS
    from arch_explorer.workload.resource_budget import limit
    if set(op.attrs)-{'batch', 'm', 'k', 'n', 'shared_b', 'dtype', 'output_dtype'}:
        raise NotImplementedError('matmul.tiled supports ordinary contiguous matrix operands')
    a, tiles = op.attrs, matrix_options(options)['tiles']
    b, m, k, n = a.get('batch', 1), a['m'], a['k'], a['n']
    if math.ceil(m/tiles['m'])*math.ceil(n/tiles['n'])+1 > limit('max_steps',MAX_STEPS):
        raise ValueError('matrix mapped step expansion bound exceeded')
    temps, steps, names, regions = [], [], [], []
    for row in range(0, m, tiles['m']):
        for col in range(0, n, tiles['n']):
            region = dict(row=row, col=col, rows=min(m-row, tiles['m']), cols=min(n-col, tiles['n']))
            name = _local(op.name, 'tile', row, col)
            temps.append(Tensor(name, (b, region['rows'], region['cols']), tensors[op.outputs[0]].dtype))
            steps.append(Operator(name, 'compiled.matmul', tuple(op.inputs), (name,),
                dict(batch=b, m=m, k=k, n=n, shared_b=a.get('shared_b', False), k_tile=tiles['k'], **region), op.label))
            names.append(name)
            regions.append(region)
    steps.append(Operator(_local(op.name, 'assemble'), 'compiled.assemble', tuple(names), tuple(op.outputs),
        dict(batch=b, m=m, n=n, tiles=regions, output_shape=list(tensors[op.outputs[0]].shape)), op.label))
    return MappingExpansion(tuple(temps), tuple(steps))


def matrix_access(op, tensors, options):
    from arch_explorer.workload.compiled_operators import access_counts
    from arch_explorer.workload.extensions import _record_source
    _record_source(access_counts)
    return access_counts(op, tensors)


def matrix_inspect(op, tensors, options):
    from arch_explorer.workload.compiled_operators import reduction_blocks
    from arch_explorer.workload.extensions import _record_source
    _record_source(reduction_blocks)
    box, region = None, None
    if op.op == 'compiled.matmul':
        a, tile = op.attrs, options['tiles']
        active = a['batch']*a['rows']*a['cols']*a['k']
        nominal = a['batch']*tile['m']*tile['n']*math.ceil(a['k']/tile['k'])*tile['k']
        box = dict(nominal_iteration_points=nominal, active_iteration_points=active,
                   masked_iteration_points=nominal-active, note='nominal box is not executed padding work')
        region = {key: a[key] for key in ('row', 'col', 'rows', 'cols')}
    return dict(region=region, iteration_box=box, reduction_blocks=reduction_blocks(op))


def expand_tensor(op, tensors, options):
    from arch_explorer.workload.tensor_ops import validate
    from arch_explorer.workload.extensions import _record_source
    _record_source(validate)
    validate(op, tensors)
    return MappingExpansion(operators=(op,))


def tensor_access(op, tensors, options):
    from arch_explorer.workload.tensor_ops import validate
    from arch_explorer.workload.extensions import _record_source
    _record_source(validate)
    validate(op, tensors)
    output = tensors[op.outputs[0]]
    size = math.prod(output.shape)
    simt = sfu = 0
    if op.op in ('add', 'sub', 'mul', 'scale','scale_tensor','relu'):
        simt = size
    elif op.op in ('exp','erf','rsqrt','sqrt'):
        sfu = size
    elif op.op == 'div':
        simt = sfu = size
    elif op.op in ('reduce_max', 'reduce_sum'):
        source = tensors[op.inputs[0]]
        simt = math.prod(source.shape[:-1])*(source.shape[-1]-1)
    elif op.op in ('affine_norm', 'gelu_tanh', 'softmax', 'gelu_exact', 'bias','quick_gelu','l2_norm'):
        from arch_explorer.workload.full_primitives import COSTS
        a,b,c,d = COSTS[op.op]
        rows = math.prod(output.shape[:-1])
        simt, sfu = a*size+c*rows, b*size+d*rows
    # Every broadcast output reads its selected input element; reductions and
    # transpose each consume their full source. Duplicate binary operands are
    # separate semantic loads rather than hidden common-subexpression reuse.
    read = output.bytes if op.op in ('broadcast','slice') else sum(tensors[n].bytes for n in op.inputs)
    input_counts = [math.prod(tensors[n].shape) for n in op.inputs]
    if op.op in ('affine_norm','bias','scale_tensor'):
        input_counts = [size]*len(op.inputs)
        from arch_explorer.components.base import dtype_bytes
        read = sum(size*dtype_bytes(tensors[n].dtype) for n in op.inputs)
    copies = ('reshape','transpose','broadcast','slice','patchify','concat')
    if op.op == 'cast':
        source_dtype = tensors[op.inputs[0]].dtype
        from arch_explorer.workload.integer import cast_conversions
        from arch_explorer.workload.extensions import _record_source
        _record_source(cast_conversions)
        conversions = cast_conversions(source_dtype,output.dtype,size)
    elif op.op in copies:
        conversions = 0
    else:
        conversions = sum(count*(tensors[n].dtype != 'fp32') for n,count in zip(op.inputs,input_counts)) + size*(output.dtype != 'fp32')
    from arch_explorer.components.base import dtype_bytes
    if op.op in ('broadcast','slice'):input_counts=[size]
    read_by_tensor={}
    for n,count in zip(op.inputs,input_counts):read_by_tensor[n]=read_by_tensor.get(n,0)+count*dtype_bytes(tensors[n].dtype)
    if sum(read_by_tensor.values())!=read:raise ValueError('tensor read decomposition differs from scalar access count')
    return dict(operations=simt+sfu+conversions, read_bytes=read, write_bytes=output.bytes, boundary='component_io',read_by_tensor=read_by_tensor,
                simt_instructions=simt, sfu_instructions=sfu,
                input_dtypes=[tensors[n].dtype for n in op.inputs],output_dtype=output.dtype,
                conversion_instructions=conversions,accumulation='copy' if op.op in (*copies,'cast') else 'fp32',
                cost='synthetic scalar arithmetic/function service counts (including abstract erf); not calibrated hardware instructions')


def tensor_inspect(op, tensors, options):
    return dict(input_shapes=[list(tensors[n].shape) for n in op.inputs],
                output_shapes=[list(tensors[n].shape) for n in op.outputs], materialized=True)


def install():
    from arch_explorer.workload.tensor_ops import KINDS
    global _DEFINITIONS
    if _DEFINITIONS is None:
        matrix = OperatorMapping('matmul.tiled', '1', 'matmul', expand_matrix, matrix_access, matrix_inspect,
            matrix_options, dict(tiles={key: 'integer 1..4096' for key in ('m', 'k', 'n')}),tensor_scope='isolated-operands')
        _DEFINITIONS = (matrix, *(OperatorMapping(kind+'.materialized', '1', kind,
            expand_tensor, tensor_access, tensor_inspect,default=True,tensor_scope='isolated-operands') for kind in KINDS))
    for definition in _DEFINITIONS:
        register_operator_mapping(definition)
    from .view_rules import install as install_views
    install_views()
    from .index_rules import install as install_index
    install_index()
    from .mla_rules import install as install_mla
    install_mla()
    from .spatial_rules import install as install_spatial
    install_spatial()
    from .array_rules import install as install_array
    install_array()
    from .block_rules import install as install_block
    install_block()
