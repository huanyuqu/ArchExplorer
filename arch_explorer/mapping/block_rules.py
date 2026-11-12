"""Physical block access accounting and ordinary component kernel conversion."""
import math
from .rules import MappingExpansion, OperatorMapping, register_operator_mapping

_DEFINITIONS = None


def expand(op, tensors, options):
    from arch_explorer.workload.block_ops import validate
    from arch_explorer.workload.extensions import _record_source
    _record_source(validate)
    if options:
        raise ValueError('block.mapping: no implicit options')
    validate(op,tensors)
    return MappingExpansion(operators=(op,))


def access(op, tensors, options):
    from arch_explorer.workload.block_ops import validate
    from arch_explorer.workload.extensions import _record_source
    _record_source(validate); validate(op,tensors)
    output = tensors[op.outputs[0]]
    reads, operations = {}, 0
    if op.op == 'block.load':
        reads[op.inputs[0]] = output.bytes
        for name in op.inputs[1:]:
            reads[name] = reads.get(name,0) + tensors[name].bytes
    elif op.op == 'block.matmul_accumulate':
        a,b,c = (tensors[n] for n in op.inputs)
        fmas = math.prod(a.shape)*b.shape[-1]
        for name, count in zip(op.inputs,(fmas*4,fmas*4,c.bytes)):
            reads[name] = reads.get(name,0)+count
        operations = 2*fmas
    return dict(operations=operations,read_bytes=sum(reads.values()),write_bytes=output.bytes,
                boundary='component_io',read_by_tensor=reads,simt_instructions=operations,sfu_instructions=0,
                conversion_instructions=0,accumulation='fp32 ordered FFMA from explicit seed',
                cost='synthetic scalar service counts; physical reads/writes are external streams')


def inspection(op, tensors, options):
    result = dict(materialized=True,output_bytes=tensors[op.outputs[0]].bytes,
                  guards=list(op.inputs[1:]) if op.op == 'block.load' else [],
                  write_region='entire independent output; no in-place update')
    if op.op == 'block.load':
        source=tensors[op.inputs[0]]
        from arch_explorer.workload.storage import resolve_layout
        from arch_explorer.workload.extensions import _record_source
        _record_source(resolve_layout)
        layout=resolve_layout(tensors,source.name)
        strides=[4*s for s in layout.strides]
        result['read_region']=dict(tensor=source.name,starts=op.attrs['starts'],shape=op.attrs['shape'],
            source_shape=list(source.shape),storage=layout.storage,byte_strides=strides,
            byte_offset=layout.offset*4+sum(a*b for a,b in zip(op.attrs['starts'],strides)),
            bytes=tensors[op.outputs[0]].bytes,
            meaning='source bytes read once for this copy; guard reads separately in read_by_tensor')
    return result


def lower_kernel(op, context):
    from arch_explorer.components.base import Kernel
    from arch_explorer.workload.block_ops import validate
    from arch_explorer.workload.extensions import _record_source
    _record_source(validate); validate(op,context.tensors)
    if context.dtype != 'fp32':
        raise ValueError('block.mapping: component kernel requires FP32')
    if op.op == 'block.matmul_accumulate':
        a,b = (context.tensors[n].shape for n in op.inputs[:2])
        return Kernel('gemm',(a[0],a[1],a[2],b[2]),'fp32')
    return Kernel('vector',(1,math.prod(context.tensors[op.outputs[0]].shape)),'fp32')


def install():
    from arch_explorer.workload.block_ops import KINDS
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS=tuple(OperatorMapping(n+'.materialized','1',n,expand,access,inspection,default=True,tensor_scope='isolated-operands') for n in KINDS)
    for definition in _DEFINITIONS:
        register_operator_mapping(definition)
