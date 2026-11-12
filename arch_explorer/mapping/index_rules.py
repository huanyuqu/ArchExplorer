"""Indexed/mask data traffic and scalar issue-count authority."""
import math
from .rules import MappingExpansion,OperatorMapping,register_operator_mapping


def expand(op,tensors,options):
    from arch_explorer.workload.graph import validate_operator
    validate_operator(op,tensors)
    return MappingExpansion(operators=(op,))


def access(op,tensors,options):
    from arch_explorer.workload.indexing import validate
    from arch_explorer.workload.extensions import _record_source
    from arch_explorer.components.base import dtype_bytes
    _record_source(validate);validate(op,tensors)
    out=tensors[op.outputs[0]];n=math.prod(out.shape);simt=sfu=conversions=read=0
    if op.op=='lookup':
        count=math.prod(tensors[op.inputs[0]].shape);simt=n+2*count
        read=count*4+out.bytes
    elif op.op=='select':
        count=out.shape[0];simt=n+2*count;read=count*4+out.bytes
    elif op.op=='argmax':
        b,s=tensors[op.inputs[0]].shape;simt=2*b*(s-1);read=b*s*4
    elif op.op=='causal_mask':simt=n
    else:
        read=n*(dtype_bytes(out.dtype)+4);simt=n
        conversions=2*n*(out.dtype!='fp32')
        if op.op=='masked_softmax':
            from arch_explorer.workload.full_primitives import COSTS,reference
            _record_source(reference)
            a,b,c,d=COSTS['softmax'];rows=math.prod(out.shape[:-1])
            simt+=a*n+c*rows;sfu=b*n+d*rows
    return dict(operations=simt+sfu+conversions,read_bytes=read,write_bytes=out.bytes,boundary='component_io',
        simt_instructions=simt,sfu_instructions=sfu,conversion_instructions=conversions,
        input_dtypes=[tensors[k].dtype for k in op.inputs],output_dtype=out.dtype,
        accumulation='fp32' if op.op in ('apply_mask','masked_softmax') else 'int32 addressing / bit-preserving copy',
        cost='synthetic shared SIMT issue counts: two bounds checks per index, one gather address per data element, strict first-max compare/update; no integer-to-float conversion',
        index_reuse='one index read per token/selected row; repeated tokens gather again, no hidden cache')


def inspect(op,tensors,options):
    return dict(materialized=True,input_shapes=[list(tensors[n].shape) for n in op.inputs],output_shape=list(tensors[op.outputs[0]].shape),
        indexing='data-dependent int32 indices supplied to logical_accesses(values=...)' if op.op in ('lookup','select') else 'declared logical order',
        mask_encoding='int32 0=keep/1=blocked; apply_mask stores finite zero payload, softmax decodes private -inf' if op.op in ('causal_mask','apply_mask','masked_softmax') else None)


_DEFINITIONS=None
def install():
    from arch_explorer.workload.indexing import KINDS
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS=tuple(OperatorMapping(k+'.materialized','1',k,expand,access,inspect,default=True) for k in KINDS)
    for definition in _DEFINITIONS:register_operator_mapping(definition)
