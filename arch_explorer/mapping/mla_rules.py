"""Service counts for generic RMS/gamma and explicit-pair rotation."""
import math
from .rules import MappingExpansion,OperatorMapping,register_operator_mapping
_DEFINITIONS=None


def expand(op,tensors,options):
    if options:raise ValueError('decode primitive has no implicit options')
    from arch_explorer.workload.mla_ops import validate
    from arch_explorer.workload.extensions import _record_source
    _record_source(validate);validate(op,tensors)
    return MappingExpansion(operators=(op,))


def access(op,tensors,options):
    output=tensors[op.outputs[0]];size=math.prod(output.shape);rows=math.prod(output.shape[:-1])
    if op.op=='rms_gamma':
        from arch_explorer.workload.full_primitives import COSTS
        from arch_explorer.workload.extensions import _record_source
        from arch_explorer.workload import full_primitives
        _record_source(full_primitives.reference)
        a,b,c,d=COSTS['rms']
        simt,sfu=(a+1)*size+c*rows,b*size+d*rows
        reads={op.inputs[0]:size*4,op.inputs[1]:size*4}
    elif op.op=='rope_pairs':
        simt,sfu=3*size,0
        reads={op.inputs[0]:size*4,op.inputs[1]:size//2*4,op.inputs[2]:size//2*4}
    else:raise ValueError('not a decode primitive')
    return dict(operations=simt+sfu,read_bytes=sum(reads.values()),write_bytes=output.bytes,boundary='component_io',
                simt_instructions=simt,sfu_instructions=sfu,read_by_tensor=reads,
                accumulation='fp32',conversion_instructions=0,cost='synthetic scalar service; no hardware calibration')


def inspection(op,tensors,options):return dict(materialized=True,input_shapes=[list(tensors[n].shape) for n in op.inputs])


def install():
    global _DEFINITIONS
    if _DEFINITIONS is None:_DEFINITIONS=tuple(OperatorMapping(n+'.materialized','1',n,expand,access,inspection,default=True,tensor_scope='isolated-operands')
                                            for n in ('rms_gamma','rope_pairs'))
    for d in _DEFINITIONS:register_operator_mapping(d)
