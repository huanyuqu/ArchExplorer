"""Materialized spatial workspace mapping, with no frontend/model knowledge."""
from .rules import OperatorMapping,MappingExpansion,register_operator_mapping

def expand(op,tensors,options):
    from arch_explorer.workload.graph import validate_operator
    validate_operator(op,tensors)
    return MappingExpansion(operators=(op,))

def access(op,tensors,options):
    from arch_explorer.workload.spatial_ops import access as counts
    from arch_explorer.workload.extensions import _record_source
    _record_source(counts)
    return counts(op,tensors)

def inspect(op,tensors,options):
    return dict(materialized=True,input_shape=list(tensors[op.inputs[0]].shape),output_shape=list(tensors[op.outputs[0]].shape),
        order='N, spatial row, spatial column, input channel, kernel row, kernel column',padding='literal zero stores with no source read')

_DEFINITION=None
def install():
    global _DEFINITION
    if _DEFINITION is None:_DEFINITION=OperatorMapping('im2col.materialized','1','im2col',expand,access,inspect,default=True)
    register_operator_mapping(_DEFINITION)
