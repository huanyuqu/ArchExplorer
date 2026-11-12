"""Selectable view expansions; materialized semantics remain the default."""
from copy import deepcopy
from .rules import OperatorMapping,MappingExpansion,MappingRuleBinding,register_operator_mapping
from arch_explorer.workload.operators import Operator


def expand_view(op,tensors,options):
    from arch_explorer.workload.storage import view_tensor
    t=view_tensor(tensors,op.inputs[0],op.outputs[0],op.op,op.attrs)
    step=Operator(op.name,'storage.view',op.inputs,op.outputs,dict(transform=op.op,**op.attrs),op.label)
    return MappingExpansion(operators=(step,),views=(t,))


def expand_preserved(op,tensors,options):
    from arch_explorer.workload.storage import validate_view
    validate_view(op,tensors)
    return MappingExpansion(operators=(op,))


def view_access(op,tensors,options):
    return dict(operations=0,read_bytes=0,write_bytes=0,boundary='component_io',
        conversion_instructions=0,readonly_view=True,accumulation='bit-preserving alias')


def view_inspect(op,tensors,options):
    from arch_explorer.workload.storage import resolve_layout
    return dict(materialized=False,layout=resolve_layout(tensors,op.outputs[0]).to_dict(),
        service='no copy or arithmetic; consumers issue real logical loads')


def layout_candidates(workload,mapping):
    """Two same-graph v3 choices; views fall back explicitly when reshape cannot share."""
    from arch_explorer.workload.api import workload_from_spec
    from arch_explorer.adf import MappingSpec
    from .compilation import map_workload,parameters
    from arch_explorer.workload.storage import view_tensor
    g=workload_from_spec(workload);p=parameters(mapping)
    p={**p,'operator_rules':dict(p.get('operator_rules',{}))}
    for op in g.operators:
        if op.op in ('transpose','reshape','slice'):
            p['operator_rules'][op.name]=MappingRuleBinding(op.op+'.materialized','1').to_dict()
    template='compile_graph_v4' if mapping.template=='compile_graph_v4' else 'compile_graph_v3'
    materialized=MappingSpec(template,deepcopy(p))
    tensors=dict(g.tensors)
    for op in g.operators:
        if op.op not in ('transpose','reshape','slice'):continue
        try:t=view_tensor(tensors,op.inputs[0],op.outputs[0],op.op,op.attrs)
        except ValueError as error:
            if op.op!='reshape' or 'reshape' not in str(error):raise
        else:
            tensors[t.name]=t
            p['operator_rules'][op.name]=MappingRuleBinding(op.op+'.view','1').to_dict()
    views=MappingSpec(template,deepcopy(p))
    map_workload(g,materialized);map_workload(g,views)
    return dict(materialized=materialized,views=views)


_DEFINITIONS=None
def install():
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS=tuple(OperatorMapping(kind+'.view','1',kind,expand_view,view_access,view_inspect,tensor_scope='isolated-operands')
                           for kind in ('transpose','reshape','slice'))+(
            OperatorMapping('storage.view.preserve','1','storage.view',expand_preserved,view_access,view_inspect,tensor_scope='isolated-operands'),)
    for definition in _DEFINITIONS:register_operator_mapping(definition)
