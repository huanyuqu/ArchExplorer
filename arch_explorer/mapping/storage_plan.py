"""Explicit root placement and immutable replicas for compile_graph_v4.

No compute/group names are interpreted. Consumers name original operations;
normal mapping provenance transports those names through rule expansion.
"""
from dataclasses import replace
from arch_explorer.workload.operators import Operator,Tensor

SCHEMA='arch-explorer.storage-plan.v1'


def validate_parameters(p):
    if type(p['storage']) is not dict or any(type(k) is not str or not k or type(v) is not str or not v for k,v in p['storage'].items()):
        raise ValueError('v4 storage requires an explicit root tensor -> actual memory object')
    if type(p['replicas']) is not list:raise ValueError('v4 replicas must be an explicit list')
    ids=set();names=set();selected=set()
    for item in p['replicas']:
        if (type(item) is not dict or set(item)!={'id','source','destinations','multicast'}
                or type(item['id']) is not str or not item['id'] or item['id'] in ids
                or type(item['source']) is not str or not item['source'] or type(item['multicast']) is not bool
                or type(item['destinations']) is not list or not item['destinations']):
            raise ValueError('replicas require unique id, source, nonempty destinations and explicit bool multicast')
        ids.add(item['id']);memories=set()
        for dst in item['destinations']:
            if (type(dst) is not dict or set(dst)!={'tensor','memory','consumers'}
                    or any(type(dst[k]) is not str or not dst[k] for k in ('tensor','memory'))
                    or dst['tensor'] in names or dst['memory'] in memories
                    or type(dst['consumers']) is not list or not dst['consumers']
                    or any(type(n) is not str or not n for n in dst['consumers'])
                    or len(set(dst['consumers']))!=len(dst['consumers'])):
                raise ValueError('replica needs unique tensor/memory and exact nonempty original consumers')
            names.add(dst['tensor']);memories.add(dst['memory'])
            for name in dst['consumers']:
                key=(item['source'],name)
                if key in selected:raise ValueError('one source/consumer cannot select two replicas')
                selected.add(key)


def apply_storage_plan(original,graph,facts,p):
    from arch_explorer.workload.extensions import _record_source
    _record_source(apply_storage_plan);_record_source(validate_parameters)
    from .compilation import placement,MAX_STEPS,MAX_TOTAL_BYTES
    from arch_explorer.workload.resource_budget import limit
    validate_parameters(p)
    produced={n for op in original.operators for n in op.outputs}
    roots=set(original.tensors)-produced
    if set(p['storage'])!=roots:raise ValueError('v4 storage must cover exactly every original external root')
    if any(original.tensors[n].alias_of is not None for n in roots):raise ValueError('v4 external roots must own independent storage')
    original_ops={op.name:op for op in original.operators}
    replacements={};copies=[];locations=dict(p['storage']);transfers={}
    for item in p['replicas']:
        src=item['source']
        if src not in roots:raise ValueError('replicas v1 source must be an external immutable root')
        if original.tensors[src].dtype!='fp32':raise ValueError('replicas v1 currently require FP32 storage')
        outputs=[]
        for dst in item['destinations']:
            if dst['tensor'] in graph.tensors:raise ValueError('replica tensor namespace collides')
            if dst['memory']==p['storage'][src]:raise ValueError('replica destination must differ from source storage')
            for consumer in dst['consumers']:
                if consumer not in original_ops or src not in original_ops[consumer].inputs:
                    raise ValueError('replica consumers must name actual direct readers of the original source')
                replacements[(consumer,src)]=dst['tensor']
            graph.tensors[dst['tensor']]=Tensor(dst['tensor'],original.tensors[src].shape,'fp32')
            locations[dst['tensor']]=dst['memory'];outputs.append(dst['tensor'])
        name='@storage/'+item['id']
        if any(op.name==name for op in graph.operators):raise ValueError('replica task namespace collides')
        op=Operator(name,'storage.copy',(src,),tuple(outputs))
        copies.append(op);transfers[name]=item
        # Copies have no arithmetic placement. The original consumer provides a
        # domain binding only; exact source/destination storage is the plan.
        facts['origins'][name]=item['destinations'][0]['consumers'][0]
        size=graph.tensors[src].bytes
        reads=size*(1 if item['multicast'] else len(outputs))
        facts['accesses'][name]=dict(operations=0,read_bytes=reads,write_bytes=size*len(outputs),boundary='component_io',
            read_by_tensor={src:reads},simt_instructions=0,sfu_instructions=0,conversion_instructions=0)
        facts['inspections'][name]=dict(storage_replica=item,service='one tree or one actual source read per unicast destination')
    mapped=[]
    for op in graph.operators:
        origin=facts['origins'][op.name]
        rename={n:replacements[(origin,n)] for n in op.inputs if (origin,n) in replacements}
        if rename:
            op=replace(op,inputs=tuple(rename.get(n,n) for n in op.inputs))
            for n in op.outputs:
                t=graph.tensors[n]
                if t.alias_of in rename:graph.tensors[n]=replace(t,alias_of=rename[t.alias_of])
            if 'read_by_tensor' in facts['accesses'][op.name]:
                reads={}
                for n,b in facts['accesses'][op.name]['read_by_tensor'].items():
                    key=rename.get(n,n);reads[key]=reads.get(key,0)+b
                facts['accesses'][op.name]={**facts['accesses'][op.name],'read_by_tensor':reads}
        if op.op=='storage.view':
            from arch_explorer.workload.storage import view_tensor,resolve_layout
            graph.tensors[op.outputs[0]]=view_tensor(graph.tensors,op.inputs[0],op.outputs[0],op.attrs['transform'],
                                                   {k:v for k,v in op.attrs.items() if k!='transform'})
            facts['inspections'][op.name]={**facts['inspections'][op.name],
                'layout':resolve_layout(graph.tensors,op.outputs[0]).to_dict()}
        mapped.append(op)
        for n in op.outputs:
            if graph.tensors[n].alias_of is None:locations[n]=placement(p,'memory',origin)
    graph.operators=[*copies,*mapped]
    if len(graph.operators)>limit('max_steps',MAX_STEPS) or sum(t.bytes for t in graph.tensors.values() if t.alias_of is None)>limit('max_total_tensor_bytes',MAX_TOTAL_BYTES):
        raise ValueError('replica expansion exceeds compile graph storage/step bounds')
    graph.meta['storage_plan']=dict(schema=SCHEMA,locations=locations,transfers=transfers)


def physical_locations(graph):
    from arch_explorer.workload.storage import resolve_layout
    plan=graph.meta['storage_plan']
    if set(plan)!={'schema','locations','transfers'} or plan['schema']!=SCHEMA:
        raise ValueError('invalid storage plan schema/fields')
    roots={n for n,t in graph.tensors.items() if t.alias_of is None}
    if set(plan['locations'])!=roots:raise ValueError('storage plan must locate every actual root exactly once')
    return {n:plan['locations'][resolve_layout(graph.tensors,n).storage] for n in graph.tensors}
