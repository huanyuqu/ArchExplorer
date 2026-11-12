"""Serializable subgraph ports and direct tensor-edge composition."""
from contextvars import ContextVar
import json

from .graph import finite_json, name, validate_graph
from .operators import Tensor, Workload

SCHEMA = 'arch-explorer.subgraph-composition.v1'
_DEPTH = ContextVar('composition_depth',default=0)


def qualified_name(instance, local):
    name(instance,'instance id');name(local,'local name')
    return '@subgraph/'+json.dumps([instance,local],separators=(',',':'),ensure_ascii=True)


def _spec(value):
    from arch_explorer.adf.evaluation import WorkloadSpec
    if type(value) is Workload:return WorkloadSpec('graph',value.to_dict()).to_dict()
    if type(value) is WorkloadSpec:return value.to_dict()
    if type(value) is dict:
        if value.get('schema') in ('arch-explorer.workload-graph.v1','arch-explorer.workload-graph.v2'):
            return WorkloadSpec('graph',finite_json(value)).to_dict()
        return WorkloadSpec(**finite_json(value)).to_dict()
    raise TypeError('subgraph workload must be WorkloadSpec, Workload or its public JSON')


def compose_workloads(*, instances, inputs, bindings, outputs, name='composition'):
    from arch_explorer.adf.evaluation import WorkloadSpec
    if type(instances) not in (list,tuple):raise TypeError('instances must be a sequence')
    serial=[]
    for item in instances:
        if type(item) is not dict or set(item) != {'id','workload','inputs','outputs'}:
            raise ValueError('subgraph instance requires exactly id/workload/inputs/outputs')
        serial.append({**item,'workload':_spec(item['workload'])})
    return WorkloadSpec('composition',finite_json(dict(schema=SCHEMA,name=name,instances=serial,
        inputs=inputs,bindings=bindings,outputs=outputs),tuples=True))


def _child(spec):
    from arch_explorer.adf.evaluation import WorkloadSpec
    from .api import workload_from_spec
    from .extensions import source_trace, _record_source, _record_identity, callable_identity
    with source_trace() as used:graph=workload_from_spec(WorkloadSpec(**spec))
    for function in used:_record_source(function)
    for identity in used.identities:_record_identity(identity)
    identities={}
    for function in used:
        identity=callable_identity(function);identities[json.dumps(identity,sort_keys=True)]=identity
    return graph,[identities[k] for k in sorted(identities)]


def build_composition(params, *, tp=1):
    if type(tp) is not int or tp != 1:raise ValueError('subgraph composition supports tp=1 only')
    depth=_DEPTH.get()
    if depth>=4:raise ValueError('subgraph nesting exceeds four levels')
    token=_DEPTH.set(depth+1)
    try:return _build(finite_json(params))
    finally:_DEPTH.reset(token)


def _build(p):
    if (type(p) is not dict or set(p) != {'schema','name','instances','inputs','bindings','outputs'}
            or p['schema'] != SCHEMA):raise ValueError('invalid subgraph composition v1 fields')
    name(p['name'],'composition name')
    if type(p['instances']) is not list or not 1<=len(p['instances'])<=16:
        raise ValueError('composition requires 1..16 subgraph instances')
    if type(p['inputs']) is not dict or type(p['outputs']) is not dict or not p['outputs'] or type(p['bindings']) is not list:
        raise ValueError('composition requires input/output objects and a binding list')
    if set(p['inputs']) & set(p['outputs']):raise ValueError('export cannot overwrite a root input')
    children={};provenance={};order=[]
    for item in p['instances']:
        if type(item) is not dict or set(item) != {'id','workload','inputs','outputs'}:
            raise ValueError('instance requires id/workload/inputs/outputs')
        ident=item['id'];name(ident,'instance id')
        if ident in children:raise ValueError('duplicate subgraph instance '+ident)
        graph,sources=_child(item['workload'])
        if graph.iterations != 1 or graph.meta.get('inputs'):
            raise ValueError('subgraphs must be structural one-iteration graphs; bind data once at the composition root')
        for tensor in graph.tensors.values():_layout(tensor)
        produced={n for op in graph.operators for n in op.outputs}
        external=set(graph.tensors)-produced
        for key in ('inputs','outputs'):
            ports=item[key]
            if type(ports) is not dict or any(type(v) is not str for v in ports.values()):
                raise ValueError('ports must map public port names to local tensor names')
            for port in ports:name(port,'port name')
            if len(set(ports.values())) != len(ports):raise ValueError('duplicate local tensor port')
        if set(item['inputs'].values()) != external:raise ValueError('input ports must cover exactly all external subgraph tensors')
        if not item['outputs'] or set(item['outputs'].values())-produced:raise ValueError('output ports must identify actual produced tensors')
        children[ident]=(item,graph);provenance[ident]=sources;order.append(ident)
    roots={};values={}
    for root in sorted(p['inputs']):
        desc=p['inputs'][root]
        _public_name(root)
        if type(desc) is not dict or set(desc)-{'shape','dtype','kind','value'} or not {'shape','dtype','kind'}<=set(desc):
            raise ValueError('root inputs require shape/dtype/kind and optional value')
        tensor=Tensor(root,tuple(desc['shape']),desc['dtype'],desc['kind']);_layout(tensor);roots[root]=tensor
        if 'value' in desc:
            from .reference_api import _input
            _input(desc['value'],tensor);values[root]=desc['value']
    targets={};edges={i:set() for i in order};used_roots=set()
    def endpoint(source):
        if type(source) is not dict or set(source) != {'instance','port'}:raise ValueError('endpoint requires exactly instance/port')
        ident,port=source['instance'],source['port']
        if type(ident) is not str or ident not in children:raise ValueError('unknown subgraph instance endpoint')
        item,g=children[ident]
        if type(port) is not str or port not in item['outputs']:raise ValueError('unknown subgraph output port')
        return ident,item['outputs'][port],g.tensors[item['outputs'][port]]
    for binding in p['bindings']:
        if type(binding) is not dict or set(binding) != {'to','from'}:raise ValueError('binding requires exactly to/from')
        target,source=binding['to'],binding['from']
        if type(target) is not dict or set(target) != {'instance','port'}:raise ValueError('binding target requires instance/port')
        ident,port=target['instance'],target['port']
        if type(ident) is not str or ident not in children:raise ValueError('unknown binding target instance')
        item,g=children[ident]
        if type(port) is not str or port not in item['inputs']:raise ValueError('unknown binding input port')
        key=(ident,item['inputs'][port])
        if key in targets:raise ValueError('input port has multiple sources')
        dest=g.tensors[key[1]]
        if type(source) is dict and set(source)=={'input'}:
            root=source['input']
            if type(root) is not str or root not in roots:raise ValueError('unknown root input')
            src=roots[root];targets[key]=(None,root);used_roots.add(root)
        else:
            parent,local,src=endpoint(source);targets[key]=(parent,local);edges[ident].add(parent)
        if (src.shape,src.dtype)!=(dest.shape,dest.dtype):raise ValueError('binding shape/dtype mismatch; use explicit materialized conversion')
    expected={(i,n) for i,(item,g) in children.items() for n in item['inputs'].values()}
    if set(targets)!=expected or used_roots!=set(roots):raise ValueError('bindings must cover each input exactly once and consume every root')
    exported={}
    for label in sorted(p['outputs']):
        source=p['outputs'][label]
        _public_name(label);ident,local,_=endpoint(source)
        if (ident,local) in exported:raise ValueError('one produced tensor cannot have multiple export names')
        exported[(ident,local)]=label
    topology=[];remaining=set(order)
    while remaining:
        ready=[i for i in order if i in remaining and not edges[i]&remaining]
        if not ready:raise ValueError('subgraph binding dependency cycle')
        topology.extend(ready);remaining.difference_update(ready)
    graph=Workload(p['name'],tensors=dict(roots),dtype=next(iter(roots.values())).dtype if roots else children[order[0]][1].dtype)
    tensor_names={};metadata=[]
    for ident in topology:
        item,g=children[ident];names={}
        for local,tensor in g.tensors.items():
            if (ident,local) in targets:
                parent,source=targets[(ident,local)]
                names[local]=source if parent is None else tensor_names[(parent,source)]
            else:
                actual=exported.get((ident,local),qualified_name(ident,local))
                if actual in graph.tensors:raise ValueError('composition namespace collision')
                names[local]=actual
                graph.tensors[actual]=Tensor(actual,tensor.shape,tensor.dtype,tensor.kind,
                    None if tensor.alias_of is None else names.get(tensor.alias_of,qualified_name(ident,tensor.alias_of)),
                    tensor.strides,tensor.offset)
            tensor_names[(ident,local)]=names[local]
        opnames={}
        for local,tensor in g.tensors.items():
            if tensor.alias_of is not None:
                actual=names[local]
                graph.tensors[actual]=Tensor(actual,tensor.shape,tensor.dtype,tensor.kind,
                    names[tensor.alias_of],tensor.strides,tensor.offset)
        for op in g.operators:
            actual=qualified_name(ident,op.name);opnames[op.name]=actual
            graph.operators.append(type(op)(actual,op.op,tuple(names[n] for n in op.inputs),tuple(names[n] for n in op.outputs),
                finite_json(op.attrs,tuples=True),op.label))
        metadata.append(dict(id=ident,workload=item['workload'],inputs=item['inputs'],outputs=item['outputs'],
            tensor_names=names,operator_names=opnames,sources=provenance[ident]))
        if len(graph.operators)>512 or len(graph.tensors)>8192:raise ValueError('composition flattening bound exceeded')
    graph.meta['inputs']=values
    graph.meta['outputs']=sorted(p['outputs'])
    graph.meta['composition']=dict(schema=SCHEMA,specification=p,instances=metadata,topological_instances=topology,
        bindings=p['bindings'],exports=p['outputs'],edge_storage='direct tensor identity; no materialized inter-subgraph copies')
    return validate_graph(graph)


def _layout(tensor):
    if (tensor.dtype not in ('fp32','fp16','bf16','int32') or tensor.kind=='kv' or not 1<=len(tensor.shape)<=4):
        raise ValueError('composition requires non-KV floating/int32 tensors, rank 1..4')


def _public_name(value):
    name(value,'public tensor name')
    if value.startswith('@'):raise ValueError('public tensor names cannot use the reserved @ namespace')
