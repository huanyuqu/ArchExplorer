"""Typed shape specialization and admission estimates without numeric root arrays."""
import math
from .resource_budget import normalize_budget,check_shape

FLOATS={1:'fp32',10:'fp16',16:'bf16'}
STORAGE={**FLOATS,6:'int32',7:'int64',9:'bool'}
WIDTHS={1:4,10:2,16:2,6:4,7:8,9:1}

def specialize(model,input_shapes):
    if type(input_shapes) is not dict:
        raise ValueError('ONNX input_shapes must be a JSON object')
    roots={v.name:v for v in model.graph.input};symbols={};shapes={}
    if set(input_shapes)-set(roots):
        raise ValueError('ONNX inputs must exactly bind graph roots; input_shapes may specialize roots only')
    for key,value in roots.items():
        t=value.type.tensor_type
        if t.elem_type not in STORAGE:
            raise ValueError('ONNX runtime roots require FP32/FP16/BF16/int32/int64/bool storage')
        explicit=input_shapes.get(key)
        if explicit is not None:
            if type(explicit) is not list:raise ValueError('ONNX root specialization must be a logical shape list')
            explicit=check_shape(explicit)
            if len(explicit)!=len(t.shape.dim):raise ValueError('ONNX root specialization rank conflicts with annotation')
        resolved=[]
        for axis,dim in enumerate(t.shape.dim):
            n=explicit[axis] if explicit is not None else (int(dim.dim_value) if dim.HasField('dim_value') else symbols.get(dim.dim_param))
            if n is None:raise ValueError('ONNX unresolved root dimension; supply explicit input_shapes')
            if dim.HasField('dim_value') and dim.dim_value!=n:raise ValueError('ONNX concrete root shape conflicts with annotation')
            if dim.HasField('dim_param') and dim.dim_param:
                if dim.dim_param in symbols and symbols[dim.dim_param]!=n:raise ValueError('ONNX symbolic dimension specializations conflict')
                symbols[dim.dim_param]=n
            resolved.append(n);dim.ClearField('dim_param');dim.dim_value=n
        shapes[key]=check_shape(resolved)
    return shapes,symbols

def preflight_model(model,input_shapes,policy,source_sizes):
    from .extensions import _record_source
    from .onnx_lowering import lower_model
    _record_source(specialize);_record_source(preflight_model)
    _record_source(lower_model)
    budget=normalize_budget({} if policy is None else policy)
    graph=lower_model(model,None,input_shapes,'onnx_preflight',_shape_only=True)
    metadata=graph.meta['onnx'];produced={n for op in graph.operators for n in op.outputs}
    external={n:t for n,t in graph.tensors.items() if n not in produced}
    facts={}
    from arch_explorer.components.base import dtype_bytes
    for key,v in metadata['values'].items():
        elements=math.prod(v['logical_shape'])
        facts[key]=dict(logical_shape=v['logical_shape'],storage_dtype=v['dtype'],elements=elements,
            bytes=elements*dtype_bytes(v['dtype']),role=v['role'],tensor=v['tensor'],materialized=v['tensor'] is not None)
    root_bytes=sum(t.bytes for t in external.values());total=sum(t.bytes for t in graph.tensors.values())
    root_host=sum(math.prod(t.shape)*40+56*sum(math.prod(t.shape[:j]) for j in range(len(t.shape))) for t in external.values())
    host=root_host*12+source_sizes['total_bytes']*8+len(graph.tensors)*4096+len(graph.operators)*8192
    totals=dict(root_bytes=root_bytes,weight_bytes=sum(t.bytes for t in external.values() if t.kind=='weight'),
        temporary_bytes=total-root_bytes,logical_tensor_bytes=total,estimated_host_bytes=host,
        nodes=len(model.graph.node),values=len(facts),lowered_operators=len(graph.operators),lowered_tensors=len(graph.tensors))
    violations=[]
    for metric,required,bound in (('source_bytes',source_sizes['total_bytes'],'max_source_bytes'),('nodes',len(model.graph.node),'max_nodes'),
        ('root_bytes',root_bytes,'max_root_bytes'),('logical_tensor_bytes',total,'max_total_tensor_bytes'),('estimated_host_bytes',host,'max_host_bytes'),
        ('lowered_operators',len(graph.operators),'max_steps')):
        if required>budget[bound]:violations.append(dict(metric=metric,required=required,limit=budget[bound]))
    return dict(schema='arch-explorer.onnx-preflight.v1',budget=budget,source=source_sizes,
        inputs={k:facts[k] for k in metadata['inputs']},values=facts,totals=totals,admitted=not violations,violations=violations,
        scope='same common lowering in descriptor mode before floating root/initializer arrays; static integer controls specialized; placement/replicas rechecked by compiler')
