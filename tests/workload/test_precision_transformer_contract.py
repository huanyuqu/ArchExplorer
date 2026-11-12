"""Consumers of frozen SIM45; execute only on the lead's installed candidate.

Collection is not acceptance. No skips, expected failures, baseline fallbacks or
production source composition hide unavailable new public APIs.
"""
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct

import pytest

from precision_transformer_expectations import (
    FORMATS, audit_materialized_storage, frozen_block_reference,
    materialized_movement, store, tensor_bytes,
)
from test_operator_mapping_attention_contract import no_backend, flat, files


INPUTS=Path(__file__).resolve().parents[1]/'fixtures/precision_transformer/INPUTS.json'
INPUT_SHA='f7538a04334fab9476e60f8e50ffece461bc2a1cf7caea9298e9cc7b8f150c06'
REJECT=(ValueError,TypeError,NotImplementedError)


@pytest.fixture
def precision_api():
    import numpy  # Declared CPU dependency before entering execution guards.
    from arch_explorer.components import Kernel  # noqa: F401
    w=importlib.import_module('arch_explorer.workload.api')
    s=importlib.import_module('arch_explorer.simulator.api')
    for name in ('round_storage','transformer_encoder_block','transformer_block_input_shapes'):
        assert callable(getattr(w,name)),name
    return w,s


def scene(dtype,mode='events',index=0):
    from arch_explorer.adf import from_dict,WorkloadSpec,MappingSpec
    assert hashlib.sha256(INPUTS.read_bytes()).hexdigest()==INPUT_SHA
    row=deepcopy(json.loads(INPUTS.read_text())[mode+'-'+dtype])
    return from_dict(row['adf']),WorkloadSpec(**row['workload']),MappingSpec(**row['mappings'][index])


def close_values(actual,expected,dtype):
    import numpy as np
    # FP32 reduction/contraction order can differ. Storage-grid boundaries and
    # cancellation/rounding primitives below are checked independently/exactly.
    rtol,atol={'fp32':(2e-5,2e-6),'fp16':(3e-3,2e-3),'bf16':(2e-2,1.56e-2)}[dtype]
    assert np.asarray(actual).shape==np.asarray(expected).shape
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)


def canonical_bytes(values,dtype):
    data=bytearray()
    for value in flat(values):
        if dtype=='bf16':
            word=struct.unpack('<I',struct.pack('<f',store(float(value),dtype)))[0]>>16
            data.extend(struct.pack('<H',word))
        else:
            data.extend(struct.pack('<f' if dtype=='fp32' else '<e',store(float(value),dtype)))
    return bytes(data)


@pytest.mark.parametrize('dtype',list(FORMATS))
def test_public_rounding_matches_rational_grid_and_preserves_subnormal_sign(precision_api,monkeypatch,dtype):
    w,_=precision_api
    p,emin,_=FORMATS[dtype];unit=2**(1-p);tiny=2**(emin-p+1)
    values=[0.,-0.,1/3,-1/7,1+unit/2,1+3*unit/2,tiny,tiny/2,-tiny/2]
    with no_backend(monkeypatch):
        actual=w.round_storage(values,dtype)
    assert list(actual)==[store(v,dtype) for v in values]
    assert math.copysign(1,float(actual[1]))==-1
    assert math.copysign(1,float(actual[-1]))==-1
    assert str(actual.dtype)==('float16' if dtype=='fp16' else 'float32')


@pytest.mark.parametrize('dtype,value',[('fp32',1e40),('fp16',65520.),('bf16',3.4e38)])
def test_public_storage_overflow_is_rejected_without_backend(precision_api,monkeypatch,dtype,value):
    w,_=precision_api
    with no_backend(monkeypatch),pytest.raises(REJECT,match='(?i)finite|overflow|range'):
        w.round_storage([value],dtype)


@pytest.mark.parametrize('dtype',list(FORMATS))
@pytest.mark.parametrize('mode',['events','roofline'])
@pytest.mark.parametrize('index',[0,1])
def test_complete_raw_mapped_block_numeric_and_actual_storage_schedule(
        precision_api,monkeypatch,tmp_path,dtype,mode,index):
    from arch_explorer.adf import WorkloadSpec
    w,s=precision_api;adf,spec,mapping=scene(dtype,mode,index)
    p=spec.params;expected=frozen_block_reference(p)
    with no_backend(monkeypatch):
        shapes=w.transformer_block_input_shapes(batch=2,sequence=3,hidden=8,heads=2,mlp_hidden=16)
        assert set(shapes)==set(p['inputs']) and len(shapes)==17
        public=w.transformer_encoder_block(**p)
        assert public.kind=='transformer_block'
        raw=w.execute_reference(public)
        compilation=w.compile_workload(adf,public,mapping)
        mapped=w.execute_reference(WorkloadSpec('graph',compilation['program']),inputs=p['inputs'])
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'run',options={'trace':True})
    assert set(raw['outputs'])==set(mapped['outputs'])=={'Y'}
    close_values(raw['outputs']['Y'],expected['output'],dtype)
    close_values(mapped['outputs']['Y'],expected['output'],dtype)
    close_values(raw['outputs']['Y'],mapped['outputs']['Y'],dtype)
    for execution in (raw,mapped):
        for name,tensor in execution['graph'].tensors.items():
            size=tensor_bytes(tensor.shape,tensor.dtype)
            assert tensor.bytes==size
            facts=execution['reference']['tensors'][name]
            assert facts['bytes']==size
            payload=canonical_bytes(execution['tensors'][name].tolist(),tensor.dtype)
            assert len(payload)==size and hashlib.sha256(payload).hexdigest()==facts['sha256']
    semantics={op.op for op in raw['graph'].operators}
    assert {'affine_norm','reshape','transpose','cast','gelu_tanh','matmul','add'}<=semantics
    assert len([op for op in raw['graph'].operators if op.op=='affine_norm'])==2
    for step in compilation['plan']['steps']:
        access=step['access']
        assert type(access['conversion_instructions']) is int and access['conversion_instructions']>=0
        if step['semantic']=='reshape':
            assert access['conversion_instructions']==0
            assert access['read_bytes']==access['write_bytes']>0
    # This is real modeled execution on declared synthetic CPU components.
    # It does not evaluate block numerical values on a GPU.
    result=s.evaluate_request(request)
    peak,intervals=audit_materialized_storage(compilation,result)
    assert peak==result['evaluation']['details']['stats']['memory_peak']['board/sram']
    compute={r['operator']:r for r in result['evaluation']['details']['trace'] if r['kind']=='compute'}
    declared={r['tensor']:r for r in compilation['plan']['storage']}
    dual=[r for r in intervals if len(set(declared[r['name']]['consumers']))>=2]
    assert dual
    assert any(r['last_reader']>min(compute[x]['end_ns'] for x in declared[r['name']]['consumers']) for r in dual)
    before=files(tmp_path/'run')
    with no_backend(monkeypatch):
        assert s.evaluate_request(result)==result
    assert before==files(tmp_path/'run')
    for mutation in ('latency','dtype','mapping'):
        changed=deepcopy(result)
        if mutation=='latency':changed['evaluation']['latency_ns']+=1
        elif mutation=='dtype':
            changed['request']['compilation']['source']['workload']['params']['dtype']='fp16' if dtype=='fp32' else 'fp32'
        else:
            changed['request']['compilation']['source']['mapping']['params']['rules']['matmul']['options']['tiles']['k']+=1
        # Recompute only the outer seal: the strict reader must still enforce
        # the nested request, selected mapping and numerical/source identity.
        def seal(document):
            body={k:v for k,v in document.items() if k!='binding'}
            return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        assert seal(result)==result['binding']['sha256']
        changed['binding']['sha256']=seal(changed)
        with no_backend(monkeypatch),pytest.raises(REJECT):
            s.evaluate_request(changed)
        assert before==files(tmp_path/'run')
    (tmp_path/'STORAGE-ORACLE.json').write_text(json.dumps(dict(peak_bytes=peak,intervals=intervals),indent=2))


def primitive(kind,input_shapes,output_shape,values,input_dtypes,output_dtype,**attrs):
    from arch_explorer.adf import WorkloadSpec
    names=['x'+str(i) for i in range(len(values))]
    tensors=[dict(name=n,shape=list(shape),dtype=dtype,kind='activation',alias_of=None,strides=None)
             for n,shape,dtype in zip(names,input_shapes,input_dtypes)]
    tensors.append(dict(name='y',shape=list(output_shape),dtype=output_dtype,kind='activation',alias_of=None,strides=None))
    return WorkloadSpec('graph',dict(schema='arch-explorer.workload-graph.v1',name='val56-'+kind,
        dtype=output_dtype,iterations=1,meta={'inputs':dict(zip(names,values))},tensors=tensors,
        operators=[dict(name='primitive',op=kind,inputs=names,outputs=['y'],
                        attrs=dict(shape=list(output_shape),**attrs),label='VAL56',type='operator')]))


@pytest.mark.parametrize('source,target',[('fp32','fp16'),('fp32','bf16'),('fp16','bf16'),
                                        ('bf16','fp16'),('bf16','fp32'),('fp16','fp16')])
def test_explicit_cast_movement_conversion_and_reference(precision_api,monkeypatch,tmp_path,source,target):
    from arch_explorer.adf import MappingSpec
    w,s=precision_api;adf,_,old=scene('fp32','roofline')
    values=[[1/3,1.0006,-2.003],[0.,2**-20,-0.]]
    spec=primitive('cast',[(2,3)],(2,3),[values],[source],target)
    mapping=MappingSpec('compile_graph_v2',dict(old.params,rules={}))
    expected=[[store(store(x,source),target) for x in row] for row in values]
    with no_backend(monkeypatch):
        compilation=w.compile_workload(adf,spec,mapping)
        reference=w.execute_reference(spec)
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'cast')
    assert reference['outputs']['y'].tolist()==expected
    access=compilation['plan']['steps'][0]['access']
    movement=materialized_movement((2,3),source,target)
    assert (access['read_bytes'],access['write_bytes'])==(movement['read_bytes'],movement['write_bytes'])
    conversions=0 if source==target else 6*((source!='fp32')+(target!='fp32'))
    assert access['conversion_instructions']==conversions
    assert s.evaluate_request(request)['evaluation']['latency_ns']>0


@pytest.mark.parametrize('mode',['events','roofline'])
@pytest.mark.parametrize('dtype',list(FORMATS))
def test_conversion_instructions_are_charged_in_actual_component_service(
        precision_api,monkeypatch,tmp_path,mode,dtype):
    from arch_explorer.adf import Param,MappingSpec
    w,s=precision_api;adf,_,old=scene(dtype,mode)
    point=adf.elements[0].elements[0]
    assert point.params['precision'].value=='fp32'
    source='explicit VAL56 controlled synthetic compute-service rate; not a device measurement'
    if mode=='events':
        point.implementation.params.update(clock_ghz=Param(1,'GHz',source),
            lanes=Param(8,'count',source),sfu_lanes=Param(1,'count',source))
        rate=8.
    else:
        point.implementation.params.update(peak_ops_per_ns=Param(16,'op/ns',source),
            bandwidth_bytes_per_ns=Param(1e12,'B/ns',source))
        rate=16.
    # A compute trace spans every occupied resource plus its access-latency
    # tail. Remove those independent bottlenecks in this dedicated service
    # fixture; the full block tests retain the original memory configuration.
    def isolate_service(node):
        for name,param in tuple(getattr(node,'params',{}).items()):
            if name=='bandwidth':
                node.params[name]=Param(1e12,param.unit,source)
            elif name=='latency':
                node.params[name]=Param(0,param.unit,source)
        for child in (*getattr(node,'elements',()),*getattr(node,'networks',())):
            isolate_service(child)
    isolate_service(adf)
    values=[[[1.,2.,3.],[4.,5.,6.]],[[2.,3.,4.],[5.,6.,7.]]]
    spec=primitive('add',[(2,3),(2,3)],(2,3),values,[dtype,dtype],dtype)
    mapping=MappingSpec('compile_graph_v2',dict(old.params,rules={}))
    conversions=0 if dtype=='fp32' else 18  # two promotions + one output rounding per element
    with no_backend(monkeypatch):
        compilation=w.compile_workload(adf,spec,mapping)
        access=compilation['plan']['steps'][0]['access']
        assert access['conversion_instructions']==conversions
        unit=4 if dtype=='fp32' else 2
        assert (access['read_bytes'],access['write_bytes'])==(12*unit,6*unit)
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'controlled',options={'trace':True})
    result=s.evaluate_request(request)
    compute=[r for r in result['evaluation']['details']['trace']
             if r['kind']=='compute' and r['operator']=='primitive']
    assert len(compute)==1
    duration=compute[0]['end_ns']-compute[0]['start_ns']
    # Isolate component service, not whole-graph time: memory can independently
    # improve for smaller tensors. Add has six logical FP32 scalar operations.
    assert duration==pytest.approx((6+conversions)/rate,rel=1e-10,abs=1e-10)


@pytest.mark.parametrize('dtype,large',[('fp16',2048.),('bf16',256.)])
def test_matmul_k_tiles_keep_fp32_fma_accumulator_until_output(precision_api,monkeypatch,dtype,large):
    from arch_explorer.adf import MappingSpec,WorkloadSpec
    w,_=precision_api;adf,_,old=scene(dtype,'events')
    values=[[[[large,1.,-large]]],[[[1.],[1.],[1.]]]]
    # Basic matmul attrs have dimensions rather than the shape attrs of unary ops.
    spec=primitive('matmul',[(1,1,3),(1,3,1)],(1,1,1),values,[dtype,dtype],dtype)
    spec.params['operators'][0]['attrs']=dict(batch=1,m=1,k=3,n=1,shared_b=False,dtype=dtype)
    with no_backend(monkeypatch):
        for k in (1,2,3):
            mapping=MappingSpec('compile_graph_v2',dict(old.params,rules={'matmul':dict(
                id='matmul.tiled',version='1',options={'tiles':dict(m=1,k=k,n=1)})}))
            compilation=w.compile_workload(adf,spec,mapping)
            actual=w.execute_reference(WorkloadSpec('graph',compilation['program']),inputs=spec.params['meta']['inputs'])
            assert actual['outputs']['y'].tolist()==[[[1.]]]


@pytest.mark.parametrize('mutation',['missing','extra','bad_heads','dropout','alias','strides','mixed_without_cast'])
def test_unsupported_block_layout_and_shape_refuse_before_prepare(precision_api,monkeypatch,tmp_path,mutation):
    from arch_explorer.adf import WorkloadSpec
    w,s=precision_api;adf,spec,mapping=scene('fp16')
    with no_backend(monkeypatch),pytest.raises(REJECT):
        if mutation=='missing':del spec.params['inputs']['bq']
        elif mutation=='extra':spec.params['inputs']['unused']=[1.]
        elif mutation=='bad_heads':spec.params['heads']=3
        elif mutation=='dropout':spec.params['dropout']=.1
        else:
            graph=w.workload_from_spec(spec).to_dict()
            tensors={t['name']:t for t in graph['tensors']}
            if mutation=='alias':tensors['X']['alias_of']='Wq'
            if mutation=='strides':tensors['X']['strides']=[24,8,1]
            if mutation=='mixed_without_cast':tensors['Wq']['dtype']='fp32'
            spec=WorkloadSpec('graph',graph)
        compilation=w.compile_workload(adf,spec,mapping)
        s.prepare_compiled(compilation,output_dir=tmp_path/'invalid')
    assert not list(tmp_path.rglob('*'))


@pytest.mark.parametrize('dtype',list(FORMATS))
def test_tiny_capacity_never_executes_on_rejection(precision_api,monkeypatch,tmp_path,dtype):
    from arch_explorer.adf import Param
    w,s=precision_api;adf,spec,mapping=scene(dtype)
    adf.elements[1].params['capacity']=Param(1,'B','explicit VAL56 impossible capacity assumption')
    with no_backend(monkeypatch),pytest.raises(REJECT,match='(?i)capacity|allocation|storage'):
        compilation=w.compile_workload(adf,spec,mapping)
        s.prepare_compiled(compilation,output_dir=tmp_path/'impossible')


@pytest.mark.parametrize('dtype',['fp16','bf16'])
def test_native_low_precision_refuses_before_sdk_or_execution(precision_api,monkeypatch,tmp_path,dtype):
    from adapter_boundary_cases import native_scene
    w,s=precision_api;adf,_,_,mapping=native_scene();_,spec,_=scene(dtype)
    with no_backend(monkeypatch),pytest.raises(REJECT,match='(?i)precision|dtype|fp32|native|unsupported'):
        compilation=w.compile_workload(adf,spec,mapping,bindings={
            mapping.params['compute']:w.AdapterBinding('accelsim','1',{'threads_per_block':32})})
        s.prepare_compiled(compilation,output_dir=tmp_path/'native')
    assert not list(tmp_path.rglob('*'))
