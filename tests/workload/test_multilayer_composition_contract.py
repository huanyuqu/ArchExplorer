"""Installed SIM46 public composition/encoder consumers; no old-API fallback."""
from copy import deepcopy
import json

import pytest

from composition_expectations import fork_values
from multilayer_contract_cases import (
    REJECT,basic_mapping,composition_api,fork_spec,observe_memory,scene,storage_oracle,
)
from multilayer_expectations import PARAMETER_NAMES,reference_for
from precision_transformer_expectations import tensor_bytes
from test_operator_mapping_attention_contract import no_backend,files
from test_precision_transformer_contract import close_values
from test_operator_mapping_extension_contract import reseal


@pytest.mark.parametrize('layers',[2,4])
@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('case',['regular','boundary'])
@pytest.mark.parametrize('mode',['events','roofline'])
@pytest.mark.parametrize('index',[0,1])
def test_complete_layer_chain_real_binding_instance_mapping_and_actual_storage(
        composition_api,monkeypatch,tmp_path,layers,dtype,case,mode,index):
    from arch_explorer.adf import WorkloadSpec
    w,s=composition_api
    adf,spec,mapping=scene(f'{mode}-{layers}-{dtype}-{case}',index)
    before=deepcopy(spec.to_dict());p=spec.params;data=p['inputs']
    expected=reference_for(dict(config={k:p[k] for k in
        ('batch','sequence','hidden','heads','mlp_hidden','epsilon')},dtype=dtype,X=data['X'],
        instances=[dict(id='layer'+str(i),parameters=value) for i,value in enumerate(data['layers'])]))
    with no_backend(monkeypatch):
        public=w.transformer_encoder(**p)
        raw=w.execute_reference(public)
        compilation=w.compile_workload(adf,public,mapping)
        mapped=w.execute_reference(WorkloadSpec('graph',compilation['program']))
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'run',options={'trace':True})
    assert spec.to_dict()==before
    graph=raw['graph'];instances={row['id']:row for row in graph.meta['composition']['instances']}
    assert set(instances)==set(expected['order'])
    assert set(raw['outputs'])==set(mapped['outputs'])=={'Y'}
    roots=set();intermediates=[]
    for i,identifier in enumerate(expected['order']):
        names=instances[identifier]['tensor_names']
        roots.update(names[key] for key in PARAMETER_NAMES)
        assert all(graph.tensors[names[key]].kind=='weight' for key in PARAMETER_NAMES)
        if i==0:roots.add(names['X'])
        else:assert names['X']==instances[expected['order'][i-1]]['tensor_names']['Y']
        if i<layers-1:intermediates.append(names['Y'])
        for result in (raw,mapped):
            close_values(result['tensors'][names['Y']],expected['layers'][identifier]['output'],dtype)
    assert len(roots)==1+16*layers
    assert set(raw['input_tensors'])==set(mapped['input_tensors'])==roots
    assert set(compilation['program']['meta']['inputs'])==roots
    assert sum(graph.tensors[n].bytes for n in roots if graph.tensors[n].kind=='weight')==layers*600*(4 if dtype=='fp32' else 2)
    for result in (raw,mapped):close_values(result['outputs']['Y'],expected['output'],dtype)
    effective=compilation['source']['mapping']['params']
    assert effective['rules']==mapping.params['rules']
    assert set(effective['operator_rules'])=={op.name for op in graph.operators}
    rules=compilation['plan']['rules']
    for name,binding in mapping.params['operator_rules'].items():
        assert effective['operator_rules'][name]==rules[name]['binding']==binding
    selections={json.dumps(row['binding'],sort_keys=True) for row in rules.values() if row['operator']=='matmul'}
    assert len(selections)>=2
    for step in compilation['plan']['steps']:
        assert step['compute']==mapping.params['compute'][step['origin']]
    conversions=sum(row['access']['conversion_instructions'] for row in compilation['plan']['steps'])
    assert conversions==0 if dtype=='fp32' else conversions>0
    with observe_memory(monkeypatch) as observations:result=s.evaluate_request(request)
    peak,intervals,tasks=storage_oracle(compilation,result,observations)
    by_name={row['name']:row for row in intervals}
    stored={row['tensor']:row for row in compilation['plan']['storage']}
    for name in intermediates:
        assert not stored[name]['returned']
        consumers=stored[name]['consumers'];assert len(set(consumers))>=2
        assert by_name[name]['release']==max(tasks[c]['end'] for c in consumers)
        assert by_name[name]['release']<result['evaluation']['latency_ns']
    before_files=files(tmp_path/'run')
    with no_backend(monkeypatch):assert s.evaluate_request(result)==result
    assert files(tmp_path/'run')==before_files
    for mutation in ('binding','selector','latency'):
        changed=deepcopy(result)
        if mutation=='binding':
            changed['request']['compilation']['source']['workload']['params']['inputs']['layers'][0]['Wq'][0][0]+=1
        elif mutation=='selector':
            choices=changed['request']['compilation']['source']['mapping']['params']['operator_rules']
            name=next(n for n,binding in choices.items() if 'tiles' in binding['options'])
            choices[name]['options']['tiles']['m']+=1
        else:changed['evaluation']['latency_ns']+=1
        with no_backend(monkeypatch),pytest.raises(REJECT):s.evaluate_request(reseal(changed))
    (tmp_path/'STORAGE-ORACLE.json').write_text(json.dumps(dict(peak=peak,intervals=intervals,observations=observations),indent=2))


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('retain',[False,True])
def test_generic_fork_serializes_reorders_and_observes_only_top_level_retention(
        composition_api,monkeypatch,tmp_path,dtype,retain):
    from arch_explorer.adf import WorkloadSpec
    w,s=composition_api;adf,_,_=scene('generic-fork')
    spec=fork_spec(w,dtype=dtype,retain=retain);before=deepcopy(spec.to_dict())
    # All subgraphs deliberately share local tensor/operator names. Declaration
    # order is not the execution dependency order.
    reverse=deepcopy(before);reverse['params']['instances'].reverse()
    with no_backend(monkeypatch):
        graph=w.workload_from_spec(WorkloadSpec(**reverse))
        raw=w.execute_reference(WorkloadSpec(**reverse))
        compilation=w.compile_workload(adf,WorkloadSpec(**reverse),basic_mapping())
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'run',options={'trace':True})
    assert spec.to_dict()==before
    assert len(graph.operators)==len({op.name for op in graph.operators})==4
    assert len(graph.tensors)==5
    assert {op.name for op in graph.operators}=={w.qualified_name(name,'sum' if name=='join' else 'apply')
                                               for name in ('make','left','right','join')}
    expected=fork_values([1.,2.,-1.,.5],dtype=dtype)
    assert raw['outputs']['Y'].tolist()==[expected['y']]
    assert set(raw['outputs'])==({'Y','SHARED'} if retain else {'Y'})
    metadata={row['id']:row for row in graph.meta['composition']['instances']}
    shared=metadata['make']['tensor_names']['y']
    assert metadata['left']['tensor_names']['x']==metadata['right']['tensor_names']['x']==shared
    assert all(not t.alias_of and t.strides is None for t in graph.tensors.values())
    with observe_memory(monkeypatch) as observed:result=s.evaluate_request(request)
    peak,intervals,tasks=storage_oracle(compilation,result,observed)
    unit=tensor_bytes((1,4),dtype)
    assert peak==(4 if retain else 3)*unit
    assert sum(row['access']['read_bytes'] for row in compilation['plan']['steps'])==5*unit
    assert sum(row['access']['write_bytes'] for row in compilation['plan']['steps'])==4*unit
    assert sum(row['access']['conversion_instructions'] for row in compilation['plan']['steps'])==(0 if dtype=='fp32' else 36)
    item=next(row for row in intervals if row['name']==shared)
    assert item['release']==(result['evaluation']['latency_ns'] if retain else
                            max(tasks[w.qualified_name(name,'apply')]['end'] for name in ('left','right')))


@pytest.mark.parametrize('mutation',['duplicate-id','unknown-input-port','unknown-output-port',
    'dangling-source','missing-binding','two-sources','cycle','self-cycle','shape','dtype',
    'unused-root','child-data','alias','strides','duplicate-export','reserved-export'])
def test_composition_refuses_invalid_edges_ports_layout_and_answer_injection(
        composition_api,monkeypatch,tmp_path,mutation):
    from arch_explorer.adf import WorkloadSpec
    w,s=composition_api;adf,_,_=scene('generic-fork');spec=fork_spec(w)
    with no_backend(monkeypatch):assert len(w.workload_from_spec(spec).operators)==4
    d=deepcopy(spec.to_dict());p=d['params']
    if mutation=='duplicate-id':p['instances'][1]['id']=p['instances'][0]['id']
    elif mutation=='unknown-input-port':p['bindings'][0]['to']['port']='missing'
    elif mutation=='unknown-output-port':p['bindings'][1]['from']['port']='missing'
    elif mutation=='dangling-source':p['bindings'][0]['from']={'input':'missing'}
    elif mutation=='missing-binding':p['bindings'].pop()
    elif mutation=='two-sources':p['bindings'].append(deepcopy(p['bindings'][1]))
    elif mutation in ('cycle','self-cycle'):
        p['bindings'][0]['from']={'instance':'join' if mutation=='cycle' else 'make','port':'out'}
        p['inputs']={}
    elif mutation=='shape':
        p['inputs']['ROOT']['shape']=[2,2]
        p['inputs']['ROOT']['value']=[[1,2],[3,4]]
    elif mutation=='dtype':p['inputs']['ROOT']['dtype']='fp16'
    elif mutation=='unused-root':p['inputs']['unused']=deepcopy(p['inputs']['ROOT'])
    elif mutation=='child-data':p['instances'][0]['workload']['params']['meta']['inputs']={'x':[[1,2,3,4]]}
    elif mutation in ('alias','strides'):
        tensor=p['instances'][0]['workload']['params']['tensors'][0]
        tensor['alias_of' if mutation=='alias' else 'strides']='y' if mutation=='alias' else [4,1]
    elif mutation=='duplicate-export':p['outputs']['alsoY']=deepcopy(p['outputs']['Y'])
    elif mutation=='reserved-export':p['outputs']['@bad']=p['outputs'].pop('Y')
    with no_backend(monkeypatch),pytest.raises(REJECT):
        invalid=WorkloadSpec(**d)
        graph=w.workload_from_spec(invalid)
        compiled=w.compile_workload(adf,invalid,basic_mapping())
        s.prepare_compiled(compiled,output_dir=tmp_path/'invalid')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('enough',[True,False])
def test_hand_checkable_fork_capacity_threshold(composition_api,monkeypatch,tmp_path,dtype,enough):
    from arch_explorer.adf import Param
    w,s=composition_api;adf,_,_=scene('generic-fork');spec=fork_spec(w,dtype=dtype)
    expected=3*tensor_bytes((1,4),dtype)
    adf.elements[1].params['capacity']=Param(expected if enough else expected-1,'B',
        'VAL57 manually derived three-live-tensor fork bound; synthetic capacity')
    try:
        with no_backend(monkeypatch):
            compilation=w.compile_workload(adf,spec,basic_mapping())
            request=s.prepare_compiled(compilation,output_dir=tmp_path/'capacity',options={'trace':True})
    except REJECT as error:
        assert not enough and any(word in str(error).lower() for word in ('capacity','storage','alloc'))
        return
    if enough:
        with observe_memory(monkeypatch) as observed:result=s.evaluate_request(request)
        assert storage_oracle(compilation,result,observed)[0]==expected
    else:
        with pytest.raises((*REJECT,MemoryError),match='(?i)capacity|storage|alloc'):
            s.evaluate_request(request)


def test_nested_exports_are_not_implicitly_returned(composition_api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec
    w,s=composition_api;adf,_,_=scene('generic-fork')
    inner=deepcopy(fork_spec(w,retain=True).to_dict())
    del inner['params']['inputs']['ROOT']['value']
    nested=w.compose_workloads(instances=[dict(id='nested',workload=WorkloadSpec(**inner),
        inputs={'x':'ROOT'},outputs={'out':'Y'})],
        inputs={'X':dict(shape=[1,4],dtype='fp32',kind='activation',value=[[1.,2.,-1.,.5]])},
        bindings=[dict(to=dict(instance='nested',port='x'),**{'from':{'input':'X'}})],
        outputs={'RESULT':dict(instance='nested',port='out')})
    with no_backend(monkeypatch):
        raw=w.execute_reference(nested)
        compilation=w.compile_workload(adf,nested,basic_mapping())
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'nested',options={'trace':True})
    assert set(raw['outputs'])=={'RESULT'} and raw['outputs']['RESULT'].tolist()==[[4.,8.,-4.,2.]]
    item=raw['graph'].meta['composition']['instances'][0]
    internal=item['tensor_names']['SHARED']
    assert not next(row for row in compilation['plan']['storage'] if row['tensor']==internal)['returned']
    with observe_memory(monkeypatch) as observed:result=s.evaluate_request(request)
    peak,intervals,_=storage_oracle(compilation,result,observed)
    assert peak==48
    assert next(row for row in intervals if row['name']==internal)['release']<result['evaluation']['latency_ns']


def test_produced_weight_label_does_not_turn_computed_storage_into_static_input(
        composition_api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec
    w,s=composition_api;adf,_,_=scene('generic-fork')
    document=fork_spec(w).to_dict()
    for instance in document['params']['instances']:
        for tensor in instance['workload']['params']['tensors']:
            if tensor['name']=='y':tensor['kind']='weight'
    spec=WorkloadSpec(**document)
    with no_backend(monkeypatch):
        raw=w.execute_reference(spec)
        compilation=w.compile_workload(adf,spec,basic_mapping())
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'produced-weight',options={'trace':True})
    assert set(raw['input_tensors'])=={'ROOT'}
    assert raw['outputs']['Y'].tolist()==[[4.,8.,-4.,2.]]
    with observe_memory(monkeypatch) as observed:result=s.evaluate_request(request)
    peak,intervals,_=storage_oracle(compilation,result,observed)
    assert peak==48
    shared=next(row for row in raw['graph'].meta['composition']['instances'] if row['id']=='make')['tensor_names']['y']
    interval=next(row for row in intervals if row['name']==shared)
    assert interval['release']==interval['last_reader']<result['evaluation']['latency_ns']
