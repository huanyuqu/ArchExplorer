"""Unknown independent FP32 extensions and cross-subgraph view roots."""
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import hashlib
import sys

import pytest

from test_operator_mapping_attention_contract import no_backend
from test_multilayer_square_compatibility import test_bundled_public_square_keeps_FP32_two_reads_and_existing_backends as square_control
from view_storage_contract_cases import REJECT,scene,view_api
from view_storage_runtime_cases import audit_storage,observe_memory


def test_unknown_nested_builder_source_change_still_invalidates_saved_requests(view_api,tmp_path):
    from test_composition_extension_provenance import test_mutating_nested_builder_source_rejects_old_compilation_and_request as consume
    consume(view_api,tmp_path)


def test_genuine_historical_bound_result_is_rejected_without_execution(view_api,monkeypatch):
    path=Path(os.environ['VAL58_HISTORICAL_RESULT']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL58_HISTORICAL_RESULT_SHA256']
    payload=json.loads(raw);assert payload['schema']=='arch-explorer.compiled-result.v3'
    _,s=view_api
    with no_backend(monkeypatch),pytest.raises(REJECT):s.evaluate_request(payload)
    assert path.read_bytes()==raw


@pytest.mark.parametrize('mode',['events','roofline'])
def test_unchanged_unknown_FP32_square_with_independent_storage(view_api,monkeypatch,tmp_path,mode):
    square_control(view_api,monkeypatch,tmp_path,mode)


def test_old_adapter_default_refuses_alias_before_its_lower_but_contiguous_control_executes(view_api,monkeypatch,tmp_path):
    from arch_explorer.adf import MappingSpec,WorkloadSpec,from_dict
    from adapter_boundary_cases import events_scene
    from test_adapter_binding_contract import external_scene
    extension=importlib.import_module('val54_adapter_extension')
    assert 'site-packages' in extension.__file__ and extension.FIRST.layouts==('independent',)
    w,s=view_api
    adf,_,spec,mapping=external_scene()
    with no_backend(monkeypatch):
        c=w.compile_workload(adf,spec,mapping)
        request=s.prepare_compiled(c,output_dir=tmp_path/'legacy-independent',options={'trace':True})
    assert s.evaluate_request(request)['evaluation']['latency_ns']>0
    _,g,_,old=events_scene();op=g.operators[0];source=op.inputs[0];shape=g.tensors[source].shape
    axes=tuple(reversed(range(len(shape))))
    w.readonly_view(g,'reverse',source,transform='transpose',shape=tuple(reversed(shape)),axes=axes)
    w.readonly_view(g,'restored','reverse',transform='transpose',shape=shape,axes=axes)
    document=g.to_dict();operators=document['operators'];operators[0]['inputs'][0]='restored'
    document['operators']=operators[-2:]+operators[:-2]
    base,_,seed=scene('events-2-fp32-regular');hardware=deepcopy(base.to_dict())
    paths=sorted(set(seed.params['compute'].values()));assert len(paths)==2
    selected=paths[1].split('/')[-1]
    def replace_component(node):
        if isinstance(node,dict):
            if node.get('name')==selected and node.get('component')=='sm':node['implementation']['id']='val54.sm'
            for value in node.values():replace_component(value)
        elif isinstance(node,list):
            for value in node:replace_component(value)
    replace_component(hardware)
    compute={row['name']:(paths[0] if row['op']=='storage.view' else paths[1]) for row in document['operators']}
    mapping=MappingSpec('compile_graph_v3',dict(old.params,compute=compute,memory='board/sram',operator_rules={}))
    calls=[];previous=sys.getprofile()
    def observe(frame,event,arg):
        if event=='call' and frame.f_code is extension.lower.__code__:calls.append(frame.f_code.co_name)
    try:
        sys.setprofile(observe)
        with no_backend(monkeypatch),pytest.raises(REJECT):
            w.compile_workload(from_dict(hardware),WorkloadSpec('graph',document),mapping)
    finally:sys.setprofile(previous)
    assert not calls


def test_unsupported_native_alias_rejects_without_emission_or_execution(view_api,monkeypatch):
    from arch_explorer.adf import WorkloadSpec,MappingSpec
    from adapter_boundary_cases import native_scene
    from arch_explorer.workload.adapters import materialized_native as native
    from view_storage_contract_cases import hand_graph
    w,_=view_api;adf,_,_,old=native_scene()
    mapping=MappingSpec('compile_graph_v3',dict(old.params,operator_rules={}))
    entered=[];previous=sys.getprofile()
    def observe(frame,event,arg):
        if event=='call' and frame.f_code is native._emitter.__code__:entered.append(frame.f_code.co_name)
    try:
        sys.setprofile(observe)
        with no_backend(monkeypatch),pytest.raises(REJECT):
            w.compile_workload(adf,WorkloadSpec('graph',hand_graph(w).to_dict()),mapping)
    finally:sys.setprofile(previous)
    assert not entered


def test_nested_composition_views_reuse_one_root_across_real_ports_and_top_exports(view_api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec,MappingSpec
    w,s=view_api
    leaf=w.Workload('slice-leaf',dtype='fp32');leaf.tensor('x',(4,6),dtype='fp32')
    w.readonly_view(leaf,'v','x',transform='slice',shape=(2,3),starts=(1,1),stops=(4,6),steps=(2,2))
    inner=w.compose_workloads(instances=[dict(id='slice',workload=leaf,inputs={'in':'x'},outputs={'out':'v'})],
        inputs={'ROOT':dict(shape=[4,6],dtype='fp32',kind='activation')},
        bindings=[dict(to=dict(instance='slice',port='in'),**{'from':{'input':'ROOT'}})],
        outputs={'V':dict(instance='slice',port='out')},name='nested-slice')
    extension=importlib.import_module('val57_composition_extension')
    assert 'site-packages' in extension.__file__
    reader=WorkloadSpec(extension.KIND,dict(shape=[2,3],factor=2.))
    spec=w.compose_workloads(instances=[dict(id='inner',workload=inner,inputs={'in':'ROOT'},outputs={'out':'V'}),
        dict(id='reader',workload=reader,inputs={'in':'x'},outputs={'out':'y'})],
        inputs={'ROOT':dict(shape=[4,6],dtype='fp32',kind='activation',value=[[r*6+c for c in range(6)] for r in range(4)])},
        bindings=[dict(to=dict(instance='inner',port='in'),**{'from':{'input':'ROOT'}}),
                  dict(to=dict(instance='reader',port='in'),**{'from':dict(instance='inner',port='out')})],
        outputs={'Y':dict(instance='reader',port='out'),'HELD':dict(instance='inner',port='out')},name='nested-view-ports')
    adf,_,_=scene('events-offset-fork')
    mapping=MappingSpec('compile_graph_v3',dict(compute='board/core/sm',memory='board/sram',rules={},operator_rules={}))
    with no_backend(monkeypatch):
        c=w.compile_workload(adf,spec,mapping);raw=w.execute_reference(spec)
        mapped=w.execute_reference(WorkloadSpec('graph',c['program']))
        request=s.prepare_compiled(c,output_dir=tmp_path/'nested',options={'trace':True})
    assert raw['outputs']['Y'].tolist()==mapped['outputs']['Y'].tolist()==[[14.,18.,22.],[38.,42.,46.]]
    with observe_memory(monkeypatch) as events:result=s.evaluate_request(request)
    facts=audit_storage(c,result,events)
    roots=[row for row in facts['intervals'] if row['bytes']==96]
    assert len(roots)==1 and roots[0]['returned']
    assert roots[0]['end']==result['evaluation']['latency_ns']
    (tmp_path/'NESTED-STORAGE.json').write_text(json.dumps(facts,indent=2)+'\n')
