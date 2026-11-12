"""Same semantic graph, explicit layout policy and preserved original selectors."""
from copy import deepcopy
import json

import pytest

from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import REJECT,scene,view_api


@pytest.mark.parametrize('index',[0,1])
def test_layout_candidates_match_frozen_pair_preserve_input_and_request_space(view_api,monkeypatch,index):
    import arch_explorer.mapping
    from arch_explorer.adf import MAPPING_TEMPLATES
    w,s=view_api;adf,spec,seed=scene('events-2-bf16-regular',index=index,strategy='materialized')
    _,_,expected_views=scene('events-2-bf16-regular',index=index,strategy='views')
    original_spec=deepcopy(spec.to_dict());original_mapping=deepcopy(seed.to_dict())
    with no_backend(monkeypatch):
        candidates=w.layout_candidates(spec,seed)
        assert set(candidates)=={'materialized','views'}
        assert candidates['materialized'].to_dict()==seed.to_dict()
        assert candidates['views'].to_dict()==expected_views.to_dict()
        graph=w.workload_from_spec(spec);operators={op.name:op for op in graph.operators}
        for strategy,mapping in candidates.items():
            params=deepcopy(mapping.params)
            space=MAPPING_TEMPLATES.get('compile_graph_v3').request_space(adf,spec,params)
            assert space=={key:[value] for key,value in params.items()}
            compiled=w.compile_workload(adf,spec,mapping)
            assert w.validate_compilation(json.loads(json.dumps(compiled)))==compiled
            effective=compiled['source']['mapping']['params']['operator_rules']
            assert set(effective)==set(operators)
            for name,binding in mapping.params['operator_rules'].items():
                assert effective[name]==binding==compiled['plan']['rules'][name]['binding']
                if operators[name].op not in ('transpose','reshape','slice'):
                    assert binding==seed.params['operator_rules'][name]
            assert mapping.params==params
    assert spec.to_dict()==original_spec and seed.to_dict()==original_mapping
    assert candidates['views'].params['compute']==seed.params['compute']
    assert candidates['views'].params['memory']==seed.params['memory']


def test_new_view_rules_do_not_change_legacy_v2_materialized_default(view_api,monkeypatch):
    from arch_explorer.adf import MappingSpec
    w,_=view_api;adf,spec,seed=scene('events-2-fp32-regular',strategy='materialized')
    legacy=MappingSpec('compile_graph_v2',{key:deepcopy(seed.params[key]) for key in ('compute','memory','rules')})
    with no_backend(monkeypatch):
        graph=w.workload_from_spec(spec)
        compiled=w.compile_workload(adf,spec,legacy)
    assert compiled['source']['mapping']['template']=='compile_graph_v2'
    for op in graph.operators:
        if op.op in ('transpose','reshape','slice'):
            assert compiled['plan']['rules'][op.name]['binding']['id']==op.op+'.materialized'
    assert all(row['alias_of'] is None for row in compiled['program']['tensors'])


@pytest.mark.parametrize('mutation',['unknown-selector','bad-rule-version','view-options'])
def test_view_policy_rejects_invalid_effective_instance_selection(view_api,monkeypatch,mutation):
    from arch_explorer.adf import MappingSpec
    w,_=view_api;adf,spec,mapping=scene('events-2-fp32-regular')
    params=deepcopy(mapping.params)
    name=next(key for key,value in params['operator_rules'].items() if value['id']=='transpose.view')
    if mutation=='unknown-selector':params['operator_rules']['absent-instance']=deepcopy(params['operator_rules'][name])
    elif mutation=='bad-rule-version':params['operator_rules'][name]['version']='not-supported'
    else:params['operator_rules'][name]['options']={'silent_pack':True}
    with no_backend(monkeypatch),pytest.raises(REJECT):w.compile_workload(adf,spec,MappingSpec('compile_graph_v3',params))


def test_explicit_transpose_reshape_policy_records_materialized_fallback(view_api,monkeypatch):
    from arch_explorer.adf import MappingSpec,WorkloadSpec
    w,_=view_api;adf,_,_=scene('events-offset-fork')
    graph=w.Workload('fallback-proof',dtype='fp32')
    graph.tensor('X',(2,3),dtype='fp32');graph.tensor('T',(3,2),dtype='fp32');graph.tensor('Y',(6,),dtype='fp32')
    graph.add('transpose','transpose',['X'],['T'],shape=[3,2],axes=[1,0])
    graph.add('reshape','reshape',['T'],['Y'],shape=[6])
    graph.meta.update(inputs={'X':[[0.,1.,2.],[3.,4.,5.]]},outputs=['Y'])
    spec=WorkloadSpec('graph',graph.to_dict())
    seed=MappingSpec('compile_graph_v3',dict(compute='board/core/sm',memory='board/sram',rules={},operator_rules={}))
    with no_backend(monkeypatch):
        candidates=w.layout_candidates(spec,seed)
        selected=candidates['views']
        assert selected.params['operator_rules']['transpose']['id']=='transpose.view'
        assert selected.params['operator_rules']['reshape']['id']=='reshape.materialized'
        compiled=w.compile_workload(adf,spec,selected)
        assert compiled['source']['mapping']['params']['operator_rules']['reshape']['id']=='reshape.materialized'
        reference=w.execute_reference(WorkloadSpec('graph',compiled['program']))
    assert reference['outputs']['Y'].tolist()==[0.,3.,1.,4.,2.,5.]
    tensors={row['name']:row for row in compiled['program']['tensors']}
    assert tensors['T']['alias_of']=='X' and tensors['Y']['alias_of'] is None
