"""Exact original operator identity survives request spaces and save/restore."""
from copy import deepcopy
import json

import pytest

from multilayer_contract_cases import REJECT,composition_api,scene
from multilayer_expectations import matmul_tiles
from test_operator_mapping_attention_contract import no_backend


def rule(tiles):return dict(id='matmul.tiled',version='1',options=dict(tiles=tiles))


def partial_mapping(w):
    from arch_explorer.adf import MappingSpec
    adf,spec,original=scene('events-2-fp32-regular')
    params=deepcopy(original.params)
    first=w.qualified_name('layer0','q_projection');second=w.qualified_name('layer1','q_projection')
    params['rules']={'matmul':rule(dict(m=2,k=3,n=3))}
    params['operator_rules']={second:rule(dict(m=3,k=5,n=5))}
    return adf,spec,MappingSpec('compile_graph_v3',params),first,second


def test_request_space_keeps_partial_requested_fields_and_compile_resolves_every_instance(
        composition_api,monkeypatch,tmp_path):
    import arch_explorer.mapping
    from arch_explorer.adf import MAPPING_TEMPLATES,MappingSpec
    w,s=composition_api;adf,spec,mapping,first,second=partial_mapping(w)
    original=deepcopy(mapping.params)
    with no_backend(monkeypatch):
        requested=MAPPING_TEMPLATES.get('compile_graph_v3').request_space(adf,spec,mapping.params)
        assert requested=={key:[value] for key,value in original.items()}
        rebuilt=MappingSpec(mapping.template,{key:choices[0] for key,choices in requested.items()})
        compilation=w.compile_workload(adf,spec,rebuilt)
        graph=w.workload_from_spec(spec)
        prepared=s.prepare_compiled(compilation,output_dir=tmp_path/'prepared')
        assert w.validate_compilation(json.loads(json.dumps(compilation)))==compilation
        assert s.assess_request(json.loads(json.dumps(prepared)))['supported']
    assert mapping.params==original
    effective=compilation['source']['mapping']['params']
    assert effective['rules']==original['rules']
    assert set(effective['operator_rules'])=={op.name for op in graph.operators}
    assert effective['operator_rules'][first]==original['rules']['matmul']
    assert effective['operator_rules'][second]==original['operator_rules'][second]
    assert effective['operator_rules'][first]!=effective['operator_rules'][second]
    assert compilation['plan']['rules'][first]['binding']==effective['operator_rules'][first]
    assert compilation['plan']['rules'][second]['binding']==effective['operator_rules'][second]
    mapped={op['name']:op for op in compilation['program']['operators']}
    originals={op.name:op for op in graph.operators}
    for name in (first,second):
        a=originals[name].attrs;tiles=effective['operator_rules'][name]['options']['tiles']
        regions=matmul_tiles(a['batch'],a['m'],a['k'],a['n'],tiles)
        # Compiled tiles carry the whole batch. Geometry is checked once;
        # batch/ascending K-loop bounds remain explicit on each real operator.
        expected={(r['rows'][0],r['columns'][0],r['rows'][1]-r['rows'][0],
                   r['columns'][1]-r['columns'][0],tiles['k']) for r in regions}
        actual=[mapped[step['name']]['attrs'] for step in compilation['plan']['steps']
                if step['origin']==name and step['semantic']=='compiled.matmul']
        assert len(actual)==len(expected)
        assert {(r['row'],r['col'],r['rows'],r['cols'],r['k_tile']) for r in actual}==expected
        assert all(r['batch']==a['batch'] and r['k']==a['k'] for r in actual)


@pytest.mark.parametrize('mutation',['unknown-instance','kind-default-as-instance','unknown-kind',
    'wrong-version','bad-option','extra-field','incomplete-compute','incomplete-memory'])
def test_instance_selector_errors_are_not_silently_ignored(composition_api,monkeypatch,mutation):
    from arch_explorer.adf import MappingSpec
    w,_=composition_api;adf,spec,mapping,_,second=partial_mapping(w)
    p=deepcopy(mapping.params)
    if mutation=='unknown-instance':p['operator_rules']['missing']=deepcopy(p['operator_rules'][second])
    elif mutation=='kind-default-as-instance':p['operator_rules']['matmul']=p['operator_rules'].pop(second)
    elif mutation=='unknown-kind':p['rules']['invented']=deepcopy(p['rules']['matmul'])
    elif mutation=='wrong-version':p['operator_rules'][second]['version']='unregistered'
    elif mutation=='bad-option':p['operator_rules'][second]['options']['tiles']['k']=0
    elif mutation=='extra-field':p['layer_tiles']={}
    elif mutation=='incomplete-compute':p['compute'].pop(next(iter(p['compute'])))
    else:p['memory']={second:'board/sram'}
    with no_backend(monkeypatch),pytest.raises(REJECT):
        w.compile_workload(adf,spec,MappingSpec('compile_graph_v3',p))


def test_v2_keeps_kind_semantics_and_migration_is_explicit_without_execution(composition_api,monkeypatch):
    from arch_explorer.adf import MappingSpec
    w,_=composition_api;adf,spec,mapping,_,second=partial_mapping(w)
    old=MappingSpec('compile_graph_v2',{k:deepcopy(v) for k,v in mapping.params.items() if k!='operator_rules'})
    before=deepcopy(old.to_dict())
    with no_backend(monkeypatch):
        migrated=w.migrate_mapping_v2(old)
        assert migrated.template=='compile_graph_v3' and migrated.params['operator_rules']=={}
        assert {k:v for k,v in migrated.params.items() if k!='operator_rules'}==old.params
        assert old.to_dict()==before
        legacy=w.compile_workload(adf,spec,old)
        assert legacy['source']['mapping']['template']=='compile_graph_v2'
        illegal=MappingSpec('compile_graph_v2',dict(old.params,operator_rules={second:mapping.params['operator_rules'][second]}))
        with pytest.raises(REJECT):w.compile_workload(adf,spec,illegal)
