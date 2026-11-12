"""Real alias readers, physical release, traffic and capacity rejection phases."""
from copy import deepcopy
import json

import pytest

from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import REJECT,hand_graph,independent_layouts,scene,view_api
from view_storage_runtime_cases import address_rows,audit_storage,fork_graph,observe_memory


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_real_logical_accesses_count_holey_elements_and_repeated_operands(view_api,monkeypatch,dtype):
    w,_=view_api
    with no_backend(monkeypatch):
        g=hand_graph(w,dtype=dtype);layouts=independent_layouts(g.to_dict())
        expected=address_rows(layouts['U'],'U','read')+address_rows(layouts['Y'],'Y','write')
        assert w.logical_accesses(g,'read')==expected
        assert sum(r['bytes'] for r in expected if r['direction']=='read')==layouts['U'].read_bytes
        assert max(r['element_offset'] for r in expected if r['direction']=='read')==23
        assert w.logical_accesses(g,'S.view')==[]
        g.tensor('Z',(2,2),dtype=dtype);g.add('double','add',['U','U'],['Z'],shape=[2,2])
        rows=w.logical_accesses(g,'double');reads=[r for r in rows if r['direction']=='read']
        # Per-output operand ordering preserves each of two actual reads.
        assert [r['element_offset'] for r in reads]==[9,9,21,21,11,11,23,23]
        assert sum(r['bytes'] for r in reads)==2*layouts['U'].read_bytes
        with pytest.raises(REJECT):w.logical_accesses(g,'double',limit=len(rows)-1)
        assert w.logical_accesses(g,'double',limit=len(rows))==rows


@pytest.mark.parametrize('mode',['events','roofline'])
@pytest.mark.parametrize('dtype',['fp32','bf16'])
@pytest.mark.parametrize('retain',[False,True])
def test_actual_root_allocation_last_reader_controls_export_and_exact_capacity(
        view_api,monkeypatch,tmp_path,mode,dtype,retain):
    from arch_explorer.adf import WorkloadSpec,MappingSpec,from_dict
    w,s=view_api;adf,_,seed=scene(mode+'-2-'+dtype+'-regular')
    paths=sorted(set(seed.params['compute'].values()));assert len(paths)==2
    observations=[]
    for swap in (False,True):
        graph=fork_graph(w,dtype=dtype,retain=retain,swap=swap);spec=WorkloadSpec('graph',graph.to_dict())
        compute={op.name:(paths[1] if op.name=='right' else paths[0]) for op in graph.operators}
        mapping=MappingSpec('compile_graph_v3',dict(compute=compute,memory='board/sram',rules={},operator_rules={}))
        with no_backend(monkeypatch):
            compilation=w.compile_workload(adf,spec,mapping)
            request=s.prepare_compiled(compilation,output_dir=tmp_path/('swap' if swap else 'original'),options={'trace':True})
        with observe_memory(monkeypatch) as events:result=s.evaluate_request(request)
        facts=audit_storage(compilation,result,events);observations.append(facts)
        root=next(row for row in facts['intervals'] if row['storage']=='X')
        assert root['aliases']==['S','T','U','X'] and root['bytes']==24*(4 if dtype=='fp32' else 2)
        assert root['end']==pytest.approx(result['evaluation']['latency_ns'] if retain else
            max(facts['task_end'][name] for name in ('left','right')))
        assert (facts['task_end']['left']>facts['task_end']['right']) is swap
        if swap:continue
        peak=facts['peaks']['board/sram'];largest=max(row['bytes'] for row in facts['intervals'])
        for capacity in (peak,peak-1):
            data=deepcopy(adf.to_dict())
            def set_capacity(node):
                if isinstance(node,dict):
                    if node.get('name')=='sram':node['params']['capacity']['value']=capacity
                    for value in node.values():set_capacity(value)
                elif isinstance(node,list):
                    for value in node:set_capacity(value)
            set_capacity(data)
            constrained=from_dict(data);directory=tmp_path/('capacity-'+str(capacity))
            if capacity<largest:
                with no_backend(monkeypatch),pytest.raises(REJECT):w.compile_workload(constrained,spec,mapping)
                phase='compile'
            else:
                with no_backend(monkeypatch):
                    compiled=w.compile_workload(constrained,spec,mapping)
                    prepared=s.prepare_compiled(compiled,output_dir=directory,options={'trace':True})
                if capacity==peak:assert s.evaluate_request(prepared)['evaluation']['latency_ns']>0
                else:
                    with pytest.raises((ValueError,RuntimeError)):s.evaluate_request(prepared)
                phase='execute'
            facts.setdefault('capacity',[]).append(dict(bytes=capacity,phase=phase,supported=capacity==peak))
    (tmp_path/'PHYSICAL-STORAGE.json').write_text(json.dumps(observations,indent=2)+'\n')


def test_static_BF16_weight_view_is_allocated_once_and_conflicting_root_placement_rejects(view_api,monkeypatch,tmp_path):
    from arch_explorer.adf import MappingSpec,WorkloadSpec
    w,s=view_api;adf,_,_=scene('events-offset-fork')
    spec=WorkloadSpec('graph',fork_graph(w,dtype='bf16',weight=True).to_dict())
    params=dict(compute='board/core/sm',memory='board/sram',rules={},operator_rules={})
    with no_backend(monkeypatch):
        compilation=w.compile_workload(adf,spec,MappingSpec('compile_graph_v3',params))
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'weight',options={'trace':True})
    with observe_memory(monkeypatch) as events:result=s.evaluate_request(request)
    facts=audit_storage(compilation,result,events)
    root=next(row for row in facts['intervals'] if row['storage']=='X')
    assert root['bytes']==48 and root['resident'] and root['end']==result['evaluation']['latency_ns']
    names=[op.name for op in w.workload_from_spec(spec).operators]
    params['memory']={name:('board/dram' if name=='right' else 'board/sram') for name in names}
    with no_backend(monkeypatch),pytest.raises(REJECT):w.compile_workload(adf,spec,MappingSpec('compile_graph_v3',params))
