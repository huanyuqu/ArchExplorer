"""Hand patch packing, broadcast bias, concat, CLS aliases and real capacity."""
from copy import deepcopy
from collections import Counter
import json

import pytest

from test_operator_mapping_attention_contract import no_backend
from view_storage_runtime_cases import address_rows,audit_storage,observe_memory
from view_storage_contract_cases import independent_layouts
from vit_contract_cases import REJECT,frozen_inputs,vit_api
from vit_expectations import Geometry,patch_addresses,shaped


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('retain',[False,True])
def test_actual_patch_bias_concat_and_shared_CLS_root_fit_and_one_byte_below(vit_api,monkeypatch,tmp_path,dtype,retain):
    from arch_explorer.adf import WorkloadSpec,MappingSpec,from_dict
    w,s = vit_api
    g=Geometry(batch=2)
    graph=w.Workload('hand-ViT-prefix',dtype=dtype)
    for name,shape,kind in [('X',(2,3,4,4),'activation'),('D',(12,),'weight'),('C',(2,1,12),'weight'),
                            ('P',(2,4,12),'activation'),('B',(2,4,12),'activation'),('Y',(2,5,12),'activation'),
                            ('L',(2,1,12),'activation'),('R',(2,1,12),'activation'),('Z',(2,1,12),'activation')]:
        graph.tensor(name,shape,dtype=dtype,kind=kind)
    graph.meta['inputs']={'X':shaped((i/128 for i in range(96)),(2,3,4,4)),
                          'D':[.25]*12,'C':[[[-.5]*12] for _ in range(2)]}
    graph.add('pack','patchify',['X'],['P'],shape=[2,4,12],patch=2)
    graph.add('bias','bias',['P','D'],['B'],shape=[2,4,12])
    graph.add('concat','concat',['C','B'],['Y'],shape=[2,5,12],axis=1)
    w.readonly_view(graph,'S','Y',transform='slice',shape=(2,1,12),starts=(0,0,0),stops=(2,1,12),steps=(1,1,1))
    graph.add('left','scale',['S'],['L'],shape=[2,1,12],factor=2.)
    graph.add('right','mul',['S','S'],['R'],shape=[2,1,12])
    graph.add('merge','add',['L','R'],['Z'],shape=[2,1,12])
    graph.meta['outputs']=['Z','S'] if retain else ['Z']
    spec=WorkloadSpec('graph',graph.to_dict())
    mapping=MappingSpec('compile_graph_v3',dict(compute={op.name:'board/core/sm1' if op.name=='right' else 'board/core/sm' for op in graph.operators},
        memory='board/sram',rules={},operator_rules={}))
    adf=from_dict(deepcopy(frozen_inputs()['hardware']['events']))
    with no_backend(monkeypatch):
        compilation=w.compile_workload(adf,spec,mapping)
        program=compilation['program'];layouts=independent_layouts(program)
        mapped_graph=w.workload_from_spec(WorkloadSpec('graph',program))
        steps={row['name']:row for row in compilation['plan']['steps']}
        unit=4 if dtype=='fp32' else 2
        expected=[]
        for b,patches in enumerate(patch_addresses(g)):
            for token,addresses in enumerate(patches):
                for feature,address in enumerate(addresses):
                    flat=address;dc=flat%4;flat//=4;dr=flat%4;flat//=4;c=flat%3;batch=flat//3
                    expected.append(dict(direction='read',tensor='X',storage='X',logical_index=[batch,c,dr,dc],
                        element_offset=address,byte_offset=address*unit,bytes=unit))
                    out=(b*4+token)*12+feature
                    expected.append(dict(direction='write',tensor='P',storage='P',logical_index=[b,token,feature],
                        element_offset=out,byte_offset=out*unit,bytes=unit))
        # The interface specifies logical accesses, not a micro-instruction
        # issue order. Compare complete rows with multiplicity preserved.
        actual=w.logical_accesses(mapped_graph,'pack')
        assert Counter(json.dumps(r,sort_keys=True) for r in actual)==Counter(json.dumps(r,sort_keys=True) for r in expected)
        assert steps['pack']['access']['read_bytes']==steps['pack']['access']['write_bytes']==96*unit
        assert steps['pack']['access']['conversion_instructions']==0
        assert steps['bias']['access']['read_bytes']==192*unit
        assert steps['bias']['access']['write_bytes']==96*unit
        assert steps['concat']['access']['read_bytes']==steps['concat']['access']['write_bytes']==120*unit
        assert steps['concat']['access']['conversion_instructions']==0
        if dtype!='fp32': assert steps['bias']['access']['conversion_instructions']==288
        assert layouts['S'].addresses()==list(range(12))+list(range(60,72))
        assert layouts['S'].read_bytes==24*unit and layouts['S'].root.bytes==120*unit
        reads=w.logical_accesses(mapped_graph,'right')
        assert sum(r['bytes'] for r in reads if r['direction']=='read')==48*unit
        assert {r['storage'] for r in reads if r['direction']=='read'}=={'Y'}
        raw=w.execute_reference(spec)
        assert raw['outputs']['Z'].reshape(-1).tolist()==[-.75]*24
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'actual',options={'trace':True})
    with observe_memory(monkeypatch) as observed: result=s.evaluate_request(request)
    facts=audit_storage(compilation,result,observed)
    y=next(row for row in facts['intervals'] if row['storage']=='Y')
    assert y['aliases']==['S','Y'] and y['bytes']==120*unit and y['returned'] is retain
    assert y['end']==pytest.approx(result['evaluation']['latency_ns'] if retain else
                                  max(facts['task_end'][name] for name in ('left','right','S.view','concat')))
    peak=facts['peaks']['board/sram'];largest=max(row['bytes'] for row in facts['intervals'])
    capacity_facts=[]
    for capacity in (peak,peak-1,largest-1):
        if capacity in [row['bytes'] for row in capacity_facts]:continue
        doc=deepcopy(adf.to_dict())
        def change(node):
            if isinstance(node,dict):
                if node.get('name')=='sram':node['params']['capacity']['value']=capacity
                for value in node.values():change(value)
            elif isinstance(node,list):
                for value in node:change(value)
        change(doc);constrained=from_dict(doc)
        if capacity<largest:
            with no_backend(monkeypatch),pytest.raises(REJECT):w.compile_workload(constrained,spec,mapping)
            phase='compile'
        else:
            with no_backend(monkeypatch):
                c=w.compile_workload(constrained,spec,mapping)
                prepared=s.prepare_compiled(c,output_dir=tmp_path/f'capacity-{capacity}',options={'trace':True})
            if capacity==peak:assert s.evaluate_request(prepared)['evaluation']['latency_ns']>0
            else:
                with pytest.raises((ValueError,RuntimeError)):s.evaluate_request(prepared)
            phase='execute'
        capacity_facts.append(dict(bytes=capacity,phase=phase,supported=capacity==peak))
    (tmp_path/'ACTUAL-PATCH-CAPACITY.json').write_text(json.dumps(dict(actual=facts,capacity=capacity_facts),indent=2)+'\n')
