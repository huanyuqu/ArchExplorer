"""Independent full CLIP semantics and actual installed component storage."""
from copy import deepcopy
import json,math
import pytest
from clip_contract_cases import canonical,clip_api,compare_reference,execution_rows,scene
from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import independent_layouts,assert_layout
from view_storage_runtime_cases import audit_storage,observe_memory

def case_id(row):
    key,mode,policy,index,execute=row
    return f'{key}-{mode}-{policy}-mapping{index}-'+('execute' if execute else 'CPU')

ROWS=list(execution_rows())
@pytest.mark.parametrize('key,mode,policy,index,execute',ROWS,ids=[case_id(r) for r in ROWS])
def test_full_rectangular_CLIP_two_towers_values_and_actual_services(clip_api,monkeypatch,tmp_path,key,mode,policy,index,execute):
    from arch_explorer.adf import WorkloadSpec
    w,s=clip_api;adf,spec,mapping=scene(key,mode,policy,index);before=deepcopy(spec.to_dict())
    with no_backend(monkeypatch):
        original=w.execute_reference(spec);compare_reference(original,key)
        c=w.compile_workload(adf,spec,mapping);mapped=w.execute_reference(WorkloadSpec('graph',c['program']));compare_reference(mapped,key)
        for n,value in original['tensors'].items():
            if n in mapped['tensors']:
                dtype=original['graph'].tensors[n].dtype
                assert canonical(value.reshape(-1).tolist(),dtype)==canonical(mapped['tensors'][n].reshape(-1).tolist(),dtype),n
        rules=c['source']['mapping']['params']['operator_rules']
        assert set(rules)=={op.name for op in original['graph'].operators}==set(c['plan']['rules'])
        for n,b in rules.items():assert c['plan']['rules'][n]['binding']==b
        layouts=independent_layouts(c['program'])
        for n,layout in layouts.items():assert_layout(w,mapped['graph'].tensors,n,layout)
        a,b=(layouts[n] for n in ('logits_per_image','logits_per_text'))
        assert a.shape==(spec.params['image_batch'],spec.params['text_batch'])
        assert b.shape==a.shape[::-1]
        assert (a.root.name==b.root.name)==(policy=='views')
        names=original['graph'].meta['clip']['tensors'];table=layouts['text.token_weight']
        assert table.root.bytes==spec.params['vocab']*spec.params['text_width']*(4 if spec.params['dtype']=='fp32' else 2)
        assert layouts['token_ids'].root.bytes==spec.params['text_batch']*spec.params['context']*4
        assert layouts['logit_scale'].root.bytes==4
        steps={step['name']:step for step in c['plan']['steps']}
        for op in c['program']['operators']:
            access=steps[op['name']]['access'];out=layouts[op['outputs'][0]]
            count=math.prod(out.shape);unit=4 if out.root.dtype in ('int32','fp32') else 2
            if op['op']=='lookup':
                indices=math.prod(layouts[op['inputs'][0]].shape)
                assert access['read_bytes']==indices*4+count*unit and access['write_bytes']==count*unit
                assert access['operations']==count+2*indices and access['conversion_instructions']==0
            elif op['op']=='argmax':
                batches,context=layouts[op['inputs'][0]].shape
                assert access['read_bytes']==batches*context*4 and access['write_bytes']==batches*4
                assert access['operations']==2*batches*(context-1) and access['conversion_instructions']==0
            elif op['op']=='select':
                batches=out.shape[0]
                assert access['read_bytes']==batches*4+count*unit and access['write_bytes']==count*unit
                assert access['operations']==count+2*batches and access['conversion_instructions']==0
            elif op['op']=='causal_mask':
                assert access['read_bytes']==0 and access['write_bytes']==count*4
                assert access['operations']==count and access['conversion_instructions']==0
            elif op['op'] in ('apply_mask','masked_softmax','scale_tensor'):
                assert access['read_bytes']==count*(unit+4) and access['write_bytes']==count*unit
                assert access['conversion_instructions']==(2*count if out.root.dtype!='fp32' else 0)
            elif op['op'] in ('quick_gelu','l2_norm'):
                assert access['read_bytes']==access['write_bytes']==count*unit
                assert access['conversion_instructions']==(2*count if out.root.dtype!='fp32' else 0)
        if execute:request=s.prepare_compiled(c,output_dir=tmp_path/'actual',options={'trace':True})
    assert spec.to_dict()==before
    info=dict(case=key,mode=mode,policy=policy,mapping=index,CPU_values_separate_from_component=True)
    if not execute:
        (tmp_path/'CPU-ONLY.json').write_text(json.dumps(info)+'\n');return
    with observe_memory(monkeypatch) as observed:result=s.evaluate_request(request)
    physical=audit_storage(c,result,observed)
    returned=[row for row in physical['intervals'] if row['returned']]
    for n in ('logits_per_image','logits_per_text'):
        root=layouts[n].root.name;row=next(r for r in returned if r['storage']==root)
        assert row['end']==result['evaluation']['latency_ns']
    with no_backend(monkeypatch):assert s.evaluate_request(result)==result
    (tmp_path/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    info.update(independent_root_lifetimes=physical,actual_peak=result['evaluation']['details']['stats']['memory_peak'],
        latency_ns=result['evaluation']['latency_ns'],conversion_instructions=sum(r['access']['conversion_instructions'] for r in c['plan']['steps']),
        new_native=False)
    (tmp_path/'FACTS.json').write_text(json.dumps(info,indent=2)+'\n')
