"""Actual ViT component services and independently expected semantic values."""
from copy import deepcopy
import json

import pytest

from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import assert_layout,independent_layouts
from view_storage_runtime_cases import audit_storage,observe_memory
from vit_contract_cases import canonical,compare_reference,execution_rows,frozen_inputs,scene,vit_api
from vit_expectations import leaves


def case_id(row):
    key,mode,policy,index,execute=row
    return f'{key}-{mode}-{policy}-mapping{index}-'+('execute' if execute else 'CPU')


@pytest.mark.parametrize('key,mode,policy,index,execute',execution_rows(),ids=[case_id(row) for row in execution_rows()])
def test_complete_ViT_values_layouts_and_actual_physical_services(vit_api,monkeypatch,tmp_path,key,mode,policy,index,execute):
    from arch_explorer.adf import WorkloadSpec
    w,s = vit_api
    adf,spec,mapping = scene(key,mode,policy,index)
    before = deepcopy(spec.to_dict())
    dtype = spec.params['dtype']
    with no_backend(monkeypatch):
        original = w.execute_reference(spec)
        compare_reference(original,key)
        compilation = w.compile_workload(adf,spec,mapping)
        program = compilation['program']
        mapped = w.execute_reference(WorkloadSpec('graph',program))
        compare_reference(mapped,key)
        for name,value in original['tensors'].items():
            if name in mapped['tensors']:
                assert canonical(value.reshape(-1).tolist(),dtype)==canonical(mapped['tensors'][name].reshape(-1).tolist(),dtype),name
        layouts = independent_layouts(program)
        for name,layout in layouts.items(): assert_layout(w,mapped['graph'].tensors,name,layout)
        rules = compilation['source']['mapping']['params']['operator_rules']
        assert set(rules)=={op.name for op in original['graph'].operators}
        assert set(compilation['plan']['rules'])==set(rules)
        for name,binding in rules.items(): assert compilation['plan']['rules'][name]['binding']==binding
        view_steps = [step for step in compilation['plan']['steps'] if step['address_map']['indexing']['semantic']=='storage.view']
        assert bool(view_steps)==(policy=='views')
        for step in view_steps:
            assert all(step['access'][k]==0 for k in ('read_bytes','write_bytes','operations','conversion_instructions'))
        # Semantic packing, broadcasts and concatenation must remain real
        # independent outputs under either policy, never zero-copy aliases.
        for op in program['operators']:
            if op['op'] in ('patchify','concat','broadcast'):
                assert all(not layouts[name].shared for name in op['outputs'])
        steps={row['name']:row for row in compilation['plan']['steps']}
        for op in program['operators']:
            access=steps[op['name']]['access']
            if op['op'] in ('patchify','concat'):
                assert access['read_bytes']==sum(layouts[n].read_bytes for n in op['inputs'])
                assert access['write_bytes']==sum(layouts[n].read_bytes for n in op['outputs'])
                assert access['conversion_instructions']==0
            elif op['op'] in ('bias','affine_norm','gelu_exact','softmax'):
                output=layouts[op['outputs'][0]]
                reads={'bias':2,'affine_norm':3,'gelu_exact':1,'softmax':1}[op['op']]
                assert access['read_bytes']==reads*output.read_bytes
                assert access['write_bytes']==output.read_bytes
                if dtype!='fp32': assert access['conversion_instructions']>0
        if execute:
            request=s.prepare_compiled(compilation,output_dir=tmp_path/'actual',options={'trace':True})
    assert spec.to_dict()==before
    if not execute:
        (tmp_path/'CPU-ONLY.json').write_text(json.dumps({'case':key,'mode':mode,'policy':policy,'mapping':index,
             'actual_component_execution':False,'independent_numeric_comparison':True})+'\n')
        return
    with observe_memory(monkeypatch) as observed: result=s.evaluate_request(request)
    physical=audit_storage(compilation,result,observed)
    with no_backend(monkeypatch): assert s.evaluate_request(result)==result
    (tmp_path/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (tmp_path/'FACTS.json').write_text(json.dumps({'case':key,'mode':mode,'policy':policy,'mapping':index,
        'CPU_reference_separate_from_component_execution':True,'independent_root_lifetimes':physical,
        'copy_read_bytes':sum(step['access']['read_bytes'] for step in steps.values()
            if step['address_map']['indexing']['semantic'] in ('patchify','concat','broadcast','reshape','transpose','slice')),
        'conversion_instructions':sum(step['access']['conversion_instructions'] for step in steps.values()),
        'actual_memory_peak':result['evaluation']['details']['stats']['memory_peak'],
        'latency_ns':result['evaluation']['latency_ns'],'native_execution':False},indent=2)+'\n')
