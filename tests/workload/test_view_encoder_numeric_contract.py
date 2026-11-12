"""Independent per-layer values for two layout policies on the identical input."""
from copy import deepcopy
import json

import pytest

from multilayer_expectations import reference_for
from test_precision_transformer_contract import close_values
from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import assert_layout,independent_layouts,scene,view_api
from view_storage_runtime_cases import audit_storage,observe_memory


@pytest.mark.parametrize('layers',[2,4])
@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('case',['regular','boundary'])
@pytest.mark.parametrize('mode',['events','roofline'])
@pytest.mark.parametrize('index',[0,1])
def test_same_encoder_inputs_per_layer_values_and_actual_policy_change(
        view_api,monkeypatch,tmp_path,layers,dtype,case,mode,index):
    from arch_explorer.adf import WorkloadSpec
    w,s=view_api
    key=f'{mode}-{layers}-{dtype}-{case}'
    adf,spec,materialized=scene(key,index=index,strategy='materialized')
    _,same_spec,views=scene(key,index=index,strategy='views')
    assert spec.to_dict()==same_spec.to_dict()
    original=deepcopy(spec.to_dict());p=spec.params
    expected=reference_for(dict(config={k:p[k] for k in ('batch','sequence','hidden','heads','mlp_hidden','epsilon')},
        dtype=dtype,X=p['inputs']['X'],instances=[dict(id='layer'+str(i),parameters=data)
        for i,data in enumerate(p['inputs']['layers'])]))
    with no_backend(monkeypatch):
        raw=w.execute_reference(spec)
    metadata={row['id']:row for row in raw['graph'].meta['composition']['instances']}
    for identifier in expected['order']:
        name=metadata[identifier]['tensor_names']['Y']
        close_values(raw['tensors'][name],expected['layers'][identifier]['output'],dtype)
    facts={}
    for policy,mapping in [('materialized',materialized),('views',views)]:
        with no_backend(monkeypatch):
            compilation=w.compile_workload(adf,spec,mapping)
            program=compilation['program'];layouts=independent_layouts(program)
            graph=w.workload_from_spec(WorkloadSpec('graph',program))
            mapped=w.execute_reference(WorkloadSpec('graph',program))
            for name,layout in layouts.items():assert_layout(w,graph.tensors,name,layout)
            request=s.prepare_compiled(compilation,output_dir=tmp_path/policy,options={'trace':True})
        assert spec.to_dict()==original
        for identifier in expected['order']:
            name=metadata[identifier]['tensor_names']['Y']
            close_values(mapped['tensors'][name],expected['layers'][identifier]['output'],dtype)
        close_values(mapped['outputs']['Y'],expected['output'],dtype)
        roots={layout.root.name:layout.root for layout in layouts.values()}
        allocations=compilation['plan']['allocations']
        assert len(allocations)==len({row['storage'] for row in allocations})
        assert {row['storage']:row['bytes'] for row in allocations}=={name:root.bytes for name,root in roots.items()}
        view_names={op['name'] for op in program['operators'] if op['op']=='storage.view'}
        for step in compilation['plan']['steps']:
            if step['name'] in view_names:
                assert step['access']['read_bytes']==step['access']['write_bytes']==0
                assert step['access'].get('conversion_instructions',0)==0
                assert step['access'].get('operations',0)==0
        with observe_memory(monkeypatch) as events:result=s.evaluate_request(request)
        physical=audit_storage(compilation,result,events)
        with no_backend(monkeypatch):assert s.evaluate_request(result)==result
        facts[policy]=dict(roots=len(roots),aliases=sum(layout.shared for layout in layouts.values()),
            view_control_tasks=len(view_names),latency_ns=result['evaluation']['latency_ns'],
            actual_memory_peak=result['evaluation']['details']['stats']['memory_peak'],
            independent_physical_storage=physical,
            declared_read_bytes=sum(step['access']['read_bytes'] for step in compilation['plan']['steps']),
            declared_write_bytes=sum(step['access']['write_bytes'] for step in compilation['plan']['steps']))
    assert facts['materialized']['view_control_tasks']==0
    assert facts['views']['view_control_tasks']>0
    assert facts['views']['roots']<facts['materialized']['roots']
    assert facts['views']['aliases']>0
    # Actual times and peaks are reported; neither policy is assumed faster.
    (tmp_path/'LAYOUT-POLICY-OBSERVATIONS.json').write_text(json.dumps(facts,indent=2)+'\n')
