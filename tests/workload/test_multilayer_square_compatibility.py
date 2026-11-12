"""Consume the exact public FP32 square distribution in the VAL57 bundle."""
from copy import deepcopy
import pytest

from multilayer_contract_cases import composition_api,REJECT
from test_operator_mapping_attention_contract import check_values,no_backend,node_spec,primitive_scene


@pytest.mark.parametrize('mode',['events','roofline'])
def test_bundled_public_square_keeps_FP32_two_reads_and_existing_backends(
        composition_api,monkeypatch,tmp_path,mode):
    from arch_explorer.adf import MappingSpec,WorkloadSpec
    from ae_operator_mappings import semantics,mapping as extension_mapping
    assert 'site-packages' in semantics.__file__
    w,s=composition_api
    spec=node_spec(semantics.KIND,[(2,3)],(2,3),{},[[[1,-2,.5],[0,.25,-.5]]])
    adf,spec,old=primitive_scene(spec,mode)
    binding=w.MappingRuleBinding(extension_mapping.MATERIALIZED.id,'1').to_dict()
    selected=MappingSpec('compile_graph_v3',dict(old.params,
        rules={semantics.KIND:binding},operator_rules={}))
    with no_backend(monkeypatch):
        raw=w.execute_reference(spec)
        compilation=w.compile_workload(adf,spec,selected)
        mapped=w.execute_reference(WorkloadSpec('graph',compilation['program']),inputs=spec.params['meta']['inputs'])
        request=s.prepare_compiled(compilation,output_dir=tmp_path/mode,options={'trace':True})
    expected=[[1,4,.25],[0,.0625,.25]]
    check_values(raw['outputs']['y'],expected);check_values(mapped['outputs']['y'],expected)
    assert {row['op'] for row in compilation['program']['operators']}=={'mul'}
    assert sum(row['access']['read_bytes'] for row in compilation['plan']['steps'])==48
    assert sum(row['access']['write_bytes'] for row in compilation['plan']['steps'])==24
    result=s.evaluate_request(request)
    assert result['evaluation']['latency_ns']>0
    with no_backend(monkeypatch):
        assert s.evaluate_request(result)==result
        for dtype in ('fp16','bf16'):
            data=deepcopy(spec.params)
            for tensor in data['tensors']:tensor['dtype']=dtype
            with pytest.raises(REJECT):w.compile_workload(adf,WorkloadSpec('graph',data),selected)
