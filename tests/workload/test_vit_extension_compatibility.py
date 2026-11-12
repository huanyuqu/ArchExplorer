"""The separate FP32 unknown square still works beside a saved full ViT."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from test_operator_mapping_attention_contract import no_backend,node_spec,primitive_scene
from vit_contract_cases import REJECT,canonical,close,independent_case,vit_api
from vit_expectations import f32,store,vit_reference


def test_independent_FP32_square_after_full_ViT_preserves_two_reads_and_rejects_low_precision(vit_api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec,MappingSpec
    from ae_operator_mappings import semantics,mapping as extension_mapping
    assert 'site-packages' in semantics.__file__
    path=Path(os.environ['VAL59_SAVED_FP32_RESULT'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==os.environ['VAL59_SAVED_FP32_RESULT_SHA256']
    saved=json.loads(path.read_text());compilation=saved['request']['compilation']
    assert compilation['source']['workload']['kind']=='vit_classifier'
    w,s=vit_api;p=compilation['source']['workload']['params'];assert p['dtype']=='fp32'
    with no_backend(monkeypatch):
        assert s.evaluate_request(saved)==saved
        actual=w.execute_reference(WorkloadSpec(**compilation['source']['workload']))['outputs']['logits']
    close(actual,vit_reference(independent_case(p))['logits'],'fp32')
    # Actual CPU output becomes the separate extension's explicit root. Its
    # expected square is scalar arithmetic on that root, not a production oracle.
    values=actual.tolist();shape=actual.shape
    spec=node_spec(semantics.KIND,[shape],shape,{},[values])
    adf,spec,seed=primitive_scene(spec,'events')
    rule=w.MappingRuleBinding(extension_mapping.MATERIALIZED.id,'1').to_dict()
    selected=MappingSpec('compile_graph_v3',dict(seed.params,rules={semantics.KIND:rule},operator_rules={}))
    with no_backend(monkeypatch):
        compiled=w.compile_workload(adf,spec,selected)
        reference=w.execute_reference(WorkloadSpec('graph',compiled['program']),inputs=spec.params['meta']['inputs'])
        wanted=[[store(f32(x*x),'fp32') for x in row] for row in values]
        assert canonical(reference['outputs']['y'].reshape(-1).tolist(),'fp32')==canonical([x for row in wanted for x in row],'fp32')
        assert {op['op'] for op in compiled['program']['operators']}=={'mul'}
        assert sum(step['access']['read_bytes'] for step in compiled['plan']['steps'])==2*actual.size*4
        assert sum(step['access']['write_bytes'] for step in compiled['plan']['steps'])==actual.size*4
        request=s.prepare_compiled(compiled,output_dir=tmp_path/'unknown-square',options={'trace':True})
    result=s.evaluate_request(request)
    with no_backend(monkeypatch):
        assert s.evaluate_request(result)==result
        for dtype in ('fp16','bf16'):
            bad=deepcopy(spec.params)
            for tensor in bad['tensors']:tensor['dtype']=dtype
            with pytest.raises(REJECT):w.compile_workload(adf,WorkloadSpec('graph',bad),selected)
