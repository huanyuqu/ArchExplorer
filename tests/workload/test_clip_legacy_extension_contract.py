"""Affected rich-attrs CLIP and independent installed FP32 extension controls."""
from copy import deepcopy
import hashlib,json,os
from pathlib import Path
import pytest
from clip_contract_cases import BLOCK,REJECT,canonical,clip_api,close,expected,frozen_inputs,independent_case,scene
from clip_expectations import clip_reference
from test_operator_mapping_attention_contract import no_backend,node_spec,primitive_scene
from test_vit_user_consumers import fresh_readonly
from precision_transformer_expectations import f32,store

@pytest.mark.parametrize('dtype',['fp32','fp16'])
def test_old_rich_attrs_CLIP_keeps_same_root_equations_and_critical_stages(clip_api,monkeypatch,dtype):
    from arch_explorer.adf import WorkloadSpec
    w,s=clip_api;p=deepcopy(frozen_inputs()['cases']['regular-2-'+dtype]['workload']['params']);values=p['inputs']
    config={k:p[k] for k in ('image_size','patch','width','layers','heads','context','vocab','text_width','text_heads','text_layers','projection')}
    config['mlp']=p['mlp_hidden']
    old=WorkloadSpec('clip',dict(profile='synthetic',dtype=dtype,image_batch=p['image_batch'],text_batch=p['text_batch'],config=config))
    root_names={'images':'images','token_ids':'token_ids','logit_scale':'logit_scale',
        'vision.patch_projection.weight':'vision.patch_weight','vision.cls_weight':'vision.cls_weight',
        'vision.position_weight':'vision.position_weight','vision.pre_norm.gamma':'vision.pre_gamma','vision.pre_norm.beta':'vision.pre_beta',
        'vision.post_norm.gamma':'vision.post_gamma','vision.post_norm.beta':'vision.post_beta','vision.projection.weight':'vision.projection_weight',
        'text.token_weight':'text.token_weight','text.position_weight':'text.position_weight',
        'text.final_norm.gamma':'text.norm_gamma','text.final_norm.beta':'text.norm_beta','text.projection.weight':'text.projection_weight'}
    for tower,count in [('vision',p['layers']),('text',p['text_layers'])]:
        for i in range(count):
            base=f'{tower}.block{i}.'
            for local,public in [('q.weight','Wq'),('k.weight','Wk'),('v.weight','Wv'),('out.weight','Wo'),
                ('q.bias_weight','bq'),('k.bias_weight','bk'),('v.bias_weight','bv'),('out.bias_weight','bo'),
                ('up.weight','W1'),('up.bias_weight','b1'),('down.weight','W2'),('down.bias_weight','b2'),
                ('ln1.gamma','norm1_gamma'),('ln1.beta','norm1_beta'),('ln2.gamma','norm2_gamma'),('ln2.beta','norm2_beta')]:
                root_names[base+local]=base+public
    with no_backend(monkeypatch):
        graph=w.workload_from_spec(old);produced={n for o in graph.operators for n in o.outputs}
        assert set(graph.tensors)-produced==set(root_names)
        legacy=w.execute_reference(old,inputs={n:deepcopy(values[p]) for n,p in root_names.items()})
        modern=w.execute_reference(w.clip_similarity(**p))
        with pytest.raises(ValueError):w.workload_from_spec(WorkloadSpec('clip',dict(old.params,dtype='bf16')))
    stages={'vision.patchify':'vision.patches','vision.pre_norm':'vision.pre_norm','vision.select_cls':'vision.CLS_select',
        'vision.post_norm':'vision.post_norm','text.lookup':'text.lookup','text.position':'text.position',
        'text.eot_index':'text.eot_index','text.select_eot':'text.EOT_select',
        'image_features':'image_features','text_features':'text_features','image_scaled':'image_scaled','logit_scale.exp':'logit_scale.exp',
        'logits_per_image':'logits_per_image','logits_per_text':'logits_per_text'}
    oracle=clip_reference(independent_case(p))
    for n,stage in stages.items():close(legacy['tensors'][n],oracle['stages'][stage],'int32' if n=='text.eot_index' else 'fp32' if n=='logit_scale.exp' else dtype)
    for tower,count in [('vision',p['layers']),('text',p['text_layers'])]:
        for i in range(count):
            close(legacy['tensors'][f'{tower}.block{i}.residual2'],oracle[f'{tower}_layers'][i],dtype)
    for n in ('logits_per_image','logits_per_text'):close(legacy['tensors'][n],modern['outputs'][n].tolist(),dtype)

def test_old_saved_ViT_identity_is_rejected_while_current_CLIP_control_is_valid(clip_api,monkeypatch):
    w,s=clip_api;p=Path(os.environ['VAL60_HISTORICAL_RESULT']);raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL60_HISTORICAL_RESULT_SHA256']
    old=json.loads(raw);adf,spec,mapping=scene('regular-2-fp32')
    with no_backend(monkeypatch):
        current=w.compile_workload(adf,spec,mapping);assert w.validate_compilation(current)==current
        with pytest.raises(REJECT):s.evaluate_request(old)

def test_unknown_FP32_square_after_saved_rectangular_CLIP_stays_an_independent_extension(clip_api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec,MappingSpec
    from ae_operator_mappings import semantics,mapping as extension
    assert 'site-packages' in semantics.__file__
    w,s=clip_api;p=Path(os.environ['VAL60_SAVED_FP32_RESULT']);raw=p.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL60_SAVED_FP32_RESULT_SHA256']
    saved=json.loads(raw);c=saved['request']['compilation'];params=c['source']['workload']['params']
    assert c['source']['workload']['kind']=='clip_similarity' and params['dtype']=='fp32'
    with no_backend(monkeypatch):
        assert s.evaluate_request(saved)==saved
        actual=w.execute_reference(WorkloadSpec(**c['source']['workload']))['outputs']['logits_per_image']
    close(actual,clip_reference(independent_case(params))['logits_per_image'],'fp32')
    values=actual.tolist();spec=node_spec(semantics.KIND,[actual.shape],actual.shape,{},[values]);adf,spec,seed=primitive_scene(spec,'events')
    rule=w.MappingRuleBinding(extension.MATERIALIZED.id,'1').to_dict();mapping=MappingSpec('compile_graph_v3',dict(seed.params,rules={semantics.KIND:rule},operator_rules={}))
    with no_backend(monkeypatch):
        c=w.compile_workload(adf,spec,mapping);ref=w.execute_reference(WorkloadSpec('graph',c['program']))
        wanted=[[store(f32(v*v),'fp32') for v in row] for row in values]
        close(ref['outputs']['y'],wanted,'fp32',exact=True)
        assert {o['op'] for o in c['program']['operators']}=={'mul'}
        assert sum(x['access']['read_bytes'] for x in c['plan']['steps'])==actual.size*8
        assert sum(x['access']['write_bytes'] for x in c['plan']['steps'])==actual.size*4
        rq=s.prepare_compiled(c,output_dir=tmp_path/'unknown',options={'trace':True})
    result=s.evaluate_request(rq);path=tmp_path/'UNKNOWN-RESULT.json';path.write_text(json.dumps(result,indent=2)+'\n')
    fresh_readonly(path,tmp_path,'compiled')
    with no_backend(monkeypatch):
        for dtype in ('fp16','bf16'):
            bad=deepcopy(spec.params)
            for tensor in bad['tensors']:tensor['dtype']=dtype
            with pytest.raises(REJECT):w.compile_workload(adf,WorkloadSpec('graph',bad),mapping)
