"""Frozen public CLIP root, integer, causal and old activation boundaries."""
from copy import deepcopy
import importlib,sys
import pytest
from clip_contract_cases import KEYS,REJECT,canonical,clip_api,compare_reference,frozen_inputs,scene,shapes
from test_operator_mapping_attention_contract import no_backend

@pytest.mark.parametrize('key',KEYS)
def test_explicit_builders_have_complete_roots_mixed_dtypes_and_no_hidden_answers(clip_api,monkeypatch,key):
    w,s=clip_api;p=deepcopy(frozen_inputs()['cases'][key]['workload']['params']);before=deepcopy(p)
    config={k:v for k,v in p.items() if k not in ('inputs','dtype','name')}
    with no_backend(monkeypatch):
        assert w.clip_input_shapes(**config)==shapes(p)
        spec=w.clip_similarity(**p);graph=w.workload_from_spec(spec)
        assert spec.kind=='clip_similarity' and spec.params==p
        structure=w.clip_structure(**{k:v for k,v in p.items() if k!='inputs'})
        assert structure.kind=='clip_structure' and 'inputs' not in structure.params
        compare_reference(w.execute_reference(structure,inputs=p['inputs']),key)
    assert p==before
    assert graph.meta['clip']['tensors']==frozen_inputs()['cases'][key]['names']
    assert graph.meta['composition']==frozen_inputs()['cases'][key]['composition']
    assert graph.tensors['token_ids'].dtype=='int32' and graph.tensors['logit_scale'].dtype=='fp32'
    assert all(op.attrs['epsilon']==1e-5 for op in graph.operators if op.op=='affine_norm')
    assert sum(op.op=='quick_gelu' for op in graph.operators)==p['layers']+p['text_layers']
    assert not any(op.op in ('gelu_exact','gelu_tanh') for op in graph.operators)

def test_structure_construction_does_not_compute_activations(clip_api,monkeypatch):
    w,s=clip_api;p=deepcopy(frozen_inputs()['cases']['regular-2-fp32']['workload']['params'])
    def forbidden(*a,**k):raise AssertionError('builder computed an activation')
    with no_backend(monkeypatch),monkeypatch.context() as patch:
        for module,name in [('precision','matmul_fp32'),('full_primitives','reference'),('reference_api','execute_reference')]:
            patch.setattr(importlib.import_module('arch_explorer.workload.'+module),name,forbidden)
        assert set(w.workload_from_spec(w.clip_similarity(**p)).meta['outputs'])=={'logits_per_image','logits_per_text'}

@pytest.mark.parametrize('failure',['missing','extra','image-shape','layer-shape','token-shape','bool','string','fraction','float-integer',
    'negative','out-of-vocab','int32-overflow','nonfinite','float-overflow','head-width','patch','zero','dtype'])
def test_invalid_roots_and_integer_values_are_rejected_before_any_execution(clip_api,monkeypatch,failure):
    w,s=clip_api;p=deepcopy(frozen_inputs()['cases']['regular-2-fp16']['workload']['params'])
    with no_backend(monkeypatch):
        assert w.workload_from_spec(w.clip_similarity(**p))
        if failure=='missing':del p['inputs']['text.block1.Wq']
        elif failure=='extra':p['inputs']['precomputed_EOT']=[0]
        elif failure=='image-shape':p['inputs']['images']=[0.]
        elif failure=='layer-shape':p['inputs']['vision.block1.W1']=[[0.]]
        elif failure=='token-shape':p['inputs']['token_ids']=[[0]]
        elif failure in ('bool','string','fraction','float-integer','negative','out-of-vocab','int32-overflow'):
            p['inputs']['token_ids'][0][0]={'bool':True,'string':'1','fraction':1.5,'float-integer':1.,'negative':-1,
                'out-of-vocab':p['vocab'],'int32-overflow':2**31}[failure]
        elif failure=='nonfinite':p['inputs']['logit_scale']=[float('inf')]
        elif failure=='float-overflow':p['inputs']['vision.patch_weight'][0][0]=1e20
        elif failure=='head-width':p['text_heads']=3
        elif failure=='patch':p['patch']=3
        elif failure=='zero':p['projection']=0
        else:p['dtype']='fp8'
        with pytest.raises(REJECT):w.workload_from_spec(w.clip_similarity(**p))

@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_actual_public_reference_causal_prefix_and_EOT_selection_are_distinct(clip_api,monkeypatch,dtype):
    w,s=clip_api;p=deepcopy(frozen_inputs()['cases']['regular-2-'+dtype]['workload']['params'])
    p['inputs']['token_ids'][0]=[5,2,1,0];q=deepcopy(p);q['inputs']['token_ids'][0][2]=p['vocab']-1
    with no_backend(monkeypatch):a=w.execute_reference(w.clip_similarity(**p));b=w.execute_reference(w.clip_similarity(**q))
    n=a['graph'].meta['clip']['tensors']
    assert a['tensors'][n['eot_index']].tolist()[0]==0 and b['tensors'][n['eot_index']].tolist()[0]==2
    for tensor in n['text_blocks']:
        assert canonical(a['tensors'][tensor][0,:2].reshape(-1).tolist(),dtype)==canonical(b['tensors'][tensor][0,:2].reshape(-1).tolist(),dtype)
    for tensor in n['text_probabilities']:
        assert canonical(a['tensors'][tensor][:p['text_heads'],:2].reshape(-1).tolist(),dtype)==canonical(b['tensors'][tensor][:p['text_heads'],:2].reshape(-1).tolist(),dtype)

@pytest.mark.parametrize('failure',['unknown-selector','unknown-rule'])
def test_instance_mapping_refusals_have_a_passing_control(clip_api,monkeypatch,failure):
    from arch_explorer.adf import MappingSpec
    w,s=clip_api;adf,spec,mapping=scene('regular-2-fp32')
    with no_backend(monkeypatch):
        assert w.compile_workload(adf,spec,mapping)
        params=deepcopy(mapping.params)
        if failure=='unknown-selector':params['operator_rules']['absent-operator']={'id':'matmul.tiled','version':'1','options':{}}
        else:next(iter(params['operator_rules'].values()))['id']='missing.rule'
        with pytest.raises(REJECT):w.compile_workload(adf,spec,MappingSpec('compile_graph_v3',params))

def test_new_CLIP_native_refuses_before_emitter_and_affected_old_activations_remain(clip_api,monkeypatch):
    from arch_explorer.adf import MappingSpec
    from arch_explorer.workload.adapters import materialized_native as native
    from adapter_boundary_cases import native_scene
    from test_vit_public_contract import test_shared_board_extraction_preserves_old_encoder_tanh_entry
    w,s=clip_api;adf,_,_,seed=native_scene();_,spec,_=scene('regular-2-fp32',policy='materialized')
    seen=[];previous=sys.getprofile()
    def observer(frame,event,arg):
        if event=='call' and frame.f_code is native._emitter.__code__:seen.append(1)
    try:
        sys.setprofile(observer)
        with no_backend(monkeypatch),pytest.raises(REJECT):w.compile_workload(adf,spec,MappingSpec('compile_graph_v3',dict(seed.params,operator_rules={})))
    finally:sys.setprofile(previous)
    assert not seen
    with no_backend(monkeypatch):
        vg=w.workload_from_spec(w.vit_classifier_structure())
        assert any(o.op=='gelu_exact' for o in vg.operators)
        assert all(o.attrs['epsilon']==1e-12 for o in vg.operators if o.op=='affine_norm')
    test_shared_board_extraction_preserves_old_encoder_tanh_entry(clip_api,monkeypatch)
