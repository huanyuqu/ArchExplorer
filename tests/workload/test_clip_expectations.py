"""Analytic VAL60 preparation controls, independent of all production APIs."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, localcontext
import math

import pytest

from clip_expectations import (
    EPSILON,Geometry,causal_mask,clip_reference,eot_addresses,eot_traffic,first_argmax,l2_norm,lookup_addresses,
    lookup_traffic,mask_traffic,parameter_shapes,prefix_softmax,quick_gelu,similarity_addresses,
    synthetic_case,token_ids,
)
from precision_transformer_expectations import dot32,f32,gelu,layer_norm,lifetime_peak,store,tensor_bytes
from vit_expectations import leaves,rounded,shaped,tensor_shape
from view_storage_expectations import Root,canonical_scalar_bytes,root_map


def test_lookup_repeated_ids_have_repeated_reads_but_one_complete_weight_root():
    ids=[[0,4,4],[3,0,4]]
    assert lookup_addresses(ids,3)==[[0,1,2],[12,13,14],[12,13,14],[9,10,11],[0,1,2],[12,13,14]]
    assert eot_addresses(ids,3)==[[3,4,5],[15,16,17]]
    assert [first_argmax(row) for row in ids]==[1,2]
    for dtype,unit in [('fp32',4),('fp16',2),('bf16',2)]:
        count=lookup_traffic(ids,vocab=5,width=3,dtype=dtype)
        assert count['index_bytes']==24
        assert count['selected_weight_bytes']==count['output_bytes']==18*unit
        assert count['root_weight_bytes']==15*unit
        assert count['integer_conversion_instructions']==0


@pytest.mark.parametrize('ids,indices',[([[10,1,10,0],[0,3,2,10],[0,0,0,0]],[0,3,0]),
    ([[4,4,4],[3,1,0],[0,1,2]],[0,0,2])])
def test_EOT_is_first_argmax_and_not_last_token_or_last_occurrence(ids,indices):
    assert [first_argmax(row) for row in ids]==indices


@pytest.mark.parametrize('bad',[True,False,'2',1.0,1.5,-1,5,2**31])
def test_integer_indices_never_truncate_cast_or_accept_invalid_values(bad):
    with pytest.raises(ValueError):token_ids([[bad,0]],batch=1,context=2,vocab=5)


def test_integer_shape_and_range_controls_do_not_allocate_an_embedding_table():
    assert token_ids([[0,2**31-1]],batch=1,context=2,vocab=2**31)==[[0,2**31-1]]
    for value in ([[0]],[[0,1],[2]],[[0,1,2]],[],[0,1]):
        with pytest.raises(ValueError):token_ids(value,batch=1,context=2,vocab=5)


def test_independent_int32_address_map_and_canonical_bytes_do_not_use_float_rounding():
    root=Root('ids',(2,3),'int32')
    assert root.bytes==24 and root_map(root).byte_addresses()==[0,4,8,12,16,20]
    assert canonical_scalar_bytes(-2**31,'int32')==bytes.fromhex('00000080')
    assert canonical_scalar_bytes(2**31-1,'int32')==bytes.fromhex('ffffff7f')
    assert canonical_scalar_bytes(0,'int32')==bytes(4)
    assert canonical_scalar_bytes(1.,'fp32')==bytes.fromhex('0000803f')
    assert canonical_scalar_bytes(1.,'fp16')==bytes.fromhex('003c')
    assert canonical_scalar_bytes(1.,'bf16')==bytes.fromhex('803f')
    for bad in (True,1.,'1',2**31,-2**31-1):
        with pytest.raises(ValueError):canonical_scalar_bytes(bad,'int32')


@pytest.mark.parametrize('dtype,unit',[('fp32',4),('fp16',2),('bf16',2)])
def test_hand_dynamic_EOT_and_mask_traffic_keeps_integer_and_score_widths_separate(dtype,unit):
    ids=[[0,4,4],[3,0,4]]
    assert eot_traffic(ids,width=3,dtype=dtype)==dict(argmax_reads=24,argmax_writes=8,
        selection_indices=8,selection_reads=6*unit,selection_writes=6*unit,feature_root_bytes=18*unit)
    assert causal_mask(3)==[[True,False,False],[True,True,False],[True,True,True]]
    assert mask_traffic(3,4,dtype=dtype,mask_item_bytes=4)==dict(constructed_mask_writes=36,
        applied_mask_reads=144,score_reads=36*unit,score_writes=36*unit)


def test_hand_traffic_never_guesses_unsupported_dtype_or_mask_encoding():
    with pytest.raises(ValueError):lookup_traffic([[0]],vocab=1,width=1,dtype='fp8')
    with pytest.raises(ValueError):eot_traffic([[0]],width=1,dtype='int32')
    with pytest.raises(ValueError):mask_traffic(2,2,dtype='bf16',mask_item_bytes=0)


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_causal_softmax_has_effective_diagonal_and_future_zero(dtype):
    assert prefix_softmax([3.,1.,10000.],0,dtype)==[1.,0.,0.]
    a=prefix_softmax([0.,0.,10000.],1,dtype)
    assert a==[.5,.5,0.]
    assert prefix_softmax([0.,0.,-10000.],1,dtype)==a
    assert prefix_softmax([10000.,9999.,9998.],2,dtype)[0]>prefix_softmax([10000.,9999.,9998.],2,dtype)[1]


def test_QuickGELU_has_an_independent_decimal_value_and_differs_from_erf_tanh():
    with localcontext() as context:
        context.prec=60
        exact=Decimal(2)/(1+(-Decimal('3.404')).exp())
    assert abs(quick_gelu(2)-float(exact))<2e-7
    assert abs(quick_gelu(2)-gelu(2,'erf'))>.015
    assert abs(quick_gelu(2)-gelu(2,'tanh'))>.015
    assert quick_gelu(0.)==0.
    assert abs((quick_gelu(2)-quick_gelu(-2))-2.)<3e-7
    assert EPSILON==1e-5
    assert layer_norm([[3.,3.]],[2.,-2.],[.25,-.5],EPSILON,'fp32')==[[.25,-.5]]


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_L2_has_no_implicit_epsilon_and_stores_the_output(dtype):
    assert l2_norm([3.,4.],dtype)==[store(.6,dtype),store(.8,dtype)]
    small=1e-4 if dtype=='fp16' else 1e-10
    actual=l2_norm(rounded([3*small,4*small],dtype),dtype)
    assert abs(actual[0]-.6)<.004 and abs(actual[1]-.8)<.004
    with pytest.raises(ValueError):l2_norm([0.,0.],dtype)
    with pytest.raises(ValueError):l2_norm([1e30,1e30],dtype)


@pytest.mark.parametrize('dtype',['fp16','bf16'])
def test_FP32_scale_cannot_move_past_the_stored_image_feature_boundary(dtype):
    differences=[]
    for image in ([.2,.9],[.6,.8],[.7,-.3]):
        for text in ([.8,.6],[.3,-.9],[.1,.7]):
            left,right=rounded(image,dtype),rounded(text,dtype)
            for raw in (.25,.5,1.):
                scale=f32(math.exp(f32(raw)))
                correct=store(dot32([store(f32(x*scale),dtype) for x in left],right,fused=True),dtype)
                moved=store(f32(dot32(left,right,fused=True)*scale),dtype)
                if correct!=moved:differences.append((image,text,raw,correct,moved))
    assert differences


def test_rectangular_similarity_transpose_uses_one_root_with_distinct_read_order():
    address=similarity_addresses(2,3)
    assert address=={'image':[0,1,2,3,4,5],'text':[0,3,1,4,2,5],'root_elements':6}
    root=[2.,4.,8.,16.,32.,64.]
    assert shaped((root[i] for i in address['text']),(3,2))==[[2.,16.],[4.,32.],[8.,64.]]


@pytest.mark.parametrize('dtype,unit',[('fp32',4),('fp16',2),('bf16',2)])
def test_hand_mixed_integer_storage_and_both_returned_logits_keep_the_root(dtype,unit):
    # Analytic intervals, not runtime observations or a prediction of CLIP's
    # complete actual peak. Release-before-allocation ties reuse the existing
    # independent interval sweep; no duplicate peak implementation is added.
    tasks={'argmax':{'end':2.},'select':{'end':5.},'similarity':{'end':8.},'transpose':{'end':9.}}
    roots=[
        dict(name='ids',shape=(2,3),dtype='int32',allocation=0.,readers=['argmax'],persistent=True),
        dict(name='table',shape=(5,3),dtype=dtype,allocation=0.,readers=[],persistent=True),
        dict(name='text_norm',shape=(2,3,3),dtype=dtype,allocation=1.,readers=['select']),
        dict(name='eot',shape=(2,),dtype='int32',allocation=2.,readers=['select']),
        dict(name='image_features',shape=(2,3),dtype=dtype,allocation=3.,readers=['similarity']),
        dict(name='text_features',shape=(3,3),dtype=dtype,allocation=5.,readers=['similarity']),
        dict(name='logits',shape=(2,3),dtype=dtype,allocation=8.,readers=['transpose'],returned=True),
    ]
    peak,intervals=lifetime_peak(roots,tasks,10.)
    assert peak==32+39*unit
    by_name={row['name']:row for row in intervals}
    assert by_name['ids']['bytes']==24 and by_name['eot']['bytes']==8
    assert by_name['table']['bytes']==15*unit and by_name['table']['release']==10.
    assert by_name['logits']['bytes']==6*unit and by_name['logits']['last_reader']==9. and by_name['logits']['release']==10.
    copied=roots+[dict(name='reverse',shape=(3,2),dtype=dtype,allocation=9.,readers=[],returned=True)]
    copied_peak,copied_intervals=lifetime_peak(copied,tasks,10.)
    assert copied_peak==peak  # another live interval dominated the peak
    assert copied_intervals[-1]['bytes']==6*unit
    assert tensor_bytes((2,3),'int32')==24
    with pytest.raises(ValueError):tensor_bytes((2,3),'fp8')


@pytest.mark.parametrize('layers,dtype',[(2,'fp32'),(4,'fp32'),(2,'fp16'),(4,'fp16'),(2,'bf16'),(4,'bf16')])
def test_complete_zero_block_towers_have_known_rectangular_logits(layers,dtype):
    g=Geometry(image_batch=2,text_batch=3,vision_layers=layers,text_layers=layers)
    data=synthetic_case(g,dtype)
    def zero(value):return [zero(x) for x in value] if isinstance(value,list) else 0.
    roots=data['roots'];roots['images']=zero(roots['images']);roots['logit_scale']=[0.]
    for tower in ('vision','text'):
        roots[tower]={k:([{n:zero(x) for n,x in p.items()} for p in v] if k=='blocks' else zero(v)) for k,v in roots[tower].items()}
    roots['vision']['post_beta']=[3.,4.,0.,0.]
    roots['text']['final_beta']=[0.,3.,4.,0.]
    roots['vision']['projection']=roots['text']['projection']=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]]
    result=clip_reference(data)
    wanted=store(dot32([store(.6,dtype),store(.8,dtype),0.],[0.,store(.6,dtype),store(.8,dtype)],fused=True),dtype)
    assert result['logits_per_image']==[[wanted]*3 for _ in range(2)]
    assert result['logits_per_text']==[[wanted]*2 for _ in range(3)]
    assert result['stages']['logit_scale.exp']==[1.]
    assert not any(leaves(result['vision_layers'])) and not any(leaves(result['text_layers']))


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_causal_prefix_is_invariant_while_EOT_selection_can_change(dtype):
    data=synthetic_case(Geometry(image_batch=1,text_batch=1,vision_width=2,vision_heads=1,
        vision_mlp=3,text_width=2,text_heads=1),dtype)
    data['roots']['token_ids']=[[5,2,1,0]]
    changed=deepcopy(data);changed['roots']['token_ids'][0][2]=10
    before,after=clip_reference(data),clip_reference(changed)
    assert before['eot_indices']==[0] and after['eot_indices']==[2]
    for a,b in zip(before['text_layers'],after['text_layers']):assert a[0][:2]==b[0][:2]
    for index in range(2):
        a,b=(r['stages'][f'text.layer{index}.softmax'] for r in (before,after))
        for head_a,head_b in zip(a,b):assert head_a[:2]==head_b[:2]
    assert before['stages']['text.EOT_select']!=after['stages']['text.EOT_select']


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_dense_root_only_rounding_cases_exercise_both_towers_and_scale(dtype):
    data=synthetic_case(Geometry(image_batch=1,text_batch=2,vision_width=2,vision_heads=1,
        vision_mlp=3,text_width=2,text_heads=1,vision_layers=2,text_layers=2),dtype,'boundary')
    saved=deepcopy(data);out=clip_reference(data)
    assert data==saved
    assert tensor_shape(out['logits_per_image'])==(1,2) and tensor_shape(out['logits_per_text'])==(2,1)
    assert all(math.isfinite(v) and store(v,dtype)==v for v in leaves(out['logits_per_image']))
    assert out['vision_layers'][0]!=out['vision_layers'][1] and out['text_layers'][0]!=out['text_layers'][1]
    assert out['stages']['vision.patch_projection']!=out['stages']['vision.position']
    assert store(out['stages']['logit_scale.exp'][0],'fp32')==out['stages']['logit_scale.exp'][0]
    assert set(data['roots'])=={'images','token_ids','vision','text','logit_scale'}


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_dense_four_layer_rectangular_towers_preserve_FP32_logit_scale(dtype):
    data=synthetic_case(Geometry(vision_layers=4,text_layers=4),dtype)
    data['roots']['logit_scale']=[.2501]
    before=deepcopy(data);out=clip_reference(data)
    assert data==before and len(out['vision_layers'])==len(out['text_layers'])==4
    assert tensor_shape(out['logits_per_image'])==(2,3)
    assert out['logits_per_text']==[list(row) for row in zip(*out['logits_per_image'])]
    assert out['stages']['logit_scale.exp']==[f32(math.exp(f32(.2501)))]
    if dtype!='fp32':assert out['stages']['logit_scale.exp']!=[f32(math.exp(store(.2501,dtype)))]
    assert out['vision_layers'][2]!=out['vision_layers'][3]
    assert out['text_layers'][2]!=out['text_layers'][3]
    for feature in out['stages']['image_features']+out['stages']['text_features']:
        assert abs(sum(x*x for x in feature)-1.)<.02


@pytest.mark.parametrize('failure',['missing','extra','shape','layer','nonfinite','scale-overflow','zero-features'])
def test_pure_reference_rejects_bad_QA_roots_without_a_production_fallback(failure):
    data=synthetic_case(Geometry(image_batch=1,text_batch=1,vision_width=2,vision_heads=1,
        vision_mlp=3,text_width=2,text_heads=1))
    roots=data['roots']
    if failure=='missing':del roots['vision']['projection']
    elif failure=='extra':roots['precomputed_features']=[[1.,0.]]
    elif failure=='shape':roots['images']=[0.]
    elif failure=='layer':roots['text']['blocks'][0]['q_weight']=[[0.]]
    elif failure=='nonfinite':roots['text']['token_weight'][0][0]=math.inf
    elif failure=='scale-overflow':roots['logit_scale']=[1000.]
    else:roots['vision']['projection']=[[0.]*3 for _ in range(2)]
    with pytest.raises(ValueError):clip_reference(data)
