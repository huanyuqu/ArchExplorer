"""Hand-derived full absorbed boundary examples, independent of production."""
from copy import deepcopy
import math

import pytest
from mla_absorbed_expectations import absorbed_decode, root_shapes


def hand_case():
    return dict(batch=1,sequence=2,width=2,query_rank=2,latent_rank=2,heads=2,
                content_dim=2,positional_dim=2,output_width=3,epsilon=5,dtype='fp32',
                inputs={'X':[2,2],'W_DQ':[1,0,0,1],'q_gamma':[3,3],
                        'latent_kv':[2,4,6,8],'pos_key':[0,0,0,0],
                        'rope_cos':[0],'rope_sin':[1],
                        'head0.WQK':[0,0,0,0],'head1.WQK':[0,0,0,0],
                        'head0.WQR':[1,0,0,0],'head1.WQR':[1,0,0,0],
                        'head0.WVO':[1,0,1,0,1,1],
                        'head1.WVO':[2,0,2,0,2,2]})


def test_hand_rms_rotary_context_and_ordered_merge():
    actual = absorbed_decode(hand_case())
    assert actual['stages']['query.down'] == [[2,2]]
    # sqrt((4+4)/2 + 5)=3; (2/3)*3 stored FP32 = 2.
    assert actual['stages']['query.norm'] == [[2,2]]
    assert actual['stages']['head0.pos_query'] == [[2,0]]
    assert actual['stages']['head0.rope'] == [[0,2]]
    assert actual['stages']['head0.softmax'] == [[.5,.5]]
    assert actual['stages']['head0.context'] == [[4,6]]
    assert actual['stages']['head0.output'] == [[4,6,10]]
    assert actual['stages']['head1.output'] == [[8,12,20]]
    assert actual['output'] == [[12,18,30]]


def test_hand_position_scores_use_declared_rotary_pair_and_content_scale():
    params = hand_case();params['inputs']['pos_key'] = [1,0,0,1]
    actual = absorbed_decode(params)['stages']
    assert actual['head0.pos_scores'] == [[0,2]]
    assert actual['head0.scale'] == [[0,1]]
    p1 = math.e/(1+math.e)
    assert actual['head0.softmax'][0] == pytest.approx([1-p1,p1],abs=8e-8)
    assert actual['head0.context'][0] == pytest.approx([2+4*p1,4+4*p1],abs=8e-7)


def test_head_sum_is_ordered_fp32_not_unordered_high_precision_sum():
    params = hand_case();params.update(heads=3)
    for h,gain in enumerate([2**25,-2**25,1]):
        params['inputs'][f'head{h}.WQK'] = [0]*4
        params['inputs'][f'head{h}.WQR'] = [0]*4
        params['inputs'][f'head{h}.WVO'] = [gain,0,0,0,0,0]
    assert absorbed_decode(params)['output'] == [[4,0,0]]
    params['inputs']['head1.WVO'],params['inputs']['head2.WVO'] = (
        params['inputs']['head2.WVO'],params['inputs']['head1.WVO'])
    assert absorbed_decode(params)['output'] == [[0,0,0]]


def test_boundary_contract_refuses_dtype_odd_position_and_precomputed_roots():
    params = hand_case()
    for dtype in ('fp16','bf16'):
        with pytest.raises(ValueError):absorbed_decode(dict(params,dtype=dtype))
    with pytest.raises(ValueError):root_shapes(dict(params,positional_dim=3))
    for bad in (dict(params['inputs'],Y=[0,0,0]),dict(params['inputs'],X=[math.inf,0])):
        with pytest.raises(ValueError):absorbed_decode(dict(params,inputs=bad))
    assert absorbed_decode(params)['output'] == [[12,18,30]]


def test_flat_dictionary_with_shaped_arrays_preserves_hand_answer():
    params = hand_case()
    params['inputs'].update(X=[[2,2]],W_DQ=[[1,0],[0,1]],
                           latent_kv=[[[2,4],[6,8]]],pos_key=[[[0,0],[0,0]]],
                           **{'head0.WQK':[[0,0],[0,0]],'head1.WQK':[[0,0],[0,0]],
                              'head0.WQR':[[1,0],[0,0]],'head1.WQR':[[1,0],[0,0]],
                              'head0.WVO':[[1,0,1],[0,1,1]],'head1.WVO':[[2,0,2],[0,2,2]]})
    assert absorbed_decode(params)['output'] == [[12,18,30]]
    bad = deepcopy(params);bad['inputs']['latent_kv'] = [[[2],[4,6,8]]]
    with pytest.raises(ValueError):absorbed_decode(bad)
