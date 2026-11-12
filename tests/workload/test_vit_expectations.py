"""Analytic checks of VAL59 preparation; no production pipeline is executed."""
from copy import deepcopy
import math

import pytest

from precision_transformer_expectations import dot32, f32, gelu, layer_norm, lifetime_peak, store
from vit_expectations import (
    EPSILON, Geometry, cls_addresses, leaves, parameter_shapes, patch_addresses,
    patchify, prefix_traffic, shaped, stable_softmax, synthetic_case, vit_reference,
)


def test_patch_order_is_patch_row_col_then_channel_local_row_col():
    g = Geometry(batch=2)
    images = [[[[1000*b + 100*c + 4*r + col for col in range(4)] for r in range(4)]
               for c in range(3)] for b in range(2)]
    result = patchify(images, g)
    assert result[0] == [
        [0,1,4,5,100,101,104,105,200,201,204,205],
        [2,3,6,7,102,103,106,107,202,203,206,207],
        [8,9,12,13,108,109,112,113,208,209,212,213],
        [10,11,14,15,110,111,114,115,210,211,214,215],
    ]
    assert result[1] == [[1000+x for x in row] for row in result[0]]
    assert sorted(leaves(patch_addresses(g))) == list(range(96))


@pytest.mark.parametrize('dtype,unit', [('fp32',4),('fp16',2),('bf16',2)])
def test_CLS_slice_keeps_batch_stride_and_the_complete_storage_root(dtype, unit):
    g = Geometry(batch=2, width=4)
    address = cls_addresses(g, dtype)
    assert address.addresses() == [0,1,2,3,20,21,22,23]
    assert address.read_bytes == 8*unit
    assert address.enclosing_span_bytes == 24*unit
    assert address.root.bytes == 40*unit
    facts = prefix_traffic(g, dtype)
    assert facts['packing'] == {'reads':96*unit,'writes':96*unit}
    assert facts['CLS_concat'] == {'reads':40*unit,'writes':40*unit}
    assert facts['CLS_select']['retained_root_bytes_if_exported'] == 40*unit


@pytest.mark.parametrize('dtype,unit', [('fp32',4),('fp16',2),('bf16',2)])
def test_hand_CLS_two_reader_intervals_hold_one_root_and_export_changes_peak(dtype,unit):
    # This is an analytic table for the future actual allocator audit. Times
    # are hand inputs, not an engine result; aliases never add allocations.
    g = Geometry(batch=2)
    tasks = {'head': {'end':5.}, 'extra-reader': {'end':7.}, 'late': {'end':8.}}
    roots = [
        dict(name='final_norm',shape=(2,5,4),dtype=dtype,allocation=2.,readers=['head','extra-reader']),
        dict(name='logits',shape=(2,3),dtype=dtype,allocation=5.,readers=[],returned=True),
        dict(name='head_weight',shape=(4,3),dtype=dtype,allocation=0.,readers=['head'],persistent=True),
        dict(name='late',shape=(16,),dtype=dtype,allocation=7.,readers=['late']),
    ]
    peak, intervals = lifetime_peak(roots,tasks,9.)
    assert peak == 58*unit  # one 40-element root + 6 logits + 12 resident weights
    assert intervals[0]['release'] == 7. and intervals[0]['bytes'] == cls_addresses(g,dtype).root.bytes
    roots[0]['returned'] = True
    exported_peak, exported = lifetime_peak(roots,tasks,9.)
    assert exported_peak == 74*unit
    assert exported[0]['release'] == 9.


def test_exact_erf_GELU_is_discriminated_from_tanh_and_other_normalization_epsilon():
    assert gelu(2, 'erf') == pytest.approx(1.9544997361036416, abs=2e-7)
    assert abs(gelu(2, 'erf') - gelu(2, 'tanh')) > 9e-5
    assert EPSILON == 1e-12
    a = layer_norm([[1.,1.000001]],[1.,1.],[0.,0.],EPSILON,'fp32')[0]
    b = layer_norm([[1.,1.000001]],[1.,1.],[0.,0.],1e-5,'fp32')[0]
    assert max(abs(x-y) for x,y in zip(a,b)) > .3
    assert layer_norm([[3.,3.]],[2.,-2.],[.25,-.5],EPSILON,'fp32') == [[.25,-.5]]


@pytest.mark.parametrize('dtype', ['fp16','bf16'])
def test_separate_matmul_bias_and_stored_scale_have_observable_boundaries(dtype):
    precision = 11 if dtype == 'fp16' else 8
    half = 2.**-precision
    # Fusing the bias into the matmul would incorrectly retain a lost half ULP.
    matmul = store(dot32([1,half],[1,1],fused=True),dtype)
    separate = store(f32(matmul + half),dtype)
    fused = store(f32(dot32([1,half],[1,1],fused=True) + half),dtype)
    assert separate == 1 and fused == 1 + 2**(1-precision)
    scores = [store(x,dtype) for x in [1.1,-.7,2.3]]
    scale = f32(1/math.sqrt(2))
    stored = [store(f32(x*scale),dtype) for x in scores]
    bypassed = [f32(x*scale) for x in scores]
    assert stored != bypassed


@pytest.mark.parametrize('dtype', ['fp32','fp16','bf16'])
def test_softmax_stability_and_one_semantic_boundary(dtype):
    assert stable_softmax([1000,1000],dtype) == [.5,.5]
    assert stable_softmax([1000,-1000],dtype) == [1.,0.]
    assert stable_softmax([0,0,0],dtype) == [store(1/3,dtype)]*3
    assert all(math.isfinite(x) for x in stable_softmax([10000,9999,9998],dtype))


@pytest.mark.parametrize('layers,dtype', [(2,'fp32'),(4,'fp32'),(2,'fp16'),(4,'bf16')])
def test_complete_zero_weight_model_has_closed_form_logits_equal_to_head_bias(layers,dtype):
    data = synthetic_case(Geometry(batch=1,layers=layers),dtype)
    roots = data['roots']
    def zero(value):
        return [zero(x) for x in value] if isinstance(value,list) else 0.
    for key in roots:
        if key != 'blocks':
            roots[key] = zero(roots[key])
    roots['blocks'] = [{key:zero(value) for key,value in block.items()} for block in roots['blocks']]
    roots['head_bias'] = [.25,-.5,1.]
    result = vit_reference(data)
    assert result['logits'] == [[.25,-.5,1.]]
    assert all(x==0 for x in leaves(result['layers']))
    assert all(x==0 for x in leaves(result['stages']['final_norm']))


def test_dense_root_only_data_exercises_patch_bias_positions_distinct_layers_and_head():
    data = synthetic_case(Geometry(batch=2,width=4,mlp=7,layers=2),'fp32')
    before = deepcopy(data)
    actual = vit_reference(data)
    assert data == before
    assert len(actual['layers'])==2 and actual['layers'][0] != actual['layers'][1]
    assert actual['logits'][0] != actual['logits'][1]
    assert any(x!=0 for x in leaves(actual['logits']))
    assert actual['stages']['layer0.q.bias'] != actual['stages']['layer0.k.bias']
    assert actual['stages']['patch_projection.mm'] != actual['stages']['patch_projection.bias']
    assert actual['stages']['CLS_concat'] != actual['stages']['position']
    assert set(data['roots']) == set(parameter_shapes(Geometry())[0]) | {'images','blocks'}
    # Changing a later-layer parameter must propagate to final logits.
    changed = deepcopy(data)
    changed['roots']['blocks'][1]['down_bias'][0] += .25
    assert vit_reference(changed)['logits'] != actual['logits']


@pytest.mark.parametrize('dtype', ['fp32','fp16','bf16'])
def test_root_rounding_boundary_inputs_keep_valid_finite_full_logits(dtype):
    data = synthetic_case(Geometry(batch=1,width=2,heads=1,mlp=3,layers=2),dtype,'boundary')
    result = vit_reference(data)
    assert all(math.isfinite(x) for value in result['stages'].values() for x in leaves(value))
    # This walk also includes each semantic stage, not just final logits.
    assert all(store(x,dtype)==x for value in result['stages'].values() for x in leaves(value))


@pytest.mark.parametrize('change', [dict(batch=True),dict(image_size=5),dict(heads=3),dict(width=0)])
def test_independent_geometry_rejects_invalid_dimensions(change):
    with pytest.raises(ValueError):
        Geometry(**change)


@pytest.mark.parametrize('failure', ['missing','extra','shape','nonfinite','overflow'])
def test_oracle_requires_complete_finite_root_parameters(failure):
    data = synthetic_case(dtype='fp16')
    if failure=='missing': del data['roots']['head_bias']
    elif failure=='extra': data['roots']['computed_logits']=[[0.]]
    elif failure=='shape': data['roots']['blocks'][0]['q_bias']=[0.]
    elif failure=='nonfinite': data['roots']['patch_bias'][0]=float('nan')
    else: data['roots']['patch_bias'][0]=1e20
    with pytest.raises(ValueError):
        vit_reference(data)
