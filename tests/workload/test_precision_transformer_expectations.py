"""Analytic sensitivity checks for the independent VAL56 references."""
from copy import deepcopy
from fractions import Fraction
import math
import struct

import pytest

from precision_transformer_expectations import (
    FORMATS, audit_materialized_storage, dot32, encoder_block, f32, gelu, ieee_rne, layer_norm,
    lifetime_peak, materialized_movement, softmax, store, tensor_bytes, tiny_inputs,
)


@pytest.mark.parametrize('dtype,p,emin', [('fp32',24,-126),('fp16',11,-14),('bf16',8,-126)])
def test_exact_rational_ties_choose_even_both_signs(dtype,p,emin):
    unit = Fraction(1, 2**(p-1))
    assert ieee_rne(1 + unit/2, dtype) == 1
    assert ieee_rne(1 + unit*3/2, dtype) == 1 + 2*unit
    assert ieee_rne(-1 - unit/2, dtype) == -1
    assert ieee_rne(-1 - unit*3/2, dtype) == -1 - 2*unit
    tiny = Fraction(1, 2**(-emin+p-1))
    assert ieee_rne(tiny/2,dtype) == 0
    assert ieee_rne(3*tiny/2,dtype) == 2*tiny
    assert math.copysign(1,ieee_rne(-tiny/2,dtype)) == -1
    # A small exact rational displacement distinguishes tie handling from truncation.
    assert ieee_rne(1 + unit/2 + unit/1024,dtype) == 1 + unit


def test_binary_half_known_extremes_subnormals_and_signed_zero():
    assert ieee_rne(65504,'fp16') == 65504
    assert ieee_rne(65519,'fp16') == 65504
    assert ieee_rne(65520,'fp16') == math.inf
    assert ieee_rne(Fraction(1,2**24),'fp16') == 2**-24
    assert math.copysign(1,ieee_rne(-0.0,'bf16')) == -1
    assert math.isnan(ieee_rne(float('nan'),'fp32'))
    for value in [True,'1',None]:
        with pytest.raises(ValueError):ieee_rne(value,'fp16')


@pytest.mark.parametrize('dtype,code', [('fp32','f'),('fp16','e')])
def test_independent_grid_matches_separate_host_ieee_conversion(dtype,code):
    # Host struct is used only as a second checker, never by the reference itself.
    values = [0.,-0.,1.,-1.,1/3,-1/7,2**-24,1.00048828125,31.9921875,65504.]
    for value in values:
        assert ieee_rne(value,dtype) == struct.unpack('>'+code,struct.pack('>'+code,value))[0]


@pytest.mark.parametrize('dtype,large,p', [('fp16',2048,11),('bf16',256,8)])
def test_accumulate_in_fp32_and_round_only_final_storage_not_each_k_tile(dtype,large,p):
    left=[store(x,dtype) for x in [large,1,-large]]
    actual=store(dot32(left,[1,1,1]),dtype)
    wrong=0.
    for x in left:wrong=store(wrong+x,dtype)
    assert actual == 1 and wrong == 0
    # A final output itself still rounds, including a tie followed by a deciding term.
    assert store(dot32([1,2**-p],[1,1]),dtype) == 1
    assert store(dot32([1,2**-p,2**(-p-1)],[1,1,1]),dtype) == 1+2**(1-p)


def test_fused_and_nonfused_fp32_oracles_are_explicit_and_discriminating():
    left=[-1.,1+2**-23];right=[1.,1-2**-23]
    assert dot32(left,right) == 0
    assert dot32(left,right,fused=True) == -(2**-46)
    with pytest.raises(ValueError):dot32([1],[1,2])


@pytest.mark.parametrize('dtype',list(FORMATS))
def test_norm_population_variance_affine_and_constant_row_closed_forms(dtype):
    assert layer_norm([[-1.,1.]],[2.,3.],[.25,-.5],0,dtype) == [[-1.75,2.5]]
    assert layer_norm([[7.,7.]],[2.,3.],[.25,-.5],1e-5,dtype) == [[.25,-.5]]


@pytest.mark.parametrize('dtype',list(FORMATS))
def test_stable_softmax_large_and_constant_rows_are_finite_and_nontrivial(dtype):
    assert softmax([1000,1000,1000],dtype) == [store(1/3,dtype)]*3
    assert softmax([1000,-1000,1000],dtype) == [.5,0,.5]
    assert softmax([-1000,-1000],dtype) == [.5,.5]


def test_gelu_formula_is_explicit_and_two_definitions_are_distinguishable():
    assert gelu(0,'erf') == 0 and gelu(0,'tanh') == 0
    assert gelu(2,'erf') == pytest.approx(1.954499736,abs=2e-7)
    assert abs(gelu(2,'erf')-gelu(2,'tanh')) > 9e-5
    with pytest.raises(ValueError):gelu(1,'unspecified')


@pytest.mark.parametrize('dtype',list(FORMATS))
def test_complete_dense_block_exercises_both_residuals_batches_heads_and_mlp(dtype):
    config,values=tiny_inputs();result=encoder_block(config,values,dtype=dtype,gelu_definition='erf')
    assert (config['batch'],config['seq'],config['hidden'],config['heads'],config['mlp'])==(2,3,8,2,16)
    assert len(result['output'])==2 and all(len(b)==3 and all(len(r)==8 for r in b) for b in result['output'])
    assert result['output'][0] != result['output'][1]
    assert result['q'] != result['k'] and result['q'] != result['v']
    flattened=[row for batch in result['output'] for row in batch]
    assert result['input'] != result['residual1']
    assert result['residual1'] != flattened
    assert len(result['probabilities'])==4 and len(result['activated'][0])==16
    for b in result['output']:
        for row in b:
            assert all(math.isfinite(x) and store(x,dtype)==x for x in row)
    mutated=deepcopy(values);mutated['w_k'],mutated['w_v']=mutated['w_v'],mutated['w_k']
    changed=encoder_block(config,mutated,dtype=dtype,gelu_definition='erf')
    assert changed['output'] != result['output']


@pytest.mark.parametrize('definition',['erf','tanh'])
def test_complete_fp32_scalar_block_matches_separate_vectorized_fp64_equations(definition):
    import numpy as np
    config,values=tiny_inputs()
    v={key:np.asarray(value,dtype=np.float64) for key,value in values.items()}

    def norm(x,index):
        centered=x-x.mean(axis=-1,keepdims=True)
        return centered/np.sqrt(np.mean(centered**2,axis=-1,keepdims=True)+config['epsilon'])*v[f'gamma{index}']+v[f'beta{index}']

    x=v['x'];normalized=norm(x,1)
    # This implementation uses explicit B,H,S,D tensor permutation, not the
    # scalar reference's flattened row/head indexing or its rounding helpers.
    projection={name:(normalized@v['w_'+name]+v['b_'+name]).reshape(2,3,2,4).transpose(0,2,1,3)
                for name in ('q','k','v')}
    scores=projection['q']@projection['k'].swapaxes(-1,-2)/2
    exponentials=np.exp(scores-scores.max(axis=-1,keepdims=True))
    probabilities=exponentials/exponentials.sum(axis=-1,keepdims=True)
    merged=(probabilities@projection['v']).transpose(0,2,1,3).reshape(2,3,8)
    residual=x+merged@v['w_o']+v['b_o']
    expanded=norm(residual,2)@v['w_fc1']+v['b_fc1']
    if definition=='erf':
        activated=expanded*.5*(1+np.vectorize(math.erf)(expanded/np.sqrt(2)))
    else:
        activated=expanded*.5*(1+np.tanh(np.sqrt(2/np.pi)*(expanded+.044715*expanded**3)))
    expected=residual+activated@v['w_fc2']+v['b_fc2']
    actual=encoder_block(config,values,dtype='fp32',gelu_definition=definition)
    np.testing.assert_allclose(actual['output'],expected,rtol=3e-6,atol=4e-7)
    np.testing.assert_allclose(np.asarray(actual['probabilities']).reshape(2,2,3,3),probabilities,
                               rtol=3e-6,atol=3e-7)


@pytest.mark.parametrize('dtype,p',[('fp16',11),('bf16',8)])
def test_block_input_rounding_and_stored_parameter_values_are_observable(dtype,p):
    config,values=tiny_inputs()
    values['x'][0][0][0]=1+3*2**(-p-1)
    values['gamma1'][0]=1+3*2**(-p-1)
    result=encoder_block(config,values,dtype=dtype,gelu_definition='erf')
    assert result['input'][0][0]==1+2**(1-p)
    assert result['input'][0][0]!=values['x'][0][0][0]
    changed=deepcopy(values);changed['gamma1'][0]=1.
    changed_result=encoder_block(config,changed,dtype=dtype,gelu_definition='erf')
    assert result['norm1']!=changed_result['norm1']


@pytest.mark.parametrize('dtype',list(FORMATS))
def test_zero_output_branches_reduce_complete_block_to_rounded_identity(dtype):
    config,values=tiny_inputs()
    for name in ['o','fc2']:
        values['w_'+name]=[[0. for _ in row] for row in values['w_'+name]]
        values['b_'+name]=[0.]*len(values['b_'+name])
    result=encoder_block(config,values,dtype=dtype,gelu_definition='tanh')
    expected=[[[store(x,dtype) for x in row] for row in b] for b in values['x']]
    assert result['output']==expected


@pytest.mark.parametrize('dtype,unit',[('fp32',4),('fp16',2),('bf16',2)])
def test_declared_storage_uses_target_bytes_not_bf16_host_container(dtype,unit):
    config,values=tiny_inputs()
    # Input48 elements; weights512 + projection/MLP bias56 + norm parameters32.
    assert sum(len(row) for name,rows in values.items() if name.startswith('w_') for row in rows)==512
    assert sum(len(row) for name,row in values.items() if name.startswith('b_'))==56
    assert sum(len(values[name]) for name in ('gamma1','beta1','gamma2','beta2'))==32
    assert tensor_bytes((2,3,8),dtype)==48*unit
    assert (48+512+56+32)*unit==648*unit
    movement=materialized_movement((2,3,2,4),dtype,dtype)
    assert movement==dict(read_bytes=48*unit,write_bytes=48*unit,allocation_bytes=48*unit)
    assert materialized_movement((2,3,8),'fp32',dtype)['read_bytes']==192
    assert materialized_movement((2,3,8),'fp32',dtype)['write_bytes']==48*unit
    with pytest.raises(ValueError):tensor_bytes((2,True,8),dtype)


@pytest.mark.parametrize('mutation',['missing','extra','shape','ragged','bool','infinite','overflow','heads','dropout'])
def test_independent_block_refuses_incomplete_or_unsupported_inputs(mutation):
    config,values=tiny_inputs()
    if mutation=='missing':del values['b_q']
    if mutation=='extra':values['unused']=[1.]
    if mutation=='shape':values['w_q']=values['w_q'][:-1]
    if mutation=='ragged':values['x'][0][0]=values['x'][0][0][:-1]
    if mutation=='bool':values['x'][0][0][0]=True
    if mutation=='infinite':values['x'][0][0][0]=float('inf')
    if mutation=='overflow':values['x'][0][0][0]=1e6
    if mutation=='heads':config['heads']=3
    if mutation=='dropout':config['dropout']=.1
    with pytest.raises(ValueError):encoder_block(config,values,dtype='fp16',gelu_definition='erf')


def test_dtype_shape_sweep_retains_weights_outputs_and_last_of_two_readers():
    tasks={'early':dict(start=1,end=2),'late':dict(start=3,end=7)}
    tensors=[dict(name='W',shape=(2,),dtype='bf16',allocation=0,readers=['early'],persistent=True),
             dict(name='T',shape=(2,),dtype='fp16',allocation=0,readers=['early','late']),
             dict(name='Y',shape=(2,),dtype='fp32',allocation=7,readers=[],returned=True)]
    peak,intervals=lifetime_peak(tensors,tasks,9)
    assert peak==12
    assert intervals[0]['last_reader']==2 and intervals[0]['release']==9
    assert intervals[1]['last_reader']==intervals[1]['release']==7
    assert intervals[2]['release']==9
    # A false early release changes the independently predicted peak.
    tensors[1]['allocation']=1;tensors[2]['allocation']=3
    assert lifetime_peak(tensors,tasks,9)[0]==16
    tensors[1]['readers']=['early']
    assert lifetime_peak(tensors,tasks,9)[0]==12


def test_actual_contract_adapter_rejects_bf16_host_byte_count_and_dependency_violation():
    compilation=dict(program={'tensors':[
        dict(name='W',shape=[2],dtype='bf16',kind='weight'),
        dict(name='Y',shape=[2],dtype='fp16',kind='activation')]},
        plan={'storage':[
            dict(tensor='W',bytes=4,producer=None,consumers=['producer']),
            dict(tensor='Y',bytes=4,producer='producer',consumers=['reader'])]})
    result={'evaluation':{'latency_ns':7,'details':{'trace':[
        dict(kind='storage',name='board:W',start_ns=0),
        dict(kind='storage',name='board:Y',start_ns=1),
        dict(kind='compute',operator='producer',start_ns=1,end_ns=3),
        dict(kind='compute',operator='reader',start_ns=4,end_ns=6)]}}}
    peak,intervals=audit_materialized_storage(compilation,result)
    assert peak==8 and intervals[0]['release']==7 and intervals[1]['release']==6
    compilation['plan']['storage'][0]['bytes']=8
    with pytest.raises(AssertionError,match='shape/dtype'):
        audit_materialized_storage(compilation,result)
    compilation['plan']['storage'][0]['bytes']=4
    result['evaluation']['details']['trace'][-1]['start_ns']=2
    with pytest.raises(AssertionError,match='producer'):
        audit_materialized_storage(compilation,result)


@pytest.mark.parametrize('dtype',list(FORMATS))
def test_frozen_sim45_inputs_bind_fma_tanh_and_fp32_softmax_without_production(dtype):
    import hashlib
    import json
    from pathlib import Path
    from precision_transformer_expectations import frozen_block_reference
    source=Path(__file__).resolve().parents[1]/'fixtures/precision_transformer/INPUTS.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='f7538a04334fab9476e60f8e50ffece461bc2a1cf7caea9298e9cc7b8f150c06'
    rows=json.loads(source.read_text())
    p=rows['events-'+dtype]['workload']['params']
    assert p==rows['roofline-'+dtype]['workload']['params']
    actual=frozen_block_reference(p)
    assert len(actual['output'])==2 and len(actual['probabilities'])==4
    probabilities=[x for head in actual['probabilities'] for row in head for x in row]
    assert all(f32(x)==x and math.isfinite(x) for x in probabilities)
    if dtype!='fp32':
        assert any(store(x,dtype)!=x for x in probabilities)
    for row in [row for head in actual['probabilities'] for row in head]:
        assert sum(row)==pytest.approx(1,abs=3e-7)
