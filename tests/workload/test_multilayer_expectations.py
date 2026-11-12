"""VAL57 independent reference sensitivity, not new production acceptance."""
from copy import deepcopy
import json

import pytest

from multilayer_expectations import (
    PARAMETER_NAMES,chain_reference,leaves,matmul_tiles,parameter_storage,
    reference_for,stack_inputs,transform,
)
from composition_expectations import fork_values,fork_storage
from precision_transformer_expectations import lifetime_peak,store


@pytest.mark.parametrize('count',[2,4])
@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('case',['regular','boundary'])
def test_distinct_layer_dataflow_is_not_independent_blocks_or_repeated_weights(count,dtype,case):
    data=stack_inputs(count,dtype=dtype,case=case);before=deepcopy(data)
    actual=reference_for(data)
    assert data==before
    assert actual['order']==['unit-'+str(i) for i in range(count)]
    assert len({json.dumps(row['parameters'],sort_keys=True) for row in data['instances']})==count
    assert all(set(row['parameters'])==PARAMETER_NAMES for row in data['instances'])
    assert all(store(value,dtype)==value for value in leaves(actual['output']))
    # A gross wiring mistake is numerically observable for these exact inputs.
    detached=chain_reference(data['config'],data['X'],[data['instances'][-1]],dtype=dtype)
    assert actual['output']!=detached['output']
    repeated=deepcopy(data)
    for row in repeated['instances']:row['parameters']=deepcopy(data['instances'][0]['parameters'])
    assert actual['output']!=reference_for(repeated)['output']
    reversed_data=deepcopy(data);reversed_data['instances'].reverse()
    assert actual['output']!=reference_for(reversed_data)['output']
    # Parameters are retained individually; no fictitious X per later layer.
    unit=4 if dtype=='fp32' else 2
    assert parameter_storage(data)=={row['id']:600*unit for row in data['instances']}


@pytest.mark.parametrize('count',[2,4])
@pytest.mark.parametrize('dtype,p',[('fp32',24),('fp16',11),('bf16',8)])
def test_closed_form_bias_only_stack_rounds_each_residual_boundary(count,dtype,p):
    data=stack_inputs(count,dtype=dtype)
    data['X']=transform(data['X'],lambda value,i:1.)
    half_ulp=2.**(-p)
    for row in data['instances']:
        row['parameters']={key:transform(value,lambda z,i:0.) for key,value in row['parameters'].items()}
        # Attention output bias and final MLP bias each add exactly half an ulp.
        for key in ('bo','b2'):
            row['parameters'][key]=transform(row['parameters'][key],lambda z,i:half_ulp)
    actual=reference_for(data)
    assert set(leaves(actual['output']))=={1.}
    assert store(1+count*2*half_ulp,dtype)>1.  # wrong delayed rounding across layers
    for stage in actual['layers'].values():
        assert set(leaves(stage['residual1']))==set(leaves(stage['output']))=={1.}


@pytest.mark.parametrize('mutation',['later-X','intermediate','duplicate-id','missing-weight'])
def test_reference_input_contract_has_only_model_X_and_actual_parameters(mutation):
    data=stack_inputs()
    if mutation=='later-X':data['instances'][1]['parameters']['X']=data['X']
    if mutation=='intermediate':data['instances'][1]['parameters']['precomputed_Y']=data['X']
    if mutation=='duplicate-id':data['instances'][1]['id']=data['instances'][0]['id']
    if mutation=='missing-weight':del data['instances'][1]['parameters']['Wq']
    with pytest.raises(ValueError):reference_for(data)


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_generic_fork_has_hand_computed_values_peak_traffic_and_last_reader(dtype):
    output=fork_values([1.,2.,-1.,.5],dtype=dtype)
    assert output['y']==[4.,8.,-4.,2.]
    case=fork_storage(dtype=dtype)
    peak,intervals=lifetime_peak(case['tensors'],case['tasks'],case['finish'])
    assert peak==case['expected_peak']==(48 if dtype=='fp32' else 24)
    assert peak<case['declared_sum']
    assert {row['name']:row['release'] for row in intervals}==case['expected_release']
    assert case['operator_read_bytes']==(80 if dtype=='fp32' else 40)
    assert case['operator_write_bytes']==(64 if dtype=='fp32' else 32)
    assert case['conversion_instructions']==(0 if dtype=='fp32' else 36)
    # A first-reader release loses a genuinely live shared activation.
    bad=deepcopy(case['tensors']);next(t for t in bad if t['name']=='u')['readers']=['left']
    _,wrong=lifetime_peak(bad,case['tasks'],case['finish'])
    assert next(row for row in wrong if row['name']=='u')['release']==2
    assert 2<case['expected_release']['u']


def test_same_kind_instances_require_distinct_nondivisible_tile_regions():
    first=matmul_tiles(1,3,5,4,dict(m=2,k=3,n=3))
    second=matmul_tiles(1,3,5,4,dict(m=3,k=2,n=4))
    assert len(first)==4 and len(second)==1
    assert first[0]['k_ranges']==[(0,3),(3,5)]
    assert second[0]['k_ranges']==[(0,2),(2,4),(4,5)]
    for regions in (first,second):
        coordinates=[(i,j) for row in regions for i in range(*row['rows']) for j in range(*row['columns'])]
        assert len(coordinates)==len(set(coordinates))==12
    # Collapsing the two same-kind bindings to either policy changes regions.
    assert first!=second
