"""Check independent VAL55 references before a production mapping API is frozen."""
from collections import Counter
from decimal import Decimal, localcontext
import math

import pytest

from adapter_boundary_expectations import interval_peak
from operator_attention_expectations import (
    attention, broadcast, declared_attention_lifetimes, f32, matmul,
    non_square_case, rectangles, reduce2, shape2, stable_softmax, transpose)


def test_scalar_attention_matches_separate_high_precision_calculation():
    case = non_square_case()
    actual = attention(case['q'],case['k'],case['v'])
    assert shape2(actual['output']) == (3,2)
    with localcontext() as context:
        context.prec = 60
        q,k,v = [[[Decimal(str(number)) for number in row] for row in case[key]]
                 for key in ('q','k','v')]
        scale = Decimal(1)/Decimal(3).sqrt()
        for row, predicted in zip(q,actual['output']):
            logits = [sum(x*y for x,y in zip(row,key))*scale for key in k]
            weights = [(value-max(logits)).exp() for value in logits]
            denom = sum(weights)
            expected = [float(sum(weights[j]*v[j][col] for j in range(5))/denom) for col in range(2)]
            assert predicted == pytest.approx(expected,rel=2e-6,abs=2e-7)
    assert shape2(actual['kt']) == (3,5)
    assert all(sum(row) == pytest.approx(1,abs=2e-7) for row in actual['normalized'])


@pytest.mark.parametrize('row,expected', [
    ([1000,1000,1000],[1/3,1/3,1/3]),
    ([1e6,-1e6,1e6],[.5,0,.5]),
    ([-1000,-1000,-1000],[1/3,1/3,1/3]),
    ([math.log(2),math.log(3),0],[1/3,.5,1/6])])
def test_stable_softmax_extreme_and_closed_form_rows(row,expected):
    result = stable_softmax([row])
    assert result['normalized'][0] == pytest.approx(expected,abs=2e-7)
    assert all(math.isfinite(v) for stage in result.values() for r in stage for v in r)
    assert max(result['shifted'][0]) == 0


def test_constant_attention_is_mean_of_values_and_not_zero_stub():
    output = attention([[0,0],[0,0]],[[1,2],[3,4],[5,6]],[[0,3],[3,6],[6,9]])['output']
    assert output == [[3,6],[3,6]]


def test_materialized_transpose_and_broadcast_are_distinct_storage():
    source = [[1,2,3],[4,5,6]]
    transposed = transpose(source)
    assert transposed == [[1,4],[2,5],[3,6]]
    materialized = broadcast([[1],[2]],(2,3))
    assert materialized == [[1,1,1],[2,2,2]]
    materialized[0][0] = 9
    assert materialized[0][1] == 1 and materialized[1][0] == 2
    transposed[0][0] = 99
    assert source[0][0] == 1
    assert math.prod(shape2(transposed))*4 == 24
    assert math.prod(shape2(materialized))*4 == 24


@pytest.mark.parametrize('axis,keepdims,expected', [
    (0,True,[[5,7,9]]),(1,True,[[6],[15]]),(-1,False,[6,15]),(-2,False,[5,7,9])])
def test_reduction_axis_and_keepdims_are_explicit(axis,keepdims,expected):
    assert reduce2([[1,2,3],[4,5,6]],'sum',axis,keepdims) == expected
    assert reduce2([[1,9,3],[4,5,6]],'max',1) == [[9],[6]]


def test_nondivisible_regions_cover_every_real_index_once():
    regions = rectangles(3,5,2,3)
    visits = Counter((i,j) for r in regions for i in range(r['row'],r['row']+r['rows'])
                     for j in range(r['col'],r['col']+r['cols']))
    assert visits == Counter((i,j) for i in range(3) for j in range(5))
    assert sum(r['rows']*r['cols'] for r in regions) == 15 < 4*2*3
    # QK^T:45 FMAs; PV:30 FMAs; each scalar softmax stage visits15 values.
    assert 3*3*5+3*5*2 == 75
    assert sum(r['rows']*r['cols'] for r in regions)*5 == 75


def test_two_attention_double_readers_release_after_last_real_use():
    tasks,intervals = declared_attention_lifetimes()
    storage = {item['name']:item for item in intervals}
    assert storage['Scaled']['readers'] == ['max','subtract']
    assert storage['Scaled']['end'] == 9 > 8
    assert storage['E']['readers'] == ['sum','normalize']
    assert storage['E']['end'] == 15 > 13
    assert interval_peak(intervals) == 196
    # Materialized K transpose overlaps live input K during conversion.
    assert storage['K']['end'] == 2 and storage['KT']['start'] == 0
    assert storage['KT']['bytes'] == storage['K']['bytes'] == 60
    assert max(t['end'] for t in tasks) == 18


@pytest.mark.parametrize('invalid', [True,float('nan'),float('inf'),1e40])
def test_independent_reference_refuses_nonfinite_or_non_numeric_scalars(invalid):
    with pytest.raises(ValueError):
        f32(invalid)


def test_reference_boundaries_reject_shape_axis_and_broadcast_mismatch():
    for operation in [lambda:shape2([[1],[2,3]]),lambda:matmul([[1,2]],[[1,2]]),
                      lambda:broadcast([[1,2]],(3,3)),
                      lambda:reduce2([[1,2]],'sum',2),
                      lambda:reduce2([[1,2]],'sum',True)]:
        with pytest.raises(ValueError):
            operation()


def test_runtime_sweep_distinguishes_persistent_weights_from_activation_inputs():
    from test_compilation_execution_contract import independently_sweep_storage
    compilation = dict(program={'tensors':[{'name':'W','kind':'weight'},{'name':'T','kind':'activation'}]},
        plan={'storage':[
            {'tensor':'W','bytes':8,'producer':None,'consumers':['read_weight']},
            {'tensor':'T','bytes':8,'producer':None,'consumers':['read_activation']}]})
    result = {'evaluation':{'latency_ns':5,'details':{'trace':[
        {'kind':'storage','name':'board:W','start_ns':0},
        {'kind':'storage','name':'board:T','start_ns':2},
        {'kind':'compute','operator':'read_weight','start_ns':0,'end_ns':2},
        {'kind':'compute','operator':'read_activation','start_ns':2,'end_ns':4}]}}}
    peak,intervals = independently_sweep_storage(compilation,result)
    weight = next(r for r in intervals if r['tensor']=='W')
    assert weight['last_reader']==2 and weight['release']==5 and weight['persistent']
    assert peak==16
    compilation['program']['tensors'][0]['kind']='activation'
    assert independently_sweep_storage(compilation,result)[0]==8
