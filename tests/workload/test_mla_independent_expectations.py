"""Hand examples and falsifiers for VAL61 oracle preparation only."""
from fractions import Fraction
import math

import pytest

from mla_independent_expectations import (
    RootLifetime, attention_core, cache_offsets, exact_absorbed_score,
    fp32_region_softmax, grouped_cache_bytes, input_shapes,
    multicast_edge_bytes, physical_peak, port_service_lower_bounds,
    synthetic_roots,
)
from precision_transformer_expectations import store


SHAPE = dict(batch=2, heads=3, sequence=5, latent=2, content=4,
             position=2, value=3, output=5)


def one_token_roots():
    shape = dict(batch=1, heads=2, sequence=1, latent=2, content=2,
                 position=2, value=2, output=3)
    roots = dict(query_content=[1, 0, 0, 1], query_position=[0, 0, 0, 0],
                 cache_latent=[2, 3], cache_position=[0, 0],
                 key_up=[1, 0, 0, 1, 1, 0, 0, 1],
                 value_up=[1, 0, 0, 1, 2, 0, 0, 2],
                 output_weight=[1, 0, 1, 0, 1, 1, 1, 0, -1, 0, 1, -1])
    return shape, roots


@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
@pytest.mark.parametrize('program', ['nonabsorbed', 'absorbed'])
def test_hand_two_head_output(dtype, program):
    shape, roots = one_token_roots()
    actual = attention_core(shape, roots, dtype=dtype, program=program,
                            softmax_policy=fp32_region_softmax)
    # One token -> probability one. Heads [2,3] and [4,6]; projection gives
    # [2+4, 3+6, 2+3-4-6]. No production reference is involved.
    assert actual['probabilities'] == [[1.0], [1.0]]
    assert actual['head_output'] == [[2.0, 3.0], [4.0, 6.0]]
    assert actual['output'] == [[6.0, 9.0, -5.0]]


@pytest.mark.parametrize('dtype', ['fp32', 'fp16', 'bf16'])
def test_rectangular_batch_head_order_and_all_stages(dtype):
    roots = synthetic_roots(SHAPE)
    for program in ('nonabsorbed', 'absorbed'):
        result = attention_core(SHAPE, roots, dtype=dtype, program=program,
                                softmax_policy=fp32_region_softmax)
        assert len(result['scores']) == 6
        assert all(len(row) == 5 for row in result['scores'])
        assert len(result['output']) == 2
        assert all(len(row) == 5 for row in result['output'])
        assert result['output'][0] != result['output'][1]
        for row in result['probabilities']:
            assert abs(sum(row) - 1) <= {'fp32': 2e-7, 'fp16': 5e-4, 'bf16': .004}[dtype]


def test_hand_zero_logits_uniform_average():
    shape, roots = one_token_roots()
    shape.update(sequence=2)
    roots.update(query_content=[0]*4, cache_latent=[2, 4, 6, 8],
                 cache_position=[0]*4)
    result = attention_core(shape, roots, dtype='bf16', program='nonabsorbed',
                            softmax_policy=fp32_region_softmax)
    assert result['scores'] == [[0, 0], [0, 0]]
    assert result['probabilities'] == [[.5, .5], [.5, .5]]
    assert result['head_output'] == [[4, 6], [8, 12]]
    assert result['output'] == [[12, 18, -10]]


def test_position_key_contributes_and_scale_uses_original_content_dimension():
    shape, roots = one_token_roots()
    shape.update(sequence=2)
    roots.update(query_content=[0]*4, query_position=[2, 0, 0, 2],
                 cache_latent=[0]*4, cache_position=[2, 0, 0, 2])
    result = attention_core(shape, roots, dtype='fp32', program='absorbed',
                            softmax_policy=fp32_region_softmax)
    assert result['scores'] == [[2, 0], [0, 2]]  # sqrt(2+2) = 2
    assert result['probabilities'][0][0] > result['probabilities'][0][1]
    assert result['probabilities'][0] == result['probabilities'][1][::-1]


def test_stable_softmax_extreme_and_ties():
    assert fp32_region_softmax([1000, 1000], 'fp32') == [.5, .5]
    assert fp32_region_softmax([-1000, 1000], 'fp16') == [0, 1]


def test_exact_association_does_not_license_low_precision_reassociation():
    q, w, cache = [1, 1], [[1, Fraction(1, 256)], [0, 1]], [1, -1]
    assert exact_absorbed_score(q, w, cache) == (Fraction(1, 256),)*2
    # BF16 materialized K=[1,-255/256] vs materialized absorbed Q=[1,1].
    # RNE maps 257/256 to one in the latter program, losing the small result.
    expanded = store(store(1, 'bf16') + store(Fraction(-255, 256), 'bf16'), 'bf16')
    absorbed = store(store(Fraction(257, 256), 'bf16') - 1, 'bf16')
    assert expanded == Fraction(1, 256)
    assert absorbed == 0


def test_manual_cache_addresses_head_aliasing_batch_separation():
    assert cache_offsets(SHAPE, batch=1, head=2, token=4) == {
        'cache_latent': (18, 19), 'cache_position': (18, 19)}
    assert cache_offsets(SHAPE, batch=1, head=0, token=4) == cache_offsets(
        SHAPE, batch=1, head=2, token=4)
    assert cache_offsets(SHAPE, batch=0, head=0, token=4)['cache_latent'] == (8, 9)
    with pytest.raises(ValueError):
        cache_offsets(SHAPE, batch=0, head=3, token=0)


def test_group_copies_follow_declared_tiles_not_head_count_or_paper_ratio():
    actual = grouped_cache_bytes(SHAPE, [[0, 1, 2], [3, 4]], unit_bytes=2)
    assert actual == {'payload': 80, 'private': 400, 'grouped': 160,
                      'group_copies': {0: 80, 1: 80}}
    assert grouped_cache_bytes(SHAPE, [[0, 1, 2, 3, 4]], unit_bytes=2)['grouped'] == 80
    with pytest.raises(ValueError):
        grouped_cache_bytes(SHAPE, [[0, 1], [1, 2]], unit_bytes=2)


def test_multicast_fanout_is_charged_and_queries_and_merge_are_distinct():
    kv = multicast_edge_bytes([['sram', 'fork', 'tile0'], ['sram', 'fork', 'tile1']], 12)
    assert sum(kv.values()) == 36  # one trunk, two branches
    links = {edge: (4, 1) for edge in kv}
    # A second KV chunk/payload version cannot merge into the first transfer.
    busy = port_service_lower_bounds([kv, kv], links)
    assert busy[('sram', 'fork')] == 8
    query0 = multicast_edge_bytes([['sram', 'fork', 'tile0']], 4)
    query1 = multicast_edge_bytes([['sram', 'fork', 'tile1']], 4)
    assert port_service_lower_bounds([query0, query1], links)[('sram', 'fork')] == 4
    merge0 = multicast_edge_bytes([['tile0', 'root']], 8)
    merge1 = multicast_edge_bytes([['tile1', 'root']], 8)
    assert sum(merge0.values()) + sum(merge1.values()) == 16


def test_tree_rejects_hidden_reconvergence_and_unpriced_edges():
    with pytest.raises(ValueError):
        multicast_edge_bytes([['s', 'a', 'd'], ['s', 'b', 'd']], 4)
    with pytest.raises(KeyError):
        port_service_lower_bounds([{('s', 'a'): 4}], {})
    with pytest.raises(ValueError):
        port_service_lower_bounds([{('s', 'a'): 4}], {('s', 'a'): (math.inf, 0)})


def test_last_reader_returned_root_actual_peak_and_one_byte_short():
    roots = [RootLifetime('cache', 12, 0, 8, (('head0', 1, 3), ('head1', 2, 8))),
             RootLifetime('scratch', 10, 2, 4),
             RootLifetime('out', 6, 4, 10, (('merge', 4, 7),), 10)]
    assert physical_peak(roots, capacity=22)['peak_bytes'] == 22
    assert physical_peak(roots[::-1])['peak_bytes'] == 22
    with pytest.raises(MemoryError, match='22 bytes, capacity 21'):
        physical_peak(roots, capacity=21)
    with pytest.raises(ValueError, match='last consumer'):
        physical_peak([RootLifetime('cache', 12, 0, 3, (('head1', 2, 8),))])
    with pytest.raises(ValueError, match='returned root'):
        physical_peak([RootLifetime('out', 6, 4, 8, (), 10)])
    with pytest.raises(ValueError, match='one positive allocation'):
        physical_peak([roots[0], roots[0]])


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_invalid_dimensions_and_root_contract(bad):
    with pytest.raises(ValueError):
        input_shapes(dict(SHAPE, latent=bad))


def test_missing_nonfinite_and_incomplete_roots_rejected():
    roots = synthetic_roots(SHAPE)
    for bad in ({}, dict(roots, query_content=[math.nan]),
                dict(roots, cache_latent=roots['cache_latent'][:-1])):
        with pytest.raises(ValueError):
            attention_core(SHAPE, bad, dtype='fp32', program='absorbed',
                           softmax_policy=fp32_region_softmax)
