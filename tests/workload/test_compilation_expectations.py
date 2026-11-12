"""Preparation oracle checks only; no new production compiler is claimed here."""
from itertools import product

import pytest

from compilation_expectations import (CASES, MatmulCase, reference_matmul,
                                      serial_special_cycles, softmax_special_counts)


@pytest.mark.parametrize('case,ops,padded,tensor_bytes', [
    (CASES[0], 60, 48, 124), (CASES[1], 60, 64, 124),
    (CASES[2], 210, 192, 284), (CASES[3], 210, 256, 284),
])
def test_independent_logical_padded_work_and_nondivisible_coverage(case, ops, padded, tensor_bytes):
    work = case.work()
    assert work['logical_ops'] == ops
    assert work['full_tile_macs_if_unguarded'] == padded
    assert work['unique_logical_tensor_bytes'] == tensor_bytes
    real, padding = case.tiled_coordinates()
    assert set(real) == set(product(*(range(size) for size in case.shape)))
    assert set(real.values()) == {1}
    assert sum(real.values()) + padding == padded
    assert padding > 0


def test_independent_numerical_reference_has_asymmetric_values_and_correct_orientation():
    assert reference_matmul([[1, 2], [3, 4], [5, 6]],
                            [[2, 0, 1, 3, 4], [1, 2, 0, 1, 2]]) == [
        [4, 4, 1, 5, 8], [10, 8, 3, 13, 20], [16, 12, 5, 21, 32]]


@pytest.mark.parametrize('shape,tiles', [((True, 2, 5), (2, 2, 2)),
                                      ((3, 0, 5), (2, 2, 2)),
                                      ((3, 2, 5), (2, -1, 2))])
def test_oracle_does_not_silently_accept_invalid_dimensions(shape, tiles):
    with pytest.raises(ValueError):
        MatmulCase('bad', shape, tiles)


def synthetic_costs():
    # Deliberately synthetic QA values; not measurements, not production rates.
    return {name: {'value': value, 'unit': 'cycle/op', 'source': 'synthetic QA arithmetic',
                   'dtype': 'fp32', 'implementation': 'synthetic-serial-instruction-sequence'}
            for name, value in [('exp', '3.5'), ('reciprocal', '5'), ('conversion', '0.25')]}


def test_special_work_and_serial_costs_keep_element_row_and_conversion_counts_separate():
    counts = softmax_special_counts(2, 3)
    assert counts == {'exp': 6, 'reciprocal': 2}
    assert serial_special_cycles(counts, synthetic_costs(), conversions=6) == 65 / 2
    assert serial_special_cycles(counts, synthetic_costs()) == 31


def test_special_resource_work_can_increase_without_increasing_memory_bound_graph_time():
    first = serial_special_cycles(softmax_special_counts(2, 3), synthetic_costs())
    second = serial_special_cycles(softmax_special_counts(2, 6), synthetic_costs())
    assert second > first
    assert max(100, first) == max(100, second) == 100


@pytest.mark.parametrize('field,value', [('source', ''), ('dtype', ''), ('implementation', ''),
                                       ('unit', 'ops/cycle'), ('value', True), ('value', 0)])
def test_special_cost_oracle_requires_explicit_usable_cost_provenance(field, value):
    costs = synthetic_costs()
    costs['exp'][field] = value
    with pytest.raises(ValueError):
        serial_special_cycles(softmax_special_counts(1, 4), costs)
