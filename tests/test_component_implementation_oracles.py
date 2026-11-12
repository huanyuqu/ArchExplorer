"""Test the independent QA oracle; passing is NOT SM implementation acceptance."""
from fractions import Fraction

import pytest

from component_implementation_oracles import (
    BoundaryCharge, ROOFLINE_CASES, check_boundary_charges, check_roofline_duration,
)


@pytest.mark.parametrize('case', ROOFLINE_CASES, ids=lambda case: case.name)
def test_frozen_answers_follow_explicit_units_and_hand_calculation(case):
    compute_ns = Fraction(case.flops, case.peak_flops_per_ns)
    memory_ns = Fraction(case.boundary_bytes, case.bandwidth_bytes_per_ns)
    assert max(compute_ns, memory_ns) == case.expected_ns
    check_roofline_duration(case, float(case.expected_ns))


@pytest.mark.parametrize('incorrect', [16, 34, 0, float('nan'), float('inf'), -1, True, '32'])
def test_observer_rejects_half_mac_count_sum_fallback_and_nonfinite(incorrect):
    with pytest.raises(AssertionError):
        check_roofline_duration(ROOFLINE_CASES[0], incorrect)


def test_sector_charges_and_distinct_physical_boundaries_are_not_duplicates():
    check_boundary_charges({('load', 'dram-icnt'): 64, ('load', 'l2-sm'): 64}, [
        BoundaryCharge('load', 'dram-icnt', 'dram', 32),
        BoundaryCharge('load', 'dram-icnt', 'dram', 32),
        BoundaryCharge('load', 'l2-sm', 'l2', 64),
    ])
    check_boundary_charges({('cached', 'dram-icnt'): 0}, [])


@pytest.mark.parametrize('charges', [
    [],  # dropped traffic
    [BoundaryCharge('load', 'dram-icnt', 'dram', 128)],
    [BoundaryCharge('load', 'dram-icnt', 'dram', 64),
     BoundaryCharge('load', 'dram-icnt', 'sm', 64)],
    [BoundaryCharge('load', 'dram-icnt', 'dram', 32),
     BoundaryCharge('load', 'dram-icnt', 'sm', 32)],
    [BoundaryCharge('other', 'dram-icnt', 'dram', 64)],
    [BoundaryCharge('load', 'other', 'dram', 64)],
    [BoundaryCharge('load', 'dram-icnt', 'dram', True)],
])
def test_accounting_rejects_missing_duplicate_split_ownership_or_renamed_transfer(charges):
    with pytest.raises(AssertionError):
        check_boundary_charges({('load', 'dram-icnt'): 64}, charges)
