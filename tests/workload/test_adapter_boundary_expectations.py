"""Check VAL54 independent expected values before candidate consumption."""
from collections import Counter
from fractions import Fraction

import pytest

from adapter_boundary_expectations import (
    region_case, serial_roofline_ns, group_lifetime_case, interval_peak,
)
from compilation_expectations import reference_matmul


def test_masked_regions_preserve_indices_reduction_and_scalar_values():
    case = region_case()
    assert reference_matmul(case['a'], case['b']) == case['output']
    visits = Counter()
    for region in case['regions']:
        for row in range(region['row'], region['row']+region['rows']):
            for col in range(region['col'], region['col']+region['cols']):
                for start, stop in region['reduction_blocks']:
                    for k in range(start, stop):
                        visits[row, k, col] += 1
    assert set(visits) == {(m, k, n) for m in range(3) for k in range(3) for n in range(5)}
    assert set(visits.values()) == {1}
    assert 2*sum(r['fmas'] for r in case['regions']) == case['operations'] == 90
    assert sum(r['read_bytes'] for r in case['regions'])+case['assembly']['read_bytes'] == 420
    assert sum(r['write_bytes'] for r in case['regions'])+case['assembly']['write_bytes'] == 120
    # Nominal 2x2x3 tiles would perform 96 FMAs; only 45 are semantic work.
    assert sum(visits.values()) == 45 < 96


def test_component_io_owns_the_entire_roofline_service_once():
    case = region_case()
    assert serial_roofline_ns(case) == Fraction(135, 2)
    # A duplicated external stream would add another 540/8 ns incorrectly.
    assert serial_roofline_ns(case) != Fraction(540, 8)*2


def test_two_resource_group_dependencies_last_reader_and_simultaneous_free():
    case = group_lifetime_case()
    tasks = {t['name']: t for t in case['tasks']}
    assert all(t['start'] >= tasks[dep]['end'] for t in tasks.values() for dep in t['deps'])
    shared = case['storage'][0]
    assert shared['end'] == max(tasks[name]['end'] for name in shared['readers'])
    assert shared['end'] > min(tasks[name]['end'] for name in shared['readers'])
    assert interval_peak(case['storage']) == case['peak_bytes'] == 64
    # At t=8 only replacement is live: a same-time allocation is not a third 32B.
    assert interval_peak([shared, case['storage'][2]]) == 32


@pytest.mark.parametrize('start,end,size', [(2, 2, 32), (3, 2, 32), (0, 1, 0)])
def test_lifetime_oracle_rejects_empty_or_reversed_intervals(start, end, size):
    with pytest.raises(ValueError):
        interval_peak([dict(start=start, end=end, bytes=size)])
