"""VAL54 expected semantics and service boundary, independent of production.

Rates are explicit synthetic test assumptions. Mapping pays real external
traffic for events, while the component_io roofline owns that service itself.
The same traffic count must never be charged by both owners.
"""
from fractions import Fraction


def region_case():
    """3x3 @ 3x5; M/N tiles 2/3 and reduction blocks [0,2), [2,3)."""
    a = [[1, 2, -1], [0, 3, 2], [-2, 1, 4]]
    b = [[2, 0, 1, -1, 3], [1, 4, -2, 2, 0], [3, -1, 0, 5, 2]]
    output = [[1, 9, -3, -2, 1], [9, 10, -6, 16, 4], [9, 0, -4, 24, 2]]
    regions = []
    for row in range(0, 3, 2):
        for col in range(0, 5, 3):
            rows, cols = min(2, 3-row), min(3, 5-col)
            fmas = rows * cols * 3
            regions.append(dict(row=row, col=col, rows=rows, cols=cols,
                                reduction_blocks=[(0, 2), (2, 3)],
                                fmas=fmas, read_bytes=8*fmas,
                                write_bytes=4*rows*cols))
    return dict(a=a, b=b, output=output, regions=regions,
                assembly=dict(read_bytes=60, write_bytes=60, operations=0),
                operations=90, read_bytes=420, write_bytes=120,
                unique_tensor_bytes=156)


def serial_roofline_ns(case, *, peak=16, bandwidth=8):
    """Independent sum for one serial compute resource; not a global DAG bound."""
    if peak <= 0 or bandwidth <= 0:
        raise ValueError('positive synthetic rates required')
    compute = sum((max(Fraction(2*r['fmas'], peak),
                       Fraction(r['read_bytes']+r['write_bytes'], bandwidth))
                   for r in case['regions']), Fraction())
    assembly = case['assembly']
    return compute + Fraction(assembly['read_bytes']+assembly['write_bytes'], bandwidth)


def group_lifetime_case():
    """Declared expected trace for dependency/resource/last-reader tests.

    A producer and two readers occupy two different compute resources in one
    execution domain. A separate temporary overlaps the shared 32-byte value.
    A third allocation at the exact release instant checks free-before-allocate.
    """
    return dict(
        tasks=[dict(name='producer', resource='a', start=0, end=2, deps=[]),
               dict(name='reader-one', resource='a', start=2, end=4, deps=['producer']),
               dict(name='reader-two', resource='b', start=2, end=8, deps=['producer'])],
        storage=[dict(name='shared', bytes=32, start=0, end=8,
                      readers=['reader-one', 'reader-two']),
                 dict(name='temporary', bytes=32, start=2, end=6, readers=[]),
                 dict(name='replacement', bytes=32, start=8, end=10, readers=[])],
        completion_ns=8, peak_bytes=64)


def interval_peak(intervals):
    """Half-open physical lifetimes: release precedes allocation on equal times."""
    changes = []
    for item in intervals:
        if item['start'] >= item['end'] or item['bytes'] <= 0:
            raise ValueError('positive size and nonempty lifetime required')
        changes.extend([(item['start'], 1, item['bytes']),
                        (item['end'], 0, -item['bytes'])])
    current = peak = 0
    for _, _, amount in sorted(changes):
        current += amount
        peak = max(peak, current)
    assert current == 0
    return peak
