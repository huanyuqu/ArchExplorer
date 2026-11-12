"""Independent VAL52 observations, not a component model or a production API.

Rates here are already FLOP/ns and B/ns. A candidate-specific test must obtain
the actual rates, boundary and duration through the frozen public contract.
These helpers neither choose a component implementation nor execute a backend.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class RooflineCase:
    name: str
    flops: int
    boundary_bytes: int
    peak_flops_per_ns: int
    bandwidth_bytes_per_ns: int
    expected_ns: Fraction


# Explicit hand calculations, independent of any author's produced result.
# FMA/MAC counts as TWO FLOPs. Bytes refer to one declared physical boundary,
# not the sum of all levels of the hierarchy or packet-header overhead.
ROOFLINE_CASES = (
    RooflineCase('compute_bound', 4096, 128, 128, 64, Fraction(32)),
    RooflineCase('memory_bound', 256, 8192, 128, 64, Fraction(128)),
    RooflineCase('ridge', 1024, 512, 128, 64, Fraction(8)),
    RooflineCase('zero_compute', 0, 128, 128, 64, Fraction(2)),
    RooflineCase('zero_boundary_traffic', 4096, 0, 128, 64, Fraction(32)),
    RooflineCase('zero_work', 0, 0, 128, 64, Fraction(0)),
    RooflineCase('fractional_ns', 3, 5, 2, 4, Fraction(3, 2)),
    # GEMM b=1,m=4,k=8,n=16: 512 MAC = 1024 FLOP; no modeled traffic.
    RooflineCase('two_flops_per_mac', 1024, 0, 128, 64, Fraction(8)),
)


def check_roofline_duration(case, observed_ns):
    """Compare an actual observation to the independently frozen hand answer."""
    assert type(observed_ns) in (int, float), 'duration must be a number, not a flag/string'
    assert math.isfinite(observed_ns) and observed_ns >= 0, 'invalid observed duration'
    assert math.isclose(observed_ns, float(case.expected_ns), rel_tol=1e-10, abs_tol=1e-12), (
        case.name, observed_ns, case.expected_ns)


@dataclass(frozen=True)
class BoundaryCharge:
    transfer: str
    boundary: str
    owner: str
    bytes: int


def check_boundary_charges(expected_bytes, observed):
    """Exactly one mechanism owns each declared transfer/boundary charge.

The adapter must retain real transfer IDs and the modeled boundary. Splitting
one charge into sectors is allowed; charging the same boundary in SM and DRAM
is not. Distinct physical boundaries are distinct keys, even for one transfer.
This does not infer bytes from time or confuse logical bytes with wire bytes.
"""
    assert all(isinstance(key, tuple) and len(key) == 2 and
               all(isinstance(part, str) and part for part in key) and
               type(size) is int and size >= 0 for key, size in expected_bytes.items())
    totals, owners = defaultdict(int), defaultdict(set)
    for charge in observed:
        assert all(isinstance(part, str) and part for part in
                   (charge.transfer, charge.boundary, charge.owner))
        assert type(charge.bytes) is int and charge.bytes > 0
        key = charge.transfer, charge.boundary
        assert key in expected_bytes, ('undeclared transfer/boundary', key)
        totals[key] += charge.bytes
        owners[key].add(charge.owner)
    for key, size in expected_bytes.items():
        assert totals[key] == size, (key, totals[key], size)
        assert len(owners[key]) == (1 if size else 0), ('competing billing owners', key, owners[key])
