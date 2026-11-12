"""Independent semantic expectations, not a proposed production IR or adapter.

Target adapters must be added against the frozen public compiler API. These
expectations do not imply a target emits full padded tiles or has perfect reuse.
"""
from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import prod
from fractions import Fraction


@dataclass(frozen=True)
class MatmulCase:
    name: str
    shape: tuple
    tiles: tuple
    dtype_bytes: int = 4

    def __post_init__(self):
        for values in (self.shape, self.tiles):
            if len(values) != 3 or any(type(x) is not int or x <= 0 for x in values):
                raise ValueError('positive exact M/K/N and abstract block extents required')
        if self.dtype_bytes != 4:
            raise ValueError('this independent first-case oracle covers FP32 only')

    def work(self):
        m, k, n = self.shape
        padded = tuple(((size + tile - 1) // tile) * tile
                       for size, tile in zip(self.shape, self.tiles))
        return {'logical_macs': prod(self.shape), 'logical_ops': 2 * prod(self.shape),
                'full_tile_macs_if_unguarded': prod(padded),
                'padded_shape_if_unguarded': padded,
                'unique_logical_tensor_bytes': self.dtype_bytes * (m*k + k*n + m*n)}

    def tiled_coordinates(self):
        """Return real MAC indices and padding positions, independently of mapping."""
        real, padding = Counter(), 0
        m, k, n = self.shape
        tm, tk, tn = self.tiles
        for bm, bk, bn in product(range(0, m, tm), range(0, k, tk), range(0, n, tn)):
            for im, ik, inn in product(range(tm), range(tk), range(tn)):
                coord = (bm+im, bk+ik, bn+inn)
                if all(c < size for c, size in zip(coord, self.shape)):
                    real[coord] += 1
                else:
                    padding += 1
        return real, padding


def reference_matmul(a, b):
    if not a or not b or any(len(row) != len(b) for row in a):
        raise ValueError('matrix dimensions do not match')
    n = len(b[0])
    if not n or any(len(row) != n for row in b):
        raise ValueError('ragged RHS')
    return [[sum(x * b[k][j] for k, x in enumerate(row)) for j in range(n)] for row in a]


CASES = tuple(MatmulCase(f'boundary-{m}-{k}-{n}-tiles-{tm}-{tk}-{tn}',
                        (m, k, n), (tm, tk, tn))
              for m, k, n in ((3, 2, 5), (5, 3, 7))
              for tm, tk, tn in ((2, 2, 2), (4, 2, 4)))

COMPILE_CHECKS = (
    ('shared-semantics', 'same semantic/tensor DAG retained by one public inspectable plan'),
    ('three-consumers', 'actual roofline/events/native artifacts bind and consume that plan'),
    ('boundary-work', 'logical work versus emitted padding/predication independently accounted'),
    ('traffic-ownership', 'each logical/physical boundary declares owner, units and timing; no duplicate charge'),
    ('temporary-two-readers', 'multi-step dependencies retain temporary until last actual reader ends'),
    ('prepare-no-execution', 'valid compile/prepare under denied subprocess/native load/search'),
    ('early-refusal', 'unsupported dtype/op/mapping/target/boundary rejects before execution'),
    ('determinism', 'saved plan identity binds semantics, values, mapping, hardware, implementation and target sources'),
    ('tamper-refusal', 'semantic/input/mapping/target/source changes cannot replay original result'),
    ('unknown-extension', 'separate installed extension consumes same compiler without main-package edits'),
    ('installed-entrypoints', 'default API/CLI/Agent converge on same installed public plan'),
    ('distribution-closure', 'affected source/sdist/wheel resources exact; no private research input exported'),
    ('ipu-exp-rcp-required', 'source-bound exp/reciprocal and declared conversions on actual public Softmax/attention; refusal alone does not close this mechanism'),
)


def softmax_special_counts(rows, cols):
    """Stable Softmax needs one exp per element and one reciprocal per row.

    These operation categories do not imply a dedicated hardware SFU. Any
    implementation using a different numerical algorithm must expose its own
    equivalent work/counting and error contract, not silently inherit this one.
    """
    if any(type(x) is not int or x <= 0 for x in (rows, cols)):
        raise ValueError('positive exact dimensions required')
    return {'exp': rows * cols, 'reciprocal': rows}


def serial_special_cycles(counts, costs, *, conversions=0):
    """QA arithmetic for an explicitly serial/shared instruction resource only.

    Costs are supplied by the case author, never production defaults or IPU
    calibration. Ordinary arithmetic is excluded; conversions are explicit.
    This is resource demand, not a prediction that graph latency must increase.
    """
    if type(conversions) is not int or conversions < 0:
        raise ValueError('explicit nonnegative conversion count required')
    result = Fraction(0)
    for op, count in {**counts, 'conversion': conversions}.items():
        if type(count) is not int or count < 0:
            raise ValueError('exact work counts required')
        if count == 0:
            continue
        entry = costs[op]
        if entry.get('unit') != 'cycle/op' or not all(entry.get(k) for k in
                                                     ('source', 'dtype', 'implementation')):
            raise ValueError('cost unit/implementation/dtype/source must be explicit')
        if type(entry['value']) is bool:
            raise ValueError('boolean cost is invalid')
        value = Fraction(str(entry['value']))
        if value <= 0:
            raise ValueError('positive cost required for this serial oracle')
        result += count * value
    return result
