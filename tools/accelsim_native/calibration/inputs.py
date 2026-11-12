"""Concrete inputs and independent Python oracle, never device addresses."""
import struct
from pathlib import Path
from common import ref

STEPS = (8, 16, 24, 32)
FAMILIES = ('dep', 'quad', 'chase')
MULTIPLIER = 1664525
ADDEND = 1013904223


def pack(values):
    return struct.pack('<' + 'I' * len(values), *values)


def make_inputs(root):
    root = Path(root)
    root.mkdir(exist_ok=True)
    values = {
        'dep': [(0x9e3779b9 * (i + 1)) & 0xffffffff for i in range(32)],
        'quad': [(0x85ebca6b * (i + 7)) & 0xffffffff for i in range(128)],
        'constants': [MULTIPLIER, ADDEND],
        'next': [(17 * i + 11) % 4096 for i in range(4096)],
        'start': [(127 * i + 3) % 4096 for i in range(32)],
    }
    for name, data in values.items():
        path = root / (name + '.bin')
        raw = pack(data)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('previous input changed')
        path.write_bytes(raw)
    return values


def expected(family, steps, values):
    if family not in FAMILIES or steps not in STEPS:
        raise ValueError('unregistered family/steps')
    if family == 'chase':
        # Compose the affine permutation analytically, separately from the C++
        # oracle which indexes the concrete array on every iteration.
        mul, add = 1, 0
        for _ in range(steps):
            mul, add = (17 * mul) % 4096, (17 * add + 11) % 4096
        return pack([(mul * index + add) % 4096 for index in values['start']])
    # Closed recurrence composition modulo 2^32, not repeated output updates.
    mul, add = 1, 0
    for _ in range(steps):
        mul, add = (MULTIPLIER * mul) % 2**32, (MULTIPLIER * add + ADDEND) % 2**32
    return pack([(mul * x + add) % 2**32 for x in values[family]])


def input_rows(root):
    return {p.stem: ref(p, Path(root).parent) for p in sorted(Path(root).glob('*.bin'))}
