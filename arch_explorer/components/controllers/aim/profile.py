"""Fixed GDDR6/AiM subset; timing values are third-party simulator assumptions.

Source: arkhadem/aim_simulator 0f28a07bdb83e42b9305ad3d45410ebd3aa2c091,
src/dram/impl/GDDR6.cpp:182-218,580-716,752-999. Defined initialization
overrides are applied; the invalid source nRFC requires a caller input.
Only RD, WR, WRGB, WRMAC16,
MAC16 and RDMAC16 are exposed. Other ISA operations are rejected.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
from arch_explorer.components.aim_parameters import SOURCE, EFFECTIVE, parameter_sources

PROFILE = 'aim-0f28a07b-effective'
OPS = ('RD', 'WR', 'WRGB', 'WRMAC16', 'MAC16', 'RDMAC16')
REG = ('WRGB', 'WRMAC16', 'RDMAC16')


@dataclass(frozen=True)
class AimProfile:
    refresh_service_ns: float | None = None
    refresh_source: str | None = None
    clock_ns: Fraction = Fraction(1)
    banks: int = 16
    rows: int = 16384
    row_bytes: int = 2048
    burst_bytes: int = 32
    queue_size: int = 33  # upstream ReqBuffer max_size=32, enqueue uses <=
    refi: int = 7800

    def __post_init__(self):
        if (self.banks, self.rows, self.row_bytes, self.burst_bytes, self.clock_ns, self.refi) != (16, 16384, 2048, 32, Fraction(1), 7800):
            raise ValueError('AiM effective profile requires 16 banks, 16K rows, 2KiB rows, 32B bursts, 1ns ticks, tREFI=7800ns')
        if (type(self.refresh_service_ns) not in (int,float) or not math.isfinite(self.refresh_service_ns)
                or self.refresh_service_ns <= 0 or not isinstance(self.refresh_source,str) or not self.refresh_source.strip()):
            raise ValueError('refresh_service_ns must be an explicit finite positive ns value with refresh_source; upstream nRFC is undefined')
        if type(self.queue_size) is not int or self.queue_size < 1:
            raise ValueError('queue size must be a positive integer')
        if self.refi <= self.rfc:
            raise ValueError('refresh interval must exceed refresh service time')

    @property
    def rfc(self):
        return math.ceil(Fraction(self.refresh_service_ns) / self.clock_ns)

    def bank_values(self):
        return dict(technology='gddr', capacity=32*1024**2, bandwidth=1., latency=86.,
                    row_bytes=2048, column_bytes=32, trcd=36., trp=32., tras=54.,
                    trrd=6., tfaw=28., tccd=2., tcl=50.,
                    trfc=self.refresh_service_ns, trefi=7800.)

    def decode(self, address):
        if type(address) is not int or address % 32 or not 0 <= address < self.banks * self.rows * self.row_bytes:
            raise ValueError('channel-local byte address must be aligned, within capacity')
        row, rem = divmod(address, self.banks * self.row_bytes)
        bank, col = divmod(rem, self.row_bytes)
        return bank, row, col // 32

    def encode(self, bank, row, column=0):
        if any(type(v) is not int for v in (bank, row, column)) or not (0 <= bank < 16 and 0 <= row < self.rows and 0 <= column < 64):
            raise ValueError('invalid bank/row/32-byte column')
        return (row * 16 + bank) * 2048 + column * 32


@dataclass(frozen=True)
class AimCommand:
    operation: str
    address: int = 0
    useful_bytes: int = 32

    def validate(self, profile):
        if self.operation not in OPS:
            raise ValueError(f'unsupported AiM command {self.operation!r}')
        bank, row, column = profile.decode(self.address)
        if self.operation not in ('RD', 'WR') and bank != 0:
            raise ValueError('channel-wide AiM commands use bank zero as the address anchor')
        if type(self.useful_bytes) is not int or not 0 < self.useful_bytes <= 32:
            raise ValueError('useful_bytes must be in [1,32]; wire transfer is always a full 32B burst')
        return bank, row, column


def timing_rules(profile):
    """Independent constraint representation of the supported source-command pairs.

    Keys are (scope, preceding, following), delay is in 1ns cycles.
    No empirical multiplier; duplicate rules take the strongest constraint.
    """
    rules = {}
    def add(scope, before, after, delay):
        for a in before.split():
            for b in after.split():
                key = (scope, a, b)
                rules[key] = max(rules.get(key, 0), delay)
    add('channel', 'RD RDMAC16', 'RD RDMAC16', 2)
    add('channel', 'WR WRGB WRMAC16', 'WR WRGB WRMAC16', 2)
    add('channel', 'RD MAC16 RDMAC16', 'RD MAC16 RDMAC16', 2)
    add('channel', 'RD', 'WR', 50 + 2 + 3 - 6 + 1)
    add('channel', 'RD', 'WRGB WRMAC16', 50 + 2 + 3 - 1 + 1)
    add('channel', 'RDMAC16', 'WR', 0 + 2 + 3 - 6 + 1)
    add('channel', 'RDMAC16', 'WRGB WRMAC16', 0 + 2 + 3 - 1 + 1)
    add('channel', 'WR', 'RD', 6 + 2 + 9)
    add('channel', 'WR', 'RDMAC16', 6 + 2 + 11)
    add('channel', 'WRGB', 'RD RDMAC16', 1 + 2 + 9)
    add('channel', 'WRMAC16', 'RD RDMAC16', 1 + 2 + 11)
    add('group', 'WR', 'RD', 6 + 2 + 11)
    add('group', 'RD', 'RD', 2)
    add('group', 'WR', 'WR', 2)
    add('group', 'ACT', 'ACT', EFFECTIVE['nRRDL'])
    add('channel', 'RD MAC16', 'PREA', 12)
    add('channel', 'MAC16', 'PRE', 12)
    add('channel', 'WR', 'PREA', 6 + 2 + 33)
    add('bank', 'RD', 'PRE', 12)
    add('bank', 'WR', 'PRE', 6 + 2 + 33)
    add('channel', 'ACT', 'ACT', EFFECTIVE['nRRDS'])
    add('channel', 'ACT ACT16', 'ACT16', 89)
    add('channel', 'ACT16', 'ACT ACT16', 89)
    add('bank', 'ACT', 'ACT', 89)
    add('channel', 'ACT ACT16', 'PREA', 54)
    add('channel', 'ACT16', 'PRE', 54)
    add('bank', 'ACT', 'PRE', 54)
    add('channel', 'PRE PREA', 'ACT16', 32)
    add('channel', 'PREA', 'ACT PRE PREA', 32)
    add('channel', 'PRE', 'PREA', 32)
    add('bank', 'PRE', 'ACT PRE', 32)
    add('channel', 'ACT ACT16', 'MAC16', 56)
    add('channel', 'ACT16', 'RD', 36)
    add('channel', 'ACT16', 'WR', 28)
    add('bank', 'ACT', 'RD', 36)
    add('bank', 'ACT', 'WR', 28)
    add('channel', 'ACT ACT16', 'REFab', 89)
    add('channel', 'PRE PREA', 'REFab', 32)
    add('channel', 'REFab', 'ACT ACT16', profile.rfc)
    add('channel', 'TMOD', ' '.join((*OPS, 'ACT', 'ACT16', 'PRE', 'PREA', 'REFab')), 32)
    return rules


LATENCY = dict(RD=52, WR=8, WRGB=3, WRMAC16=3, MAC16=1, RDMAC16=2)
