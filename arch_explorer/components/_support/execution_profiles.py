"""Explicit architecture policies for instruction and transaction execution.

Default values are simulator configuration, NOT calibrated device parameters.
Synthetic layouts are useful mechanism tests; they are not device presets.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AddressLayout:
    channels: int = 32
    ranks: int = 2
    groups: int = 4
    banks_per_group: int = 4
    columns: int = 16
    burst_bytes: int = 64
    rows: int = 32768

    def __post_init__(self):
        for value in self.__dict__.values():
            if type(value) is not int or value < 1 or value & (value - 1):
                raise ValueError('address dimensions must be positive powers of two')

    @property
    def banks(self):
        return self.groups * self.banks_per_group

    @property
    def row_stride(self):
        return self.burst_bytes*self.columns*self.channels*self.banks*self.ranks

    def decode(self, address):
        if type(address) is not int or not 0 <= address < self.row_stride*self.rows:
            raise ValueError('address outside physical layout')
        value = address // self.burst_bytes
        fields = []
        for size in (self.columns, self.channels, self.banks, self.ranks):
            fields.append(value % size)
            value //= size
        col, channel, bank, rank = fields
        return channel, rank, bank, value, col

    def encode(self, channel=0, rank=0, bank=0, row=0, column=0):
        for value, size in zip((channel, rank, bank, row, column),
                              (self.channels, self.ranks, self.banks, self.rows, self.columns)):
            if type(value) is not int or not 0 <= value < size:
                raise ValueError('address coordinate outside layout')
        return ((((row*self.ranks+rank)*self.banks+bank)*self.channels+channel)*self.columns+column)*self.burst_bytes

    def stripe(self, address):
        """Swap low channel/column fields for the profile's first-touch stream."""
        low = address % self.burst_bytes
        upper, fields = divmod(address // self.burst_bytes, self.channels*self.columns)
        column, channel = divmod(fields, self.channels)
        return ((upper*self.channels+channel)*self.columns+column)*self.burst_bytes+low


@dataclass(frozen=True)
class ControllerPolicy:
    name: str = 'neupims-f299af3-dual-buffer'
    write_threshold: int = 8
    pim_queue_target: int = 2
    activation_window: int = 4
    min_comps: int = 6
    pim_enabled: bool = True

    def __post_init__(self):
        for key in ('write_threshold', 'pim_queue_target', 'activation_window', 'min_comps'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('positive integer policy limits required')


@dataclass(frozen=True)
class InstructionPolicy:
    name: str = 'neupims-f299af3'
    spad_bytes: int = 134218*1024
    accum_bytes: int = 134218*1024
    size_divisor: int = 8
    minimum_issue: int = 4
    vector_slots: int = 8
    vector_bytes_per_tick: int = 128
    element_bytes: int = 2

    def address_ring_bytes(self, hidden, tp):
        if any(type(v) is not int or v<1 for v in (hidden,tp)):
            raise ValueError('positive hidden dimension and tensor parallel degree required')
        # Common.cc::from_instruction: circular cursor limit, not tensor layout.
        return hidden*hidden*5*self.element_bytes//tp

    def __post_init__(self):
        for key, value in self.__dict__.items():
            if key != 'name' and (type(value) is not int or value < 1):
                raise ValueError('positive integer instruction policy values required')


NEUPIMS_LAYOUT = AddressLayout()
NEUPIMS_CONTROLLER = ControllerPolicy()
NEUPIMS_INSTRUCTIONS = InstructionPolicy()
PROFILE_SOURCE = ('NeuPIMs f299af3: Common.cc initialize_config; MatMul.cc; '
                  'NeuPIMSystolicWS.cc; NewtonSim configuration.cc, timing.cc, '
                  'neupims_controller.cc, HBM2_8Gb_x128_dualpim.ini. '
                  'Capacities in B, timing in integer 1 ns ticks; no fitting.')
