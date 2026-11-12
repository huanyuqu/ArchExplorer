"""Reusable logical regions and explicit instruction/tile IR; units are bytes."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Region:
    tensor: str
    row: int
    col: int
    rows: int
    cols: int
    stride: int
    element_bytes: int

    def __post_init__(self):
        if not self.tensor:
            raise ValueError('region requires a tensor identity')
        if any(type(v) is not int or v<0 for v in (self.row,self.col)):
            raise ValueError('region offsets must be nonnegative integers')
        if any(type(v) is not int or v<1 for v in (self.rows,self.cols,self.stride,self.element_bytes)):
            raise ValueError('region dimensions and element bytes must be positive integers')
        if self.col+self.cols>self.stride:
            raise ValueError('region crosses its logical row stride')

    @property
    def bytes(self):
        return self.rows * self.cols * self.element_bytes

    def spans(self, base):
        for r in range(self.row, self.row + self.rows):
            yield base + (r*self.stride+self.col)*self.element_bytes, self.cols*self.element_bytes


@dataclass(frozen=True)
class Instruction:
    opcode: str
    dest: tuple
    sources: tuple = ()
    region: Region | None = None
    size: int = 0  # upstream compute size, NOT bytes; MOVIN size = region.bytes
    shape: tuple = ()


@dataclass(frozen=True)
class Tile:
    index: tuple
    accum: bool
    footprint: int
    instructions: tuple
