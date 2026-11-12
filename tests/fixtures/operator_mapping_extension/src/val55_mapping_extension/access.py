"""Logical square demand, delegated by the registered access callback."""
from math import prod


def square_access(operator,tensors):
    elements = prod(tensors[operator.outputs[0]].shape)
    return dict(operations=elements,read_bytes=8*elements,write_bytes=4*elements,
                boundary='component_io',fixture='val55-square')
