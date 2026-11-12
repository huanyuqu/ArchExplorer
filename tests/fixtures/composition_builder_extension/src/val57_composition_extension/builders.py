"""Separate builder file enables a real nested-source byte mutation test."""
from copy import deepcopy
import math

from arch_explorer.workload.api import Workload


def build_scale(params,*,tp=1):
    if tp!=1:raise ValueError('external scale has no TP communication')
    if set(params)-{'shape','factor','inputs'}:raise ValueError('unsupported external scale field')
    shape=params.get('shape',[1,4]);factor=params.get('factor',2.)
    if (not isinstance(shape,(list,tuple)) or not 1<=len(shape)<=4
            or any(type(n) is not int or n<1 for n in shape)):
        raise ValueError('external scale requires positive contiguous rank1..4 FP32 shape')
    if type(factor) not in (int,float) or not math.isfinite(factor):
        raise ValueError('external scale factor must be finite')
    inputs=params.get('inputs',{})
    if type(inputs) is not dict or set(inputs)-{'x'}:raise ValueError('external scale has only input port x')
    graph=Workload('external_leaf',dtype='fp32')
    x=graph.tensor('x',tuple(shape),dtype='fp32')
    y=graph.tensor('y',tuple(shape),dtype='fp32')
    graph.add('scale','scale',[x],[y],shape=list(shape),factor=factor)
    graph.meta['inputs']=deepcopy(inputs)
    return graph
