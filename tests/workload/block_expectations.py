"""Small independent FP32 block math, rectangular coverage and logical IO.

No product code; no device Events or scale graph is fabricated. Large coverage
enumerations are deliberately outside this reference's small-control domain.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import math
import struct
import numpy as np
from typed_onnx_expectations import cast
from precision_transformer_expectations import ieee_rne


def positive(shape):
    if not shape or any(type(n) is not int or n<=0 for n in shape):raise ValueError('block oracle positive integer shape required')
    return tuple(shape)


def fp32(value):
    a=cast(value,'fp32')
    if not np.isfinite(a).all():raise ValueError('block oracle requires finite FP32 values')
    return a


def payload_digest(value):
    a=fp32(value)
    return hashlib.sha256(b''.join(struct.pack('<f',float(v)) for v in a.flat)).hexdigest()


def partition_grid(shape,block_shape):
    shape=positive(shape);block_shape=positive(block_shape)
    if len(shape)!=len(block_shape):raise ValueError('block oracle partition rank differs')
    grids=[range(0,n,b) for n,b in zip(shape,block_shape)]
    if math.prod(len(g) for g in grids)>4096:raise ValueError('block oracle exceeds 4096 fragments')
    return [dict(starts=list(at),shape=[min(b,n-s) for n,b,s in zip(shape,block_shape,at)],
                 bytes=4*math.prod(min(b,n-s) for n,b,s in zip(shape,block_shape,at)))
            for at in itertools.product(*grids)]


def partition_coverage(shape,blocks):
    shape=positive(shape)
    if math.prod(shape)>1000000:raise ValueError('block coverage oracle is limited to small controls')
    seen=set();fragments=[]
    for block in blocks:
        sizes=positive(block['shape']);starts=block['starts']
        if len(sizes)!=len(shape) or len(starts)!=len(shape) or any(type(s) is not int or s<0 for s in starts):
            raise ValueError('block oracle fragment rank/start differs')
        if any(s+b>n for s,b,n in zip(starts,sizes,shape)):raise ValueError('block oracle fragment outside source')
        if block['bytes']!=4*math.prod(sizes):raise ValueError('block oracle fragment byte width differs')
        addresses=[]
        for local in np.ndindex(sizes):
            at=tuple(s+i for s,i in zip(starts,local))
            if at in seen:raise ValueError('block oracle fragment overlap')
            seen.add(at);addresses.append(int(np.ravel_multi_index(at,shape)))
        fragments.append(addresses)
    if len(seen)!=math.prod(shape):raise ValueError('block oracle fragment coverage hole')
    return dict(elements=len(seen),bytes=4*len(seen),flat_indices_by_fragment=fragments)


def load(source,*,starts,shape,guards=(),source_name='source',guard_names=None):
    source=fp32(source);sizes=positive(shape)
    if len(starts)!=source.ndim or len(sizes)!=source.ndim or any(type(s) is not int or s<0 for s in starts):
        raise ValueError('block oracle load rank/start differs')
    if any(s+n>limit for s,n,limit in zip(starts,sizes,source.shape)):raise ValueError('block oracle load outside source')
    names=list(guard_names) if guard_names is not None else [f'guard{i}' for i in range(len(guards))]
    if len(names)!=len(guards):raise ValueError('block oracle guard name count differs')
    reads=[(source_name,tuple(s+i for s,i in zip(starts,at))) for at in np.ndindex(sizes)]
    for name,guard in zip(names,guards):
        value=fp32(guard)
        reads.extend((name,at) for at in np.ndindex(value.shape))
    result=source[tuple(slice(s,s+n) for s,n in zip(starts,sizes))].copy()
    return result,ledger(reads,4*math.prod(sizes))


def zero(shape):
    shape=positive(shape)
    return np.zeros(shape,dtype=np.float64),ledger([],4*math.prod(shape))


def matmul_accumulate(left,right,seed,*,names=('A','B','C')):
    a,b,c=fp32(left),fp32(right),fp32(seed)
    if a.ndim!=3 or b.ndim!=3 or c.ndim!=3 or a.shape[0]!=b.shape[0] or a.shape[2]!=b.shape[1] or c.shape!=(a.shape[0],a.shape[1],b.shape[2]):
        raise ValueError('block oracle batch matmul/seed shapes differ')
    output=np.empty_like(c);reads=[]
    for batch,row,col in np.ndindex(c.shape):
        reads.append((names[2],(batch,row,col)));value=float(c[batch,row,col])
        for k in range(a.shape[2]):
            reads.extend([(names[0],(batch,row,k)),(names[1],(batch,k,col))])
            value=ieee_rne(Fraction(value)+Fraction(float(a[batch,row,k]))*Fraction(float(b[batch,k,col])),'fp32')
        output[batch,row,col]=value
    if not np.isfinite(output).all():raise ValueError('block oracle result is nonfinite')
    return output,ledger(reads,4*c.size)


def ledger(reads,write_bytes):
    counts=Counter(name for name,_ in reads)
    return dict(reads=reads,read_by_tensor={n:4*v for n,v in counts.items()},read_bytes=4*len(reads),write_bytes=write_bytes,
        scope='logical FP32 IO; device locations/lifetimes/routes require actual candidate observations')
