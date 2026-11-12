"""Independent typed elementary oracle, using exact rational IEEE rounding.

Arrays carry decoded values as float64; dtype is explicit, including BF16.
No product parser, lowering, layout resolver or runtime is imported here.
Only stash_type=FLOAT is prepared. This is a chosen sequential FP32 reduction
reference; parallel framework reductions can differ and are compared by error.
"""
from fractions import Fraction
import math
import numpy as np
from precision_transformer_expectations import ieee_rne, dot32


def cast(value, dtype):
    if dtype not in ('fp32','fp16','bf16'):
        raise ValueError('oracle: unsupported floating dtype')
    a=np.asarray(value,dtype=np.float64)
    return np.array([ieee_rne(float(x),dtype) for x in a.flat],dtype=np.float64).reshape(a.shape)


def elementwise(op, *values, dtype):
    arrays=np.broadcast_arrays(*[np.asarray(x,dtype=np.float64) for x in values])
    funcs={'add':lambda a,b:Fraction(float(a))+Fraction(float(b)),
           'sub':lambda a,b:Fraction(float(a))-Fraction(float(b)),
           'mul':lambda a,b:Fraction(float(a))*Fraction(float(b)),
           'div':lambda a,b:Fraction(float(a))/Fraction(float(b)),
           'sqrt':lambda a:math.sqrt(float(a)), 'erf':lambda a:math.erf(float(a)),
           'tanh':lambda a:math.tanh(float(a))}
    if op not in funcs:raise ValueError('oracle: unsupported elementary operation '+op)
    output=[ieee_rne(ieee_rne(funcs[op](*scalars),'fp32'),dtype) for scalars in zip(*(x.flat for x in arrays))]
    return np.asarray(output,dtype=np.float64).reshape(arrays[0].shape)


def matmul(left,right,*,dtype):
    a,b=cast(left,dtype),cast(right,dtype)
    if a.ndim<2 or b.ndim<2 or a.shape[-1]!=b.shape[-2]:
        raise ValueError('oracle: rank>=2 matmul with matching K required')
    batch=np.broadcast_shapes(a.shape[:-2],b.shape[:-2])
    a=np.broadcast_to(a,batch+a.shape[-2:]);b=np.broadcast_to(b,batch+b.shape[-2:])
    result=np.empty(batch+(a.shape[-2],b.shape[-1]))
    for index in np.ndindex(batch or ()):
        for m in range(a.shape[-2]):
            for n in range(b.shape[-1]):
                result[index+(m,n)]=ieee_rne(dot32(a[index][m].tolist(),b[index][:,n].tolist(),fused=True),dtype)
    return result


def reduce_mean32(value,axes):
    x=np.asarray(value,dtype=np.float64)
    axes=tuple(sorted(a % x.ndim for a in axes))
    kept=tuple(i for i in range(x.ndim) if i not in axes)
    matrix=x.transpose(kept+axes).reshape(-1,math.prod(x.shape[i] for i in axes))
    rows=[]
    for row in matrix:
        total=0.0
        for v in row:total=ieee_rne(Fraction(total)+Fraction(float(v)),'fp32')
        rows.append(ieee_rne(Fraction(total)/len(row),'fp32'))
    shape=tuple(1 if i in axes else x.shape[i] for i in range(x.ndim))
    return np.asarray(rows).reshape(shape)


def layer_norm(x,scale,bias=None,*,dtype,axis=-1,epsilon=1e-5,stash_type=1):
    if stash_type!=1:raise ValueError('oracle: only explicit FLOAT stash_type=1 prepared')
    x=cast(x,dtype);scale=cast(scale,dtype)
    bias=np.zeros_like(scale) if bias is None else cast(bias,dtype)
    if x.ndim==0 or not -x.ndim<=axis<x.ndim or not math.isfinite(epsilon) or epsilon<0:
        raise ValueError('oracle: invalid LayerNormalization axis/epsilon')
    axis%=x.ndim;axes=tuple(range(axis,x.ndim))
    if scale.shape!=x.shape[axis:] or bias.shape!=x.shape[axis:]:
        raise ValueError('oracle: Scale/B must have the complete normalization suffix shape')
    mean=reduce_mean32(x,axes)
    centered=elementwise('sub',x,mean,dtype='fp32')
    variance=reduce_mean32(elementwise('mul',centered,centered,dtype='fp32'),axes)
    std=elementwise('sqrt',elementwise('add',variance,cast(epsilon,'fp32'),dtype='fp32'),dtype='fp32')
    inv=elementwise('div',1,std,dtype='fp32')
    normalized=cast(elementwise('mul',centered,inv,dtype='fp32'),dtype)
    scaled=elementwise('mul',normalized,scale,dtype=dtype)
    y=elementwise('add',scaled,bias,dtype=dtype)
    return dict(Y=y,Mean=mean,InvStdDev=inv,Normalized=normalized,NormalizedScaled=scaled)


def gelu_erf_decomposition(x,*,dtype):
    """Explicit opset17 Div/Erf/Add/Mul/Mul export path, every output stored."""
    x=cast(x,dtype)
    d=elementwise('div',x,cast(math.sqrt(2),dtype),dtype=dtype)
    e=elementwise('erf',d,dtype=dtype)
    a=elementwise('add',e,cast(1,dtype),dtype=dtype)
    m=elementwise('mul',x,a,dtype=dtype)
    return elementwise('mul',m,cast(.5,dtype),dtype=dtype)


def numeric_error(actual,expected):
    a,b=np.asarray(actual,dtype=np.float64),np.asarray(expected,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('oracle: shape mismatch or nonfinite comparison')
    d=np.abs(a-b); nonzero=np.abs(b)>0
    return dict(shape=list(a.shape),max_absolute_error=float(d.max(initial=0)),
                max_relative_error=None if not nonzero.any() else float((d[nonzero]/np.abs(b[nonzero])).max()),
                zero_reference_count=int((~nonzero).sum()))
