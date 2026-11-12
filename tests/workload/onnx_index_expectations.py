"""Independent opset17 integer/index/shape oracle. No product or ONNX imports.

All index/shape values stay signed integers, including values above 2**53.
These functions describe logical tensors, not physical placement or traffic.
"""
import numpy as np


def integer_array(value):
    if isinstance(value,(list,tuple)) and not value:
        return np.asarray(value,dtype=np.int64)
    array=np.asarray(value)
    if array.dtype not in (np.dtype('int32'),np.dtype('int64')):
        raise ValueError('oracle index/shape requires signed int32/int64 without float or bool coercion')
    return array


def vector(value):
    array=integer_array(value)
    if array.ndim!=1:raise ValueError('oracle index/shape parameter must be rank one')
    return [int(v) for v in array]


def axis(value,rank):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not -rank<=value<rank:
        raise ValueError('oracle axis outside rank')
    return int(value)%rank


def first_argmax(data,axis_value=0,keepdims=1,select_last_index=0):
    a=np.asarray(data);ax=axis(axis_value,a.ndim)
    if keepdims not in (0,1) or select_last_index not in (0,1) or a.shape[ax]==0:
        raise ValueError('oracle invalid ArgMax flags or empty reduction')
    if a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ValueError('oracle ArgMax requires finite numeric values')
    output=np.argmax(np.flip(a,ax) if select_last_index else a,axis=ax).astype(np.int64)
    if select_last_index:output=a.shape[ax]-1-output
    return np.expand_dims(output,ax) if keepdims else output


def gather(data,indices,axis_value=0):
    a=np.asarray(data);index=integer_array(indices);ax=axis(axis_value,a.ndim)
    if np.any(index < -a.shape[ax]) or np.any(index >= a.shape[ax]):
        raise ValueError('oracle Gather index outside axis bounds')
    return np.take(a,index,axis=ax)


def gather_elements(data,indices,axis_value=0):
    a=np.asarray(data);index=integer_array(indices);ax=axis(axis_value,a.ndim)
    if index.ndim!=a.ndim or any(index.shape[i]>a.shape[i] for i in range(a.ndim) if i!=ax):
        raise ValueError('oracle GatherElements shape differs')
    output=np.empty(index.shape,dtype=a.dtype)
    for location in np.ndindex(index.shape):
        selected=int(index[location]);limit=a.shape[ax]
        if not -limit<=selected<limit:raise ValueError('oracle GatherElements index outside axis bounds')
        source=list(location);source[ax]=selected%limit;output[location]=a[tuple(source)]
    return output


def shape_of(data,start=0,end=None):
    rank=np.asarray(data).ndim
    for v in (start,end):
        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,np.integer))):
            raise ValueError('oracle Shape bounds must be integer')
    return np.asarray(np.asarray(data).shape,dtype=np.int64)[slice(start,end)]


def concat(values,axis_value=0):
    arrays=[np.asarray(v) for v in values]
    if not arrays:raise ValueError('oracle Concat requires inputs')
    ax=axis(axis_value,arrays[0].ndim)
    if any(a.dtype!=arrays[0].dtype or a.ndim!=arrays[0].ndim or
           any(a.shape[i]!=arrays[0].shape[i] for i in range(a.ndim) if i!=ax) for a in arrays):
        raise ValueError('oracle Concat shape/dtype differs')
    return np.concatenate(arrays,axis=ax)


def slice_tensor(data,starts,ends,axes=None,steps=None):
    a=np.asarray(data);starts=vector(starts);ends=vector(ends)
    axes=list(range(len(starts))) if axes is None else vector(axes)
    steps=[1]*len(starts) if steps is None else vector(steps)
    if not len(starts)==len(ends)==len(axes)==len(steps) or any(s==0 for s in steps):
        raise ValueError('oracle Slice lengths or zero step')
    axes=[axis(i,a.ndim) for i in axes]
    if len(axes)!=len(set(axes)):raise ValueError('oracle Slice duplicate axis')
    selection=[slice(None)]*a.ndim
    for ax,start,end,step in zip(axes,starts,ends,steps):selection[ax]=slice(start,end,step)
    return a[tuple(selection)].copy()


def expand(data,shape):
    a=np.asarray(data);wanted=tuple(vector(shape))
    if any(v<0 for v in wanted):raise ValueError('oracle Expand dimensions must be nonnegative')
    return np.broadcast_to(a,np.broadcast_shapes(a.shape,wanted)).copy()


def unsqueeze(data,axes):
    a=np.asarray(data);axes=vector(axes);rank=a.ndim+len(axes)
    normalized=[axis(i,rank) for i in axes]
    if len(normalized)!=len(set(normalized)):raise ValueError('oracle Unsqueeze duplicate axis')
    shape=list(a.shape)
    for i in sorted(normalized):shape.insert(i,1)
    return a.reshape(shape)


def squeeze(data,axes=None):
    a=np.asarray(data)
    if axes is None:return np.squeeze(a)
    normalized=[axis(i,a.ndim) for i in vector(axes)]
    if len(normalized)!=len(set(normalized)) or any(a.shape[i]!=1 for i in normalized):
        raise ValueError('oracle Squeeze axes must be distinct singleton dimensions')
    return np.squeeze(a,axis=tuple(normalized))


def integer_binary(op,left,right):
    """Exact checked int64 arithmetic for shape work, division towards zero.

    Overflow is outside this reference's admitted domain, never float-rounded.
    """
    a,b=np.broadcast_arrays(integer_array(left),integer_array(right));output=[]
    for av,bv in zip(a.flat,b.flat):
        av,bv=int(av),int(bv)
        if op=='Add':value=av+bv
        elif op=='Sub':value=av-bv
        elif op=='Mul':value=av*bv
        elif op=='Div':
            if bv==0:raise ValueError('oracle integer division by zero')
            value=(abs(av)//abs(bv))*(-1 if (av<0)!=(bv<0) else 1)
        else:raise ValueError('oracle unsupported integer arithmetic')
        if not -(2**63)<=value<2**63:raise ValueError('oracle integer result outside checked int64 domain')
        output.append(value)
    return np.asarray(output,dtype=np.int64).reshape(a.shape)
