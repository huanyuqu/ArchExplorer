"""Exact integer/boolean storage and explicit Cast boundaries shared by frontends."""
import numpy as np

INTEGER_DTYPES=('int32','int64')
TYPED_DTYPES=('fp32','fp16','bf16',*INTEGER_DTYPES,'bool')


def exact_storage(value,dtype,*,shape=None):
    """Validate integer leaves before conversion; never pass through a float."""
    from .extensions import _record_source
    _record_source(exact_storage)
    if dtype not in (*INTEGER_DTYPES,'bool'):raise ValueError('exact storage requires int32/int64/bool')
    width=32 if dtype=='int32' else 64
    low,high=-(1<<(width-1)),(1<<(width-1))-1
    def valid(item):
        if isinstance(item,(list,tuple)):
            return all(valid(x) for x in item)
        if isinstance(item,np.ndarray):
            if dtype=='bool':return item.dtype.kind=='b'
            if item.dtype.kind not in 'iu':return False
            if not item.size:return True
            return low<=int(item.min()) and int(item.max())<=high
        if dtype=='bool':return isinstance(item,(bool,np.bool_))
        return (isinstance(item,(int,np.integer)) and not isinstance(item,(bool,np.bool_))
                and low<=int(item)<=high)
    if not valid(value):raise ValueError('exact '+dtype+' storage requires in-range integer leaves' if dtype!='bool' else 'bool storage requires boolean leaves')
    result=np.array(value,dtype=dtype,order='C',copy=True)
    if shape is not None and result.shape!=tuple(shape):raise ValueError('exact '+dtype+' storage shape differs')
    return result


def cast_storage(value,dtype):
    """Only an explicit Cast may truncate a finite float or convert bool/integer."""
    from .extensions import _record_source
    _record_source(cast_storage)
    if dtype not in TYPED_DTYPES:raise ValueError('unsupported Cast storage dtype')
    array=np.asarray(value)
    if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():raise ValueError('Cast requires finite typed values')
    if dtype in ('fp32','fp16','bf16'):
        from .precision import round_storage
        return round_storage(array,dtype)
    if dtype=='bool':return np.not_equal(array,0)
    if array.dtype.kind=='f':
        bits=32 if dtype=='int32' else 64
        truncated=np.trunc(array)
        if np.any(truncated<-(1<<(bits-1))) or np.any(truncated>=(1<<(bits-1))):
            raise ValueError('Cast integer storage range differs')
        return np.array(truncated,dtype=dtype,order='C',copy=True)
    if array.dtype.kind=='b':return np.array(array,dtype=dtype,order='C',copy=True)
    return exact_storage(array,dtype)


def cast_conversions(source,target,count):
    """Declared scalar issue counts, retaining the existing floating Cast policy."""
    if source not in TYPED_DTYPES or target not in TYPED_DTYPES:raise ValueError('unsupported Cast storage dtype')
    if source==target:return 0
    floats=('fp32','fp16','bf16')
    if source in floats and target in floats:return count*((source!='fp32')+(target!='fp32'))
    return count*(1+(source in floats and source!='fp32')+(target in floats and target!='fp32'))
