"""Static ONNX shape/control specialization, using common semantics for arithmetic.

Runtime root values never enter this folder. Only literal controls and Shape
metadata from the explicit specialization may be evaluated at compile time.
"""
import math


def axis(value,rank):
    if type(value) is not int or not -rank<=value<rank:raise ValueError('ONNX axis outside tensor rank')
    return value%rank


def reshape(source,target,allowzero=0):
    if type(allowzero) is not int or allowzero not in (0,1) or any(type(n) is not int or n<-1 for n in target) or target.count(-1)>1:
        raise ValueError('ONNX Reshape requires allowzero 0/1 and at most one inferred -1')
    if allowzero and 0 in target:raise ValueError('ONNX zero-sized Reshape is outside positive static domain')
    result=[]
    for j,n in enumerate(target):
        if n==0:
            if j>=len(source):raise ValueError('ONNX Reshape zero-copy dimension exceeds source rank')
            n=source[j]
        result.append(n)
    known=math.prod(n for n in result if n!=-1);total=math.prod(source)
    if -1 in result:
        if not known or total%known:raise ValueError('ONNX Reshape inferred dimension is not integral')
        result[result.index(-1)]=total//known
    if math.prod(result)!=total:raise ValueError('ONNX Reshape element count differs')
    return tuple(result)


def squeeze(source,axes=None):
    selected=[j for j,n in enumerate(source) if n==1] if axes is None else [axis(int(j),len(source)) for j in axes]
    if len(set(selected))!=len(selected) or any(source[j]!=1 for j in selected):raise ValueError('ONNX Squeeze requires distinct unit dimensions')
    return tuple(n for j,n in enumerate(source) if j not in selected)


def unsqueeze(source,axes):
    rank=len(source)+len(axes)
    selected=[axis(int(j),rank) for j in axes]
    if len(set(selected))!=len(selected):raise ValueError('ONNX Unsqueeze requires distinct axes')
    values=iter(source)
    return tuple(1 if j in selected else next(values) for j in range(rank))


def slices(source,starts,ends,axes=None,steps=None):
    axes=list(range(len(starts))) if axes is None else [axis(int(j),len(source)) for j in axes]
    steps=[1]*len(starts) if steps is None else steps
    if not len(starts)==len(ends)==len(axes)==len(steps) or len(set(axes))!=len(axes) or any(type(n) is not int or n==0 for n in steps):
        raise ValueError('ONNX Slice controls require distinct axes and nonzero steps')
    selection=[slice(None)]*len(source)
    for a,b,j,step in zip(starts,ends,axes,steps):selection[j]=slice(a,b,step)
    normalized=[s.indices(n) for s,n in zip(selection,source)]
    target=tuple(len(range(a,b,step)) for a,b,step in normalized)
    return selection,target,normalized


def fold(kind,attrs,names,shapes,types,constants):
    import numpy as np
    from .resource_budget import check_shape,limit
    from .extensions import _record_source,get_operator
    from .operators import Operator
    _record_source(fold)
    if kind=='Shape':
        rank=len(shapes[names[0]])
        def boundary(name,default):
            v=attrs.get(name,default)
            if type(v) is not int:raise ValueError('ONNX Shape bounds must be integers')
            return min(rank,max(0,v+rank if v<0 else v))
        return np.asarray(shapes[names[0]][boundary('start',0):boundary('end',rank)],dtype=np.int64),7
    if any(name not in constants for name in names):return None
    values=[constants[name] for name in names]
    if kind=='ConstantOfShape':
        from .onnx_import import _onnx
        onnx=_onnx();shape_value=values[0]
        if shape_value.dtype!=np.dtype('int64') or shape_value.ndim!=1:raise ValueError('ONNX ConstantOfShape requires a static int64 vector')
        target=check_shape(shape_value.tolist());count=math.prod(target)
        if count>limit('max_tensor_elements',1_000_000):raise ValueError('ONNX constant tensor element budget exceeded')
        literal=attrs.get('value');dtype=1
        if literal is None:value=np.asarray(0,dtype=np.float32)
        else:
            dtype=literal.data_type;value=onnx.numpy_helper.to_array(literal)
            if value.size!=1:raise ValueError('ONNX ConstantOfShape value must contain one element')
        from .onnx_resources import STORAGE
        if dtype not in STORAGE:raise ValueError('ONNX ConstantOfShape dtype unsupported')
        if dtype in (6,7,9):
            from .integer import exact_storage
            value=exact_storage(value,STORAGE[dtype])
        else:
            from .precision import round_storage
            value=round_storage(value,STORAGE[dtype])
        return np.broadcast_to(value.reshape(()),target),dtype
    if any(types[name] not in (6,7,9) for name in names):return None
    dtype=types[names[0]];x=values[0]
    if kind=='Identity':return x.copy(),dtype
    if kind=='Concat':return np.concatenate(values,axis=axis(attrs.get('axis',0),x.ndim)),dtype
    if kind=='Slice':
        if any(v.dtype!=np.dtype('int64') or v.ndim!=1 for v in values[1:]):raise ValueError('ONNX Slice controls must be static int64 vectors')
        selection,_,_=slices(x.shape,values[1].tolist(),values[2].tolist(),values[3].tolist() if len(values)>3 else None,values[4].tolist() if len(values)>4 else None)
        return x[tuple(selection)].copy(),dtype
    if kind=='Unsqueeze':return x.reshape(unsqueeze(x.shape,values[1].tolist())),dtype
    if kind=='Squeeze':return x.reshape(squeeze(x.shape,values[1].tolist() if len(values)>1 else None)),dtype
    if kind=='Reshape':return x.reshape(reshape(x.shape,values[1].tolist(),attrs.get('allowzero',0))),dtype
    if kind=='Transpose':return x.transpose(attrs.get('perm',tuple(reversed(range(x.ndim))))).copy(),dtype
    if kind=='Flatten':
        rank=x.ndim;a=attrs.get('axis',1)
        if type(a) is not int or not -rank<=a<=rank:raise ValueError('ONNX Flatten axis outside tensor rank')
        if a<0:a+=rank
        return x.reshape((math.prod(x.shape[:a]),math.prod(x.shape[a:]))),dtype
    if kind=='Expand':
        target=check_shape(values[1].tolist())
        if math.prod(target)>limit('max_tensor_elements',1_000_000):raise ValueError('ONNX constant tensor element budget exceeded')
        return np.broadcast_to(x,target).copy(),dtype
    if kind=='Range':
        if dtype not in (6,7) or any(v.ndim!=0 for v in values):raise ValueError('ONNX Range requires static integer scalars')
        start,end,step=(int(v) for v in values)
        if step==0:raise ValueError('ONNX Range delta cannot be zero')
        seq=range(start,end,step)
        try:count=len(seq)
        except OverflowError as error:raise ValueError('ONNX Range exceeds integer element budget') from error
        if count>limit('max_tensor_elements',1_000_000):raise ValueError('ONNX Range exceeds integer element budget')
        return np.asarray(list(seq),dtype=x.dtype),dtype
    if kind=='Cast':
        from .onnx_resources import STORAGE
        from .integer import cast_storage
        target=attrs.get('to')
        if target not in STORAGE:raise ValueError('ONNX Cast storage dtype unsupported')
        return cast_storage(x,STORAGE[target]),target
    semantic={'Add':'integer_add','Sub':'integer_sub','Mul':'integer_mul','Div':'integer_div','Equal':'equal','Greater':'greater','Less':'less','Where':'where','Gather':'gather','GatherElements':'gather_elements'}.get(kind)
    if semantic is None:return None
    if kind in ('Gather','GatherElements'):
        a=axis(attrs.get('axis',0),x.ndim)
        target=x.shape[:a]+values[1].shape+x.shape[a+1:] if kind=='Gather' else values[1].shape
        actual=values
        options=dict(shape=list(target or (1,)),axis=a)
    else:
        target=np.broadcast_shapes(*(v.shape for v in values));actual=[np.broadcast_to(v,target) for v in values]
        options=dict(shape=list(target or (1,)))
    function=get_operator(semantic).reference
    if function is None:raise ValueError('ONNX static control lacks common semantics: '+kind)
    _record_source(function)
    physical=[v.reshape(v.shape or (1,)) for v in actual]
    op=Operator('static.control',semantic,tuple('x'+str(j) for j in range(len(physical))),('out',),options)
    result=function(op,physical)
    if isinstance(result,dict):result=result['out']
    dtype=9 if kind in ('Equal','Greater','Less') else types[names[1]] if kind=='Where' else dtype
    return np.asarray(result).reshape(target),dtype
