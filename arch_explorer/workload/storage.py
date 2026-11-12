"""One element-address contract for independent allocations and readonly views."""
from dataclasses import dataclass
from collections import ChainMap
import itertools
import math


def contiguous_strides(shape):
    return tuple(math.prod(shape[i+1:]) for i in range(len(shape)))


@dataclass(frozen=True)
class StorageLayout:
    storage: str
    shape: tuple
    dtype: str
    strides: tuple
    offset: int
    allocation_bytes: int

    def to_dict(self):
        return dict(storage=self.storage,shape=list(self.shape),dtype=self.dtype,
            strides=list(self.strides),offset=self.offset,allocation_bytes=self.allocation_bytes,
            units='elements',readonly=True)


def resolve_layout(tensors, name, _active=None):
    from .extensions import _record_source
    _record_source(resolve_layout)
    active=set() if _active is None else _active
    if name in active:raise ValueError('tensor storage alias cycle')
    if name not in tensors:raise ValueError('dangling tensor storage source: '+name)
    t=tensors[name]
    if (t.dtype not in ('fp32','fp16','bf16','int32','int64','bool') or t.kind=='kv' or not 1<=len(t.shape)<=4):
        raise ValueError('readonly storage requires floating/int32 non-KV rank 1..4')
    if type(t.offset) is not int or t.offset<0:raise ValueError('offset must be a nonnegative integer in elements')
    if t.alias_of is None:
        if t.offset or t.strides is not None:raise ValueError('independent storage must have zero offset and implicit contiguous strides')
        return StorageLayout(name,t.shape,t.dtype,contiguous_strides(t.shape),0,t.bytes)
    active.add(name)
    try:source=resolve_layout(tensors,t.alias_of,active)
    finally:active.remove(name)
    if source.dtype!=t.dtype:raise ValueError('readonly view cannot reinterpret dtype')
    strides=t.strides or contiguous_strides(t.shape)
    if len(strides)!=len(t.shape) or any(type(s) is not int or s<=0 for s in strides):
        raise ValueError('readonly view strides must be positive integers')
    offset=source.offset+t.offset
    from arch_explorer.components.base import dtype_bytes
    if offset+sum((n-1)*s for n,s in zip(t.shape,strides))>=source.allocation_bytes//dtype_bytes(t.dtype):
        raise ValueError('readonly view exceeds its root allocation')
    return StorageLayout(source.storage,t.shape,t.dtype,tuple(strides),offset,source.allocation_bytes)


def element_offset(layout, index):
    if (not isinstance(index,(tuple,list)) or len(index)!=len(layout.shape)
            or any(type(i) is not int or not 0<=i<n for i,n in zip(index,layout.shape))):
        raise ValueError('logical tensor index out of bounds')
    return layout.offset+sum(i*s for i,s in zip(index,layout.strides))


def reshape_strides(shape, strides, target):
    """Split/coalesce contiguous logical chunks without ever making a copy."""
    if math.prod(shape)!=math.prod(target):raise ValueError('view reshape element count differs')
    # Unit dimensions do not interrupt a contiguous chunk.
    dims=[(n,s) for n,s in zip(shape,strides) if n!=1]
    if not dims:return contiguous_strides(target)
    chunks=[];size,base=dims[-1]
    for n,s in reversed(dims[:-1]):
        if s==base*size:size*=n
        else:chunks.append((size,base));size,base=n,s
    chunks.append((size,base))
    result=[0]*len(target);j=len(target)-1
    for size,base in chunks:
        used=1
        while j>=0 and (used<size or target[j]==1):
            result[j]=base*used;used*=target[j];j-=1
        if used!=size:raise ValueError('reshape cannot share this noncontiguous logical traversal')
    if j>=0:raise ValueError('reshape crosses a noncontiguous storage chunk')
    return tuple(result)


def view_tensor(tensors, source, output, kind, attrs):
    from .operators import Tensor
    from .graph import shape
    from .extensions import _record_source
    _record_source(view_tensor);_record_source(reshape_strides)
    src=tensors[source];layout=resolve_layout(tensors,source)
    target=shape(attrs['shape']);offset=0
    if not 1<=len(target)<=4:raise ValueError('view target rank must be 1..4')
    if kind=='transpose':
        if set(attrs)!={'shape','axes'}:raise ValueError('transpose view requires shape/axes')
        axes=attrs['axes']
        if (not isinstance(axes,(tuple,list)) or any(type(a) is not int for a in axes)
                or sorted(axes)!=list(range(len(src.shape))) or target!=tuple(src.shape[a] for a in axes)):
            raise ValueError('view transpose axes/shape differ')
        strides=tuple(layout.strides[a] for a in axes)
    elif kind=='reshape':
        if set(attrs)!={'shape'}:raise ValueError('reshape view requires shape')
        strides=reshape_strides(src.shape,layout.strides,target)
    elif kind=='slice':
        if set(attrs)!={'shape','starts','stops','steps'}:raise ValueError('slice requires shape/starts/stops/steps')
        rank=len(src.shape);starts,stops,steps=(attrs[k] for k in ('starts','stops','steps'))
        if any(not isinstance(x,(tuple,list)) or len(x)!=rank for x in (starts,stops,steps)):
            raise ValueError('slice requires one explicit start/stop/step per source dimension')
        if any(type(a) is not int or type(b) is not int or type(s) is not int or not 0<=a<b<=n or s<=0
               for n,a,b,s in zip(src.shape,starts,stops,steps)):
            raise ValueError('slice must be nonempty, in bounds and have positive steps')
        if target!=tuple((b-a+s-1)//s for a,b,s in zip(starts,stops,steps)):
            raise ValueError('slice declared shape differs')
        strides=tuple(s*v for s,v in zip(layout.strides,steps))
        offset=sum(a*s for a,s in zip(starts,layout.strides))
    else:raise ValueError('unsupported readonly view transform')
    result=Tensor(output,target,src.dtype,src.kind,source,strides,offset)
    resolve_layout(ChainMap({output:result},tensors),output)
    return result


def validate_view(op,tensors):
    if len(op.inputs)!=1 or len(op.outputs)!=1 or 'transform' not in op.attrs:
        raise ValueError('storage.view requires one source/output and explicit transform')
    a={k:v for k,v in op.attrs.items() if k!='transform'}
    expected=view_tensor(tensors,op.inputs[0],op.outputs[0],op.attrs['transform'],a)
    if expected!=tensors[op.outputs[0]]:raise ValueError('view descriptor differs from its declared transform')


def reference_view(op,inputs):
    value=inputs[0];a=op.attrs;kind=a['transform']
    if kind=='transpose':return value.transpose(a['axes'])
    if kind=='reshape':
        import numpy as np
        result=value.reshape(a['shape'])
        if not np.shares_memory(value,result):raise ValueError('view reference attempted an implicit reshape copy')
        return result
    if kind=='slice':return value[tuple(slice(a,b,s) for a,b,s in zip(a['starts'],a['stops'],a['steps']))]
    raise ValueError('unsupported view transform')


def readonly_view(graph,name,source,*,transform,shape,**attrs):
    """Append an explicit readonly view to a public Workload and return its name."""
    if name in graph.tensors:raise ValueError('duplicate view tensor name')
    if any(op.name==name+'.view' for op in graph.operators):raise ValueError('duplicate view operator name')
    tensor=view_tensor(graph.tensors,source,name,transform,dict(shape=shape,**attrs))
    graph.tensors[name]=tensor
    graph.add(name+'.view','storage.view',[source],[name],transform=transform,shape=list(shape),**attrs)
    return name


def logical_accesses(graph,operator,*,limit=100000,values=None):
    """Bounded audit of actual logical loads/stores, never a scheduler or oracle."""
    from arch_explorer.components.base import dtype_bytes
    if type(limit) is not int or not 1<=limit<=1000000:raise ValueError('access audit limit must be 1..1000000')
    op=next((o for o in graph.operators if o.name==operator),None)
    if op is None:raise ValueError('unknown access-audit operator')
    from .tensor_ops import KINDS
    from .indexing import KINDS as INDEX_KINDS
    from .spatial_ops import KINDS as SPATIAL_KINDS
    from .array_ops import KINDS as ARRAY_KINDS
    from .block_ops import KINDS as BLOCK_KINDS
    if op.op not in (*KINDS,*INDEX_KINDS,*SPATIAL_KINDS,*ARRAY_KINDS,*BLOCK_KINDS,'storage.view','compiled.matmul','compiled.assemble'):
        raise NotImplementedError('logical accesses require a declared builtin indexing contract')
    layouts={n:resolve_layout(graph.tensors,n) for n in (*op.inputs,*op.outputs)}
    rows=[]
    def flat_index(shape,index):
        values=[]
        for stride in contiguous_strides(shape):values.append(index//stride);index%=stride
        return tuple(values)
    def access(direction,name,index):
        if len(rows)>=limit:raise ValueError('logical access audit bound exceeded')
        l=layouts[name];width=dtype_bytes(l.dtype);offset=element_offset(l,index)
        rows.append(dict(direction=direction,tensor=name,storage=l.storage,logical_index=list(index),
            element_offset=offset,byte_offset=offset*width,bytes=width))
    def flat(direction,name,index):access(direction,name,flat_index(layouts[name].shape,index))
    if op.op=='storage.view':return rows
    a=op.attrs;output=op.outputs[0]
    if op.op in (*INDEX_KINDS,*SPATIAL_KINDS,*ARRAY_KINDS,*BLOCK_KINDS):
        if op.op in INDEX_KINDS:
            from .indexing import access_indices
        elif op.op in SPATIAL_KINDS:
            from .spatial_ops import access_indices
        elif op.op in ARRAY_KINDS:
            from .array_ops import access_indices
        else:
            from .block_ops import access_indices
        from .extensions import _record_source
        _record_source(access_indices)
        for n,index in access_indices(op,graph.tensors,graph.meta.get('inputs',{}) if values is None else values):
            access('read',n,index)
    elif op.op=='compiled.matmul':
        for b,r,c,k in itertools.product(range(a['batch']),range(a['rows']),range(a['cols']),range(a['k'])):
            flat('read',op.inputs[0],(b*a['m']+r+a['row'])*a['k']+k)
            flat('read',op.inputs[1],((0 if a['shared_b'] else b)*a['k']+k)*a['n']+c+a['col'])
    elif op.op=='compiled.assemble':
        for n in op.inputs:
            for i in range(math.prod(layouts[n].shape)):flat('read',n,i)
    elif op.op in ('reduce_sum','reduce_max'):
        for i in range(math.prod(layouts[op.inputs[0]].shape)):flat('read',op.inputs[0],i)
    elif op.op=='concat':
        axis=a['axis']%len(layouts[output].shape)
        for index in itertools.product(*(range(n) for n in layouts[output].shape)):
            offset=index[axis]
            for n in op.inputs:
                if offset<layouts[n].shape[axis]:
                    selected=list(index);selected[axis]=offset
                    access('read',n,selected)
                    break
                offset-=layouts[n].shape[axis]
    elif op.op=='patchify':
        source=op.inputs[0];p=a['patch'];_,c,h,w=layouts[source].shape
        for b,token,inner in itertools.product(*(range(n) for n in layouts[output].shape)):
            channel,local=divmod(inner,p*p)
            py,px=divmod(token,w//p);dy,dx=divmod(local,p)
            access('read',source,(b,channel,py*p+dy,px*p+dx))
    else:
        for index in itertools.product(*(range(n) for n in layouts[output].shape)):
            pos=sum(i*s for i,s in zip(index,contiguous_strides(layouts[output].shape)))
            for n in op.inputs:
                if op.op=='transpose':
                    inverse=[0]*len(index)
                    for i,axis in enumerate(a['axes']):inverse[axis]=index[i]
                    selected=tuple(inverse)
                elif op.op=='slice':selected=tuple(start+i*step for start,i,step in zip(a['starts'],index,a['steps']))
                elif op.op=='broadcast':
                    aligned=index[len(index)-len(layouts[n].shape):]
                    selected=tuple(0 if dim==1 else i for i,dim in zip(aligned,layouts[n].shape))
                elif op.op in ('affine_norm','bias') and n!=op.inputs[0]:selected=(index[-1],)
                elif op.op=='scale_tensor' and n==op.inputs[1]:selected=(0,)
                else:selected=flat_index(layouts[n].shape,pos)
                access('read',n,selected)
    for i in range(math.prod(layouts[output].shape)):flat('write',output,i)
    return rows


def install():
    from .extensions import OperatorDefinition,register_operator
    global _DEFINITION
    if _DEFINITION is None:_DEFINITION=OperatorDefinition('storage.view',validate=validate_view,reference=reference_view)
    register_operator(_DEFINITION)


_DEFINITION=None
