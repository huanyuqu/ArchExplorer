"""Materialized NCHW patch extraction shared by all graph frontends."""
import itertools
import math

KINDS=('im2col',)


def geometry(source,kernel,strides,pads,dilations):
    if len(source)!=4:raise ValueError('im2col requires NCHW rank-four source')
    for field,count,label,zero in ((kernel,2,'kernel',False),(strides,2,'strides',False),(pads,4,'pads',True),(dilations,2,'dilations',False)):
        if not isinstance(field,(tuple,list)) or len(field)!=count or any(type(n) is not int or n<(0 if zero else 1) for n in field):
            raise ValueError('im2col '+label+' dimensions differ')
    n,c,h,w=source;kh,kw=kernel;sh,sw=strides;top,left,bottom,right=pads;dh,dw=dilations
    oh=(h+top+bottom-(dh*(kh-1)+1))//sh+1
    ow=(w+left+right-(dw*(kw-1)+1))//sw+1
    if min(oh,ow)<1:raise ValueError('im2col requires positive output spatial dimensions')
    return (n,oh*ow,c*kh*kw),(oh,ow)


def validate(op,tensors):
    if op.op not in KINDS or len(op.inputs)!=1 or len(op.outputs)!=1 or set(op.attrs)!={'shape','kernel','strides','pads','dilations'}:
        raise ValueError('im2col fields/arity differ')
    x=tensors[op.inputs[0]];y=tensors[op.outputs[0]]
    if x.dtype not in ('fp32','fp16','bf16') or y.dtype!=x.dtype or y.alias_of is not None or y.strides is not None or x.kind=='kv' or y.kind=='kv':
        raise ValueError('im2col requires floating storage and an independent output')
    shape,_=geometry(x.shape,op.attrs['kernel'],op.attrs['strides'],op.attrs['pads'],op.attrs['dilations'])
    if tuple(op.attrs['shape'])!=shape or y.shape!=shape:raise ValueError('im2col declared output shape differs')


def indices(op,source_shape):
    """Yield output indices and either a real source index or explicit padding."""
    shape,(oh,ow)=geometry(source_shape,op.attrs['kernel'],op.attrs['strides'],op.attrs['pads'],op.attrs['dilations'])
    n,c,h,w=source_shape;kh,kw=op.attrs['kernel'];sh,sw=op.attrs['strides'];top,left,_,_=op.attrs['pads'];dh,dw=op.attrs['dilations']
    for b,row,col,channel,i,j in itertools.product(range(n),range(oh),range(ow),range(c),range(kh),range(kw)):
        y=row*sh-top+i*dh;x=col*sw-left+j*dw
        yield (b,row*ow+col,channel*kh*kw+i*kw+j),((b,channel,y,x) if 0<=y<h and 0<=x<w else None)


def reference(op,values):
    import numpy as np
    from .extensions import _record_source
    _record_source(indices);_record_source(geometry)
    source=values[0];out=np.empty(op.attrs['shape'],dtype=source.dtype)
    for target,index in indices(op,source.shape):out[target]=source[index] if index is not None else 0
    return out


def access(op,tensors):
    from arch_explorer.components.base import dtype_bytes
    from .extensions import _record_source
    _record_source(validate);_record_source(geometry);validate(op,tensors)
    x=tensors[op.inputs[0]];y=tensors[op.outputs[0]]
    _,(oh,ow)=geometry(x.shape,op.attrs['kernel'],op.attrs['strides'],op.attrs['pads'],op.attrs['dilations'])
    _,c,h,w=x.shape;kh,kw=op.attrs['kernel'];sh,sw=op.attrs['strides'];top,left,_,_=op.attrs['pads'];dh,dw=op.attrs['dilations']
    valid_y=sum(0<=row*sh-top+i*dh<h for row,i in itertools.product(range(oh),range(kh)))
    valid_x=sum(0<=col*sw-left+j*dw<w for col,j in itertools.product(range(ow),range(kw)))
    reads=x.shape[0]*c*valid_y*valid_x*dtype_bytes(x.dtype);elements=math.prod(y.shape);simt=3*elements
    return dict(operations=simt,read_bytes=reads,write_bytes=y.bytes,read_by_tensor={x.name:reads},boundary='component_io',
        simt_instructions=simt,sfu_instructions=0,conversion_instructions=0,input_dtypes=[x.dtype],output_dtype=y.dtype,
        accumulation='bit-preserving patch copy; explicit zero padding',cost='synthetic two bounds tests and one output-address issue per patch element; no calibrated ISA claim')


def access_indices(op,tensors,values):
    for _,index in indices(op,tensors[op.inputs[0]].shape):
        if index is not None:yield op.inputs[0],index


def install():
    from .extensions import OperatorDefinition,register_operator
    register_operator(OperatorDefinition('im2col',validate,reference=reference))
