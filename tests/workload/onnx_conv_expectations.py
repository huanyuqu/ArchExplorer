"""Independent NCHW Conv indexing and ordered FP32 FMA, with explicit storage.

Chosen low-precision contract: input/weight storage, FP32 ordered FMA, FP32
bias addition, output storage. Does not import product lowering or a framework.
"""
import numpy as np
from precision_transformer_expectations import dot32,ieee_rne
from typed_onnx_expectations import cast


def conv_geometry(input_shape,weight_shape,*,strides=(1,1),pads=(0,0,0,0),dilations=(1,1),group=1,auto_pad='NOTSET'):
    shapes=[*input_shape,*weight_shape,*strides,*pads,*dilations,group]
    if any(type(v) is not int for v in shapes):raise ValueError('oracle Conv dimensions must be exact integers')
    if len(input_shape)!=4 or len(weight_shape)!=4 or len(strides)!=2 or len(pads)!=4 or len(dilations)!=2:
        raise ValueError('oracle Conv requires NCHW two-dimensional convolution')
    n,c,h,w=input_shape;m,cg,kh,kw=weight_shape
    if min(n,c,h,w,m,cg,kh,kw,*strides,*dilations,group)<=0 or min(pads)<0:
        raise ValueError('oracle Conv requires positive dimensions/strides/dilations and nonnegative pads')
    if c!=cg*group or m%group:raise ValueError('oracle Conv channels/group differ')
    effective=[d*(k-1)+1 for d,k in zip(dilations,(kh,kw))]
    actual=list(pads)
    if auto_pad not in ('NOTSET','VALID','SAME_UPPER','SAME_LOWER'):raise ValueError('oracle unsupported Conv auto_pad')
    if auto_pad!='NOTSET':
        if any(pads):raise ValueError('oracle Conv auto_pad conflicts with explicit pads')
        actual=[0]*4
        if auto_pad.startswith('SAME'):
            for i,(size,stride,kernel) in enumerate(zip((h,w),strides,effective)):
                total=max(0,((size+stride-1)//stride-1)*stride+kernel-size)
                before=total//2 if auto_pad=='SAME_UPPER' else (total+1)//2
                actual[i]=before;actual[i+2]=total-before
    oh=(h+actual[0]+actual[2]-effective[0])//strides[0]+1
    ow=(w+actual[1]+actual[3]-effective[1])//strides[1]+1
    if min(oh,ow)<=0:raise ValueError('oracle Conv output dimension is nonpositive')
    return dict(output_shape=[n,m,oh,ow],pads=actual,group=group,kernel=[kh,kw],
                strides=list(strides),dilations=list(dilations),ordered_fmas=n*m*oh*ow*cg*kh*kw)


def conv2d(data,weight,bias=None,*,dtype='fp32',**options):
    x=cast(data,dtype);weights=cast(weight,dtype)
    geometry=conv_geometry(list(x.shape),list(weights.shape),**options)
    n,m,oh,ow=geometry['output_shape'];group=geometry['group'];cg=weights.shape[1]
    b=None if bias is None else cast(bias,dtype)
    if b is not None and b.shape!=(m,):raise ValueError('oracle Conv bias shape differs')
    out=np.empty((n,m,oh,ow),dtype=np.float64)
    for batch,channel,row,col in np.ndindex(out.shape):
        left=[];right=[];base=(channel//(m//group))*cg
        for local,ky,kx in np.ndindex(weights.shape[1:]):
            iy=row*geometry['strides'][0]-geometry['pads'][0]+ky*geometry['dilations'][0]
            ix=col*geometry['strides'][1]-geometry['pads'][1]+kx*geometry['dilations'][1]
            left.append(float(x[batch,base+local,iy,ix]) if 0<=iy<x.shape[2] and 0<=ix<x.shape[3] else 0.)
            right.append(float(weights[channel,local,ky,kx]))
        value=dot32(left,right,fused=True)
        if b is not None:value=ieee_rne(value+float(b[channel]),'fp32')
        out[batch,channel,row,col]=ieee_rne(value,dtype)
    if not np.isfinite(out).all():raise ValueError('oracle Conv stored result is nonfinite')
    return out
