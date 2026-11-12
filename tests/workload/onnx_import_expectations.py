"""Independent closed-form ONNX expectations, without ONNX or product imports.

NumPy FP64 dot/pointwise stages are rounded to FP32 at every graph value.
This bounds BLAS/FMA accumulation differences; it does not require bit equality.
"""
import math
import numpy as np

ABS_TOL=4e-6
REL_TOL=2e-5
POLICY='FP64 closed-form operation -> FP32 stored value at each node; abs4e-6 + rel2e-5, bounded K<=4 and finite small roots'

def stored(value):return np.asarray(value,dtype=np.float32)
def dot(left,right):return stored(np.matmul(np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)))

def gemm(a,b,c=None,*,alpha=1,beta=1,transA=0,transB=0):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2:raise ValueError('Gemm expects rank two A/B')
    if transA:a=a.T
    if transB:b=b.T
    if a.shape[1]!=b.shape[0]:raise ValueError('Gemm contraction differs')
    product=alpha*(a@b)
    if c is not None:
        c=np.asarray(c,dtype=np.float64)
        if c.ndim>2 or np.broadcast_shapes(c.shape,product.shape)!=product.shape:
            raise ValueError('Gemm C must unidirectionally broadcast to M,N')
        product=product+beta*c
    return stored(product)

def softmax(value,axis=-1):
    value=np.asarray(value,dtype=np.float64)
    if not -value.ndim<=axis<value.ndim:raise ValueError('softmax axis outside rank')
    shifted=value-value.max(axis=axis,keepdims=True)
    exps=np.exp(shifted)
    return stored(exps/exps.sum(axis=axis,keepdims=True))

def reshape_shape(input_shape,requested,*,allowzero=0):
    if allowzero not in (0,1):raise ValueError('allowzero must be zero or one')
    if any(type(x) is not int or x<0 for x in input_shape):raise ValueError('invalid input dimensions')
    if any(type(x) is not int or x< -1 for x in requested):raise ValueError('invalid target dimension')
    shape=list(requested)
    if shape.count(-1)>1:raise ValueError('at most one inferred dimension')
    if allowzero and 0 in shape and -1 in shape:raise ValueError('allowzero cannot mix zero and inferred dimensions')
    for i,size in enumerate(shape):
        if size==0 and not allowzero:
            if i>=len(input_shape):raise ValueError('copied axis outside input rank')
            shape[i]=input_shape[i]
    source=math.prod(input_shape);known=math.prod(x for x in shape if x!=-1)
    if -1 in shape:
        if known==0 or source%known:raise ValueError('cannot infer integral dimension')
        shape[shape.index(-1)]=source//known
    elif source!=known:raise ValueError('reshape element count differs')
    return tuple(shape)

def mlp(roots):
    x,w,b,w2,b2=(roots[k] for k in ('X','W1','b1','W2','b2'))
    h=gemm(x,w,b,alpha=.5,beta=1.5,transB=1)
    a0=stored(np.maximum(h,0));a=a0.reshape(reshape_shape(a0.shape,[0,-1]))
    z=gemm(a,w2,b2,alpha=1.25,beta=-.5)
    return dict(H=h,A0=a0,shape=np.array([0,-1],np.int64),A=a,Z=z,Y=z.copy(),shared=stored(a.astype(np.float64)+b))

def attention(roots):
    x,wqk,wv,kb,scale,mask=(roots[k] for k in ('X','Wqk','Wv','kb','scale','mask'))
    q=dot(x,wqk);k0=dot(x,wqk);k=stored(k0.astype(np.float64)+kb)
    v=dot(x,wv);kt=k.transpose(0,2,1);scores=dot(q,kt)
    scaled=stored(scores.astype(np.float64)*scale);masked=stored(scaled.astype(np.float64)+mask)
    p=softmax(masked,-1);c=dot(p,v);t=c.transpose(1,0,2);r=t.reshape(3,6)
    return dict(Q=q,K0=k0,K=k,V=v,KT=kt,scores=scores,scaled=scaled,masked=masked,P=p,C=c,T=t,
                shape=np.array([3,6],np.int64),R=r,Y=r.copy())
