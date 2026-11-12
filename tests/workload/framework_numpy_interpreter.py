"""Independent interpreter for exported standard nodes; extensions are opt-in.

Reads original ModelProto only. BF16 storage is decoded explicitly; NumPy
uint16 values are never interpreted as numeric BF16. Dtypes come from actual
original types and ONNX shape inference, without production reference helpers.
"""
from fractions import Fraction
import math
import numpy as np
from typed_onnx_expectations import cast, elementwise, matmul, layer_norm
from precision_transformer_expectations import ieee_rne, dot32

TYPES={1:'fp32',10:'fp16',16:'bf16',7:'int64',6:'int32',9:'bool'}


def decode_bf16(value):
    bits=np.asarray(value).view(np.uint16).astype(np.uint32)
    return np.left_shift(bits,np.uint32(16)).view(np.float32)


def tensor_array(tensor):
    import onnx
    if tensor.data_type==16:
        bits=np.frombuffer(tensor.raw_data,dtype='<u2') if tensor.raw_data else np.asarray(tensor.int32_data,dtype=np.uint16)
        return decode_bf16(bits).astype(np.float64).reshape(tensor.dims)
    return onnx.numpy_helper.to_array(tensor).copy()


def gemm(a,b,c,attrs,dtype):
    a,b=cast(a,dtype),cast(b,dtype)
    if attrs.get('transA',0):a=a.T
    if attrs.get('transB',0):b=b.T
    out=np.empty((a.shape[0],b.shape[1]))
    bias=np.broadcast_to(c,out.shape)
    for m,n in np.ndindex(out.shape):
        acc=dot32(a[m].tolist(),b[:,n].tolist(),fused=True)
        alpha=ieee_rne(Fraction(acc)*Fraction(attrs.get('alpha',1.0)),'fp32')
        beta=ieee_rne(Fraction(float(bias[m,n]))*Fraction(attrs.get('beta',1.0)),'fp32')
        out[m,n]=ieee_rne(ieee_rne(Fraction(alpha)+Fraction(beta),'fp32'),dtype)
    return out


def softmax(x,axis,dtype):
    x=np.asarray(x);axis%=x.ndim
    moved=np.moveaxis(x,axis,-1);result=np.empty_like(moved,dtype=np.float64)
    for idx in np.ndindex(moved.shape[:-1]):
        row=moved[idx];maximum=max(row)
        exps=[ieee_rne(math.exp(ieee_rne(float(v-maximum),'fp32')),'fp32') for v in row]
        total=0.
        for v in exps:total=ieee_rne(Fraction(total)+Fraction(v),'fp32')
        result[idx]=[ieee_rne(ieee_rne(Fraction(v)/Fraction(total),'fp32'),dtype) for v in exps]
    return np.moveaxis(result,-1,axis)


def evaluate(model, feeds, *, extension=None):
    import onnx
    if model.ir_version!=8 or [(o.domain,o.version) for o in model.opset_import]!=[('',17)]:
        raise ValueError('QA interpreter requires actual IR8/opset17')
    inferred=onnx.shape_inference.infer_shapes(model,strict_mode=True)
    types={v.name:TYPES[v.type.tensor_type.elem_type] for v in [*inferred.graph.input,*inferred.graph.value_info,*inferred.graph.output]}
    values={t.name:tensor_array(t) for t in model.graph.initializer}
    types.update({t.name:TYPES[t.data_type] for t in model.graph.initializer})
    cast_feed=cast if extension is None else extension.cast_input
    if extension is not None:values={name:cast_feed(value,types[name]) for name,value in values.items()}
    for name,value in feeds.items():values[name]=cast_feed(value,types[name])
    stages={};rows=[]
    for index,node in enumerate(model.graph.node):
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        args=[values[n] for n in node.input if n];dtype=types[node.output[0]]
        op=node.op_type
        out=None if extension is None else extension.execute(op,args,attrs,dtype)
        if out is not None:pass
        elif op=='Constant':out=[tensor_array(attrs['value'])]
        elif op=='Cast':out=[cast(args[0],TYPES[attrs['to']])]
        elif op=='Identity':out=[args[0].copy()]
        elif op=='Gemm':out=[gemm(*args,attrs,dtype)]
        elif op=='MatMul':out=[matmul(*args,dtype=dtype)]
        elif op in ('Add','Sub','Mul','Div','Erf','Sqrt','Tanh'):
            out=[elementwise(op.lower(),*args,dtype=dtype)]
        elif op=='Transpose':out=[args[0].transpose(attrs.get('perm',tuple(reversed(range(args[0].ndim)))))]
        elif op=='Softmax':out=[softmax(args[0],attrs.get('axis',-1),dtype)]
        elif op=='LayerNormalization':
            r=layer_norm(*args,dtype=dtype,axis=attrs.get('axis',-1),epsilon=attrs.get('epsilon',1e-5),stash_type=attrs.get('stash_type',1))
            out=[r[n] for n in ['Y','Mean','InvStdDev'][:len(node.output)]]
        else:raise NotImplementedError('No independent QA semantics for '+op)
        if len(out)!=len(node.output):raise ValueError('QA output arity differs at '+node.name)
        for name,value in zip(node.output,out):
            if extension is not None:value=cast_feed(value,types[name])
            values[name]=value;stages[name]=value
        rows.append(dict(index=index,name=node.name,op_type=op,inputs=list(node.input),outputs=list(node.output),
                         output_dtypes=[types[n] for n in node.output]))
    return dict(outputs={v.name:values[v.name] for v in model.graph.output},stages=stages,types=types,nodes=rows)
