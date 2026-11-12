"""Independent typed opset17 extension for frozen vision/dual-tower exports.

The shared interpreter retains its previous default. This explicit extension
adds logical operators only; no product lowering or physical plan is consumed.
"""
from fractions import Fraction
import math
import numpy as np
import onnx_index_expectations as index
from onnx_conv_expectations import conv2d
from precision_transformer_expectations import ieee_rne
from typed_onnx_expectations import cast

INTEGER={'int32':np.dtype('int32'),'int64':np.dtype('int64')}
FORMATS={1:'fp32',10:'fp16',16:'bf16',6:'int32',7:'int64',9:'bool'}


def cast_input(value,dtype):
    if dtype in INTEGER:
        a=index.integer_array(value);limits=np.iinfo(INTEGER[dtype])
        if np.any(a<limits.min) or np.any(a>limits.max):raise ValueError('oracle integer feed outside dtype range')
        return a.astype(INTEGER[dtype])
    if dtype=='bool':
        a=np.asarray(value)
        if a.dtype!=np.dtype('bool'):raise ValueError('oracle bool feed requires bool values')
        return a
    result=cast(value,dtype)
    if not np.isfinite(result).all():raise ValueError('oracle floating value is nonfinite')
    return result


def explicit_cast(value,dtype):
    a=np.asarray(value)
    if dtype in INTEGER:
        limits=np.iinfo(INTEGER[dtype]);values=[]
        for v in a.flat:
            if not math.isfinite(v):raise ValueError('oracle integer Cast requires finite input')
            n=int(v)
            if not limits.min<=n<=limits.max:raise ValueError('oracle integer Cast outside dtype range')
            values.append(n)
        return np.asarray(values,dtype=INTEGER[dtype]).reshape(a.shape)
    if dtype=='bool':
        if not np.isfinite(a).all():raise ValueError('oracle bool Cast requires finite input')
        return a!=0
    # Exact integer -> chosen floating grid, without a binary64 intermediate.
    values=[ieee_rne(int(v) if a.dtype.kind in 'iub' else float(v),dtype) for v in a.flat]
    output=np.asarray(values,dtype=np.float64).reshape(a.shape)
    if not np.isfinite(output).all():raise ValueError('oracle floating Cast result is nonfinite')
    return output


def reduction(op,value,axes,keepdims,dtype):
    x=np.asarray(value,dtype=np.float64)
    axes=list(range(x.ndim)) if axes is None else index.vector(axes)
    axes=sorted(index.axis(a,x.ndim) for a in axes)
    if len(axes)!=len(set(axes)) or keepdims not in (0,1):raise ValueError('oracle reduction axes/keepdims differ')
    kept=[i for i in range(x.ndim) if i not in axes]
    matrix=x.transpose(kept+axes).reshape(math.prod(x.shape[i] for i in kept),math.prod(x.shape[i] for i in axes))
    if matrix.shape[1]==0:raise ValueError('oracle empty reduction outside admitted domain')
    rows=[]
    for row in matrix:
        if op=='ReduceMax':answer=float(max(row))
        else:
            answer=0.
            for v in row:
                operand=Fraction(float(v))
                if op=='ReduceL2':operand=Fraction(ieee_rne(operand*operand,'fp32'))
                answer=ieee_rne(Fraction(answer)+operand,'fp32')
            if op=='ReduceMean':answer=ieee_rne(Fraction(answer)/len(row),'fp32')
            if op=='ReduceL2':answer=ieee_rne(math.sqrt(answer),'fp32')
        rows.append(ieee_rne(answer,dtype))
    shape=tuple(1 if i in axes else x.shape[i] for i in range(x.ndim)) if keepdims else tuple(x.shape[i] for i in kept)
    return np.asarray(rows).reshape(shape)


def execute(op,args,attrs,dtype):
    if op=='Cast':return [explicit_cast(args[0],FORMATS[attrs['to']])]
    if op in ('Add','Sub','Mul','Div') and dtype in INTEGER:
        return [cast_input(index.integer_binary(op,*args),dtype)]
    if op=='ArgMax':return [index.first_argmax(args[0],attrs.get('axis',0),attrs.get('keepdims',1),attrs.get('select_last_index',0))]
    if op=='Gather':return [index.gather(*args,axis_value=attrs.get('axis',0))]
    if op=='GatherElements':return [index.gather_elements(*args,axis_value=attrs.get('axis',0))]
    if op=='Shape':return [index.shape_of(args[0],attrs.get('start',0),attrs.get('end'))]
    if op=='Concat':return [index.concat(args,attrs['axis'])]
    if op=='Slice':return [index.slice_tensor(*args)]
    if op=='Expand':return [index.expand(*args)]
    if op=='Unsqueeze':return [index.unsqueeze(*args)]
    if op=='Squeeze':return [index.squeeze(*args)]
    if op=='Reshape':
        shape=index.vector(args[1]);a=args[0]
        if any(v<-1 for v in shape) or shape.count(-1)>1:raise ValueError('oracle invalid Reshape shape')
        if not attrs.get('allowzero',0):shape=[a.shape[i] if v==0 else v for i,v in enumerate(shape)]
        return [a.reshape(shape)]
    if op=='Flatten':
        a=args[0];axis=attrs.get('axis',1)
        if not -a.ndim<=axis<=a.ndim:raise ValueError('oracle Flatten axis outside rank')
        axis=axis+a.ndim if axis<0 else axis
        return [a.reshape(math.prod(a.shape[:axis]),math.prod(a.shape[axis:]))]
    if op=='Equal':return [np.equal(*args)]
    if op=='Where':
        if args[0].dtype!=np.dtype('bool'):raise ValueError('oracle Where condition must be bool')
        return [np.where(*args)]
    if op=='Range':
        if dtype not in INTEGER or any(a.ndim!=0 for a in args):raise ValueError('oracle Range requires integer scalars')
        start,limit,delta=[int(a) for a in args]
        if not delta:raise ValueError('oracle Range step is zero')
        values=range(start,limit,delta)
        if len(values)>1000000:raise ValueError('oracle Range exceeds reference budget')
        return [cast_input(list(values),dtype)]
    if op=='ConstantOfShape':
        from framework_numpy_interpreter import tensor_array
        shape=index.vector(args[0]);value=tensor_array(attrs['value']) if 'value' in attrs else np.array([0.],dtype=np.float32)
        if value.size!=1 or any(n<0 for n in shape):raise ValueError('oracle ConstantOfShape shape/value differs')
        return [np.full(shape,value.item(),dtype=value.dtype)]
    if op=='Conv':
        opts={k:attrs[k] for k in ('strides','pads','dilations','group','auto_pad') if k in attrs}
        if isinstance(opts.get('auto_pad'),bytes):opts['auto_pad']=opts['auto_pad'].decode('ascii')
        if 'kernel_shape' in attrs and list(args[1].shape[2:])!=attrs['kernel_shape']:raise ValueError('oracle Conv kernel_shape differs')
        return [conv2d(*args,dtype=dtype,**opts)]
    if op in ('ReduceSum','ReduceMean','ReduceMax','ReduceL2'):
        axes=args[1] if len(args)>1 else attrs.get('axes')
        if axes is not None and len(axes)==0 and attrs.get('noop_with_empty_axes',0):return [args[0].copy()]
        return [reduction(op,args[0],axes,attrs.get('keepdims',1),dtype)]
    if op in ('Exp','Relu'):
        function=math.exp if op=='Exp' else lambda v:max(0.,v)
        out=np.asarray([ieee_rne(ieee_rne(function(float(v)),'fp32'),dtype) for v in args[0].flat]).reshape(args[0].shape)
        if not np.isfinite(out).all():raise ValueError('oracle activation result is nonfinite')
        return [out]
    return None


def evaluate(model,feeds):
    import sys
    from framework_numpy_interpreter import evaluate as shared_evaluate
    return shared_evaluate(model,feeds,extension=sys.modules[__name__])
