"""Independent ONNX numeric interpretation; no product parsing or lowering."""
import numpy as np
from onnx_import_expectations import dot,gemm,softmax,reshape_shape,stored

def evaluate_model(model,feeds):
    import onnx
    assert model.ir_version==8
    assert [(o.domain,o.version) for o in model.opset_import]==[('',17)]
    values={t.name:onnx.numpy_helper.to_array(t).copy() for t in model.graph.initializer}
    values.update({name:np.asarray(value).copy() for name,value in feeds.items()})
    stages={}
    for node in model.graph.node:
        attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
        args=[values[n] for n in node.input if n]
        if node.op_type=='MatMul':value=dot(*args)
        elif node.op_type=='Gemm':value=gemm(*args,**attrs)
        elif node.op_type=='Add':value=stored(args[0].astype(np.float64)+args[1])
        elif node.op_type=='Mul':value=stored(args[0].astype(np.float64)*args[1])
        elif node.op_type=='Relu':value=stored(np.maximum(args[0],0))
        elif node.op_type=='Softmax':value=softmax(args[0],attrs.get('axis',-1))
        elif node.op_type=='Transpose':
            value=args[0].transpose(attrs.get('perm',tuple(reversed(range(args[0].ndim)))))
        elif node.op_type=='Reshape':
            shape=reshape_shape(args[0].shape,[int(n) for n in args[1]],allowzero=attrs.get('allowzero',0))
            value=args[0].reshape(shape)
        elif node.op_type=='Identity':value=args[0].copy()
        elif node.op_type=='Constant':value=onnx.numpy_helper.to_array(attrs['value']).copy()
        else:raise NotImplementedError('QA interpreter has no independent '+node.op_type)
        if len(node.output)!=1:raise ValueError('QA operator family is single-output')
        values[node.output[0]]=value;stages[node.output[0]]=value
    return dict(stages=stages,outputs={o.name:values[o.name] for o in model.graph.output})
