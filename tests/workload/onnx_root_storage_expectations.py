"""Root storage from original ModelProto metadata; no product graph or arrays."""
from math import prod

PROTO_STORAGE={1:('fp32',4),6:('int32',4),7:('int64.shape',8),10:('fp16',2),16:('bf16',2)}


def roots_from_model(model,inputs,dtype):
    size={'fp32':4,'fp16':2,'bf16':2}[dtype]
    roots={name:dict(logical_shape=list(shape),storage_dtype=dtype,elements=prod(shape),
                     bytes=prod(shape)*size,role='input') for name,shape in inputs.items()}
    def weight(name,tensor):
        storage,item=PROTO_STORAGE[tensor.data_type];shape=list(tensor.dims)
        if name in roots:raise ValueError('Frozen root names overlap')
        roots[name]=dict(logical_shape=shape,storage_dtype=storage,elements=prod(shape),
                         bytes=prod(shape)*item,role='weight')
    for tensor in model.graph.initializer:weight(tensor.name,tensor)
    for node in model.graph.node:
        if node.op_type=='Constant':
            attrs={a.name:a for a in node.attribute}
            if set(attrs)!={'value'} or len(node.output)!=1:
                raise ValueError('Expected frozen Tensor-valued Constant')
            weight(node.output[0],attrs['value'].t)
    return roots


def frozen_roots(model_path,row):
    import onnx
    return roots_from_model(onnx.load(str(model_path),load_external_data=False),row['input_shapes'],row['dtype'])
