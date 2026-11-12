"""ONNX support-domain validation and mechanical expansion into common operators.

All numerical execution remains in the registered common semantic definitions.
Only literal FP32 constants and int64 Reshape shape values are folded.
"""
import math
from .operators import Tensor, Operator, Workload
from .graph import validate_graph, name as check_name
from .resource_budget import check_shape
from .onnx_resources import FLOATS,STORAGE,specialize

ATTRIBUTES = dict(MatMul=set(), Gemm={'alpha', 'beta', 'transA', 'transB'}, Add=set(), Mul=set(),
                  Relu=set(), Softmax={'axis'}, Transpose={'perm'}, Reshape={'allowzero'},
                  Identity=set(), Constant={'value'},Cast={'to'},Div=set(),Sub=set(),Erf=set(),
                  LayerNormalization={'axis','epsilon','stash_type'},Conv={'auto_pad','dilations','group','kernel_shape','pads','strides'},
                  Shape={'start','end'},Gather={'axis'},GatherElements={'axis'},ArgMax={'axis','keepdims','select_last_index'},
                  Slice=set(),Concat={'axis'},Expand=set(),Unsqueeze=set(),Squeeze=set(),Flatten={'axis'},
                  Equal=set(),Greater=set(),Less=set(),Where=set(),ConstantOfShape={'value'},Range=set(),
                  ReduceL2={'axes','keepdims'},ReduceMean={'axes','keepdims'},Exp=set(),Sqrt=set())


def _shape(shape):
    return check_shape(shape)


def _broadcast(left, right):
    result = []
    for a, b in zip((1,) * max(0, len(right) - len(left)) + left,
                    (1,) * max(0, len(left) - len(right)) + right):
        if a != b and a != 1 and b != 1:
            raise ValueError('ONNX incompatible multidirectional broadcast')
        result.append(max(a, b))
    return _shape(result)


def lower_model(model, inputs, input_shapes, name, *, _shape_only=False):
    import numpy as np
    from .onnx_import import _onnx
    onnx = _onnx()
    check_name(name)
    if (not _shape_only and type(inputs) is not dict) or type(input_shapes) is not dict:
        raise ValueError('ONNX inputs/input_shapes must be JSON objects')
    g = model.graph
    roots = {v.name: v for v in g.input}
    initializers = {t.name: t for t in g.initializer}
    if len(roots) != len(g.input) or len(initializers) != len(g.initializer) or set(roots) & set(initializers):
        raise ValueError('ONNX duplicate root/initializer or initializer also listed in graph.input is unsupported')
    if (not _shape_only and set(inputs) != set(roots)) or set(input_shapes) - set(roots):
        raise ValueError('ONNX inputs must exactly bind graph roots; input_shapes may specialize roots only')
    root_shapes,symbols = specialize(model,input_shapes)
    shapes, types, arrays = {}, {}, {}
    declarations = {}
    for value in (*g.input, *g.value_info, *g.output):
        if not value.name or not value.type.HasField('tensor_type') or not value.type.tensor_type.HasField('shape'):
            raise ValueError('ONNX values require named typed tensor shapes')
        declarations.setdefault(value.name, []).append(value.type.tensor_type)
    for key, value in roots.items():
        t = value.type.tensor_type
        if t.elem_type not in STORAGE:
            raise ValueError('ONNX runtime roots require FP32/FP16/BF16/int32/int64/bool storage')
        explicit = input_shapes.get(key)
        if explicit is not None:
            if type(explicit) is not list:
                raise ValueError('ONNX root specialization must be a logical shape list')
            explicit = _shape(explicit)
            if len(explicit) != len(t.shape.dim):
                raise ValueError('ONNX root specialization rank conflicts with annotation')
        resolved = []
        for axis, dim in enumerate(t.shape.dim):
            n = explicit[axis] if explicit is not None else (int(dim.dim_value) if dim.HasField('dim_value') else symbols.get(dim.dim_param))
            if n is None:
                raise ValueError('ONNX unresolved root dimension; supply explicit input_shapes')
            if dim.HasField('dim_value') and dim.dim_value != n:
                raise ValueError('ONNX concrete root shape conflicts with annotation')
            if dim.HasField('dim_param') and dim.dim_param:
                if dim.dim_param in symbols and symbols[dim.dim_param] != n:
                    raise ValueError('ONNX symbolic dimension specializations conflict')
                symbols[dim.dim_param] = n
            resolved.append(n)
            dim.ClearField('dim_param')
            dim.dim_value = n
        shapes[key] = root_shapes[key]
        types[key] = t.elem_type
        if _shape_only:
            arrays[key]=None
            continue
        if types[key] in (6,7,9):
            from .integer import exact_storage
            arrays[key]=exact_storage(inputs[key],STORAGE[types[key]],shape=shapes[key])
            continue
        try:
            def numeric(value):
                return all(numeric(v) for v in value) if type(value) is list else type(value) in (int, float)
            if not numeric(inputs[key]):
                raise ValueError('nonnumeric JSON leaf')
            raw = np.asarray(inputs[key])
            if raw.dtype.kind not in 'iuf' or raw.shape != shapes[key] or not np.isfinite(raw).all():
                raise ValueError('shape/type/nonfinite')
            with np.errstate(over='raise', invalid='raise'):
                from .precision import round_storage
                arrays[key] = round_storage(raw,FLOATS[types[key]])
            if not np.isfinite(arrays[key]).all():
                raise ValueError('FP32 overflow')
        except (ValueError, TypeError, FloatingPointError) as error:
            raise ValueError('ONNX root ' + key + ' must contain exactly shaped finite '+FLOATS[types[key]]+' numbers') from error
    for node in g.node:
        if node.domain or node.op_type not in ATTRIBUTES:
            raise ValueError('ONNX unsupported node/domain: ' + node.domain + ':' + node.op_type)
        fields = [a.name for a in node.attribute]
        if len(set(fields)) != len(fields) or set(fields) - ATTRIBUTES[node.op_type]:
            raise ValueError('ONNX unsupported/duplicate attributes for ' + node.op_type)
        if any(a.type in (a.GRAPH, a.GRAPHS, a.SPARSE_TENSOR, a.SPARSE_TENSORS) for a in node.attribute):
            raise ValueError('ONNX graph/sparse attributes are unsupported')
    try:
        onnx.checker.check_model(model, full_check=False)
        inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    except Exception as error:
        raise ValueError('ONNX checker/shape inference rejected model: ' + str(error)) from error
    for v in (*inferred.graph.value_info, *inferred.graph.output):
        declarations.setdefault(v.name, []).append(v.type.tensor_type)
    constants = {}
    for tensor in g.initializer:
        if _shape_only and tensor.data_type in FLOATS:
            shapes[tensor.name]=_shape(list(tensor.dims));types[tensor.name]=tensor.data_type
            arrays[tensor.name]=None
            continue
        try:
            a = onnx.numpy_helper.to_array(tensor).copy()
        except Exception as error:
            raise ValueError('ONNX initializer payload invalid: ' + tensor.name) from error
        constants[tensor.name] = a
        shapes[tensor.name] = tuple(a.shape) if tensor.data_type in (6,7,9) else _shape(list(a.shape))
        types[tensor.name] = tensor.data_type
    for key, a in constants.items():
        if types[key] in FLOATS:
            if not np.isfinite(a).all():
                raise ValueError('ONNX nonfinite initializer')
            from .precision import round_storage
            arrays[key] = round_storage(a,FLOATS[types[key]])
        elif types[key] not in (6,7,9):
            raise ValueError('ONNX initializer storage dtype unsupported')
        else:
            from .integer import exact_storage
            constants[key]=exact_storage(a,STORAGE[types[key]])
    tensors, operators, value_ids, node_facts = {}, [], {}, []
    def value_id(key):
        if key not in value_ids:
            value_ids[key] = 'v' + str(len(value_ids)).zfill(5)
        return value_ids[key]
    def tensor(key, shape, kind='activation',storage_dtype=None):
        n = value_id(key)
        tensors[n] = Tensor(n, _shape(list(shape)) or (1,), storage_dtype or STORAGE[types.get(key,1)], kind)
        return n
    def materialize(key):
        n=value_id(key)
        if n not in tensors:
            if key not in constants:raise ValueError('ONNX runtime value lacks a common tensor: '+key)
            a=constants[key]
            if not a.size:raise ValueError('ONNX empty control value cannot be a runtime tensor')
            arrays[key]=a;tensor(key,shapes[key],'weight')
        return n
    for key in (*roots, *initializers):
        value_id(key)
        if key in arrays:
            tensor(key, shapes[key], 'weight' if key in initializers else 'activation')
    def annotation(key, shape, dtype):
        for t in declarations.get(key, ()):
            if t.elem_type != dtype or not t.HasField('shape') or len(t.shape.dim) != len(shape):
                raise ValueError('ONNX value type/rank conflicts with inferred value: ' + key)
            for dim, n in zip(t.shape.dim, shape):
                if dim.HasField('dim_value') and dim.dim_value != n:
                    raise ValueError('ONNX value dimensions conflict: ' + key)
                if dim.HasField('dim_param') and dim.dim_param:
                    if dim.dim_param in symbols and symbols[dim.dim_param] != n:
                        raise ValueError('ONNX symbolic value dimensions conflict: ' + key)
                    symbols[dim.dim_param] = n
    for key in shapes:
        annotation(key, shapes[key], types[key])
    for index, node in enumerate(g.node):
        identity = 'n' + str(index).zfill(5) + '.' + node.op_type
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        arities={'Constant':(0,), 'Gemm':(2,3), 'LayerNormalization':(2,3),'Conv':(2,3),
            'MatMul':(2,),'Add':(2,),'Mul':(2,),'Sub':(2,),'Div':(2,),'Reshape':(2,),
            'Gather':(2,),'GatherElements':(2,),'Expand':(2,),'Unsqueeze':(2,),'Squeeze':(1,2),
            'Equal':(2,),'Greater':(2,),'Less':(2,),'Where':(3,),'Range':(3,),'Slice':(3,4,5)}
        allowed=tuple(range(1,1025)) if node.op_type=='Concat' else arities.get(node.op_type,(1,))
        required=3 if node.op_type=='Slice' else 2 if node.op_type in ('Conv','LayerNormalization') else len(node.input)
        if (not 1<=len(node.output)<=(3 if node.op_type=='LayerNormalization' else 1) or not node.output[0]
                or any(n in shapes for n in node.output if n)
                or len(node.input) not in allowed
                or any(not n or n not in shapes for n in node.input[:required])
                or any(n and n not in shapes for n in node.input[required:])):
            raise ValueError('ONNX requires SSA topological nodes, supported arity and one output: ' + identity)
        start = len(operators)
        output = node.output[0]
        value_id(output)
        def emit(kind, operands, shape, part, target=None,storage_dtype=None, **a):
            dt=storage_dtype or tensors[operands[0]].dtype
            key = target if target is not None else identity + '.' + part
            if target is None:
                n = key
                tensors[n] = Tensor(n, _shape(list(shape)) or (1,), dt)
            else:
                n = tensor(key, shape,storage_dtype=dt)
            operators.append(Operator(identity + '.' + part, kind, tuple(operands), (n,),
                                      dict(shape=list(tensors[n].shape), **a), node.name))
            return n
        def broadcast(operand, target_shape, part):
            actual = tensors[operand].shape
            desired = target_shape or (1,)
            if actual == desired:
                return operand
            if _broadcast(actual, desired) != desired:
                raise ValueError('ONNX operand cannot broadcast to output')
            return emit('broadcast', (operand,), desired, part)
        def transpose(operand, axes, part):
            return emit('transpose', (operand,), tuple(tensors[operand].shape[i] for i in axes), part, axes=list(axes))
        def matmul(left, right, left_shape, right_shape, target=None):
            if min(len(left_shape), len(right_shape)) < 2:
                raise ValueError('ONNX MatMul supports rank 2..4 operands; vector promotion is unsupported')
            if left_shape[-1] != right_shape[-2]:
                raise ValueError('ONNX MatMul inner dimensions differ')
            batch_shape = _broadcast(left_shape[:-2], right_shape[:-2])
            result_shape = _shape(list(batch_shape + (left_shape[-2], right_shape[-1])))
            left = broadcast(left, batch_shape + left_shape[-2:], 'batch.left')
            shared = len(right_shape) == 2
            if not shared:
                right = broadcast(right, batch_shape + right_shape[-2:], 'batch.right')
            canonical_shape = (math.prod(batch_shape), left_shape[-2], right_shape[-1])
            direct = bool(target) and canonical_shape == result_shape
            n = tensor(output, result_shape,storage_dtype=tensors[left].dtype) if direct else identity + '.matrix.raw'
            if not direct:
                tensors[n] = Tensor(n, canonical_shape, tensors[left].dtype)
            operators.append(Operator(identity + '.matmul', 'matmul', (left, right), (n,),
                dict(batch=math.prod(batch_shape), m=left_shape[-2], k=left_shape[-1], n=right_shape[-1], shared_b=shared), node.name))
            if canonical_shape != result_shape:
                n = emit('reshape', (n,), result_shape, 'matrix.shape', output if target else None)
            return n, result_shape
        from .onnx_shapes import fold
        from .extensions import _record_source
        _record_source(fold)
        provided=[n for n in node.input if n]
        folded=None if node.op_type=='Constant' else fold(node.op_type,attrs,provided,shapes,types,constants)
        if folded is not None:
            a,dtype=folded;shape=tuple(a.shape);constants[output]=a
            if dtype in FLOATS:
                arrays[output]=a if not _shape_only else None
                types[output]=dtype;tensor(output,shape,'weight')
        elif node.op_type == 'Constant':
            if set(attrs) != {'value'}:
                raise ValueError('ONNX Constant requires tensor attribute value')
            dtype = attrs['value'].data_type
            shape=tuple(attrs['value'].dims)
            a=None if _shape_only and dtype in FLOATS else onnx.numpy_helper.to_array(attrs['value']).copy()
            if dtype in FLOATS and (_shape_only or np.isfinite(a).all()):
                from .precision import round_storage
                arrays[output] = None if _shape_only else round_storage(a,FLOATS[dtype])
                tensor(output, shape, 'weight',FLOATS[dtype])
            elif dtype in (6,7,9):
                from .integer import exact_storage
                a=exact_storage(a,STORAGE[dtype])
            else:raise ValueError('ONNX Constant storage dtype unsupported/nonfinite')
            constants[output] = a
        else:
            control_ops=('Reshape','Expand','Unsqueeze','Squeeze','Slice')
            float_inputs = node.input[:1] if node.op_type in control_ops else [n for n in node.input if n]
            if any(types[n] not in STORAGE for n in float_inputs):raise ValueError('ONNX runtime storage dtype unsupported')
            operands = [materialize(n) for n in float_inputs]
            shape = shapes[node.input[0]]
            dtype = types[node.input[0]]
            if node.op_type in ('MatMul','Gemm','LayerNormalization','Softmax','Relu','Erf','Exp','Sqrt','ReduceL2','ReduceMean') and dtype not in FLOATS:
                raise ValueError('ONNX '+node.op_type+' requires floating storage in this declared domain')
            if node.op_type not in ('Cast','LayerNormalization','Gather','GatherElements','Where') and any(types[n]!=dtype for n in float_inputs):
                raise ValueError('ONNX mixed runtime types require explicit Cast')
            if node.op_type=='Conv':
                from .spatial_ops import geometry
                _record_source(geometry)
                if dtype not in (1,10):raise ValueError('ONNX opset17 Conv requires FLOAT/FLOAT16; BF16 needs explicit FP32 Conv boundary')
                source_shape,weight_shape=(shapes[n] for n in node.input[:2])
                if len(source_shape)!=4 or len(weight_shape)!=4 or attrs.get('group',1)!=1 or source_shape[1]!=weight_shape[1]:
                    raise ValueError('ONNX Conv supports NCHW rank-four group=1 with matching input channels')
                mode=attrs.get('auto_pad',b'NOTSET')
                if mode not in ('NOTSET',b'NOTSET'):raise ValueError('ONNX Conv requires explicit padding, auto_pad=NOTSET')
                kernel=attrs.get('kernel_shape',list(weight_shape[-2:]));strides=attrs.get('strides',[1,1]);pads=attrs.get('pads',[0,0,0,0]);dilations=attrs.get('dilations',[1,1])
                if tuple(kernel)!=weight_shape[-2:]:raise ValueError('ONNX Conv kernel_shape conflicts with weight dimensions')
                packed,(oh,ow)=geometry(source_shape,kernel,strides,pads,dilations)
                if len(operands)==3 and shapes[node.input[2]]!=(weight_shape[0],):raise ValueError('ONNX Conv bias must match output channels')
                if dtype!=1:operands=[emit('cast',(n,),tensors[n].shape,'conv.to.fp32.'+str(j),storage_dtype='fp32') for j,n in enumerate(operands)]
                patches=emit('im2col',(operands[0],),packed,'conv.patches',kernel=list(kernel),strides=list(strides),pads=list(pads),dilations=list(dilations))
                weights=emit('reshape',(operands[1],),(weight_shape[0],packed[-1]),'conv.weights.rows')
                weights=transpose(weights,(1,0),'conv.weights.transpose')
                product,_=matmul(patches,weights,packed,(packed[-1],weight_shape[0]))
                if len(operands)==3:product=emit('add',(product,broadcast(operands[2],tensors[product].shape,'conv.bias.broadcast')),tensors[product].shape,'conv.bias')
                product=transpose(product,(0,2,1),'conv.channels')
                shape=(source_shape[0],weight_shape[0],oh,ow)
                product=emit('reshape',(product,),shape,'conv.spatial',output if dtype==1 else None)
                if dtype!=1:emit('cast',(product,),shape,'conv.storage',output,storage_dtype=STORAGE[dtype])
            elif node.op_type in ('Shape','ConstantOfShape','Range'):
                raise ValueError('ONNX '+node.op_type+' requires statically specialized shape/control inputs')
            elif node.op_type in ('Gather','GatherElements'):
                from .onnx_shapes import axis
                if types[node.input[1]] not in (6,7):raise ValueError('ONNX Gather indices require int32/int64 storage')
                a=axis(attrs.get('axis',0),len(shape));index_shape=shapes[node.input[1]]
                if node.op_type=='Gather':
                    shape=shape[:a]+index_shape+shape[a+1:]
                    physical=tensors[operands[0]].shape[:a]+tensors[operands[1]].shape+tensors[operands[0]].shape[a+1:]
                    value=emit('gather',operands,physical,'gather.values',output if physical==(shape or (1,)) else None,axis=a)
                    if physical!=(shape or (1,)):emit('reshape',(value,),shape,'output',output)
                else:
                    shape=index_shape
                    emit('gather_elements',operands,shape,'output',output,axis=a)
            elif node.op_type=='ArgMax':
                from .onnx_shapes import axis
                a=axis(attrs.get('axis',0),len(shape));keep=attrs.get('keepdims',1);last=attrs.get('select_last_index',0)
                if type(keep) is not int or keep not in (0,1) or type(last) is not int or last not in (0,1):raise ValueError('ONNX ArgMax requires keepdims/select_last_index 0/1')
                shape=shape[:a]+((1,) if keep else ())+shape[a+1:];dtype=7
                emit('argmax_axis',operands,shape,'output',output,storage_dtype='int64',axis=a,keepdims=bool(keep),select_last_index=bool(last))
            elif node.op_type in ('Squeeze','Unsqueeze','Flatten','Expand','Slice'):
                from .onnx_shapes import squeeze,unsqueeze,slices
                _record_source(squeeze);_record_source(unsqueeze);_record_source(slices)
                controls=[constants.get(n) for n in node.input[1:] if n]
                if any(a is None or a.dtype!=np.dtype('int64') or a.ndim!=1 for a in controls):raise ValueError('ONNX shape/index controls require static int64 vectors')
                if node.op_type=='Flatten':
                    rank=len(shape);a=attrs.get('axis',1)
                    if type(a) is not int or not -rank<=a<=rank:raise ValueError('ONNX Flatten axis outside tensor rank')
                    if a<0:a+=rank
                    shape=(math.prod(shape[:a]),math.prod(shape[a:]));emit('reshape',operands,shape,'output',output)
                elif node.op_type=='Squeeze':
                    shape=squeeze(shape,controls[0].tolist() if controls else None);emit('reshape',operands,shape,'output',output)
                elif node.op_type=='Unsqueeze':
                    shape=unsqueeze(shape,controls[0].tolist());emit('reshape',operands,shape,'output',output)
                elif node.op_type=='Expand':
                    shape=_broadcast(shape,_shape(controls[0].tolist()));emit('broadcast',operands,shape,'output',output)
                else:
                    _,shape,normalized=slices(shape,controls[0].tolist(),controls[1].tolist(),controls[2].tolist() if len(controls)>2 else None,controls[3].tolist() if len(controls)>3 else None)
                    if not all(n>0 for n in shape) or any(step<1 for _,_,step in normalized):raise ValueError('ONNX runtime Slice requires positive strides and nonempty output')
                    emit('slice',operands,shape,'output',output,starts=[a for a,_,_ in normalized],stops=[b for _,b,_ in normalized],steps=[step for _,_,step in normalized])
            elif node.op_type=='Concat':
                from .onnx_shapes import axis
                a=axis(attrs.get('axis',0),len(shape));all_shapes=[shapes[n] for n in node.input]
                if any(len(s)!=len(shape) or any(x!=y for j,(x,y) in enumerate(zip(s,shape)) if j!=a) for s in all_shapes):raise ValueError('ONNX Concat non-axis dimensions differ')
                shape=shape[:a]+(sum(s[a] for s in all_shapes),)+shape[a+1:]
                emit('concat' if len(operands)>1 else 'reshape',operands,shape,'output',output,**({'axis':a} if len(operands)>1 else {}))
            elif node.op_type in ('Equal','Greater','Less','Where'):
                if node.op_type=='Where':
                    if types[node.input[0]]!=9 or types[node.input[1]]!=types[node.input[2]]:raise ValueError('ONNX Where needs bool condition and equal branch storage')
                    dtype=types[node.input[1]]
                else:dtype=9
                shape=()
                for n in node.input:shape=_broadcast(shape,shapes[n])
                expanded=[broadcast(n,shape,'broadcast.'+str(j)) for j,n in enumerate(operands)]
                emit(node.op_type.lower(),expanded,shape,'output',output,storage_dtype=STORAGE[dtype])
            elif node.op_type in ('ReduceL2','ReduceMean'):
                from .onnx_shapes import axis
                selected=attrs.get('axes');selected=list(range(len(shape))) if not selected else [axis(j,len(shape)) for j in selected]
                keep=attrs.get('keepdims',1)
                if len(set(selected))!=len(selected) or type(keep) is not int or keep not in (0,1) or dtype not in FLOATS:raise ValueError('ONNX reduction axes/keepdims/dtype unsupported')
                selected=sorted(selected);remaining=[j for j in range(len(shape)) if j not in selected];order=remaining+selected
                value=operands[0]
                if dtype!=1:value=emit('cast',(value,),shape,'reduce.to.fp32',storage_dtype='fp32')
                if order!=list(range(len(shape))):value=transpose(value,order,'reduce.axes')
                rows=math.prod(shape[j] for j in remaining);cols=math.prod(shape[j] for j in selected)
                value=emit('reshape',(value,),(rows,cols),'reduce.rows')
                if node.op_type=='ReduceL2':value=emit('mul',(value,value),(rows,cols),'reduce.squares')
                value=emit('reduce_sum',(value,),(rows,1),'reduce.sum',axis=-1,keepdims=True)
                value=emit('sqrt' if node.op_type=='ReduceL2' else 'scale',(value,),(rows,1),'reduce.final',**({} if node.op_type=='ReduceL2' else {'factor':float(1/cols)}))
                shape=tuple(1 if j in selected else n for j,n in enumerate(shape)) if keep else tuple(shape[j] for j in remaining)
                value=emit('reshape',(value,),shape,'reduce.shape',output if dtype==1 else None)
                if dtype!=1:emit('cast',(value,),shape,'output',output,storage_dtype=STORAGE[dtype])
            elif node.op_type == 'MatMul':
                _, shape = matmul(*operands, *(shapes[n] for n in node.input), target=output)
            elif node.op_type == 'Gemm':
                if any(len(shapes[n]) != 2 for n in node.input[:2]):
                    raise ValueError('ONNX Gemm requires rank-two A and B')
                a_shape, b_shape = (shapes[n] for n in node.input[:2])
                for flag in ('transA', 'transB'):
                    if type(attrs.get(flag, 0)) is not int or attrs.get(flag, 0) not in (0, 1):
                        raise ValueError('ONNX Gemm transpose flags must be 0 or 1')
                if attrs.get('transA', 0):
                    operands[0] = transpose(operands[0], (1, 0), 'transA')
                    a_shape = a_shape[::-1]
                if attrs.get('transB', 0):
                    operands[1] = transpose(operands[1], (1, 0), 'transB')
                    b_shape = b_shape[::-1]
                # Gemm owns one final stored result. Its internal alpha/beta
                # and bias arithmetic remains FP32, with explicit conversion
                # and materialized temporaries for low-storage operands.
                if dtype!=onnx.TensorProto.FLOAT:
                    operands=[emit('cast',(n,),tensors[n].shape,'gemm.to.fp32.'+str(j),storage_dtype='fp32') for j,n in enumerate(operands)]
                product, shape = matmul(*operands[:2], a_shape, b_shape)
                alpha, beta = attrs.get('alpha', 1.0), attrs.get('beta', 1.0)
                if not all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= np.finfo(np.float32).max for v in (alpha, beta)):
                    raise ValueError('ONNX Gemm alpha/beta must be finite FP32')
                product = emit('scale', (product,), shape, 'alpha', factor=float(alpha))
                if len(node.input) == 3:
                    bias = broadcast(operands[2], shape, 'bias.broadcast')
                    bias = emit('scale', (bias,), shape, 'beta', factor=float(beta))
                    if dtype==onnx.TensorProto.FLOAT:emit('add', (product, bias), shape, 'add', output)
                    else:
                        product=emit('add',(product,bias),shape,'add')
                        emit('cast',(product,),shape,'output',output,storage_dtype=FLOATS[dtype])
                else:
                    emit('reshape' if dtype==onnx.TensorProto.FLOAT else 'cast', (product,), shape, 'output', output,storage_dtype=FLOATS[dtype])
            elif node.op_type in ('Add', 'Mul','Sub','Div'):
                shape = _broadcast(*(shapes[n] for n in node.input))
                expanded = [broadcast(n, shape, 'broadcast.' + str(j)) for j, n in enumerate(operands)]
                if dtype==9:raise ValueError('ONNX arithmetic does not accept bool storage')
                emit(('integer_' if dtype in (6,7) else '')+node.op_type.lower(), expanded, shape, 'output', output)
            elif node.op_type == 'Transpose':
                axes = attrs.get('perm', list(reversed(range(len(shape)))))
                if (any(type(i) is not int for i in axes) or sorted(axes) != list(range(len(shape)))):
                    raise ValueError('ONNX Transpose perm must be a rank permutation')
                shape = tuple(shape[i] for i in axes)
                if not shape:
                    emit('reshape', operands, shape, 'output', output)
                else:
                    emit('transpose', operands, shape, 'output', output, axes=list(axes))
            elif node.op_type == 'Reshape':
                target = node.input[1]
                if target not in constants or types[target] != onnx.TensorProto.INT64 or constants[target].ndim != 1:
                    raise ValueError('ONNX Reshape shape must be a literal rank-one int64 initializer/Constant')
                values = constants[target].tolist()
                allowzero = attrs.get('allowzero', 0)
                if type(allowzero) is not int or allowzero not in (0, 1) or any(n < -1 for n in values) or values.count(-1) > 1:
                    raise ValueError('ONNX Reshape requires allowzero 0/1 and at most one inferred -1')
                if allowzero and 0 in values:
                    raise ValueError('ONNX zero-sized Reshape is outside positive static domain')
                resolved = []
                for j, n in enumerate(values):
                    if n == 0:
                        if j >= len(shape):
                            raise ValueError('ONNX Reshape zero-copy dimension exceeds source rank')
                        n = shape[j]
                    resolved.append(n)
                known = math.prod(n for n in resolved if n != -1)
                total = math.prod(shape)
                if -1 in resolved:
                    if not known or total % known:
                        raise ValueError('ONNX Reshape inferred dimension is not integral')
                    resolved[resolved.index(-1)] = total // known
                if math.prod(resolved) != total:
                    raise ValueError('ONNX Reshape element count differs')
                shape = _shape(resolved)
                emit('reshape', operands, shape, 'output', output)
            elif node.op_type == 'Softmax':
                axis = attrs.get('axis', -1)
                if not shape or type(axis) is not int or not -len(shape) <= axis < len(shape):
                    raise ValueError('ONNX Softmax requires a valid axis on rank 1..4')
                axis %= len(shape)
                axes = tuple(j for j in range(len(shape)) if j != axis) + (axis,)
                permuted = tuple(shape[j] for j in axes)
                operand = operands[0]
                if axis != len(shape) - 1:
                    operand = transpose(operand, axes, 'axis.to.last')
                    result = emit('softmax', (operand,), permuted, 'softmax')
                    emit('transpose', (result,), shape, 'output', output, axes=[axes.index(j) for j in range(len(shape))])
                else:
                    emit('softmax', (operand,), shape, 'output', output)
            elif node.op_type=='Cast':
                dtype=attrs.get('to')
                if type(dtype) is not int or dtype not in STORAGE:
                    raise ValueError('ONNX Cast supports declared floating/int32/int64/bool runtime storage only')
                extras={'to':STORAGE[dtype]} if types[node.input[0]] not in FLOATS or dtype not in FLOATS else {}
                emit('cast',operands,shape,'output',output,storage_dtype=STORAGE[dtype],**extras)
            elif node.op_type=='LayerNormalization':
                axis=attrs.get('axis',-1);stash=attrs.get('stash_type',1);epsilon=attrs.get('epsilon',1e-5)
                if type(axis) is not int or not shape or not -len(shape)<=axis<len(shape) or stash!=1 or type(stash) is not int:
                    raise ValueError('ONNX LayerNormalization requires a valid suffix axis and stash_type=1')
                if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or not 0<=epsilon<=np.finfo(np.float32).max:
                    raise ValueError('ONNX LayerNormalization epsilon must be finite nonnegative FP32')
                axis%=len(shape);suffix=shape[axis:];rows=math.prod(shape[:axis]);cols=math.prod(suffix)
                if any(types[n]!=dtype or shapes[n]!=suffix for n in float_inputs[1:]):
                    raise ValueError('ONNX LayerNormalization Scale/B must match normalized suffix shape and X dtype')
                value=operands[0]
                if dtype!=1:value=emit('cast',(value,),shape,'norm.to.fp32',storage_dtype='fp32')
                flat=emit('reshape',(value,),(rows,cols),'norm.rows')
                mean=emit('reduce_sum',(flat,),(rows,1),'norm.sum',axis=-1,keepdims=True)
                mean=emit('scale',(mean,),(rows,1),'norm.mean',factor=float(1/cols))
                centered=emit('sub',(flat,broadcast(mean,(rows,cols),'norm.mean.broadcast')),(rows,cols),'norm.center')
                squares=emit('mul',(centered,centered),(rows,cols),'norm.squares')
                var=emit('reduce_sum',(squares,),(rows,1),'norm.variance.sum',axis=-1,keepdims=True)
                var=emit('scale',(var,),(rows,1),'norm.variance',factor=float(1/cols))
                key='@onnx/'+identity+'/epsilon'
                if key in shapes:raise ValueError('ONNX generated scalar namespace collision')
                arrays[key]=np.asarray(epsilon,dtype=np.float32);types[key]=1;shapes[key]=()
                constants[key]=arrays[key];eps=tensor(key,(),'weight')
                ve=emit('add',(var,broadcast(eps,(rows,1),'norm.epsilon.broadcast')),(rows,1),'norm.epsilon')
                inv=emit('rsqrt',(ve,),(rows,1),'norm.inverse_std')
                normal=emit('mul',(centered,broadcast(inv,(rows,cols),'norm.inv.broadcast')),(rows,cols),'norm.normalized')
                normal=emit('reshape',(normal,),shape,'norm.original.shape')
                if dtype!=1:normal=emit('cast',(normal,),shape,'norm.to.storage',storage_dtype=FLOATS[dtype])
                scale=broadcast(operands[1],shape,'norm.scale.broadcast')
                product=emit('mul',(normal,scale),shape,'norm.affine.scale',output if len(operands)==2 else None)
                if len(operands)==3:
                    emit('add',(product,broadcast(operands[2],shape,'norm.bias.broadcast')),shape,'output',output)
                saved_shape=shape[:axis]+(1,)*(len(shape)-axis)
                for extra,value in zip(node.output[1:],(mean,inv)):
                    if extra:
                        shapes[extra]=saved_shape;types[extra]=1
                        emit('reshape',(value,),saved_shape,'norm.saved.'+extra,extra,storage_dtype='fp32')
                        annotation(extra,saved_shape,1)
            elif node.op_type in ('Identity', 'Relu','Erf','Exp','Sqrt'):
                emit({'Identity':'reshape','Relu':'relu','Erf':'erf','Exp':'exp','Sqrt':'sqrt'}[node.op_type], operands, shape, 'output', output)
            else:
                raise ValueError('ONNX unsupported operation')
        shapes[output], types[output] = shape, dtype
        annotation(output, shape, dtype)
        node_facts.append(dict(index=index, id=identity, display_name=node.name, op_type=node.op_type,
            inputs=list(node.input), outputs=list(node.output), graph_operators=[o.name for o in operators[start:]],
            constant_folded=node.op_type == 'Constant' or folded is not None))
    outputs = []
    output_facts = []
    produced = {n for op in operators for n in op.outputs}
    for index, v in enumerate(g.output):
        if v.name not in shapes or types[v.name] not in STORAGE:
            raise ValueError('ONNX graph outputs must be known declared runtime storage values')
        key = materialize(v.name)
        if key not in produced:
            copied = 'output.' + str(index).zfill(5)
            tensors[copied] = Tensor(copied, tensors[key].shape, tensors[key].dtype)
            operators.append(Operator(copied + '.copy', 'reshape', (key,), (copied,), dict(shape=list(tensors[key].shape))))
            key = copied
        outputs.append(key)
        output_facts.append(dict(name=v.name, tensor=key, logical_shape=list(shapes[v.name])))
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError('ONNX graph requires unique FP32 outputs')
    metadata = dict(schema='arch-explorer.onnx-import.v1', ir_version=8, domain='', opset=17,
        graph_name=g.name, scalar_storage='logical [] is physical [1]; reshape conversion is explicit',
        initializer_input_overlap='rejected', dead_nodes='runtime nodes preserved and charged; literal/static shape controls may be folded',
        nodes=node_facts, values={n:dict(tensor=value_ids[n] if value_ids[n] in tensors else None,
            logical_shape=list(shapes[n]), dtype=STORAGE[types[n]],
            role='input' if n in roots else 'initializer' if n in initializers else 'constant' if n in constants else 'intermediate') for n in shapes},
        inputs={n:dict(tensor=value_ids[n], logical_shape=list(shapes[n]), dtype=STORAGE[types[n]]) for n in roots},
        initializers=list(initializers), outputs=output_facts, input_shapes=input_shapes)
    external_values = {value_id(n): a.reshape(tensors[value_id(n)].shape).tolist() for n, a in arrays.items() if a is not None and not _shape_only}
    return validate_graph(Workload(name, tensors, operators, 'fp32', 1,
        dict(inputs=external_values, outputs=outputs, onnx=metadata)))
