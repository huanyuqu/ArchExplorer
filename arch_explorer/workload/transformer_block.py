"""Explicit pre-LN encoder block composition, without backend or model timing."""
import math

from .graph import finite_json
from .operators import Workload, Tensor


def transformer_block_input_shapes(*, batch=2, sequence=3, hidden=8, heads=2, mlp_hidden=16):
    values = (batch, sequence, hidden, heads, mlp_hidden)
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError('block dimensions must be positive integers')
    if batch > 8 or sequence > 64 or hidden > 128 or mlp_hidden > 512 or hidden % heads:
        raise ValueError('block bounds B<=8/S<=64/H<=128/MLP<=512 and heads dividing hidden required')
    return dict(X=(batch, sequence, hidden), **{n:(hidden,hidden) for n in ('Wq','Wk','Wv','Wo')},
        **{n:(hidden,) for n in ('bq','bk','bv','bo','norm1_gamma','norm1_beta','norm2_gamma','norm2_beta')},
        W1=(hidden,mlp_hidden),b1=(mlp_hidden,),W2=(mlp_hidden,hidden),b2=(hidden,))


def transformer_encoder_block(*, inputs, batch=2, sequence=3, hidden=8, heads=2,
                              mlp_hidden=16, dtype='fp32', epsilon=1e-5, name='pre_ln_encoder'):
    from arch_explorer.adf.evaluation import WorkloadSpec
    params = dict(inputs=inputs, batch=batch, sequence=sequence, hidden=hidden, heads=heads,
                  mlp_hidden=mlp_hidden, dtype=dtype, epsilon=epsilon, name=name)
    _configuration(params)
    return WorkloadSpec('transformer_block', finite_json(params))


def _configuration(params, *, require_inputs=True):
    defaults = dict(batch=2,sequence=3,hidden=8,heads=2,mlp_hidden=16,dtype='fp32',epsilon=1e-5,name='pre_ln_encoder')
    allowed = {'inputs',*defaults} if require_inputs else set(defaults)
    if (type(params) is not dict or (require_inputs and 'inputs' not in params)
            or set(params)-allowed):
        raise ValueError('block requires explicit inputs and supported configuration fields')
    p = {**defaults, **finite_json(params)}
    shapes = transformer_block_input_shapes(**{k:p[k] for k in ('batch','sequence','hidden','heads','mlp_hidden')})
    if p['dtype'] not in ('fp32','fp16','bf16'):
        raise ValueError('block dtype must be fp32/fp16/bf16')
    if type(p['name']) is not str or not p['name'] or p['name'].strip() != p['name']:
        raise ValueError('block name must be nonempty and unpadded')
    if (type(p['epsilon']) not in (int,float) or not math.isfinite(p['epsilon'])
            or not 0 < p['epsilon'] <= 3.4028234663852886e38):
        raise ValueError('block epsilon must be positive finite FP32')
    from .precision import round_storage
    if float(round_storage(p['epsilon'], 'fp32')) <= 0:
        raise ValueError('block epsilon must remain positive after FP32 decoding')
    if not require_inputs:
        return p, shapes
    if type(p['inputs']) is not dict or set(p['inputs']) != set(shapes):
        raise ValueError('block inputs must exactly bind X, projections, biases, norms and MLP weights')
    from .reference_api import _input
    for name, shape in shapes.items():
        _input(p['inputs'][name], Tensor(name, shape, p['dtype']))
    return p, shapes


def build_block(params, *, tp=1):
    p, _ = _configuration(params)
    graph = build_structure({k:v for k,v in p.items() if k != 'inputs'}, tp=tp)
    graph.meta['inputs'] = p['inputs']
    return graph


def transformer_block_structure(*, batch=2, sequence=3, hidden=8, heads=2,
                                mlp_hidden=16, dtype='fp32', epsilon=1e-5, name='pre_ln_encoder'):
    from arch_explorer.adf.evaluation import WorkloadSpec
    params = dict(batch=batch,sequence=sequence,hidden=hidden,heads=heads,mlp_hidden=mlp_hidden,
                  dtype=dtype,epsilon=epsilon,name=name)
    _configuration(params,require_inputs=False)
    return WorkloadSpec('transformer_block_structure',finite_json(params))


def build_structure(params, *, tp=1):
    if tp != 1:
        raise ValueError('block has no tensor-parallel communication implementation')
    p, shapes = _configuration(params,require_inputs=False)
    b,s,h,heads,f = (p[k] for k in ('batch','sequence','hidden','heads','mlp_hidden'))
    d, dtype = h//heads, p['dtype']
    graph = Workload(p['name'], dtype=dtype)
    for name, shape in shapes.items():
        graph.tensor(name, shape, dtype=dtype, kind='activation' if name == 'X' else 'weight')
    graph.meta['block'] = dict(kind='single_pre_ln_encoder',batch=b,sequence=s,hidden=h,heads=heads,
        mlp_hidden=f,storage_dtype=dtype,accumulation='fp32',epsilon=p['epsilon'],gelu='tanh',causal=False,dropout=0)

    def op(name, kind, inputs, shape, storage=None, **attrs):
        output = graph.tensor(name, tuple(shape), dtype=storage or dtype)
        graph.add(name, kind, inputs, [output], shape=list(shape), **attrs)
        return output

    def matrix(name, left, right, batch, rows, inner, cols, shared):
        output = graph.tensor(name,(batch,rows,cols),dtype=dtype)
        graph.add(name,'matmul',[left,right],[output],batch=batch,m=rows,k=inner,n=cols,
                  shared_b=shared,dtype=dtype,output_dtype=dtype)
        return output

    def bias(name, x, source, shape):
        expanded = op(name+'_broadcast','broadcast',[source],shape)
        return op(name,'add',[x,expanded],shape)

    shape = (b,s,h)
    norm = op('norm1','affine_norm',['X','norm1_gamma','norm1_beta'],shape,epsilon=p['epsilon'])
    qkv = {}
    for tag in ('q','k','v'):
        projection = matrix(tag+'_projection',norm,'W'+tag,b,s,h,h,True)
        biased = bias(tag+'_bias',projection,'b'+tag,shape)
        split = op(tag+'_split','reshape',[biased],(b,s,heads,d))
        axes = [0,2,3,1] if tag == 'k' else [0,2,1,3]
        layout = (b,heads,d,s) if tag == 'k' else (b,heads,s,d)
        transposed = op(tag+'_transpose','transpose',[split],layout,axes=axes)
        qkv[tag] = op(tag+'_batches','reshape',[transposed],(b*heads,d,s) if tag == 'k' else (b*heads,s,d))
    logits = matrix('logits',qkv['q'],qkv['k'],b*heads,s,d,s,False)
    logits32 = op('logits_fp32','cast',[logits],(b*heads,s,s),storage='fp32')
    scaled = op('scaled','scale',[logits32],(b*heads,s,s),storage='fp32',factor=1/math.sqrt(d))
    maximum = op('row_max','reduce_max',[scaled],(b*heads,s,1),storage='fp32',axis=-1,keepdims=True)
    max_grid = op('max_grid','broadcast',[maximum],(b*heads,s,s),storage='fp32')
    centered = op('centered','sub',[scaled,max_grid],(b*heads,s,s),storage='fp32')
    exponentials = op('exponentials','exp',[centered],(b*heads,s,s),storage='fp32')
    summed = op('row_sum','reduce_sum',[exponentials],(b*heads,s,1),storage='fp32',axis=-1,keepdims=True)
    sum_grid = op('sum_grid','broadcast',[summed],(b*heads,s,s),storage='fp32')
    probabilities = op('probabilities_fp32','div',[exponentials,sum_grid],(b*heads,s,s),storage='fp32')
    stored = op('probabilities','cast',[probabilities],(b*heads,s,s))
    attended = matrix('av',stored,qkv['v'],b*heads,s,s,d,False)
    unbatch = op('unbatch_heads','reshape',[attended],(b,heads,s,d))
    merged = op('merge_transpose','transpose',[unbatch],(b,s,heads,d),axes=[0,2,1,3])
    merged = op('merge_heads','reshape',[merged],shape)
    projected = matrix('output_projection',merged,'Wo',b,s,h,h,True)
    projected = bias('output_bias',projected,'bo',shape)
    residual = op('residual1','add',['X',projected],shape)
    norm = op('norm2','affine_norm',[residual,'norm2_gamma','norm2_beta'],shape,epsilon=p['epsilon'])
    expanded = matrix('mlp_up',norm,'W1',b,s,h,f,True)
    expanded = bias('mlp_up_bias',expanded,'b1',(b,s,f))
    activated = op('mlp_gelu','gelu_tanh',[expanded],(b,s,f))
    lowered = matrix('mlp_down',activated,'W2',b,s,f,h,True)
    lowered = bias('mlp_down_bias',lowered,'b2',shape)
    op('Y','add',[residual,lowered],shape)
    return graph
