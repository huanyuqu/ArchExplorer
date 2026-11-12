"""Multi-layer encoder composition; the single block owns all mathematics."""
from .graph import finite_json
from .transformer_block import transformer_block_input_shapes


def transformer_encoder_input_shapes(*, layer_count=2,batch=2,sequence=3,hidden=8,heads=2,mlp_hidden=16):
    if type(layer_count) is not int or not 2<=layer_count<=4:raise ValueError('encoder requires 2..4 layers')
    shapes=transformer_block_input_shapes(batch=batch,sequence=sequence,hidden=hidden,heads=heads,mlp_hidden=mlp_hidden)
    return dict(X=shapes['X'],layers=[{k:v for k,v in shapes.items() if k!='X'} for _ in range(layer_count)])


def _configuration(params):
    defaults=dict(batch=2,sequence=3,hidden=8,heads=2,mlp_hidden=16,dtype='fp32',epsilon=1e-5,name='encoder')
    if type(params) is not dict or 'inputs' not in params or set(params)-{'inputs',*defaults}:
        raise ValueError('encoder requires inputs and supported configuration fields')
    p={**defaults,**finite_json(params)};data=p['inputs']
    if type(data) is not dict or set(data)!={'X','layers'} or type(data['layers']) is not list:
        raise ValueError('encoder inputs require exactly one X and a layers list; no precomputed later X')
    shapes=transformer_encoder_input_shapes(layer_count=len(data['layers']),**{k:p[k] for k in ('batch','sequence','hidden','heads','mlp_hidden')})
    from .transformer_block import _configuration as block_config
    from .reference_api import _input
    from .operators import Tensor
    block_config({k:v for k,v in p.items() if k!='inputs'},require_inputs=False)
    _input(data['X'],Tensor('X',shapes['X'],p['dtype']))
    for index,(layer,expected) in enumerate(zip(data['layers'],shapes['layers'])):
        if type(layer) is not dict or set(layer)!=set(expected):raise ValueError('each encoder layer requires exactly 16 parameter tensors')
        for key,shape in expected.items():_input(layer[key],Tensor('layer'+str(index)+'.'+key,shape,p['dtype']))
    return p,shapes


def transformer_encoder(*, inputs,batch=2,sequence=3,hidden=8,heads=2,mlp_hidden=16,dtype='fp32',epsilon=1e-5,name='encoder'):
    from arch_explorer.adf.evaluation import WorkloadSpec
    p=dict(inputs=inputs,batch=batch,sequence=sequence,hidden=hidden,heads=heads,mlp_hidden=mlp_hidden,dtype=dtype,epsilon=epsilon,name=name)
    _configuration(p)
    return WorkloadSpec('transformer_encoder',finite_json(p))


def build_encoder(params, *, tp=1):
    from .api import workload_from_spec
    from .composition import compose_workloads
    from .transformer_block import transformer_block_structure
    if type(tp) is not int or tp!=1:raise ValueError('encoder supports tp=1 only')
    p,shapes=_configuration(params);instances=[];bindings=[]
    roots={'X':dict(shape=list(shapes['X']),dtype=p['dtype'],kind='activation',value=p['inputs']['X'])}
    config={k:p[k] for k in ('batch','sequence','hidden','heads','mlp_hidden','dtype','epsilon')}
    for index,layer in enumerate(p['inputs']['layers']):
        ident='layer'+str(index)
        instances.append(dict(id=ident,workload=transformer_block_structure(**config),inputs={'X':'X',**{k:k for k in sorted(layer)}},outputs={'Y':'Y'}))
        bindings.append({'to':dict(instance=ident,port='X'),'from':{'input':'X'} if not index else dict(instance='layer'+str(index-1),port='Y')})
        for key in sorted(layer):
            value=layer[key]
            root=ident+'.'+key
            roots[root]=dict(shape=list(shapes['layers'][index][key]),dtype=p['dtype'],kind='weight',value=value)
            bindings.append({'to':dict(instance=ident,port=key),'from':{'input':root}})
    spec=compose_workloads(name=p['name'],instances=instances,inputs=roots,bindings=bindings,
        outputs={'Y':dict(instance='layer'+str(len(instances)-1),port='Y')})
    graph=workload_from_spec(spec)
    graph.meta['encoder']=dict(layer_count=len(instances),batch=p['batch'],sequence=p['sequence'],hidden=p['hidden'],
        heads=p['heads'],mlp_hidden=p['mlp_hidden'],storage_dtype=p['dtype'],causal=False,dropout=0)
    return graph
