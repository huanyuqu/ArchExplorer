"""Absorbed-projection MLA decode, with a complete immutable cache boundary.

WQK absorbs the per-head non-positional query/key up projections. WVO absorbs
the value up projection and that head's output projection block. Absorption
defines this FP32 contract; it is not a rounding-equivalence claim about an
unabsorbed implementation. No hidden state, weights or cache initialization.
"""
import math
from .structural import Graph
from .operators import Tensor


def mla_input_shapes(*,batch=1,sequence=5,width=6,query_rank=3,latent_rank=4,
                     heads=4,content_dim=4,positional_dim=2,output_width=5):
    p=locals()
    if any(type(v) is not int or not 1<=v<=8192 for v in p.values()):
        raise ValueError('MLA dimensions require positive integers up to 8192')
    if heads>128 or positional_dim%2:raise ValueError('MLA supports at most 128 heads and even positional width')
    shapes={'X':(batch,width),'W_DQ':(width,query_rank),'q_gamma':(query_rank,),
            'latent_kv':(batch,sequence,latent_rank),'pos_key':(batch,sequence,positional_dim),
            'rope_cos':(positional_dim//2,),'rope_sin':(positional_dim//2,)}
    for h in range(heads):
        shapes.update({f'head{h}.WQK':(query_rank,latent_rank),f'head{h}.WQR':(query_rank,positional_dim),
                       f'head{h}.WVO':(latent_rank,output_width)})
    return shapes


def mla_decode(*,inputs,batch=1,sequence=5,width=6,query_rank=3,latent_rank=4,
               heads=4,content_dim=4,positional_dim=2,output_width=5,epsilon=1e-6,dtype='fp32',budget=None,
               resource_budget=None):
    from .extensions import _record_source
    from .graph import finite_json
    from .mla_budget import mla_preflight,require_admitted
    _record_source(mla_decode);_record_source(mla_input_shapes)
    policy=None
    if resource_budget is not None:
        from .resource_budget import normalize_budget
        policy=normalize_budget(resource_budget)
    if dtype!='fp32':raise ValueError('absorbed MLA decode v1 explicitly supports FP32 only')
    dims=dict(batch=batch,sequence=sequence,width=width,query_rank=query_rank,latent_rank=latent_rank,
              heads=heads,content_dim=content_dim,positional_dim=positional_dim,output_width=output_width)
    shapes=mla_input_shapes(**dims)
    if type(inputs) is not dict or set(inputs)!=set(shapes):raise ValueError('MLA requires every exact root input and no intermediate/unknown roots')
    if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or not 1e-38<=epsilon<=1:
        raise ValueError('MLA RMS epsilon must be in [1e-38,1]')
    preflight=mla_preflight(**dims,dtype=dtype,budget=budget)
    _record_source(require_admitted)
    require_admitted(preflight,stage='build')
    roots={n:Tensor(n,s,'fp32','activation' if n=='X' else 'weight') for n,s in shapes.items()}
    g=Graph('mla_decode_absorbed','fp32',roots)
    q=g.linear('query.down','X','W_DQ')
    q=g.op('query.norm','rms_gamma',[q,'q_gamma'],(batch,query_rank),epsilon=epsilon)
    instances={'query':{'operators':[op.name for op in g.w.operators]}}
    partial=[]
    for h in range(heads):
        start=len(g.w.operators);prefix=f'head{h}'
        qa=g.linear(prefix+'.query',q,prefix+'.WQK')
        qa=g.op(prefix+'.query.rows','reshape',[qa],(batch,1,latent_rank))
        qr=g.linear(prefix+'.pos_query',q,prefix+'.WQR')
        qr=g.op(prefix+'.rope','rope_pairs',[qr,'rope_cos','rope_sin'],(batch,positional_dim))
        qr=g.op(prefix+'.rope.rows','reshape',[qr],(batch,1,positional_dim))
        ck=g.op(prefix+'.latent.transpose','transpose',['latent_kv'],(batch,latent_rank,sequence),axes=[0,2,1])
        pk=g.op(prefix+'.pos.transpose','transpose',['pos_key'],(batch,positional_dim,sequence),axes=[0,2,1])
        a=g.mm(prefix+'.content_scores',qa,ck,batch,1,latent_rank,sequence)
        b=g.mm(prefix+'.pos_scores',qr,pk,batch,1,positional_dim,sequence)
        score=g.op(prefix+'.scores','add',[a,b],(batch,1,sequence))
        score=g.op(prefix+'.scale','scale',[score],(batch,1,sequence),factor=(content_dim+positional_dim)**-.5)
        prob=g.op(prefix+'.softmax','softmax',[score],(batch,1,sequence))
        context=g.mm(prefix+'.context',prob,'latent_kv',batch,1,sequence,latent_rank)
        y=g.mm(prefix+'.output',context,prefix+'.WVO',batch,1,latent_rank,output_width,True)
        partial.append(y)
        instances[prefix]={'operators':[op.name for op in g.w.operators[start:]]}
    start=len(g.w.operators);y=partial[0]
    for i,x in enumerate(partial[1:],1):y=g.op('merge.'+str(i),'add',[y,x],(batch,1,output_width))
    g.op('Y','reshape',[y],(batch,output_width))
    instances['merge']={'operators':[op.name for op in g.w.operators[start:]]}
    g.w.meta.update(inputs=finite_json(inputs),outputs=['Y'],mla=dict(schema='arch-explorer.mla-absorbed.v1',dimensions=dims,
        epsilon=epsilon,dtype=dtype,cache='immutable normalized latent and already-RoPE positional keys, including current token',
        boundary='absorbed weights are roots; no cache append, dense/MoE block, tokenizer or hardware accuracy',
        instances=instances))
    # An explicit request is part of the saved semantic identity. Default-budget
    # legacy inputs retain their exact graph payload, including old metadata.
    if budget is not None:g.w.meta['mla']['budget']=preflight['budget']
    if policy is not None:g.w.meta['resource_budget']=policy
    return g.w


def build_decode(params,*,tp=1):
    if tp!=1:raise ValueError('MLA decode uses explicit head placement, not implicit TP')
    return mla_decode(**params)
