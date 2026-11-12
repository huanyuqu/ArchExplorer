"""Independent integer ledger for decoder performance graphs; no product imports.

Counts one tensor-parallel rank. This is logical storage/work, with no weights
allocated, no target timing inferred and no physical replica/capacity answer.
"""
from math import prod


def tiny_request(*, phase='prefill', dtype='fp16', layers=2, iterations=3):
    return dict(model=dict(name='qa-metadata',hidden=4,heads=2,kv_heads=1,head_dim=2,
        intermediate=6,layers=2,vocab=17,gated=True,norm='rmsnorm',activation='silu',
        source='independent synthetic hand calculation'),phase=phase,batch=1,seq_len=3,
        layers=layers,dtype=dtype,iterations=iterations)


def decoder_ledger(request, tp):
    model=request['model']
    batch=request['batch'];sequence=request['seq_len'];layers=request['layers']
    iterations=request.get('iterations',1);phase=request['phase']
    if phase not in ('prefill','decode'):raise ValueError('unknown phase')
    values=(batch,sequence,layers,iterations,tp,model['hidden'],model['heads'],
            model['kv_heads'],model['head_dim'],model['intermediate'])
    if any(type(v) is not int or v<1 for v in values):raise ValueError('positive integer dimensions required')
    if model['heads']%tp or model['intermediate']%tp:raise ValueError('incompatible query/FFN sharding')
    if model['kv_heads']%tp and tp%model['kv_heads']:raise ValueError('incompatible KV sharding')
    if type(model['gated']) is not bool:raise ValueError('gated must be boolean')
    size={'fp16':2,'bf16':2,'fp32':4}[request['dtype']]
    hidden=model['hidden'];head_dim=model['head_dim']
    query_heads=model['heads']//tp;kv_heads=max(1,model['kv_heads']//tp)
    inter=model['intermediate']//tp;query=sequence if phase=='prefill' else 1
    tokens=batch*query;up=(2 if model['gated'] else 1)*inter
    qkv=(query_heads+2*kv_heads)*head_dim
    # Four projections; gate/up are stored together. Attention has two products.
    weights=[(hidden,qkv),(query_heads*head_dim,hidden),(hidden,up),(inter,hidden)]
    kv=(2,batch,sequence,kv_heads,head_dim)
    score=(batch*query_heads,query,sequence)
    x=(tokens,hidden)
    # Two norms, attention projection/residual, FFN down/residual: six full-width
    # outputs. Other outputs are QKV, two score arrays, context, up and gate.
    temporaries=[x]*6+[(tokens,qkv),score,score,(tokens,query_heads*head_dim),
                       (tokens,up),(tokens,inter)]
    if tp>1:temporaries += [x,x]
    weight_elements=sum(prod(shape) for shape in weights)
    matrix=layers*(tokens*weight_elements+2*batch*query_heads*query*sequence*head_dim)
    weight_bytes=layers*weight_elements*size
    kv_bytes=layers*prod(kv)*size
    root_bytes=prod(x)*size+weight_bytes+kv_bytes
    temporary_bytes=layers*sum(prod(shape) for shape in temporaries)*size
    shapes=[x,*weights,kv,*temporaries]
    return dict(operators=layers*(12+(2 if tp>1 else 0)),tensors=1+layers*(17+(2 if tp>1 else 0)),
        root_bytes=root_bytes,materialized_root_bytes=0,weight_bytes=weight_bytes,kv_bytes=kv_bytes,
        temporary_bytes=temporary_bytes,total_tensor_bytes=root_bytes+temporary_bytes,
        root_python_bytes=0,max_rank=max(map(len,shapes)),
        max_dimension=max(max(shape) for shape in shapes),
        max_tensor_elements=max(prod(shape) for shape in shapes),
        matrix_fmas_per_iteration=matrix,total_matrix_fmas=matrix*iterations,
        vector_elements_per_iteration=layers*(4*prod(x)+prod(score)+tokens*inter),
        allreduce_bytes_per_iteration=layers*2*prod(x)*size if tp>1 else 0)
