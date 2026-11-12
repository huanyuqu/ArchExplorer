"""Absorbed MLA composed from physical blocks and ordinary semantic operators.

Scores retain the original full-row softmax. Context chunks accumulate using
the same increasing sequence-index FFMA order, seeded by explicit zero storage.
No graph executor, implicit SRAM, replicas or uncharged device copies live here.
"""
import hashlib
import json
import math


def _source_digest(inputs):
    return hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def mla_stream(*, inputs, batch=1, sequence=5, width=6, query_rank=3, latent_rank=4,
               heads=4, content_dim=4, positional_dim=2, output_width=5,
               epsilon=1e-6, dtype='fp32', chunk=64, slots=64,
               cache_mode='sharded', resource_budget=None):
    """Build a complete graph from canonical MLA roots and explicit placement slots.

    ``sharded`` stores one physical piece per batch/sequence chunk, distributed
    round-robin across slots. ``external`` retains the canonical roots at the
    declared external memory. Both modes perform explicit local workspace loads.
    Slots are logical indexes only; block_mapping binds them to actual ADF paths.
    """
    from .resource_budget import budget_scope
    with budget_scope(resource_budget):
        return _build(inputs=inputs,batch=batch,sequence=sequence,width=width,query_rank=query_rank,
            latent_rank=latent_rank,heads=heads,content_dim=content_dim,positional_dim=positional_dim,
            output_width=output_width,epsilon=epsilon,dtype=dtype,chunk=chunk,slots=slots,
            cache_mode=cache_mode,resource_budget=resource_budget)


def _build(*,inputs,batch,sequence,width,query_rank,latent_rank,heads,content_dim,positional_dim,
           output_width,epsilon,dtype,chunk,slots,cache_mode,resource_budget):
    from .mla_decode import mla_input_shapes
    from .operators import Tensor
    from .structural import Graph
    from .block_ops import partition_input
    from .reference_api import _input
    from .extensions import _record_source
    _record_source(mla_stream); _record_source(_build); _record_source(_input)
    dims=dict(batch=batch,sequence=sequence,width=width,query_rank=query_rank,latent_rank=latent_rank,
              heads=heads,content_dim=content_dim,positional_dim=positional_dim,output_width=output_width)
    shapes=mla_input_shapes(**dims)
    if (dtype!='fp32' or type(chunk) is not int or not 1<=chunk<=4096
            or type(slots) is not int or not 1<=slots<=4096 or cache_mode not in ('sharded','external')):
        raise ValueError('mla_stream.contract: FP32, positive chunk/slots and declared cache_mode required')
    if type(epsilon) not in (int,float) or not math.isfinite(epsilon) or not 1e-38<=epsilon<=1:
        raise ValueError('mla_stream.contract: epsilon outside [1e-38,1]')
    if type(inputs) is not dict or set(inputs)!=set(shapes):
        raise ValueError('mla_stream.contract: every exact canonical root required')
    original={n:Tensor(n,s,'fp32','activation' if n=='X' else 'weight') for n,s in shapes.items()}
    normalized={n:_input(inputs[n],t).tolist() for n,t in original.items()}
    roots, values, partitions, root_slots = {}, {}, {}, {}
    for n,t in original.items():
        if n in ('latent_kv','pos_key') and cache_mode=='sharded':
            pieces=partition_input(t,normalized[n],block_shape=(1,chunk,t.shape[-1]))
            partitions[n]=pieces['partition']
            roots.update({k:Tensor(**v) for k,v in pieces['tensors'].items()})
            values.update(pieces['inputs'])
            for i,p in enumerate(pieces['partition']['blocks']): root_slots[p['tensor']]=i%slots
        else:
            roots[n]=t; values[n]=normalized[n]
            root_slots[n]='external' if n in ('latent_kv','pos_key') else int(n.split('.')[0][4:])%slots if n.startswith('head') else 0
    g=Graph('mla_decode_block_stream','fp32',roots)
    placements={}; current=0
    def op(name,kind,ins,shape,**attrs):
        result=g.op(name,kind,ins,shape,**attrs);placements[name]=current;return result
    def mm(name,a,b,batch_,m,k,n,shared=False):
        result=g.mm(name,a,b,batch_,m,k,n,shared);placements[name]=current;return result
    def linear(name,x,w):
        b,k=g.w.tensors[x].shape;n=g.w.tensors[w].shape[-1]
        y=mm(name+'.mm',x,w,1,b,k,n,True)
        return op(name+'.reshape','reshape',[y],(b,n))
    def load(name,source,shape,starts,guards=()):
        return op(name,'block.load',[source,*guards],shape,starts=list(starts))
    def join(name,items,shape,axis):
        return items[0] if len(items)==1 else op(name,'concat',items,shape,axis=axis)
    def cache(name,b,s):
        if cache_mode=='external':return name,(b,s,0)
        i=b*math.ceil(sequence/chunk)+s//chunk
        return partitions[name]['blocks'][i]['tensor'],(0,0,0)
    q=linear('query.down','X','W_DQ')
    q=op('query.norm','rms_gamma',[q,'q_gamma'],(batch,query_rank),epsilon=epsilon)
    previous, partial={},[]
    for h in range(heads):
        current=h%slots;prefix=f'head{h}'
        gate=() if current not in previous else (previous[current],)
        query=load(prefix+'.query.load',q,(batch,query_rank),(0,0),gate)
        qa=linear(prefix+'.query',query,prefix+'.WQK')
        qr=linear(prefix+'.pos_query',query,prefix+'.WQR')
        qr=op(prefix+'.rope','rope_pairs',[qr,'rope_cos','rope_sin'],(batch,positional_dim))
        batches=[]
        for b in range(batch):
            p=prefix+f'.batch{b}'
            gate=(batches[-1],) if batches else (query,)
            a=load(p+'.query.load',qa,(1,latent_rank),(b,0),gate)
            a=op(p+'.query.rows','reshape',[a],(1,1,latent_rank))
            r=load(p+'.rope.load',qr,(1,positional_dim),(b,0),gate)
            r=op(p+'.rope.rows','reshape',[r],(1,1,positional_dim))
            scores=[]
            for s in range(0,sequence,chunk):
                n=min(chunk,sequence-s);p2=p+f'.chunk{s}'
                guards=(scores[-1],) if scores else (a,r)
                key,start=cache('latent_kv',b,s)
                ck=load(p2+'.key.load',key,(1,n,latent_rank),start,guards)
                key,start=cache('pos_key',b,s)
                pk=load(p2+'.pos.load',key,(1,n,positional_dim),start,guards)
                ck=op(p2+'.key.T','transpose',[ck],(1,latent_rank,n),axes=[0,2,1])
                pk=op(p2+'.pos.T','transpose',[pk],(1,positional_dim,n),axes=[0,2,1])
                c=mm(p2+'.content_scores',a,ck,1,1,latent_rank,n)
                z=mm(p2+'.pos_scores',r,pk,1,1,positional_dim,n)
                score=op(p2+'.scores','add',[c,z],(1,1,n))
                scores.append(op(p2+'.scale','scale',[score],(1,1,n),factor=(content_dim+positional_dim)**-.5))
            full=join(p+'.scores.join',scores,(1,1,sequence),2)
            prob=op(p+'.softmax','softmax',[full],(1,1,sequence))
            acc=op(p+'.context.zero','block.zero',[],(1,1,latent_rank))
            for i,s in enumerate(range(0,sequence,chunk)):
                n=min(chunk,sequence-s);p2=p+f'.context{s}'
                key,start=cache('latent_kv',b,s)
                ck=load(p2+'.load',key,(1,n,latent_rank),start,(acc,prob) if i==0 else (acc,))
                pr=op(p2+'.prob','slice',[prob],(1,1,n),starts=[0,0,s],stops=[1,1,s+n],steps=[1,1,1])
                acc=op(p2+'.accumulate','block.matmul_accumulate',[pr,ck,acc],(1,1,latent_rank))
            batches.append(mm(p+'.output',acc,prefix+'.WVO',1,1,latent_rank,output_width,True))
        y=join(prefix+'.output',batches,(batch,1,output_width),0)
        previous[current]=y;partial.append(y)
    current=0;y=partial[0]
    for h,x in enumerate(partial[1:],1):y=op('merge.'+str(h),'add',[y,x],(batch,1,output_width))
    op('Y','reshape',[y],(batch,output_width))
    g.w.meta.update(inputs=values,outputs=['Y'],block_stream=dict(schema='arch-explorer.mla-block-stream.v1',
        dimensions=dims,epsilon=epsilon,chunk=chunk,slots=slots,cache_mode=cache_mode,
        source_inputs_sha256=_source_digest(normalized),partitions=partitions,
        root_slots=root_slots,operator_slots=placements,
        numerics='absorbed FP32; dense full-row softmax; sequence-ordered seeded FFMA; ordered head merge',
        boundary='preloaded physical roots; independent block workspaces; explicit charged load and guard reads'))
    if resource_budget is not None:
        from .resource_budget import normalize_budget
        g.w.meta['resource_budget']=normalize_budget(resource_budget)
    return g.w


def block_mapping(workload, *, compute, memories, external_memory=None):
    """Bind the saved logical slots to complete physical v4 placement fields."""
    from arch_explorer.adf import MappingSpec
    from .extensions import _record_source
    _record_source(block_mapping)
    meta=workload.meta['block_stream'];slots=meta['slots']
    if (type(compute) not in (list,tuple) or type(memories) not in (list,tuple)
            or len(compute)!=slots or len(memories)!=slots
            or len(set(compute))!=slots or len(set(memories))!=slots
            or any(type(p) is not str or not p for p in (*compute,*memories))):
        raise ValueError('block.mapping: one distinct actual compute/memory path per slot required')
    if meta['cache_mode']=='external' and (type(external_memory) is not str or not external_memory
                                          or external_memory in memories):
        raise ValueError('block.mapping: distinct actual external memory path required')
    binding=lambda n,options={}:dict(id=n,version='1',options=options)
    rules={'matmul':binding('matmul.tiled',dict(tiles=dict(m=16,k=512,n=128)))}
    rules.update({kind:binding(kind+'.view') for kind in ('transpose','reshape','slice')})
    kinds={op.op for op in workload.operators}
    rules={kind:value for kind,value in rules.items() if kind in kinds}
    return MappingSpec('compile_graph_v4',dict(
        compute={n:compute[i] for n,i in meta['operator_slots'].items()},
        memory={n:memories[i] for n,i in meta['operator_slots'].items()},rules=rules,operator_rules={},
        storage={n:external_memory if i=='external' else memories[i] for n,i in meta['root_slots'].items()},
        replicas=[]))


def mla_stream_plan(*,batch=1,sequence=5,width=6,query_rank=3,latent_rank=4,heads=4,
                    content_dim=4,positional_dim=2,output_width=5,chunk=64,slots=64,
                    capacity=1024**2,cache_mode='sharded'):
    """Allocation-free per-slot root ledger and conservative workspace allowance.

    No graph, numpy arrays, input values or simulator service is constructed.
    The upper bound assumes the block_mapping view policy and serial guards.
    It is admission planning, not a successful compiler or aggregate-peak receipt.
    """
    from .mla_decode import mla_input_shapes
    from .extensions import _record_source
    _record_source(mla_stream_plan)
    dims=dict(batch=batch,sequence=sequence,width=width,query_rank=query_rank,latent_rank=latent_rank,
        heads=heads,content_dim=content_dim,positional_dim=positional_dim,output_width=output_width)
    shapes=mla_input_shapes(**dims)
    if (cache_mode not in ('sharded','external') or any(type(n) is not int or n<1 for n in (chunk,slots,capacity))
            or chunk>4096 or slots>4096):
        raise ValueError('mla_stream.plan: invalid chunk/slots/capacity/mode')
    root=[0]*slots;external=0
    for name,shape in shapes.items():
        if name in ('latent_kv','pos_key'):
            if cache_mode=='external':external+=4*math.prod(shape)
            else:
                for b in range(batch):
                    for s in range(0,sequence,chunk):
                        i=b*math.ceil(sequence/chunk)+s//chunk
                        root[i%slots]+=min(chunk,sequence-s)*shape[-1]*4
        else:
            i=int(name.split('.')[0][4:])%slots if name.startswith('head') else 0
            root[i]+=4*math.prod(shape)
    # Sequential score guards mean the previous key/position last readers
    # finish before allocating the next copy. Context loads depend on the
    # previous accumulator; the first also depends on complete softmax.
    # Thus score-copy, full-row assembly/softmax and context-copy workspaces
    # occupy separate phases. Allow all eager zero seeds and retained partials.
    c=min(chunk,sequence);owned=math.ceil(heads/slots)
    common=4*(8*batch*(query_rank+latent_rank+positional_dim)+
              owned*batch*(latent_rank+4*output_width)+8*latent_rank+8*output_width)
    phases=dict(score=4*(c*(latent_rank+positional_dim)+sequence+16*c),
                softmax=8*sequence,context=4*(c*latent_rank+sequence+4*latent_rank))
    workspace=common+max(phases.values())
    peaks=[n+workspace for n in root]
    chunks=math.ceil(sequence/chunk)
    operators=3+heads*(7+(batch>1)+batch*(7+(chunks>1)+11*chunks))
    # Every original matmul expands to tiles plus one real assembly. The
    # seeded accumulator is already a single semantic matrix task.
    score_tiles=sum(math.ceil(min(chunk,sequence-s)/128) for s in range(0,sequence,chunk))
    added_matrix_steps=math.ceil(batch/16)*math.ceil(query_rank/128)+heads*(
        math.ceil(batch/16)*(math.ceil(latent_rank/128)+math.ceil(positional_dim/128))+
        batch*(2*score_tiles+math.ceil(output_width/128)))
    roots_count=3*heads+7 if cache_mode=='external' else 3*heads+5+2*batch*chunks
    per_batch=(2*(latent_rank+positional_dim)+2*sequence*(latent_rank+positional_dim)+4*sequence+
               (sequence if chunks>1 else 0)+sequence+latent_rank+
               sequence*(latent_rank+1)+chunks*latent_rank+output_width)
    graph_output_elements=(3*batch*query_rank+heads*(batch*query_rank+2*batch*latent_rank+
        3*batch*positional_dim+batch*per_batch+(batch*output_width if batch>1 else 0))+heads*batch*output_width)
    matrix_tile_elements=batch*query_rank+heads*(batch*(latent_rank+positional_dim)+batch*(2*sequence+output_width))
    return dict(schema='arch-explorer.mla-block-plan.v1',dimensions=dims,chunk=chunk,slots=slots,
        cache_mode=cache_mode,capacity_bytes=capacity,root_bytes_by_slot=root,
        common_workspace_upper_bound=common,phase_workspace_upper_bounds=phases,
        workspace_upper_bound_per_slot=workspace,peak_upper_bound_by_slot=peaks,
        root_bytes=sum(root),external_bytes=external,root_fits=all(n<=capacity for n in root),
        conservative_plan_fits=all(n<=capacity for n in peaks),
        graph_operators=operators,mapped_steps=operators+added_matrix_steps,
        graph_tensors=roots_count+operators,mapped_tensors=roots_count+operators+added_matrix_steps,
        graph_total_tensor_bytes=sum(root)+external+4*graph_output_elements,
        mapped_total_tensor_bytes=sum(root)+external+4*(graph_output_elements+matrix_tile_elements),
        cache_pass_bytes=heads*batch*sequence*(2*latent_rank+positional_dim)*4,
        copy_boundary='key and position once for scores; key again for context; guards and arithmetic reads additional',
        status='static-plan-only; exact scheduled aggregate peak and service require actual execution',
        next_condition='external mode requires explicitly declared memory capacity and routed network; no free host storage')
