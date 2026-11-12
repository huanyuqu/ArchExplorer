"""Allocation-free MLA accounting and explicit build/plan admission limits.

Storage and operation counts describe the unchanged absorbed FP32 graph. Host
and JSON estimates are declared conservative accounting allowances, not measured
RSS, peak device capacity, or a replacement for the actual mapper/provider.
"""
import math

SCHEMA = 'arch-explorer.mla-budget.v1'
DEFAULT_LIMITS = {
    'root_bytes': 32 * 1024**2,
    'tensor_bytes': 256 * 1024**2,
    'temporary_bytes': 16 * 1024**3,
    'replica_bytes': 128 * 1024**2,
    'tensors': 8192,
    'operators': 8192,
    'graph_json_bytes': 512 * 1024**2,
    'plan_steps': 16384,
    'plan_json_bytes': 512 * 1024**2,
    'build_host_bytes': 2 * 1024**3,
}


def mla_budget(**limits):
    """Return a complete JSON budget; unknown/nonpositive/non-int limits refuse."""
    from .extensions import _record_source
    _record_source(mla_budget)
    if set(limits) - set(DEFAULT_LIMITS):
        raise ValueError('MLA budget unknown limits: ' + ','.join(sorted(set(limits) - set(DEFAULT_LIMITS))))
    result = {**DEFAULT_LIMITS, **limits}
    if any(type(v) is not int or v < 1 for v in result.values()):
        raise ValueError('MLA budget limits must be positive integers')
    return dict(schema=SCHEMA, limits=result)


def _budget(value):
    if value is None:
        return mla_budget()
    if type(value) is not dict or set(value) != {'schema', 'limits'} or value['schema'] != SCHEMA:
        raise ValueError('MLA budget requires exactly schema/limits, version v1')
    if type(value['limits']) is not dict or set(value['limits']) != set(DEFAULT_LIMITS):
        raise ValueError('MLA budget limits must list every v1 field; use mla_budget for defaults')
    return mla_budget(**value['limits'])


def mla_preflight(*, batch=1, sequence=5, width=6, query_rank=3, latent_rank=4,
                  heads=4, content_dim=4, positional_dim=2, output_width=5,
                  dtype='fp32', budget=None, cache_replicas=0, tiles=None):
    """Return exact scalar storage/FMA counts and explicit estimated plan limits.

    This function allocates neither roots, numpy arrays, Graph nor plan. Each
    cache replica is one extra copy of both full cache roots. ``tiles`` is the
    chosen common matmul.tiled shape; the default is m16/k512/n128. The returned
    ``admitted`` applies this ledger only. It never asserts mapper support,
    device capacity, actual host usage, successful execution or numerical proof.
    """
    from .extensions import _record_source
    from .mla_decode import mla_input_shapes
    _record_source(mla_preflight)
    _record_source(_budget)
    dims = dict(batch=batch, sequence=sequence, width=width, query_rank=query_rank,
                latent_rank=latent_rank, heads=heads, content_dim=content_dim,
                positional_dim=positional_dim, output_width=output_width)
    shapes = mla_input_shapes(**dims)
    if dtype != 'fp32':
        raise ValueError('absorbed MLA decode v1 explicitly supports FP32 only')
    if type(cache_replicas) is not int or cache_replicas < 0:
        raise ValueError('MLA cache_replicas must be a nonnegative integer')
    tiles = {'m': 16, 'k': 512, 'n': 128} if tiles is None else tiles
    if (type(tiles) is not dict or set(tiles) != {'m', 'k', 'n'}
            or any(type(v) is not int or not 1 <= v <= 4096 for v in tiles.values())):
        raise ValueError('MLA plan tiles must contain integer m/k/n in 1..4096')
    selected = _budget(budget)
    b, s, w, q, l, h, p, o = batch, sequence, width, query_rank, latent_rank, heads, positional_dim, output_width
    # Each record is a produced tensor. These are the same shapes as Graph's
    # materialized reshape and linear expansion; no payload or operator object.
    outputs = [('query.down.mm', (1,b,q), 'matmul', (1,b,w,q)),
               ('query.down.reshape', (b,q), 'reshape', None),
               ('query.norm', (b,q), 'rms_gamma', None)]
    for head in range(h):
        prefix = 'head' + str(head) + '.'
        for name, shape, kind, mm in (
            ('query.mm',(1,b,l),'matmul',(1,b,q,l)),
            ('query.reshape',(b,l),'reshape',None),
            ('query.rows',(b,1,l),'reshape',None),
            ('pos_query.mm',(1,b,p),'matmul',(1,b,q,p)),
            ('pos_query.reshape',(b,p),'reshape',None),
            ('rope',(b,p),'rope_pairs',None),
            ('rope.rows',(b,1,p),'reshape',None),
            ('latent.transpose',(b,l,s),'transpose',None),
            ('pos.transpose',(b,p,s),'transpose',None),
            ('content_scores',(b,1,s),'matmul',(b,1,l,s)),
            ('pos_scores',(b,1,s),'matmul',(b,1,p,s)),
            ('scores',(b,1,s),'add',None),
            ('scale',(b,1,s),'scale',None),
            ('softmax',(b,1,s),'softmax',None),
            ('context',(b,1,l),'matmul',(b,1,s,l)),
            ('output',(b,1,o),'matmul',(b,1,l,o))):
            outputs.append((prefix+name,shape,kind,mm))
    outputs += [('merge.'+str(j),(b,1,o),'add',None) for j in range(1,h)]
    outputs.append(('Y',(b,o),'reshape',None))
    root_elements = {n:math.prod(v) for n,v in shapes.items()}
    roots = sum(root_elements.values()) * 4
    cache = (root_elements['latent_kv'] + root_elements['pos_key']) * 4
    temporary = sum(math.prod(v)*4 for _,v,_,_ in outputs)
    tensor_count = len(shapes)+len(outputs)
    matrix = [mm for _,_,_,mm in outputs if mm is not None]
    matrix_tiles = sum(math.ceil(m/tiles['m'])*math.ceil(n/tiles['n']) for _,m,_,n in matrix)
    mapped_steps = len(outputs) + matrix_tiles
    mapped_tensors = tensor_count + matrix_tiles
    # Each tile output duplicates exactly its matrix output coverage; assembly
    # retains the original output. No padding work or per-K intermediate copy.
    matrix_tile_bytes = sum(mb*m*n*4 for mb,m,_,n in matrix)
    replicas = cache * cache_replicas
    # Fixed allowances are explicitly accounting estimates. Numeric JSON assumes
    # finite FP32 leaves; unexpected input containers are checked by the builder.
    graph_json = sum(root_elements.values())*32 + tensor_count*1024 + len(outputs)*2048
    plan_json = mapped_steps*8192 + mapped_tensors*2048
    build_host = sum(root_elements.values())*128 + tensor_count*2048 + len(outputs)*4096
    metrics = dict(root_bytes=roots,
                   tensor_bytes=max([4*e for e in root_elements.values()]+[math.prod(v)*4 for _,v,_,_ in outputs]),
                   temporary_bytes=temporary, replica_bytes=replicas,
                   tensors=tensor_count, operators=len(outputs), graph_json_bytes=graph_json,
                   plan_steps=mapped_steps, plan_json_bytes=plan_json, build_host_bytes=build_host)
    violations = [dict(metric=k,required=metrics[k],limit=v) for k,v in selected['limits'].items() if metrics[k]>v]
    build_violations=[v for v in violations if v['metric'] not in ('replica_bytes','plan_steps','plan_json_bytes')]
    return dict(schema='arch-explorer.mla-preflight.v1', dimensions=dims, dtype=dtype,
        root_shapes={n:list(v) for n,v in shapes.items()}, budget=selected, admitted=not violations,
        build_admitted=not build_violations,violations=violations, metrics=metrics,
        storage=dict(root_bytes=roots, root_activation_bytes=root_elements['X']*4,
            root_parameter_bytes=roots-cache-root_elements['X']*4, immutable_cache_bytes=cache,
            materialized_temporary_bytes=temporary, materialized_graph_bytes=roots+temporary,
            matrix_tile_bytes=matrix_tile_bytes, cache_replicas=cache_replicas, replica_bytes=replicas,
            mapped_materialized_bytes=roots+temporary+matrix_tile_bytes+replicas,
            view_candidate_copy_bytes=sum(math.prod(v)*4 for _,v,k,_ in outputs if k in ('reshape','transpose'))),
        graph=dict(tensors=tensor_count,operators=len(outputs),matrix_operators=len(matrix)),
        work=dict(matrix_fmas=sum(mb*m*k*n for mb,m,k,n in matrix),
            scalar_floating_storage='4 bytes; original K-ordered FP32 arithmetic retained'),
        plan=dict(tiles=dict(tiles),matrix_tiles=matrix_tiles,steps=mapped_steps,tensors=mapped_tensors,
            excludes='ADF placement/transfer/storage/control tasks; obtain these from the actual mapped plan'),
        host_estimate=dict(build_bytes=build_host,graph_json_bytes=graph_json,plan_json_bytes=plan_json,
            policy='128 bytes per root scalar; 2 KiB/tensor + 4 KiB/operator build overhead; JSON 32 bytes/scalar + metadata allowances; not measured RSS'),
        scope='shape-only admission, no allocation, reference, compile, capacity or service success claim')


def require_admitted(preflight, *, stage='all'):
    """Raise before copying; graph construction does not select a mapping plan."""
    if stage not in ('all','build'):raise ValueError('MLA admission stage must be all or build')
    violations=[v for v in preflight['violations'] if stage=='all' or v['metric'] not in ('replica_bytes','plan_steps','plan_json_bytes')]
    if violations:
        raise ValueError('MLA resource budget exceeded: ' + '; '.join(
            v['metric']+' required='+str(v['required'])+' limit='+str(v['limit']) for v in violations))
