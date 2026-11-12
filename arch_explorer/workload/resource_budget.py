"""Saved, per-request engineering budgets; no process-global limit mutation."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math

SCHEMA = 'arch-explorer.resource-budget.v1'
DEFAULTS = dict(max_source_bytes=16*1024**2, max_nodes=256, max_rank=4,
    max_dimension=4096, max_tensor_elements=1_000_000, max_root_bytes=32*1024**2,
    max_total_tensor_bytes=32*1024**2, max_steps=512, max_matrix_fmas=100_000_000,
    max_tiles=4096, max_host_bytes=8*1024**3, reference_mode='execute')
_ACTIVE = ContextVar('arch_explorer_resource_budget', default=None)

def normalize_budget(value):
    """Expand an explicit JSON policy; omitted policy preserves legacy guards."""
    from .extensions import _record_source
    _record_source(normalize_budget)
    if type(value) is not dict or set(value)-set(DEFAULTS)-{'schema'}:
        raise ValueError('resource_budget requires known JSON policy fields')
    if 'schema' in value and value['schema'] != SCHEMA:
        raise ValueError('resource_budget schema differs')
    p = dict(DEFAULTS);p.update({k:v for k,v in value.items() if k!='schema'})
    for key in set(DEFAULTS)-{'reference_mode'}:
        if type(p[key]) is not int or not 1 <= p[key] <= 2**63-1:
            raise ValueError('resource_budget '+key+' must be a positive integer')
    if p['reference_mode'] not in ('execute','inputs'):
        raise ValueError('resource_budget reference_mode must be execute or inputs')
    if p['max_rank']>32 or p['max_dimension']>2**31-1 or p['max_host_bytes']>16*1024**3:
        raise ValueError('resource_budget exceeds the declared rank/int32-index/16GiB host domain')
    if p['max_source_bytes']*8>p['max_host_bytes'] or p['max_root_bytes']>p['max_host_bytes']:
        raise ValueError('resource_budget source/root policy exceeds host reserve')
    return dict(schema=SCHEMA,**p)

@contextmanager
def budget_scope(value):
    # None inherits an enclosing validated graph/compiler scope.
    token = None if value is None else _ACTIVE.set(normalize_budget(value))
    try:yield _ACTIVE.get()
    finally:
        if token is not None:_ACTIVE.reset(token)

def limit(name, legacy):
    from .extensions import _record_source
    _record_source(limit)
    value=_ACTIVE.get()
    return legacy if value is None else value[name]

def policy_for_spec(spec):
    p=spec.params
    return p.get('resource_budget',p.get('meta',{}).get('resource_budget'))

def graph_budget(function):
    @wraps(function)
    def scoped(graph,*args,**kwargs):
        with budget_scope(graph.meta.get('resource_budget')):
            return function(graph,*args,**kwargs)
    return scoped

def check_shape(shape, *, legacy_rank=4, legacy_dimension=4096, legacy_elements=1_000_000):
    if (len(shape)>limit('max_rank',legacy_rank)
            or any(type(n) is not int or not 1<=n<=limit('max_dimension',legacy_dimension) for n in shape)
            or math.prod(shape)>limit('max_tensor_elements',legacy_elements)):
        raise ValueError('resource_budget logical shape/rank/dimension/elements exceeded')
    return tuple(shape)

def estimate_graph(graph, *, initialize_roots=False):
    produced={n for op in graph.operators for n in op.outputs}
    roots={n:t for n,t in graph.tensors.items() if n not in produced}
    root_bytes=sum(t.bytes for t in roots.values())
    tensor_bytes=sum(t.bytes for t in graph.tensors.values() if t.alias_of is None)
    # Lists plus numeric leaves are copied by finite-JSON source/plan snapshots.
    # This is a conservative admission estimate, separate from physical SRAM.
    stored=graph.meta.get('inputs',{})
    initialized=[t for n,t in roots.items() if initialize_roots or (type(stored) is dict and n in stored)]
    root_host=sum(math.prod(t.shape)*40+56*sum(math.prod(t.shape[:j]) for j in range(len(t.shape))) for t in initialized)
    metadata_bytes=256*(len(graph.tensors)+len(graph.operators))+64*sum(len(op.inputs)+len(op.outputs) for op in graph.operators)
    return dict(root_bytes=root_bytes,materialized_root_bytes=sum(t.bytes for t in initialized),
        weight_bytes=sum(t.bytes for t in roots.values() if t.kind=='weight'),
        temporary_bytes=tensor_bytes-root_bytes,total_tensor_bytes=tensor_bytes,root_python_bytes=root_host,
        estimated_host_bytes=root_host*12+metadata_bytes*12,operators=len(graph.operators),tensors=len(graph.tensors),
        physical_memory_note='host admission estimate; actual memory placement/replica/lifetime is in the compilation plan')

def check_graph_budget(graph, *, initialize_roots=False):
    value=_ACTIVE.get()
    if value is None:return
    facts=estimate_graph(graph,initialize_roots=initialize_roots)
    for field,bound in (('materialized_root_bytes','max_root_bytes'),('total_tensor_bytes','max_total_tensor_bytes'),
                        ('estimated_host_bytes','max_host_bytes'),('operators','max_steps')):
        if facts[field]>value[bound]:
            raise ValueError('resource_budget '+field+' exceeded '+bound+': '+str(facts[field])+' > '+str(value[bound]))
