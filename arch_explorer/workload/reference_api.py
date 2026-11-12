"""Public bounded graph reference execution, independent of GPU providers.

Registered workload initializers own synthetic/model inputs. OperatorDefinition
reference callbacks own numerical semantics. No timing or native execution here.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import math
from collections import Counter
import numpy as np

from .extensions import _record_source, get_operator, source_trace
from .reference_execution import (_DTYPES, _array_facts, _builtin_preflight,
                                  _builtin_initialize, _check_values, REFERENCE_LIMITS)
from .graph import finite_json

_REFERENCES = {}

@dataclass(frozen=True)
class ReferenceDefinition:
    kind:str
    version:str
    initialize:object
    preflight:object=None
    check_values:object=None

def register_reference(definition):
    if not isinstance(definition,ReferenceDefinition):
        raise TypeError('register_reference requires ReferenceDefinition')
    if any(type(x) is not str or not x or x.strip()!=x for x in (definition.kind,definition.version)):
        raise ValueError('reference kind/version must be nonempty identities')
    inspect.signature(definition.initialize).bind(None,None,[])
    if definition.preflight is not None:
        inspect.signature(definition.preflight).bind(None)
    if definition.check_values is not None:
        inspect.signature(definition.check_values).bind(None,None,[])
    if definition.kind in _REFERENCES and _REFERENCES[definition.kind] is not definition:
        raise ValueError('reference initializer identity already registered: '+definition.kind)
    _REFERENCES[definition.kind]=definition
    return definition

def _source(function):
    path=inspect.getsourcefile(function)
    if not path:
        raise ValueError('reference implementation source unavailable')
    from pathlib import Path
    from hashlib import sha256
    return {'module':function.__module__,'qualname':function.__qualname__,
            'file_name':Path(path).name,'sha256':sha256(Path(path).read_bytes()).hexdigest()}

def _input(value,tensor):
    if tensor.dtype in ('int64','bool'):
        from .integer import exact_storage
        array=exact_storage(value,tensor.dtype,shape=tensor.shape)
        array.flags.writeable=False
        return array
    def numeric(item):
        if isinstance(item,(list,tuple)):
            return all(numeric(v) for v in item)
        if isinstance(item,np.ndarray):
            return item.dtype.kind in 'fiu'
        return type(item) in (int,float) or (isinstance(item,np.number) and item.dtype.kind in 'fiu')
    if not numeric(value):
        raise ValueError('input must contain numeric values without bool/string coercion: '+tensor.name)
    array=np.asarray(value)
    if array.dtype.kind not in 'fiu' or array.shape!=tensor.shape or not np.isfinite(array).all():
        raise ValueError('input shape/type/finite contract differs: '+tensor.name)
    if tensor.dtype=='int32' and (np.any(array!=np.trunc(array))
            or np.any(array<-(2**31)) or np.any(array>=2**31)):
        raise ValueError('input integer storage range differs: '+tensor.name)
    with np.errstate(over='raise',invalid='raise'):
        if tensor.dtype in ('fp32','fp16','bf16'):
            from .precision import round_storage
            array=round_storage(array,tensor.dtype)
        else:
            array=np.array(array,dtype=_DTYPES[tensor.dtype],order='C',copy=True)
    if not np.isfinite(array).all():
        raise ValueError('input overflowed declared storage: '+tensor.name)
    array.flags.writeable=False
    return array

def execute_reference(spec, *, inputs=None, _initialize_only=False):
    from .resource_budget import budget_scope, policy_for_spec
    with budget_scope(policy_for_spec(spec)):
        return _execute_reference(spec,inputs=inputs,_initialize_only=_initialize_only)


def _execute_reference(spec, *, inputs=None, _initialize_only=False):
    """Execute a complete graph using registered semantics, returning readonly arrays.

    Inputs may be explicit or graph.meta['inputs']; a registered initializer is
    used only when neither is supplied. Missing/extra inputs are never inferred.
    """
    from .api import workload_from_spec
    from .full_primitives import FullOperator,reference as full_reference
    from .extensions import _discover
    _discover()
    adapter=_REFERENCES.get(spec.kind)
    with source_trace() as used:
        if adapter is not None and adapter.preflight is not None:
            _record_source(adapter.preflight);adapter.preflight(spec)
        graph=workload_from_spec(spec)
        # Performance-only graphs may declare large logical weights without
        # allocating them. Numeric/root initialization must reserve all roots
        # before a registered initializer or array conversion can allocate.
        from .resource_budget import check_graph_budget
        check_graph_budget(graph,initialize_roots=True)
        sizes=[math.prod(t.shape) for t in graph.tensors.values()]
        total_bytes=sum(t.bytes for t in graph.tensors.values())
        if (not _initialize_only and (any(t.dtype not in _DTYPES for t in graph.tensors.values())
                or max(sizes,default=0)>REFERENCE_LIMITS['max_tensor_elements']
                or total_bytes>REFERENCE_LIMITS['max_total_tensor_bytes']
                or len(graph.operators)>4096)):
            raise ValueError('graph reference storage/operator bound exceeded')
        produced={n for op in graph.operators for n in op.outputs}
        external=[n for n in graph.tensors if n not in produced]
        from .graph import graph_outputs
        outputs=graph_outputs(graph)
        metadata={}
        used_initializer=False
        values=inputs if inputs is not None else graph.meta.get('inputs')
        if values is None and adapter is not None:
            _record_source(adapter.initialize)
            used_initializer=True
            initialization=adapter.initialize(spec,graph,external)
            if (type(initialization) is not dict
                    or set(initialization)-{'inputs','outputs','metadata'} or 'inputs' not in initialization):
                raise ValueError('initializer must return inputs and optional outputs/metadata')
            values=initialization['inputs'];outputs=initialization.get('outputs',outputs)
            metadata=finite_json(initialization.get('metadata',{}),tuples=True)
        if type(values) is not dict or set(values)!=set(external):
            raise ValueError('reference inputs must exactly bind every external input/weight')
        if (not isinstance(outputs,(list,tuple)) or not outputs or len(set(outputs))!=len(outputs)
                or any(n not in produced for n in outputs)):
            raise ValueError('reference outputs must identify unique produced tensors')
        arrays={n:_input(values[n],graph.tensors[n]) for n in external}
        facts={n:_array_facts(arrays[n],graph.tensors[n]) for n in external}
        if _initialize_only:
            return {'graph':graph,'tensors':arrays,'input_tensors':external,'outputs':{},
                    'reference':dict(complete=False,executed_operators=0,input_facts=facts,
                        mode='root validation and storage rounding only; no operator reference execution')}
        versions,steps,producers=[],[],{}
        for op in graph.operators:
            definition=get_operator(op.op)
            function=getattr(definition,'reference',None)
            # FullOperator is an explicit builtin semantic adapter, not a model/name inference.
            if function is None and type(op) is FullOperator:
                function=full_reference
            if not callable(function):
                raise NotImplementedError('missing numerical reference for '+op.name+'/'+op.op)
            _record_source(function)
            operands=[arrays[n] for n in op.inputs]
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                returned=function(op,operands)
            if not isinstance(returned,dict):
                if len(op.outputs)!=1:
                    raise ValueError('multi-output reference must return exact output-name dictionary')
                returned={op.outputs[0]:returned}
            if set(returned)!=set(op.outputs):
                raise ValueError('reference returned missing/extra outputs at '+op.name)
            for name in op.outputs:
                tensor=graph.tensors[name]
                value=np.asarray(returned[name])
                if value.dtype.kind not in 'bfiu':
                    raise ValueError('reference output must use numeric storage: '+name)
                if value.shape!=tensor.shape:
                    raise ValueError('reference output shape mismatch: '+op.name+'/'+name)
                if tensor.alias_of is None:
                    if tensor.dtype in ('fp32','fp16','bf16') and np.isfinite(value).all():
                        from .precision import round_storage
                        value=round_storage(value,tensor.dtype)
                    elif tensor.dtype in ('int64','bool'):
                        from .integer import exact_storage
                        value=exact_storage(value,tensor.dtype,shape=tensor.shape)
                    else:
                        value=np.array(value,dtype=_DTYPES[tensor.dtype],order='C',copy=True)
                else:
                    if value.dtype!=_DTYPES[tensor.dtype] or not np.shares_memory(value,arrays[tensor.alias_of]):
                        raise ValueError('reference alias materialized or changed precision: '+name)
                    if tensor.strides is not None and tuple(s//value.itemsize for s in value.strides)!=tensor.strides:
                        raise ValueError('reference alias strides mismatch: '+name)
                    from .storage import resolve_layout
                    layout=resolve_layout(graph.tensors,name)
                    root=arrays[layout.storage]
                    offset=(value.__array_interface__['data'][0]-root.__array_interface__['data'][0])//value.itemsize
                    if offset!=layout.offset:raise ValueError('reference alias offset mismatch: '+name)
                    value.setflags(write=False)
                if adapter is not None and adapter.check_values is not None:
                    _record_source(adapter.check_values);adapter.check_values(op,value,operands)
                elif not np.isfinite(value).all():
                    raise ValueError('reference returned nonfinite result: '+name)
                value.flags.writeable=False;arrays[name]=value;facts[name]=_array_facts(value,tensor)
            versions.append({'node':op.name,'operator':op.op,'version':getattr(definition,'version','1'),
                             'reference':_source(function)})
            steps.append({'index':len(steps),'name':op.name,'op':op.op,'inputs':list(op.inputs),
                          'outputs':list(op.outputs),
                          'dependencies':sorted({producers[n] for n in op.inputs if n in producers}),
                          'status':'executed'})
            producers.update({n:op.name for n in op.outputs})
        if set(arrays)!=set(graph.tensors) or any(not np.isfinite(arrays[n]).all() for n in outputs):
            raise ValueError('incomplete graph reference or nonfinite final outputs')
    reference={'schema':'arch-explorer.graph-reference.v1','kind':'synthetic',
        'model_kind':spec.kind,'storage_dtype':graph.dtype,'numpy_version':np.__version__,
        'input_tensors':external,'outputs':list(outputs),'tensors':facts,'steps':steps,
        'operator_coverage':dict(Counter(op.op for op in graph.operators)),
        'complete':True,'executed_operators':len(steps),'missing_operators':[],
        'declared_tensor_bytes':total_bytes,'limits':dict(REFERENCE_LIMITS),
        'semantic_implementations':versions,'initializer':None if not used_initializer else {
            'kind':adapter.kind,'version':adapter.version,'source':_source(adapter.initialize)},
        'metadata':metadata,'source':{'implementation':[_source(fn) for fn in used],
                                   'identities':list(used.identities)},
        'limitations':['synthetic bound inputs/reference; no framework compiler equivalence, native execution or hardware accuracy']}
    finite_json(reference,tuples=True)
    return {'graph':graph,'tensors':arrays,'outputs':{n:arrays[n] for n in outputs},
            'input_tensors':external,'reference':reference,'_source_objects':tuple(used)}

def initialize_reference_inputs(spec):
    """Validate every root through the same initializer/storage authority, executing no operator."""
    return execute_reference(spec,_initialize_only=True)

_BUILTINS=tuple(ReferenceDefinition(kind,'1',_builtin_initialize,_builtin_preflight,_check_values)
                for kind in ('vit','clip'))
for _definition in _BUILTINS:
    register_reference(_definition)

