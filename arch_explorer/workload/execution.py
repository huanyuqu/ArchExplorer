"""Workload target implementations and the materialized events lowering IR.

Selection knows semantic identities and declared target capabilities, not
model kinds, component identities or scheduling. Target-specific products
remain opaque here; only the common events Kernel/Program product is checked.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
from array import array
import math

from arch_explorer.components.base import Kernel, dtype_bytes
from .extensions import (_discover, _name, _signature, _record_source,
                         _record_identity, _record_operator, callable_identity,
                         get_operator)
from .graph import validate_operator, validate_tensor
from .operators import Operator, Tensor


def _tokens(value, field, *, tuple_only=False):
    if not isinstance(value, (tuple,) if tuple_only else (tuple, list, set, frozenset)):
        raise TypeError(f'{field} must be ' + ('a tuple' if tuple_only else 'a token sequence/set'))
    for item in value:
        _name(item)
    if len(set(value)) != len(value):
        raise ValueError(f'{field} contains duplicate tokens')
    return tuple(sorted(value))


@dataclass(frozen=True)
class ExecutionContext:
    target: str
    dtype: str
    tensors: dict
    options: dict = field(default_factory=dict)
    capabilities: tuple = ()

    def __post_init__(self):
        _name(self.target)
        if type(self.dtype) is not str:
            raise TypeError('execution dtype must be a string')
        dtype_bytes(self.dtype)
        if type(self.tensors) is not dict:
            raise TypeError('execution tensors must be a Tensor dictionary')
        for key, tensor in self.tensors.items():
            if type(tensor) is not Tensor or key != tensor.name:
                raise ValueError('execution tensor key must match its Tensor.name')
            validate_tensor(tensor)
        if type(self.options) is not dict:
            raise TypeError('execution options must be a dictionary')
        # The emitter is an evaluation-local object, not a public JSON option.
        object.__setattr__(self, 'options', dict(self.options))
        object.__setattr__(self, 'tensors', dict(self.tensors))
        object.__setattr__(self, 'capabilities', _tokens(self.capabilities, 'capabilities'))


def _value_snapshot(value):
    """Independent native JSON/tuple state; flat primitive rows use C loops."""
    kind=type(value)
    if value is None or kind in (bool,int,str):
        return kind,value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError('callback state requires finite native JSON values')
        return kind,array('d',[value]).tobytes()
    if kind is dict:
        if set(map(type,value))-{str}:
            raise TypeError('callback JSON keys must be native strings')
        return kind,{k:_value_snapshot(v) for k,v in value.items()}
    if kind in (list,tuple):
        leaf=type(value[0]) if value else None
        if leaf in (bool,int,float,str,type(None)) and set(map(type,value))=={leaf}:
            if leaf is float:
                if not all(map(math.isfinite,value)):
                    raise ValueError('callback state requires finite native JSON values')
                return kind,array('d',value).tobytes(),leaf,len(value)
            return kind,value.copy() if kind is list else tuple(value),leaf,len(value)
        return kind,tuple(_value_snapshot(v) for v in value),None,len(value)
    raise TypeError('callback state requires native JSON values, got '+kind.__name__)


def _matches_value(value, snapshot):
    kind,baseline=snapshot[:2]
    if type(value) is not kind:
        return False
    if kind is dict:
        return (value.keys()==baseline.keys() and not set(map(type,value))-{str}
                and all(_matches_value(value[k],saved) for k,saved in baseline.items()))
    if kind in (list,tuple):
        leaf,size=snapshot[2:]
        if len(value)!=size:
            return False
        if leaf is not None:
            if set(map(type,value))!={leaf}:
                return False
            return array('d',value).tobytes()==baseline if leaf is float else value==baseline
        return all(_matches_value(v,saved) for v,saved in zip(value,baseline))
    if kind is float:
        return array('d',[value]).tobytes()==baseline
    return value==baseline


_ADAPTER_GUARD=object()


def _callback_checkpoint(op, context):
    if type(context) is not ExecutionContext or type(context.tensors) is not dict:
        raise TypeError('callback requires an ExecutionContext Tensor dictionary')
    if any(type(t) is not Tensor or type(n) is not str or n!=t.name for n,t in context.tensors.items()):
        raise ValueError('callback Tensor membership/type differs')
    return (type(op),_value_snapshot(vars(op)),frozenset(vars(context)),
            _value_snapshot((context.target,context.dtype,context.capabilities,context.options)),
            {n:_value_snapshot(vars(t)) for n,t in context.tensors.items()})


def _callback_unchanged(op, context, saved):
    cls,operation,keys,fields,tensors=saved
    return (type(op) is cls and _matches_value(vars(op),operation)
            and type(context) is ExecutionContext and frozenset(vars(context))==keys
            and getattr(context,'_adapter_guard',None) is _ADAPTER_GUARD
            and _matches_value((context.target,context.dtype,context.capabilities,context.options),fields)
            and type(context.tensors) is dict and context.tensors.keys()==tensors.keys()
            and not set(map(type,context.tensors))-{str}
            and all(type(t) is Tensor and _matches_value(vars(t),tensors[n]) for n,t in context.tensors.items()))


def _fork_context(validated, *, options=None, operator=None, tensor_scope='graph'):
    """Internal evaluation-local fork of a freshly validated group context.

    Public construction always runs __post_init__. The adapter owns the
    validated template and checks callbacks plus the complete group graph;
    each callback still receives independent tensor and option dictionaries.
    """
    if type(validated) is not ExecutionContext or type(validated.tensors) is not dict:
        raise TypeError('context fork requires a validated ExecutionContext')
    options={} if options is None else options
    if type(options) is not dict:
        raise TypeError('execution options must be a dictionary')
    if tensor_scope not in ('graph','isolated-operands'):
        raise ValueError('implementation tensor_scope must be graph or isolated-operands')
    tensors=validated.tensors
    if tensor_scope=='isolated-operands':
        if operator is None:
            raise ValueError('isolated execution context requires its operator')
        selected={}
        for operand in (*operator.inputs,*operator.outputs):
            name=operand;trail=set()
            while name is not None:
                if name in trail:
                    raise ValueError('execution operand alias cycle')
                if name in selected:
                    break
                if name not in tensors:
                    raise ValueError('execution operand/alias is undeclared: '+str(name))
                trail.add(name);selected[name]=tensors[name];name=tensors[name].alias_of
        tensors={n:deepcopy(t) for n,t in selected.items()}
    else:
        tensors=dict(tensors)
    result=object.__new__(ExecutionContext)
    for name in ('target','dtype','capabilities'):
        object.__setattr__(result,name,getattr(validated,name))
    object.__setattr__(result,'tensors',tensors)
    object.__setattr__(result,'options',deepcopy(options))
    object.__setattr__(result,'_adapter_guard',_ADAPTER_GUARD)
    return result


@dataclass(frozen=True)
class ImplementationDefinition:
    id: str
    version: str
    operator: str
    target: str
    lower: object
    requirements: tuple = ()
    dtypes: tuple = ()
    supports: object = None
    tensor_scope: str = 'graph'


_IMPLEMENTATIONS = {}
_IMPLEMENTATION_STATES = {}


def _state(definition):
    def callback(fn):
        return (type(fn),id(fn),id(getattr(fn,'__code__',None)),
                id(getattr(type(fn).__call__,'__code__',None)),
                getattr(fn,'__module__',None),getattr(fn,'__qualname__',None),
                id(getattr(fn,'__defaults__',None)),id(getattr(fn,'__kwdefaults__',None)))
    def tokens(value):
        return (type(value),tuple((type(v),v) for v in value) if type(value) is tuple else None)
    return (type(definition),frozenset(vars(definition)),
            *((type(v),v) for v in (definition.id,definition.version,definition.operator,definition.target,definition.tensor_scope)),
            callback(definition.lower),tokens(definition.requirements),tokens(definition.dtypes),callback(definition.supports))


def _descriptor(definition):
    if not isinstance(definition, ImplementationDefinition):
        raise TypeError('implementation requires ImplementationDefinition')
    for value in (definition.id, definition.version, definition.operator, definition.target):
        _name(value)
    _tokens(definition.requirements, 'requirements', tuple_only=True)
    _tokens(definition.dtypes, 'dtypes', tuple_only=True)
    for dtype in definition.dtypes:
        dtype_bytes(dtype)
    _signature(definition.lower, None, None)
    if definition.supports is not None:
        _signature(definition.supports, None, None)
    if type(definition.tensor_scope) is not str or definition.tensor_scope not in ('graph','isolated-operands'):
        raise ValueError('implementation tensor_scope must be graph or isolated-operands')
    return definition


def register_implementation(definition):
    """Register one immutable ID/version identity; repeat only the same object."""
    _descriptor(definition)
    if definition.id.startswith('legacy:'):
        raise ValueError('implementation prefix legacy: is reserved for compatibility adapters')
    previous = _IMPLEMENTATIONS.get(definition.id)
    if previous is not None:
        if previous is not definition:
            raise ValueError(f'implementation {definition.id!r} already registered; identity/version conflict')
        _checked(definition)
        return definition
    _IMPLEMENTATIONS[definition.id] = definition
    _IMPLEMENTATION_STATES[definition.id] = _state(definition)
    return definition


@dataclass(frozen=True)
class _LegacyImplementation(ImplementationDefinition):
    semantic_type: str = 'operator'


def _legacy_lower(op, context):
    return op.kernel(context.dtype, rows=context.options.get('rows'), tile=context.options.get('tile'))


def _legacy(op, target):
    from .full_primitives import FullOperator
    if target != 'events':
        return None
    if type(op) is FullOperator:
        version, kind = '1', 'full'
    else:
        semantic = get_operator(op.op)
        if semantic.lower is None:
            return None
        version, kind = semantic.version, 'operator'
    return _LegacyImplementation('legacy:'+op.op, version, op.op, target,
                                 _legacy_lower, semantic_type=kind)


def _checked(definition):
    if not isinstance(definition, ImplementationDefinition):
        raise TypeError('implementation requires ImplementationDefinition')
    if type(definition) is _LegacyImplementation:
        _descriptor(definition)
        if definition.lower is not _legacy_lower or definition.id != 'legacy:'+definition.operator:
            raise ValueError('invalid legacy implementation adapter')
        return definition
    if (_IMPLEMENTATIONS.get(definition.id) is not definition
            or _IMPLEMENTATION_STATES.get(definition.id) != _state(definition)):
        raise ValueError(f'implementation {definition.id!r} is unregistered or was mutated')
    return definition


def _implementation_by_id(op, target, identifier):
    """Resolve an explicit immutable descriptor before choosing its view."""
    _discover();_name(identifier)
    definition=_IMPLEMENTATIONS.get(identifier)
    if definition is None:
        legacy=_legacy(op,target)
        definition=legacy if legacy is not None and legacy.id==identifier else None
    if definition is None:
        raise ValueError(_diagnostic(op,target,identifier,'explicit implementation does not exist'))
    _checked(definition)
    return definition


def _invoke_implementation(definition, function, op, context):
    # Public callers keep their full context and opaque emitter option contract.
    # Internal adapter forks carry only guarded JSON options.
    guarded=getattr(context,'_adapter_guard',None) is _ADAPTER_GUARD
    before=_callback_checkpoint(op,context) if guarded else None
    try:
        return function(op,context)
    finally:
        _checked(definition)
        if guarded and not _callback_unchanged(op,context,before):
            raise ValueError('execution callback mutated its operator, tensor context or options')


def implementation_source_objects(definition):
    """Return lower/supports callables; do not select, lower, or rebuild a graph."""
    _checked(definition)
    return tuple(value for value in (definition.lower, definition.supports) if value is not None)


def implementation_identity(definition):
    _checked(definition)
    value = dict(kind='implementation', id=definition.id, version=definition.version,
                 operator=definition.operator, target=definition.target,
                 requirements=sorted(definition.requirements), dtypes=sorted(definition.dtypes),
                 tensor_scope=definition.tensor_scope,
                 lower=callable_identity(definition.lower),
                 supports=None if definition.supports is None else callable_identity(definition.supports))
    if type(definition) is _LegacyImplementation:
        value['semantic_type'] = definition.semantic_type
    return value


def _diagnostic(op, context, identifier, reason):
    target=context if type(context) is str else context.target
    return (f'node={op.name!r} operator={op.op!r} implementation={identifier!r} '
            f'target={target!r}: {reason}')


def _compatible(definition, op, context):
    _checked(definition)
    reasons = []
    if definition.operator != op.op:
        reasons.append('operator differs')
    if definition.target != context.target:
        reasons.append('target differs')
    missing = set(definition.requirements) - set(context.capabilities)
    if missing:
        reasons.append('missing capabilities '+repr(sorted(missing)))
    if definition.dtypes and context.dtype not in definition.dtypes:
        reasons.append('unsupported dtype '+repr(context.dtype))
    if type(definition) is _LegacyImplementation:
        expected = _legacy(op, context.target)
        if expected is None or _state(expected) != _state(definition) or expected.semantic_type != definition.semantic_type:
            reasons.append('legacy semantic identity differs')
    if not reasons and definition.supports is not None:
        _record_source(definition.supports)
        try:
            supported = _invoke_implementation(definition,definition.supports,op,context)
        except (ValueError, NotImplementedError) as error:
            reasons.append(f'shape/layout constraint rejected: {error}')
        else:
            if type(supported) is not bool:
                raise TypeError(_diagnostic(op, context, definition.id, 'supports must return bool'))
            if not supported:
                reasons.append('shape/layout constraint rejected')
    return reasons


def select_implementation(op, target, *, dtype, tensors, capabilities=(), implementation=None, options=None):
    """Select the sole compatible implementation, or require an explicit ID."""
    context = ExecutionContext(target, dtype, tensors, {} if options is None else options, capabilities)
    return _select_in_context(op,context,implementation=implementation)


def _select_in_context(op, context, *, implementation=None):
    """Shared selection authority; internal callers supply a validated fork."""
    if type(context) is not ExecutionContext:
        raise TypeError('implementation selection requires ExecutionContext')
    target=context.target
    _discover()
    validate_operator(op, context.tensors)
    if implementation is not None:
        _name(implementation)
    if implementation is not None:
        selected = _implementation_by_id(op,target,implementation)
        reasons = _compatible(selected, op, context)
        if reasons:
            raise ValueError(_diagnostic(op, context, implementation, '; '.join(reasons)))
        return selected
    legacy = _legacy(op, target)
    candidates = list(_IMPLEMENTATIONS.values())
    if legacy is not None:
        candidates.append(legacy)
    compatible, rejected = [], []
    for definition in sorted(candidates, key=lambda item: item.id):
        if definition.operator != op.op or definition.target != target:
            continue
        reasons = _compatible(definition, op, context)
        if reasons:
            rejected.append(definition.id+': '+'; '.join(reasons))
        else:
            compatible.append(definition)
    if len(compatible) != 1:
        reason = ('ambiguous compatible implementations '+repr([d.id for d in compatible])
                  if compatible else 'no compatible implementation; rejected='+repr(rejected))
        raise ValueError(_diagnostic(op, context, implementation, reason))
    return compatible[0]


def lower_implementation(definition, op, context):
    """Lower through the selected identity; leave non-events products opaque."""
    if type(context) is not ExecutionContext:
        raise TypeError('lower_implementation requires ExecutionContext')
    validate_operator(op, context.tensors)
    reasons = _compatible(definition, op, context)
    if reasons:
        raise ValueError(_diagnostic(op, context, definition.id, '; '.join(reasons)))
    _record_operator(op)
    _record_source(definition.lower)
    # Identity computation is only needed when collecting provenance. Consumers
    # can request it directly outside a source_trace context.
    from .extensions import _SOURCE_TRACE
    if _SOURCE_TRACE.get() is not None:
        _record_identity(implementation_identity(definition))
    product = _invoke_implementation(definition,definition.lower,op,context)
    return as_program(product, op, context) if context.target == 'events' else product


def validate_kernel(kernel):
    """Single authority shared by legacy Operator.kernel and events programs."""
    if not isinstance(kernel, Kernel):
        raise TypeError('events lower must return Kernel or ExecutionProgram')
    if type(kernel.precision) is not str:
        raise TypeError('Kernel precision must be a string')
    dtype_bytes(kernel.precision)
    arity = {'gemm': 4, 'mvm': 3, 'vector': 2}.get(kernel.op)
    if arity is None:
        raise NotImplementedError(f'{kernel.op!r} is not represented by the common Kernel IR')
    if (not isinstance(kernel.shape, (tuple, list)) or len(kernel.shape) != arity
            or any(type(n) not in (int, float) or not math.isfinite(n) or n <= 0 for n in kernel.shape)):
        raise ValueError('lowered Kernel shape must be finite and positive with the declared arity')
    for count in (kernel.simt, kernel.sfu, kernel.row_simt, kernel.row_sfu):
        if type(count) not in (int, float) or not math.isfinite(count) or count < 0:
            raise ValueError('lowered Kernel operation counts must be finite and nonnegative')
    return kernel


def _names(names, field):
    if type(names) is not tuple:
        raise TypeError(f'{field} must be a tuple')
    for name in names:
        _name(name)


@dataclass(frozen=True)
class ExecutionStep:
    name: str
    kernel: Kernel
    inputs: tuple
    outputs: tuple
    deps: tuple = ()

    def __post_init__(self):
        _name(self.name)
        validate_kernel(self.kernel)
        for names, field in ((self.inputs, 'step inputs'), (self.outputs, 'step outputs'), (self.deps, 'step deps')):
            _names(names, field)
        if not self.outputs or len(set(self.outputs)) != len(self.outputs):
            raise ValueError('step outputs must be nonempty and unique')
        if len(set(self.deps)) != len(self.deps):
            raise ValueError('step deps must be unique')
        if set(self.inputs) & set(self.outputs):
            raise ValueError('in-place execution step writes are unsupported')


@dataclass(frozen=True)
class ExecutionProgram:
    tensors: tuple = ()
    steps: tuple = ()

    def __post_init__(self):
        if type(self.tensors) is not tuple or type(self.steps) is not tuple:
            raise TypeError('ExecutionProgram tensors/steps must be tuples')
        if not self.steps or any(type(step) is not ExecutionStep for step in self.steps):
            raise ValueError('ExecutionProgram requires nonempty ExecutionStep objects')
        for tensor in self.tensors:
            if type(tensor) is not Tensor:
                raise TypeError('ExecutionProgram tensors must be Tensor objects')
            validate_tensor(tensor)
            if tensor.kind != 'activation' or tensor.alias_of is not None or tensor.strides is not None:
                raise ValueError('temporary tensors require independent materialized activation storage')

    def to_dict(self):
        """JSON-safe IR facts; callbacks and target objects never appear here."""
        from .graph import finite_json
        return finite_json(dict(schema='arch-explorer.execution-program.v1',
                                tensors=[asdict(t) for t in self.tensors],
                                steps=[asdict(s) for s in self.steps]), tuples=True)


def as_program(value, op, context):
    """Validate the events product without invoking a lower or reference hook."""
    if type(context) is not ExecutionContext or context.target != 'events':
        raise ValueError('as_program only validates the events Kernel/Program target')
    # Also verifies original tensor and semantic contracts before use.
    validate_operator(op, context.tensors)
    if isinstance(value, Kernel):
        value = ExecutionProgram(steps=(ExecutionStep(op.name, value, tuple(op.inputs), tuple(op.outputs)),))
    if type(value) is not ExecutionProgram:
        raise TypeError('events lower must return Kernel or ExecutionProgram')
    value.__post_init__()
    temporary = {t.name: t for t in value.tensors}
    if len(temporary) != len(value.tensors):
        raise ValueError(f'{op.name}: duplicate temporary tensor name')
    if set(temporary) & set(context.tensors):
        raise ValueError(f'{op.name}: temporary tensor collides with graph tensor')
    allowed = set(op.inputs) | set(op.outputs) | set(temporary)
    writable = set(op.outputs) | set(temporary)
    produced, ancestors, used = {}, {}, set()
    for step in value.steps:
        step.__post_init__()
        if step.name in ancestors:
            raise ValueError(f'{op.name}: duplicate execution step {step.name!r}')
        if any(dep not in ancestors for dep in step.deps):
            raise ValueError(f'{op.name}/{step.name}: deps must reference preceding steps')
        parents = set(step.deps)
        for dep in step.deps:
            parents.update(ancestors[dep])
        for name in step.inputs:
            if name not in allowed or (name not in op.inputs and name not in produced):
                raise ValueError(f'{op.name}/{step.name}: undeclared or unproduced input {name!r}')
            if name in produced and produced[name] not in parents:
                raise ValueError(f'{op.name}/{step.name}: missing dependency on producer of {name!r}')
            used.add(name)
        for name in step.outputs:
            if name not in writable:
                raise ValueError(f'{op.name}/{step.name}: undeclared output or input overwrite {name!r}')
            if name in produced:
                raise ValueError(f'{op.name}/{step.name}: duplicate tensor producer {name!r}')
            produced[name] = step.name
        ancestors[step.name] = parents
    if set(produced) != writable:
        raise ValueError(f'{op.name}: incomplete output/temporary production {sorted(writable-set(produced))}')
    if not set(op.inputs) <= used:
        raise ValueError(f'{op.name}: unconsumed operator inputs {sorted(set(op.inputs)-used)}')
    if not set(temporary) <= used:
        raise ValueError(f'{op.name}: unused temporary tensors {sorted(set(temporary)-used)}')
    return value
