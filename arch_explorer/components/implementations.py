"""Component simulation implementations, separate from workload lowering.

Registration is descriptor-only. Instance construction is explicit and checks
the execution domain before calling any factory. No native library is loaded.
"""
from dataclasses import asdict, dataclass, field
import json
import math
import re

from arch_explorer.adf.model import Param, SpacePoint
from arch_explorer.adf.registry import COMPONENTS, Conventions, ParamSpec

_IMPLEMENTATIONS = {}
_IMPLEMENTATION_STATES = {}
_LEGACY = {}
_ALIASES = {}


def _token(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', value):
        raise ValueError('invalid component implementation '+label)
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def _parameter_descriptor(spec):
    result=asdict(spec)
    if isinstance(spec.domain,tuple) and len(spec.domain)==2:
        result['domain']={'min':spec.domain[0],'max':spec.domain[1]}
    return result


@dataclass(frozen=True)
class ImplementationBinding:
    id: str
    version: str
    params: dict = field(default_factory=dict)

    def to_dict(self):
        _token(self.id, 'id'); _token(self.version, 'version')
        if not isinstance(self.params, dict) or any(not isinstance(p, Param) for p in self.params.values()):
            raise TypeError('implementation params must contain Param values')
        return dict(id=self.id, version=self.version,
                    params={k: p.to_dict() for k, p in sorted(self.params.items())})

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {'id', 'version', 'params'} or not isinstance(data['params'], dict):
            raise ValueError('exact implementation id/version/params required')
        result = cls(data['id'], data['version'], {k: Param.from_dict(v) for k, v in data['params'].items()})
        result.to_dict()
        return result


@dataclass(frozen=True)
class ComponentImplementation:
    component: str
    id: str
    version: str
    source: object
    factory: object
    execution_domain: str
    input_contract: str
    result_scope: str
    params: dict = field(default_factory=dict)
    execution_modes: tuple = ('demand',)
    capabilities: dict = field(default_factory=dict)
    conventions: object = None
    validate: object = None
    hardware_parameters: object = None  # None: original full hardware validation; tuple: explicit consumed subset
    adapters: tuple = ()  # Explicit task converters; storage/network mechanisms need none.

    def descriptor(self):
        for key in ('component', 'id', 'version', 'execution_domain', 'input_contract'):
            _token(getattr(self, key), key)
        if not self.source or not isinstance(self.source, (str, dict)) or not self.result_scope:
            raise ValueError('implementation source and result_scope required')
        if not callable(self.factory) or self.validate is not None and not callable(self.validate):
            raise TypeError('implementation factory/validator must be callable')
        if (not isinstance(self.params, dict) or any(not isinstance(v, ParamSpec) for v in self.params.values())
                or not isinstance(self.capabilities, dict)):
            raise TypeError('implementation parameter declarations/capabilities required')
        if (not isinstance(self.execution_modes, tuple) or not self.execution_modes
                or set(self.execution_modes)-{'demand', 'stateful', 'native'}):
            raise ValueError('explicit demand/stateful/native execution modes required')
        if self.conventions is not None and not isinstance(self.conventions, Conventions):
            raise TypeError('implementation conventions must be Conventions')
        if self.hardware_parameters is not None and (not isinstance(self.hardware_parameters, tuple)
                or len(set(self.hardware_parameters)) != len(self.hardware_parameters)):
            raise TypeError('hardware_parameters must be None or a tuple of unique parameter names')
        if type(self.adapters) is not tuple or len(set(self.adapters)) != len(self.adapters):
            raise TypeError('adapters must be a tuple of unique IDs')
        for adapter in self.adapters:
            _token(adapter, 'adapter')
        def identity(fn):
            return None if fn is None else fn.__module__+':'+fn.__qualname__
        result = {k: getattr(self, k) for k in ('component', 'id', 'version', 'source', 'execution_domain',
                  'input_contract', 'result_scope', 'capabilities')}
        result.update(parameters={k: _parameter_descriptor(v) for k, v in self.params.items()},
                      execution_modes=list(self.execution_modes), factory=identity(self.factory),
                      validator=identity(self.validate),
                      hardware_parameters=None if self.hardware_parameters is None else list(self.hardware_parameters),
                      adapters=list(self.adapters),
                      conventions=None if self.conventions is None else asdict(self.conventions))
        return json.loads(_json(result))


def register_component_implementation(definition):
    if not isinstance(definition, ComponentImplementation):
        raise TypeError('ComponentImplementation required (not an operator ImplementationDefinition)')
    state = _json(definition.descriptor())
    if definition.component not in COMPONENTS._items:
        raise ValueError('register the hardware component before its implementation')
    key = (definition.component, definition.id, definition.version)
    if key in _IMPLEMENTATIONS:
        if _IMPLEMENTATIONS[key] is definition and _IMPLEMENTATION_STATES[key] == state:
            return definition
        raise ValueError('component implementation already registered: '+str(key))
    overlap = set(definition.params) & set(COMPONENTS._items[definition.component].params)
    if overlap:
        raise ValueError('implementation parameters overlap common hardware parameters: '+str(sorted(overlap)))
    _IMPLEMENTATIONS[key] = definition
    _IMPLEMENTATION_STATES[key] = state
    return definition


def _check(definition):
    key = (definition.component, definition.id, definition.version)
    if _IMPLEMENTATION_STATES.get(key) != _json(definition.descriptor()):
        raise ValueError('registered component implementation was mutated')


def component_implementations(component=None):
    from .extensions import discover_components
    discover_components()
    result = []
    for key, definition in sorted(_IMPLEMENTATIONS.items()):
        if component is None or key[0] == component:
            _check(definition)
            result.append(definition.descriptor())
    return result


def register_legacy_implementation(cls):
    """Freeze the pre-multiple-implementation default of an existing type."""
    key = cls.name
    if key not in _LEGACY:
        _LEGACY[key] = ComponentImplementation(
            key, 'builtin.'+key, '1', {'basis': cls.__module__+':'+cls.__qualname__}, cls.build,
            'events', 'events.component.v1', 'Existing event model; original declared limitations unchanged',
            execution_modes=tuple(cls.execution_modes), adapters=('events',) if cls.role == 'compute' else (),
            capabilities={'legacy_default': True,
                          'operations': 'kernel_demand / memory access / route, according to declared hardware role; factory validates support',
                          'precision': 'explicit hardware precision when consumed; no implicit conversion',
                          'shape': 'Kernel gemm=(batch,m,k,n), mvm=(batch,m,k), vector=(rows,cols); applicable subset validated by factory',
                          'port_protocol': 'events.demand.v1',
                          **getattr(cls, 'implementation_capabilities', {})}, conventions=cls.conventions)
    register_component_implementation(_LEGACY[key])


def _parameters(point, declarations):
    unknown = set(point.params)-set(declarations)
    if unknown:
        raise ValueError('unknown implementation parameters: '+str(sorted(unknown)))
    for key, spec in declarations.items():
        if key not in point.params:
            if spec.required:
                raise ValueError('implementation requires '+key)
            continue
        p = point.params[key]
        expected = {'int': int, 'float': (int, float), 'str': str, 'bool': bool}[spec.kind]
        if (not isinstance(p, Param) or p.unit != spec.unit
                or isinstance(p.value, bool) and spec.kind != 'bool' or not isinstance(p.value, expected)):
            raise ValueError('implementation parameter type/unit differs: '+key)
        if isinstance(p.value, (int, float)) and not math.isfinite(p.value):
            raise ValueError('implementation parameter must be finite: '+key)
        if spec.domain is not None:
            ok = (spec.domain['min'] <= p.value <= spec.domain['max'] if isinstance(spec.domain, dict)
                  else spec.domain[0] <= p.value <= spec.domain[1] if isinstance(spec.domain, tuple) and len(spec.domain)==2
                  else p.value in spec.domain)
            if not ok:
                raise ValueError('implementation parameter outside declared domain: '+key)


class ResolvedComponent:
    def __init__(self, hardware, implementation, point, binding):
        self.hardware, self.implementation, self.point, self.binding = hardware, implementation, point, binding
        self.name, self.role = hardware.name, hardware.role
        self.params = dict(hardware.params, **implementation.params)
        self.conventions = implementation.conventions or hardware.conventions
        self.execution_modes = implementation.execution_modes
        self.assembly_hooks = hardware.assembly_hooks

    def __getattr__(self, name):
        return getattr(self.hardware, name)

    def materialize(self):
        return SpacePoint(self.point.name, self.name, dict(self.point.params, **self.binding.params), self.point.repeat)

    def identity(self):
        return dict(component=self.name, implementation=self.binding.to_dict(),
                    declaration=self.implementation.descriptor())

    def build(self, point, context):
        # Old ComponentType.build methods may use super()/isinstance(self,...).
        # Only their frozen original adapter retains that exact receiver.
        receiver=self.hardware if self.implementation is _LEGACY.get(self.name) else self
        return self.implementation.factory(receiver, self.materialize(), context)


def resolve_component(point, *, execution_domain=None):
    if not isinstance(point, SpacePoint):
        raise TypeError('SpacePoint required')
    from .extensions import discover_components
    discover_components()
    alias = _ALIASES.get(point.component)
    if alias:
        if point.implementation is not None:
            raise ValueError('legacy native alias already binds an exact implementation')
        name, ident, version = alias
        common = COMPONENTS.get(name).params
        point = SpacePoint(point.name, name, {k:v for k,v in point.params.items() if k in common}, point.repeat,
                           ImplementationBinding(ident, version, {k:v for k,v in point.params.items() if k not in common}))
    hardware = COMPONENTS.get(point.component)
    binding = point.implementation
    if binding is None:
        legacy = _LEGACY.get(point.component)
        if legacy is None:
            raise ValueError('explicit component implementation required: '+point.component)
        binding = ImplementationBinding(legacy.id, legacy.version)
    if not isinstance(binding, ImplementationBinding):
        raise TypeError('implementation must be ImplementationBinding')
    binding.to_dict()
    key = (point.component, binding.id, binding.version)
    if key not in _IMPLEMENTATIONS:
        raise ValueError('unknown component implementation/version: '+str(key))
    implementation = _IMPLEMENTATIONS[key]
    _check(implementation)
    if implementation.hardware_parameters is None:
        hardware.validate(point)
    else:
        allowed = set(implementation.hardware_parameters)
        if allowed-set(hardware.params) or set(point.params)-allowed:
            raise ValueError('common hardware parameters incompatible with selected implementation')
        _parameters(point, {k: hardware.params[k] for k in allowed})
    _parameters(SpacePoint(point.name, point.component, binding.params), implementation.params)
    result = ResolvedComponent(hardware, implementation, point, binding)
    if implementation.validate is not None:
        implementation.validate(result.materialize())
    if execution_domain is not None and implementation.execution_domain != execution_domain:
        raise ValueError('component execution domain requires an explicit bridge: '+implementation.execution_domain+' -> '+execution_domain)
    return result


def build_component(point, context, *, execution_domain='events'):
    resolved = resolve_component(point, execution_domain=execution_domain)
    return resolved.build(point, context)


def implementation_identities(adf):
    from arch_explorer.adf.model import walk
    return [dict(path=path, **resolve_component(node).identity())
            for path, node in walk(adf) if isinstance(node, SpacePoint)]


def component_candidates(point, bindings):
    """Validate explicit search choices without inventing implementation inputs."""
    from copy import deepcopy
    if not isinstance(point, SpacePoint) or not isinstance(bindings, (list, tuple)) or not bindings:
        raise ValueError('point and nonempty explicit implementation bindings required')
    result, seen = [], set()
    for binding in bindings:
        candidate = deepcopy(point)
        candidate.implementation = deepcopy(binding)
        resolved = resolve_component(candidate)
        identity = _json(resolved.binding.to_dict())
        if identity in seen:
            raise ValueError('duplicate component implementation search choice')
        seen.add(identity)
        result.append(candidate)
    return result


def split_binding(point, num):
    """Only implementation-declared extensive quantities partition."""
    if point.implementation is None:
        return None
    resolved=resolve_component(point)
    params=dict(resolved.binding.params)
    for key,param in params.items():
        spec=resolved.implementation.params[key]
        if not spec.extensive:continue
        if spec.kind=='int' and param.value % num:
            raise ValueError('implementation extensive integer cannot split evenly: '+key)
        value=param.value//num if spec.kind=='int' else param.value/num
        params[key]=Param(value,param.unit,'split('+str(num)+') of '+param.source)
    return ImplementationBinding(resolved.binding.id,resolved.binding.version,params)


def merge_bindings(points):
    bindings=[p.implementation for p in points]
    if all(b is None for b in bindings):return None
    if any(b is None for b in bindings) or len({(b.id,b.version) for b in bindings})!=1:
        raise ValueError('cannot merge components with different implementation bindings')
    resolved=[resolve_component(p) for p in points]
    first=resolved[0]
    if any(set(r.binding.params)!=set(first.binding.params) for r in resolved):
        raise ValueError('implementation parameter sets differ on merge')
    params={}
    for key,param in first.binding.params.items():
        spec=first.implementation.params[key];values=[r.binding.params[key].value for r in resolved]
        if spec.extensive:params[key]=Param(sum(values),param.unit,'merge of '+str(len(points))+' '+first.name)
        elif any(v!=param.value for v in values):raise ValueError('intensive implementation parameter differs: '+key)
        else:params[key]=param
    return ImplementationBinding(first.binding.id,first.binding.version,params)


def component_catalog(registry=COMPONENTS):
    """One JSON catalog for API/CLI/Agent consumers; no simulation side effects."""
    rows = component_implementations() if registry is COMPONENTS else []
    result = []
    for name in registry.names():
        if registry is COMPONENTS and name in _ALIASES:
            continue
        hardware = registry.get(name)
        choices = [d for d in rows if d['component'] == name]
        if registry is not COMPONENTS and getattr(hardware, 'implementation_policy', 'legacy') == 'legacy':
            choices = [ComponentImplementation(name, 'builtin.'+name, '1',
                {'basis':type(hardware).__module__+':'+type(hardware).__qualname__}, type(hardware).build,
                'events', 'events.component.v1', 'Custom registry original component model',
                execution_modes=tuple(hardware.execution_modes), conventions=hardware.conventions).descriptor()]
        result.append(dict(name=name, role=hardware.role, parameters={k: _parameter_descriptor(v) for k,v in hardware.params.items()},
                           conventions=asdict(hardware.conventions), implementations=choices,
                           directory='arch_explorer/components/'+name if type(hardware).__module__.startswith('arch_explorer.') else None,
                           coverage='implementation-limited'))
    return dict(schema='arch-explorer.component-catalog.v1', components=result)
