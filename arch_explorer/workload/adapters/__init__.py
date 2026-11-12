"""Descriptor-only task conversion and execution-domain composition contracts.

No concrete adapter, topology, component mechanism or timing formula lives in
this registry. Installed component registrars own their adapter relationships.
"""
from dataclasses import dataclass, field
import inspect
import json

from ..graph import finite_json

_ADAPTERS = {}
_ADAPTER_STATES = {}
_DOMAINS = {}
_DOMAIN_STATES = {}


@dataclass(frozen=True)
class AdapterBinding:
    id: str
    version: str
    options: dict = field(default_factory=dict)

    def to_dict(self):
        from ..extensions import _name
        _name(self.id); _name(self.version)
        if type(self.options) is not dict:
            raise TypeError('adapter options must be a JSON object')
        return dict(id=self.id, version=self.version, options=finite_json(self.options))

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {'id', 'version', 'options'}:
            raise ValueError('adapter binding requires exact id/version/options')
        result = cls(**value)
        result.to_dict()
        return result


@dataclass(frozen=True)
class AdapterContext:
    adf: object
    program: object
    mapping: object
    tasks: tuple
    compute: str
    memory: str
    component: object
    options: dict
    implementations: dict
    accesses: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptedGroup:
    payload: dict
    scope: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TargetInput:
    provider: str
    workload: object
    mapping: object
    scope: dict


@dataclass(frozen=True)
class TaskAdapter:
    id: str
    version: str
    execution_domain: str
    semantics: tuple
    lower: object
    restore: object = None
    parameters: dict = field(default_factory=dict)
    grouping: str = 'component'
    layouts: tuple = ('independent',)


@dataclass(frozen=True)
class AdapterDomain:
    id: str
    compose: object


def _signature(function, count):
    if not callable(function):
        raise TypeError('adapter callback must be callable')
    inspect.signature(function).bind(*([None] * count))
    return (id(function), id(getattr(function, '__code__', None)))


def _state(definition):
    from ..extensions import _name
    for value in (definition.id, definition.version, definition.execution_domain):
        _name(value)
    if (type(definition.semantics) is not tuple or not definition.semantics
            or len(set(definition.semantics)) != len(definition.semantics)):
        raise ValueError('adapter requires explicit unique semantic kinds')
    for name in definition.semantics:
        _name(name)
    if definition.grouping != 'component':
        raise ValueError('supported task group boundary is component')
    if (type(definition.layouts) is not tuple or not definition.layouts
            or len(set(definition.layouts))!=len(definition.layouts)
            or set(definition.layouts)-{'independent','readonly_strided'}):
        raise ValueError('adapter layouts must explicitly declare independent/readonly_strided')
    if type(definition.parameters) is not dict:
        raise TypeError('adapter parameter descriptors must be an object')
    for key, spec in definition.parameters.items():
        _name(key)
        if (type(spec) is not dict or set(spec)-{'type', 'choices', 'min', 'max', 'default'}
                or spec.get('type') not in ('int', 'float', 'str', 'bool')):
            raise ValueError('invalid adapter option descriptor: '+key)
        finite_json(spec)
    return (definition.id, definition.version, definition.execution_domain,
            definition.semantics, definition.grouping,definition.layouts,
            json.dumps(definition.parameters, sort_keys=True, allow_nan=False),
            _signature(definition.lower, 1),
            None if definition.restore is None else _signature(definition.restore, 1))


def register_task_adapter(definition):
    if type(definition) is not TaskAdapter:
        raise TypeError('register_task_adapter requires TaskAdapter')
    state = _state(definition)
    old = _ADAPTERS.get(definition.id)
    if old is not None and (old is not definition or _ADAPTER_STATES[definition.id] != state):
        raise ValueError('adapter duplicate or mutated identity: '+definition.id)
    _ADAPTERS[definition.id] = definition
    _ADAPTER_STATES[definition.id] = state
    return definition


def register_adapter_domain(definition):
    from ..extensions import _name
    if type(definition) is not AdapterDomain:
        raise TypeError('register_adapter_domain requires AdapterDomain')
    _name(definition.id)
    state = _signature(definition.compose, 3)
    old = _DOMAINS.get(definition.id)
    if old is not None and (old is not definition or _DOMAIN_STATES[definition.id] != state):
        raise ValueError('adapter domain duplicate or mutated identity: '+definition.id)
    _DOMAINS[definition.id] = definition
    _DOMAIN_STATES[definition.id] = state
    return definition


def get_adapter(identifier):
    from ..extensions import _discover
    _discover()
    definition = _ADAPTERS.get(identifier)
    if definition is None or _ADAPTER_STATES[identifier] != _state(definition):
        raise ValueError('unknown/unavailable or mutated task adapter: '+str(identifier))
    return definition


def get_domain(identifier):
    definition = _DOMAINS.get(identifier)
    if definition is None or _DOMAIN_STATES[identifier] != _signature(definition.compose, 3):
        raise ValueError('unknown/unavailable or mutated adapter domain: '+identifier)
    return definition


def adapter_identity(definition):
    from ..extensions import callable_identity
    return dict(id=definition.id, version=definition.version,
                execution_domain=definition.execution_domain, semantics=list(definition.semantics),
                grouping=definition.grouping, layouts=list(definition.layouts),parameters=finite_json(definition.parameters),
                lower=callable_identity(definition.lower),
                restore=None if definition.restore is None else callable_identity(definition.restore))


def task_adapters():
    from ..extensions import _discover
    from arch_explorer.components.implementations import component_implementations
    _discover()
    implementations = component_implementations()
    return [dict(adapter_identity(get_adapter(key)), implementations=[
        {k:item[k] for k in ('component','id','version','execution_domain','input_contract')}
        for item in implementations if key in item['adapters']]) for key in sorted(_ADAPTERS)]


def options_for(definition, value):
    value = finite_json(value)
    if type(value) is not dict or set(value)-set(definition.parameters):
        raise ValueError('unknown adapter options: '+definition.id)
    result = dict(value)
    types = dict(int=int, float=(int, float), str=str, bool=bool)
    for key, spec in definition.parameters.items():
        if key not in result:
            if 'default' not in spec:
                raise ValueError('explicit adapter option required: '+definition.id+'.'+key)
            result[key] = spec['default']
        item = result[key]
        if type(item) not in ((types[spec['type']],) if spec['type'] != 'float' else types['float']):
            raise ValueError('adapter option has wrong type: '+key)
        if ('choices' in spec and item not in spec['choices']
                or 'min' in spec and item < spec['min'] or 'max' in spec and item > spec['max']):
            raise ValueError('adapter option outside declared domain: '+key)
    return finite_json(result)


def bind_adapter(component, semantics, binding=None):
    declared = component.implementation.adapters
    candidates = []
    for identifier in declared:
        adapter = get_adapter(identifier)
        if (adapter.execution_domain == component.implementation.execution_domain
                and set(semantics) <= set(adapter.semantics)):
            candidates.append(adapter)
    if binding is None:
        if len(candidates) != 1:
            raise ValueError('component requires an unambiguous compatible adapter binding: '+component.binding.id)
        adapter = candidates[0]
        binding = AdapterBinding(adapter.id, adapter.version)
    elif type(binding) is dict:
        binding = AdapterBinding.from_dict(binding)
    if type(binding) is not AdapterBinding:
        raise TypeError('bindings must contain AdapterBinding or its exact JSON form')
    adapter = get_adapter(binding.id)
    if adapter not in candidates or binding.version != adapter.version:
        raise ValueError('adapter binding does not match component/domain/semantics/version')
    return adapter, AdapterBinding(adapter.id, adapter.version, options_for(adapter, binding.options))
