"""Source-bound operator mapping contracts over the existing semantic graph.

Rules own expansion, accesses and region inspection. This registry does not
implement numerical semantics, backend conversion or component timing.
"""
from dataclasses import dataclass, field
import inspect
import json

from arch_explorer.workload.graph import finite_json

_RULES = {}
_RULE_STATES = {}


@dataclass(frozen=True)
class MappingRuleBinding:
    id: str
    version: str
    options: dict = field(default_factory=dict)

    def to_dict(self):
        from arch_explorer.workload.extensions import _name
        _name(self.id); _name(self.version)
        if type(self.options) is not dict:
            raise TypeError('mapping rule options must be a JSON object')
        return dict(id=self.id, version=self.version, options=finite_json(self.options))

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {'id', 'version', 'options'}:
            raise ValueError('mapping rule binding requires exact id/version/options')
        result = cls(**value)
        result.to_dict()
        return result


@dataclass(frozen=True)
class MappingExpansion:
    tensors: tuple = ()
    operators: tuple = ()
    views: tuple = ()  # replacements of original output descriptors; no allocation/write


@dataclass(frozen=True)
class OperatorMapping:
    id: str
    version: str
    operator: str
    expand: object
    access: object
    inspect: object = None
    validate_options: object = None
    parameters: dict = field(default_factory=dict)
    default: bool = False
    tensor_scope: str = 'graph'


def _callback(function, count):
    if not callable(function):
        raise TypeError('mapping rule callback must be callable')
    inspect.signature(function).bind(*([None] * count))
    return id(function), id(getattr(function, '__code__', None))


def _state(definition):
    from arch_explorer.workload.extensions import _name
    if type(definition) is not OperatorMapping:
        raise TypeError('register_operator_mapping requires OperatorMapping')
    for name in (definition.id, definition.version, definition.operator):
        _name(name)
    if type(definition.parameters) is not dict:
        raise TypeError('mapping rule parameters must be a JSON object')
    if type(definition.default) is not bool:raise TypeError('mapping default must be bool')
    if definition.tensor_scope not in ('graph','isolated-operands'):
        raise ValueError('mapping tensor_scope must be graph or isolated-operands')
    return (definition.id, definition.version, definition.operator,definition.default,
            definition.tensor_scope,
            *(_callback(getattr(definition, name), count) if getattr(definition, name) is not None else None
              for name, count in (('expand', 3), ('access', 3), ('inspect', 3), ('validate_options', 1))),
            json.dumps(finite_json(definition.parameters), sort_keys=True, allow_nan=False))


def register_operator_mapping(definition):
    state = _state(definition)
    if definition.expand is None or definition.access is None:
        raise TypeError('mapping rule requires expand and access callbacks')
    from arch_explorer.workload.extensions import get_operator
    get_operator(definition.operator)
    previous = _RULES.get(definition.id)
    if previous is not None and (previous is not definition or _RULE_STATES[definition.id] != state):
        raise ValueError('mapping rule identity/version conflict or mutated definition: '+definition.id)
    _RULES[definition.id], _RULE_STATES[definition.id] = definition, state
    return definition


def rule_identity(definition):
    from arch_explorer.workload.extensions import callable_identity
    if _RULE_STATES.get(definition.id) != _state(definition):
        raise ValueError('registered mapping rule was mutated: '+definition.id)
    return dict(id=definition.id, version=definition.version, operator=definition.operator,default=definition.default,
                tensor_scope=definition.tensor_scope,
                parameters=finite_json(definition.parameters), **{
                    name: None if getattr(definition, name) is None else callable_identity(getattr(definition, name))
                    for name in ('expand', 'access', 'inspect', 'validate_options')})


def operator_mappings(operator=None):
    from arch_explorer.components.extensions import discover_components
    discover_components()
    return [rule_identity(rule) for key, rule in sorted(_RULES.items())
            if operator is None or rule.operator == operator]


def select_rule(operator, value=None):
    from arch_explorer.components.extensions import discover_components
    from arch_explorer.workload.extensions import _record_source, _record_identity
    discover_components()
    if value is None:
        matches = [r for r in _RULES.values() if r.operator == operator]
        if len(matches)>1:
            defaults=[r for r in matches if r.default]
            if len(defaults)==1:matches=defaults
        if len(matches) != 1:
            raise ValueError('mapping rule selection is absent or ambiguous for '+operator)
        rule = matches[0]
        binding = MappingRuleBinding(rule.id, rule.version)
    else:
        binding = value if type(value) is MappingRuleBinding else MappingRuleBinding.from_dict(value)
        if binding.id not in _RULES:
            raise ValueError('unknown mapping rule: '+binding.id)
        rule = _RULES[binding.id]
    binding.to_dict()
    identity = rule_identity(rule)
    if binding.version != rule.version or rule.operator != operator:
        raise ValueError('mapping rule operator/version differs')
    supplied = finite_json(binding.options)
    if rule.validate_options is None:
        if supplied:
            raise ValueError('mapping rule accepts no options: '+rule.id)
        options = {}
    else:
        _record_source(rule.validate_options)
        options = finite_json(rule.validate_options(finite_json(supplied)))
        if type(options) is not dict:
            raise TypeError('mapping rule validate_options must return a JSON object')
    _record_identity(dict(kind='operator-mapping', definition=identity))
    return rule, MappingRuleBinding(rule.id, rule.version, options)


def invoke(rule, name, op, tensors, options):
    from arch_explorer.workload.extensions import _record_source
    function = getattr(rule, name)
    if function is None:
        return {}
    _record_source(function)
    return function(op, tensors, options)


def validate_access(value):
    """Validate a supplied boundary count, without deriving semantic formulas."""
    value = finite_json(value)
    if type(value) is not dict or value.get('boundary') != 'component_io':
        raise ValueError('mapping access requires the component_io data boundary')
    for key in ('operations', 'read_bytes', 'write_bytes'):
        if type(value.get(key)) not in (int, float) or value[key] < 0:
            raise ValueError('mapping access '+key+' must be finite and nonnegative')
    return value
