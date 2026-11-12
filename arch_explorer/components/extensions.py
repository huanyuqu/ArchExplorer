"""Descriptor-only extension registration and isolated installed discovery.

No engine, component mechanism, or backend name belongs in this registry.
An entry point executes the same no-argument registrar as explicit Python use.
"""
from __future__ import annotations

import copy
from importlib import import_module
from importlib import metadata
import json
import re

API_VERSION = '1'
ENTRY_POINT_GROUP = 'arch_explorer.components'
_PROVIDERS = {}
_DESCRIPTORS = {}
_SCHEMAS = {}
_LOADED = set()
_FAILURES = {}
_DISCOVERING = False
_BUILTINS_LOADED = False
_DISCOVERED = False


def _label(value, field):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError(f'invalid provider {field}: {value!r}')
    return value


def _descriptor(provider):
    if getattr(provider, 'api_version', API_VERSION) != API_VERSION:
        raise ValueError('incompatible component extension API version')
    result = {key: _label(getattr(provider, key, None), key) for key in ('id', 'version')}
    for key in ('component_types', 'request_schemas', 'result_schemas'):
        values = getattr(provider, key, None)
        if (not isinstance(values, tuple) or len(set(values)) != len(values)
                or any(not isinstance(v, str) or not v or len(v) > 256 for v in values)):
            raise ValueError(f'provider {key} must be a tuple of unique nonempty strings')
        result[key] = list(values)
    if not result['request_schemas'] or not result['result_schemas']:
        raise ValueError('provider must declare request and result schemas')
    if getattr(provider, 'execution_domain', None) not in ('single', 'joint'):
        raise ValueError('provider execution_domain must be single or joint')
    result['execution_domain'] = provider.execution_domain
    if not isinstance(getattr(provider, 'capabilities', None), dict):
        raise TypeError('provider capabilities must be a JSON object')
    result['capabilities'] = json.loads(json.dumps(provider.capabilities, allow_nan=False))
    if result['capabilities'].get('open_component_types') is True:
        contracts = result['capabilities'].get('component_contracts')
        if (result['execution_domain'] != 'single' or not isinstance(contracts, list)
                or not contracts or len(set(contracts)) != len(contracts)
                or set(contracts) - {'demand', 'stateful'}):
            raise ValueError('open component domain requires explicit demand/stateful contracts and single domain')
    elif not result['component_types']:
        raise ValueError('closed provider must declare its component types')
    for method in ('prepare_request', 'assess_request', 'evaluate_request'):
        if not callable(getattr(provider, method, None)):
            raise TypeError(f'provider missing {method}')
    return result


def register_provider(provider):
    """Register a provider instance; competing identities/schemas are refused."""
    descriptor = _descriptor(provider)
    key = descriptor['id']
    if key in _PROVIDERS:
        if _PROVIDERS[key] is provider and descriptor == _DESCRIPTORS[key]:
            return provider
        raise ValueError(f'provider {key!r} already registered; version/identity conflict')
    schemas = descriptor['request_schemas'] + descriptor['result_schemas']
    if len(set(schemas)) != len(schemas):
        raise ValueError('provider request/result schemas must be disjoint')
    for schema in schemas:
        if schema in _SCHEMAS or schema in ('arch-explorer.simulator-request.v1', 'arch-explorer.simulator-result.v1'):
            raise ValueError(f'provider schema {schema!r} already registered/reserved')
    _PROVIDERS[key] = provider
    _DESCRIPTORS[key] = descriptor
    _SCHEMAS.update({schema: key for schema in schemas})
    return provider


def _transaction(function):
    """Rollback a broken registrar without disturbing other extensions."""
    from arch_explorer.adf.registry import COMPONENTS, MAPPING_TEMPLATES
    from arch_explorer.adf.evaluation import _BACKENDS
    from arch_explorer.workload.extensions import _OPERATORS, _WORKLOADS, _OPERATOR_STATES
    from arch_explorer.workload.execution import _IMPLEMENTATIONS, _IMPLEMENTATION_STATES
    from arch_explorer.workload.adapters import _ADAPTERS, _ADAPTER_STATES, _DOMAINS, _DOMAIN_STATES
    from arch_explorer.mapping.rules import _RULES, _RULE_STATES
    from arch_explorer.components.implementations import (
        _IMPLEMENTATIONS as component_implementations,
        _IMPLEMENTATION_STATES as component_states, _LEGACY, _ALIASES,
    )
    dictionaries = (COMPONENTS._items, MAPPING_TEMPLATES._items, _BACKENDS,
                    _PROVIDERS, _DESCRIPTORS, _SCHEMAS, _OPERATORS, _WORKLOADS, _OPERATOR_STATES,
                    _IMPLEMENTATIONS, _IMPLEMENTATION_STATES, component_implementations, component_states,
                    _LEGACY, _ALIASES, _ADAPTERS, _ADAPTER_STATES, _DOMAINS, _DOMAIN_STATES,
                    _RULES, _RULE_STATES)
    # The reference API arrives in the disjoint target child; retain its fixed
    # dictionary identity as part of the same discovery transaction when present.
    from importlib.util import find_spec
    if find_spec('arch_explorer.workload.reference_api') is not None:
        from arch_explorer.workload.reference_api import _REFERENCES
        dictionaries += (_REFERENCES,)
    saved = [dict(d) for d in dictionaries]
    try:
        function()
    except BaseException:
        for target, previous in zip(dictionaries, saved):
            target.clear()
            target.update(previous)
        raise


def discover_components():
    """Load installed registrars once; an extension failure is local and visible.

    The in-package registrar is also callable explicitly when working from
    source. Installed discovery uses the identical registrar and registration
    contracts; builtin provider names are never encoded here.
    """
    global _DISCOVERING, _BUILTINS_LOADED, _DISCOVERED
    if _DISCOVERING or _DISCOVERED:
        return {'loaded': sorted(_LOADED), 'failures': dict(_FAILURES)}
    _DISCOVERING = True
    try:
        if not _BUILTINS_LOADED:
            from .builtins import registrars
            for name, target in registrars().items():
                identity = 'arch_explorer.builtin.'+name
                try:
                    module, attribute = target.split(':')
                    _transaction(lambda: getattr(import_module(module), attribute)())
                    _LOADED.add(identity)
                except (Exception, SystemExit) as error:
                    _FAILURES[identity] = f'{type(error).__name__}: {error}'
            _BUILTINS_LOADED = True
        try:
            entries = metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception as error:
            _FAILURES['entry-point-discovery'] = f'{type(error).__name__}: {error}'
            entries = ()
        for entry in sorted(entries, key=lambda ep: (ep.name, ep.value)):
            dist = getattr(entry, 'dist', None)
            identity = f'{getattr(dist, "name", "unknown")}:{entry.name}={entry.value}'
            if identity in _LOADED or identity in _FAILURES:
                continue
            try:
                def register_entry(entry=entry):
                    # Loading a module can itself register workload semantics.
                    # Include import and invocation in the same transaction.
                    registrar = entry.load()
                    if not callable(registrar):
                        raise TypeError('component entry point must be a callable register()')
                    registrar()
                _transaction(register_entry)
                _LOADED.add(identity)
            except (Exception, SystemExit) as error:
                _FAILURES[identity] = f'{type(error).__name__}: {error}'
        _DISCOVERED = True
    finally:
        _DISCOVERING = False
    return {'loaded': sorted(_LOADED), 'failures': dict(_FAILURES)}


def get_provider(identifier):
    discover_components()
    if identifier not in _PROVIDERS:
        raise NotImplementedError(f'unknown/unavailable provider {identifier!r}; '
                                  f'available: {sorted(_PROVIDERS)}; extensions: {_FAILURES}')
    value = _PROVIDERS[identifier]
    if _descriptor(value) != _DESCRIPTORS[identifier]:
        raise ValueError('registered provider descriptor was mutated')
    return value


def provider_for_schema(schema):
    discover_components()
    if schema not in _SCHEMAS:
        raise ValueError(f'unregistered request/result schema {schema!r}')
    return get_provider(_SCHEMAS[schema])


def providers():
    discover_components()
    return [copy.deepcopy(_DESCRIPTORS[key]) for key in sorted(_DESCRIPTORS)]
