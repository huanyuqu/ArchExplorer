"""Provider-independent preparation, capability checking, execution and replay."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
import math

from arch_explorer.adf.evaluation import Evaluation, MappingSpec, WorkloadSpec
from arch_explorer.components.extensions import get_provider, provider_for_schema

REQUEST_SCHEMA = 'arch-explorer.simulator-request.v1'
RESULT_SCHEMA = 'arch-explorer.simulator-result.v1'
_REQUEST_FIELDS = {'schema', 'provider', 'adf', 'workload', 'mapping', 'options', 'payload', 'binding'}
_RESULT_FIELDS = {'schema', 'provider', 'request', 'evaluation', 'payload', 'binding'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _copy(value):
    return json.loads(_canonical(value))


def _bound(value):
    result = _copy(value)
    result['binding'] = {'sha256': hashlib.sha256(_canonical(result)).hexdigest()}
    return result


def _check(value, fields):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('incomplete/extra generic simulator envelope fields')
    binding = value['binding']
    if not isinstance(binding, dict) or set(binding) != {'sha256'}:
        raise ValueError('complete simulator envelope binding required')
    actual = hashlib.sha256(_canonical({k: v for k, v in value.items() if k != 'binding'})).hexdigest()
    if binding['sha256'] != actual:
        raise ValueError('simulator envelope content binding differs')
    selector = value['provider']
    if not isinstance(selector, dict) or set(selector) != {'id', 'version'}:
        raise ValueError('exact provider id/version required')
    provider = get_provider(selector['id'])
    if selector['version'] != provider.version:
        raise ValueError('unsupported provider version')
    return provider


def _validate_payload(provider, payload, *, result=False):
    schemas = provider.result_schemas if result else provider.request_schemas
    if not isinstance(payload, dict) or payload.get('schema') not in schemas:
        raise ValueError('provider returned/matched an undeclared payload schema')
    _canonical(payload)


def _request_provider(request):
    if not isinstance(request, dict):
        raise TypeError('simulator request must be a JSON object')
    if request.get('schema') == REQUEST_SCHEMA:
        provider = _check(request, _REQUEST_FIELDS)
        _validate_payload(provider, request['payload'])
        # The outer truth and component payload must describe the same specs.
        # Providers may transform specs internally, but must echo these inputs
        # in their prepared request to keep the authority explicit.
        for key in ('adf', 'workload', 'mapping', 'options'):
            if _canonical(request['payload'].get(key)) != _canonical(request[key]):
                raise ValueError(f'provider payload differs from generic {key}')
        return provider, request['payload']
    provider = provider_for_schema(request.get('schema'))
    _validate_payload(provider, request, result=request.get('schema') in provider.result_schemas)
    return provider, request


def _assessment(provider, payload):
    from arch_explorer.adf.model import SpacePoint, from_dict, walk
    from arch_explorer.adf.registry import COMPONENTS
    request = payload.get('request') if payload.get('schema') in provider.result_schemas else payload
    if not isinstance(request, dict) or not isinstance(request.get('adf'), dict):
        raise ValueError('provider request must retain its complete ADF execution domain')
    gaps = []
    points = [node for _, node in walk(from_dict(request['adf'])) if isinstance(node, SpacePoint)]
    from arch_explorer.components.implementations import resolve_component
    resolved = [resolve_component(node) for node in points]
    types = {r.implementation.capabilities.get('provider_types', {}).get(provider.id, r.name) for r in resolved}
    if provider.capabilities.get('open_component_types') is True:
        contracts = set(provider.capabilities['component_contracts'])
        for r in resolved:
            name = r.name
            try:
                component = r
            except KeyError:
                gaps.append('unregistered component type: '+name)
                continue
            if not contracts.intersection(component.execution_modes):
                gaps.append('component requires a separate execution domain: '+name)
    else:
        gaps.extend('component outside declared provider execution domain: '+name
                    for name in sorted(types-set(provider.component_types)))
    if gaps:
        return {'supported': False, 'gaps': gaps}
    result = provider.assess_request(_copy(payload))
    if (not isinstance(result, dict) or type(result.get('supported')) is not bool
            or not isinstance(result.get('gaps'), list)):
        raise TypeError('provider assessment requires supported:bool and gaps:list')
    if result['supported'] and result['gaps']:
        raise ValueError('supported assessment must have no capability gaps')
    return _copy(result)


def _result_request(provider, result, request):
    """Associate a component result with this exact prepared source request."""
    expected = request['payload'] if request.get('schema') == REQUEST_SCHEMA else request
    if expected.get('schema') in provider.result_schemas:
        expected = expected['request']
    if _canonical(result.get('request')) != _canonical(expected):
        raise ValueError('provider result belongs to a different outer request')


def prepare_request(adf, workload, mapping, *, output_dir, provider, options=None):
    """Bind public specs and prepare provider artifacts, without executing."""
    if not isinstance(workload, WorkloadSpec) or not isinstance(mapping, MappingSpec):
        raise TypeError('WorkloadSpec and MappingSpec required')
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise TypeError('provider options must be a JSON object')
    options = _copy(options)
    selected = get_provider(provider)
    payload = selected.prepare_request(adf, workload, mapping, output_dir=output_dir,
                                       options=copy.deepcopy(options))
    _validate_payload(selected, payload)
    request = _bound({'schema': REQUEST_SCHEMA, 'provider': {'id': selected.id, 'version': selected.version},
                      'adf': adf.to_dict(), 'workload': workload.to_dict(),
                      'mapping': mapping.to_dict(), 'options': options, 'payload': payload})
    _request_provider(request)
    return request


def assess_request(request):
    """Check the registered provider only; no fallback or external evaluation."""
    from .compilation import REQUEST_SCHEMA as COMPILED_REQUEST, RESULT_SCHEMA as COMPILED_RESULT, assess_compiled
    if isinstance(request, dict) and request.get('schema') in (COMPILED_REQUEST, COMPILED_RESULT):
        return assess_compiled(request)
    if isinstance(request, dict) and request.get('schema') == RESULT_SCHEMA:
        provider = _check(request, _RESULT_FIELDS)
        source, payload = _request_provider(request['request'])
        if provider is not source:
            raise ValueError('result/request provider mismatch')
        _validate_payload(provider, request['payload'], result=True)
        _result_request(provider, request['payload'], request['request'])
        return _assessment(provider, request['payload'])
    provider, payload = _request_provider(request)
    return _assessment(provider, payload)


def _evaluation(value):
    if not isinstance(value, dict) or set(value) != set(Evaluation(0).to_dict()):
        raise ValueError('complete Evaluation.to_dict result required')
    latency = value['latency_ns']
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
        raise ValueError('evaluation latency_ns must be finite and positive')
    _canonical(value)
    return _copy(value)


def evaluate_request(request_or_result):
    """Execute a prepared request, or read-only validate a bound existing result."""
    from .compilation import REQUEST_SCHEMA as COMPILED_REQUEST, RESULT_SCHEMA as COMPILED_RESULT, evaluate_compiled
    if isinstance(request_or_result, dict) and request_or_result.get('schema') in (COMPILED_REQUEST, COMPILED_RESULT):
        return evaluate_compiled(request_or_result)
    if not isinstance(request_or_result, dict):
        raise TypeError('simulator request/result must be a JSON object')
    value = _copy(request_or_result)
    replay = value.get('schema') == RESULT_SCHEMA
    if replay:
        provider = _check(value, _RESULT_FIELDS)
        selected, _ = _request_provider(value['request'])
        if selected is not provider:
            raise ValueError('result/request provider mismatch')
        _validate_payload(provider, value['payload'], result=True)
        _result_request(provider, value['payload'], value['request'])
        payload = value['payload']
        request = value['request']
    else:
        provider, payload = _request_provider(value)
        request = value
    # Providers may reuse their own validation products only inside this
    # evaluation. Domain assessment, request binding, current-source checks
    # and provider execution remain their original public operations.
    scope = getattr(provider, '_evaluation_scope', None)
    with scope() if callable(scope) else nullcontext():
        assessment = _assessment(provider, payload)
        if not assessment['supported']:
            raise ValueError('unsupported provider request: '+json.dumps(assessment['gaps'], ensure_ascii=False))
        # value owns an independent native JSON copy. Keep its original
        # provider document as bytes while the callback uses that subtree.
        # Restore the authority after leaving the model scope: a provider may
        # put arbitrary private state there, which must never be traversed.
        original = _canonical(payload)
        generic = value.get('schema') in (REQUEST_SCHEMA, RESULT_SCHEMA)
        outcome = provider.evaluate_request(payload)
    # Drop our decoded private-input references before decoding the preserved
    # source. There is no second decoded input during the provider execution,
    # and no inspection of state the provider chose to keep in its own input.
    if generic:
        value['payload'] = None
    else:
        value = None
    request = payload = None
    payload = json.loads(original)
    if generic:
        value['payload'] = payload
    else:
        value = payload
    request = value['request'] if replay else value
    del original
    _validate_payload(provider, outcome, result=True)
    original_payload = payload.get('request') if payload.get('schema') in provider.result_schemas else payload
    if _canonical(outcome.get('request')) != _canonical(original_payload):
        raise ValueError('provider result belongs to a different request')
    _result_request(provider, outcome, request)
    evaluation = _evaluation(outcome.get('evaluation'))
    if replay:
        if _canonical(outcome) != _canonical(payload) or _canonical(evaluation) != _canonical(value['evaluation']):
            raise ValueError('provider replay differs from bound result/evaluation')
        return value
    return _bound({'schema': RESULT_SCHEMA, 'provider': {'id': provider.id, 'version': provider.version},
                   'request': request, 'evaluation': evaluation, 'payload': outcome})


def prepare_compiled(compilation, *, output_dir, options=None, registry=None):
    """Prepare a bound common compilation through its registered provider."""
    from .compilation import prepare_compiled as prepare
    return prepare(compilation, output_dir=output_dir, options=options,registry=registry)
