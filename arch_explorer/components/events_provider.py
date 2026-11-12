"""Bound event-program preparation, execution receipts and read-only replay.

The provider owns this file format. The event scheduler sees only the graph
and its explicit ExecutionPlan. A result consumes the original raw execution
receipt; replay never calls simulate(), evaluate_events(), or a state factory.
"""
from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
import json
from functools import partial
from pathlib import Path

from arch_explorer.adf.evaluation import Evaluation, MappingSpec, WorkloadSpec, register_backend
from arch_explorer.adf.model import from_dict
from arch_explorer.adf.registry import MAPPING_TEMPLATES
from arch_explorer.components.api import register_provider

REQUEST_SCHEMA = 'arch-explorer.events-request.v1'
RESULT_SCHEMA = 'arch-explorer.events-result.v1'
RAW_SCHEMA = 'arch-explorer.events-execution.v1'
_REQUEST_KEYS = {'schema', 'adf', 'workload', 'mapping', 'options', 'output_dir', 'sources', 'plan', 'binding'}
_RESULT_KEYS = {'schema', 'request', 'evaluation', 'raw', 'binding'}
_OPTIONS = {'scheduler', 'systolic_model', 'check_capacity', 'iterations', 'trace'}
_VALIDATED_PREPARE = ContextVar('events_validated_prepare', default=None)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _matches_request_file(path, payload):
    # Preserve exact canonical bytes plus one newline. Serialize before
    # opening the file, and compare bounded reads to a view: neither the
    # prepared file nor a newline-appended copy must coexist in memory.
    expected = memoryview(_canonical(payload))
    block = 1024 * 1024
    with path.open('rb') as stream:
        for offset in range(0, len(expected), block):
            piece = expected[offset:offset + block]
            if stream.read(len(piece)) != piece:
                return False
        return stream.read(2) == b'\n'


def _copy(value):
    return json.loads(_canonical(value))


def _seal(value):
    value = _copy(value)
    value['binding'] = {'sha256': hashlib.sha256(_canonical(value)).hexdigest()}
    return value


def _check(value, keys, schema):
    if not isinstance(value, dict) or set(value) != keys or value['schema'] != schema:
        raise ValueError('invalid events payload fields/schema')
    actual = hashlib.sha256(_canonical({k: v for k, v in value.items() if k != 'binding'})).hexdigest()
    if value['binding'] != {'sha256': actual}:
        raise ValueError('events payload content binding differs')


def _options(options):
    if not isinstance(options, dict) or set(options) - _OPTIONS:
        raise ValueError('unsupported events provider options')
    for name in ('trace', 'check_capacity'):
        if name in options and type(options[name]) is not bool:
            raise TypeError(f'{name} must be bool')
    if options.get('systolic_model', 'native') != 'native':
        raise ValueError('bound events programs currently require systolic_model=native')
    return _copy(options)


def _source_files(simulator, mapping):
    """Bind the complete Python event/mapping implementation and used extensions.

    The builtin event domain has a deliberately closed source set: the generic
    driver, all its mapping strategies, ordinary components and stateful
    implementations. Other provider packages are not imported or bound.
    External component/mapping packages contribute their whole Python package.
    """
    root = Path(__file__).resolve().parents[1]
    files = {}
    for folder in ('simulator/events', 'mapping'):
        for path in sorted((root / folder).glob('*.py')):
            if path.name != 'accelsim_program.py' and path.name != 'accelsim.py':
                files[path.relative_to(root).as_posix()] = path
    for path in sorted((root/'components').rglob('*.py')):
        files[path.relative_to(root).as_posix()] = path
    for folder in ('adf', 'workload', 'model_specs'):
        for path in sorted((root / folder).rglob('*.py')):
            files[path.relative_to(root).as_posix()] = path
    files['simulator/plan.py'] = root / 'simulator/plan.py'
    subjects = [type(point) for point in simulator.points.values()]
    subjects += [type(point.component_type) for point in simulator.points.values()]
    subjects += [type(point.component_type.hardware) for point in simulator.points.values()
                 if hasattr(point.component_type, 'hardware')]
    for point in simulator.points.values():
        implementation = getattr(point.component_type, 'implementation', None)
        if implementation is not None:
            subjects.append(implementation.factory)
            if implementation.validate is not None:
                subjects.append(implementation.validate)
    subjects.append(type(MAPPING_TEMPLATES.get(mapping.template)))
    subjects.extend(simulator._events_workload_sources)
    for subject in subjects:
        while isinstance(subject, partial):
            subject = subject.func
        if not (inspect.isclass(subject) or inspect.isroutine(subject)):
            subject = type(subject)
        for parent in subject.__mro__ if inspect.isclass(subject) else (subject,):
            path = inspect.getsourcefile(parent) if parent.__module__ != 'builtins' else None
            if not path:
                if parent.__module__ == 'builtins':
                    continue
                raise ValueError(f'implementation source unavailable: {parent!r}')
            path = Path(path).resolve()
            if path.is_relative_to(root):
                continue
            package = path.parent
            if (package / '__init__.py').is_file():
                while (package.parent / '__init__.py').is_file():
                    package = package.parent
                for item in sorted(package.rglob('*.py')):
                    key = f'extension:{package.name}/{item.relative_to(package).as_posix()}'
                    if key in files and files[key] != item:
                        raise ValueError(f'ambiguous extension source identity: {key}')
                    files[key] = item
            else:
                key = f'extension:{parent.__module__}'
                if key in files and files[key] != path:
                    raise ValueError(f'ambiguous extension source identity: {key}')
                files[key] = path
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}


def _plan(simulator):
    ctx = simulator.mapping
    result = dict(constraints=[asdict(c) for c in ctx.plan.constraints], scheduler=ctx.plan.scheduler,
                iterations=ctx.graph.iterations, tasks=len(ctx.graph.tasks),
                execution_models=sorted(ctx.graph.execution_models),
                execution_owners=dict(ctx.graph.execution_owners),
                resources={key: asdict(value) for key, value in sorted(ctx.graph.resources.items())})
    if 'execution_graph' in ctx.details:
        # Stable implementation identity and the exact consumed multi-step IR
        # are part of the prepared request, not an unbound diagnostic hint.
        result['execution_graph'] = _copy(ctx.details['execution_graph'])
    if 'compilation' in ctx.details:
        result['compilation'] = _copy(ctx.details['compilation'])
    return result


def _prepare(adf, workload, mapping, options):
    from arch_explorer.simulator.events.backend import prepare_mapping
    from arch_explorer.workload.extensions import source_trace
    allowed = _options(options)
    with source_trace() as sources:
        simulator = prepare_mapping(adf, workload, mapping, **allowed)
    simulator._events_workload_sources = tuple(sources)
    return simulator


def _load_request(request, *, read_only=False):
    _check(request, _REQUEST_KEYS, REQUEST_SCHEMA)
    options = _options(request['options'])
    path = Path(request['output_dir'])
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('events output directory must be canonical and absolute')
    # _check has verified the complete JSON content, including native scalar
    # encodings and negative zero. Reuse only an identical validated request
    # in the current API call; live plan and source checks still run below.
    scope = _VALIDATED_PREPARE.get()
    key = request['binding']['sha256']
    previous = scope.get('validated') if scope is not None else None
    mapping = MappingSpec(**request['mapping'])
    if previous is not None and previous[0] == key:
        sim = previous[1]
    else:
        adf = from_dict(request['adf'])
        workload = WorkloadSpec(**request['workload'])
        sim = _prepare(adf, workload, mapping, options)
    if _plan(sim) != request['plan']:
        raise ValueError('events implementation or prepared plan differs')
    # This provider's current source vector is its authority. Historical v1
    # receipts keep their producer installation; native layout recognition is
    # neither an events dependency nor a source-identity bypass.
    if request['sources'] != _source_files(sim, mapping):
        raise ValueError('events semantic/component/mapping sources differ; retain the original producer for historical replay')
    if scope is not None:
        scope['validated'] = (key, sim)
    return sim


def _read_result(value):
    _check(value, _RESULT_KEYS, RESULT_SCHEMA)
    request = value['request']
    _load_request(request, read_only=True)  # Builds/checks; no execution models are instantiated.
    ref = value['raw']
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('original events execution receipt required')
    raw_path = Path(ref['path'])
    root = Path(request['output_dir'])
    if raw_path != root / 'RAW.json' or not raw_path.is_file() or raw_path.is_symlink():
        raise ValueError('events raw receipt must be the original output file')
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != ref['sha256']:
        raise ValueError('events raw execution bytes differ')
    raw = json.loads(raw_bytes)
    if (set(raw) != {'schema', 'status', 'request', 'plan', 'evaluation', 'trace'} or raw['schema'] != RAW_SCHEMA
            or raw['status'] != 'complete' or _canonical(raw['request']) != _canonical(request)
            or raw['plan'] != request['plan'] or _canonical(raw['evaluation']) != _canonical(value['evaluation'])):
        raise ValueError('events result differs from the original execution receipt')
    latency = raw['evaluation']['latency_ns']
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency <= 0:
        raise ValueError('invalid events latency')
    records = raw['trace']['records']
    if not records or raw['trace']['latency_ns'] != latency or max(r['end_ns'] for r in records) != latency:
        raise ValueError('events raw trace completion boundary differs')
    return _copy(value)


class EventsProvider:
    id = 'events'
    version = '1'
    # An ordinary protocol domain accepts installed demand/stateful components;
    # compatibility is checked by actual build and mapping, rather than a list
    # of private builtin names. Joint provider-only components cannot build here.
    component_types = ()
    execution_domain = 'single'
    capabilities = {'component_contracts': ['demand', 'stateful'], 'open_component_types': True,
                    'state_model': 'submit/next_event_time/advance', 'unit': 'ns',
                    'replay': 'read_only_original_receipt', 'systolic_models': ['native']}
    request_schemas = (REQUEST_SCHEMA,)
    result_schemas = (RESULT_SCHEMA,)

    @contextmanager
    def _evaluation_scope(self):
        token = _VALIDATED_PREPARE.set({})
        try:
            yield
        finally:
            _VALIDATED_PREPARE.reset(token)

    def prepare_request(self, adf, workload, mapping, *, output_dir, options=None):
        options = _options({} if options is None else options)
        sim = _prepare(adf, workload, mapping, options)
        directory = Path(output_dir).expanduser().resolve()
        request = _seal(dict(schema=REQUEST_SCHEMA, adf=adf.to_dict(), workload=workload.to_dict(),
            mapping=mapping.to_dict(), options=options, output_dir=str(directory),
            sources=_source_files(sim, mapping), plan=_plan(sim)))
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'REQUEST.json').write_bytes(_canonical(request) + b'\n')
        return request

    def assess_request(self, payload):
        try:
            if payload.get('schema') == RESULT_SCHEMA:
                _read_result(payload)
            else:
                _load_request(payload)
        except (ValueError, TypeError, KeyError, OSError, NotImplementedError) as error:
            return {'supported': False, 'gaps': [f'{type(error).__name__}: {error}']}
        return {'supported': True, 'gaps': []}

    def evaluate_request(self, payload):
        if payload.get('schema') == RESULT_SCHEMA:
            return _read_result(payload)
        sim = _load_request(payload)
        directory = Path(payload['output_dir'])
        request_path = directory / 'REQUEST.json'
        if (request_path.is_symlink() or not request_path.is_file()
                or not _matches_request_file(request_path, payload)):
            raise ValueError('original prepared events request differs/missing')
        # O_EXCL marks one execution attempt, including failures. Failed or
        # completed requests cannot silently execute a second time on replay.
        with (directory / 'ATTEMPT.json').open('x') as attempt:
            attempt.write(_canonical({'schema': 'arch-explorer.events-attempt.v1',
                                      'request_binding': payload['binding']}).decode() + '\n')
        try:
            return self._execute(payload, sim)
        except Exception as error:
            failure = dict(schema='arch-explorer.events-failure.v1', status='failed', request=payload,
                           error={'type': type(error).__name__, 'message': str(error)})
            with (directory / 'FAILURE.json').open('xb') as output:
                output.write(_canonical(_seal(failure)) + b'\n')
            raise

    def _execute(self, payload, sim):
        directory = Path(payload['output_dir'])
        from arch_explorer.simulator.events.engine import simulate
        from arch_explorer.simulator.events.backend import evaluation_from_trace
        trace = simulate(sim.mapping.graph, sim.mapping.plan.scheduler)
        with (directory / 'TRACE.json').open('xb') as output:
            output.write(_canonical(asdict(trace)) + b'\n')
        evaluation = evaluation_from_trace(sim, trace, trace=payload['options'].get('trace', False),
                                          check_capacity=payload['options'].get('check_capacity', True))
        evaluation.backend = self.id
        data = evaluation.to_dict()
        if set(data) != set(Evaluation(0).to_dict()):
            raise TypeError('events result requires complete Evaluation')
        raw = dict(schema=RAW_SCHEMA, status='complete', request=payload, plan=payload['plan'],
                   evaluation=data, trace=asdict(trace))
        raw_bytes = _canonical(raw) + b'\n'
        with (directory / 'RAW.json').open('xb') as output:
            output.write(raw_bytes)
        result = _seal(dict(schema=RESULT_SCHEMA, request=payload, evaluation=data,
            raw={'path': str(directory / 'RAW.json'), 'sha256': hashlib.sha256(raw_bytes).hexdigest()}))
        with (directory / 'RESULT.json').open('xb') as output:
            output.write(_canonical(result) + b'\n')
        return result


PROVIDER = EventsProvider()


def register():
    """Descriptor-only, idempotent registration for explicit and installed use."""
    from arch_explorer.simulator.events.backend import evaluate_events
    evaluate_events.model_spec_contracts = {'pi05': 'openpi.pi05.v1'}
    register_backend('events', evaluate_events)
    register_provider(PROVIDER)
