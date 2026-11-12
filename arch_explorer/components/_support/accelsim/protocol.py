# Concrete Accel-Sim implementation; owned by the component provider.
"""Read-only closure of a completed native prediction and its frozen inputs.

This is a posthoc software review, not a signed provenance certificate, a new
execution, or a claim that an old prediction was frozen before measurement.
Trace/config/engine/statistics semantics belong to their production parsers.
"""
import argparse
import json
import math
from pathlib import Path
import re
import tarfile

from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
from arch_explorer.components.accelsim.backend import REQUEST_SCHEMA
from arch_explorer.components.accelsim.result_contract import (RESULT_SCHEMAS, DETAIL_SCHEMA, COMPONENT_MAP_SCHEMA, producer_contract,
                              detail_fields, component_map)
from arch_explorer.components.accelsim.capabilities import assess
from arch_explorer.components.accelsim.config import compile_adf
from arch_explorer.components.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA
from arch_explorer.components.accelsim.runner import load_engine
from arch_explorer.components.accelsim.stats import parse_stats
from arch_explorer.components.accelsim.trace import digest, identity, load_json, load_trace, same_json
from arch_explorer.components.accelsim.trace_stream import file_digest

SCHEMA = 'arch-explorer.accelsim-protocol.v1'
RESULT = 'arch-explorer.accelsim-protocol-preflight.v1'
SOURCE_SCHEMA = 'arch-explorer.accelsim-execution-source.v1'
HISTORY_SCHEMA = 'arch-explorer.accelsim-history.v1'
SOURCE_PREFIX = 'arch_explorer/'
SOURCE_REQUIRED = {'__init__.py', 'backend.py', 'capabilities.py', 'config.py', 'example.py',
                   'isa.py', 'profile.py', 'runner.py', 'stats.py', 'trace.py', 'trace_stream.py', 'trace_v2.py'}
PROTOCOL_FIELDS = {'schema', 'purpose', 'review_kind', 'request', 'frozen', 'history',
                   'case_identity', 'target', 'compile_trace_binding'}
EVALUATION_FIELDS = {'latency_ns', 'breakdown', 'utilization', 'area_mm2', 'cost_usd',
                     'warnings', 'provenance', 'backend', 'details'}
RUN_FIELDS = {'schema', 'argv', 'cwd', 'environment', 'engine', 'engine_manifest_sha256',
              'config_sha256', 'adf_sha256', 'limits', 'files', 'status',
              'adapter_sources_sha256', 'pid', 'exit_code', 'stdout_sha256',
              'stderr_sha256', 'stats', 'trace_manifest_sha256'}


class ClosureError(ValueError):
    def __init__(self, kind, at, reason):
        self.blocker = {'kind': kind, 'at': at, 'reason': reason}
        super().__init__(reason)


def check(condition, kind, at, reason):
    if not condition:
        raise ClosureError(kind, at, reason)


def fields(value, required, kind, at):
    check(type(value) is dict and set(value) == required, kind, at,
          'expected exactly fields '+str(sorted(required)))


def text(value, at):
    check(type(value) is str and bool(value.strip()), 'protocol_type', at, 'nonempty string required')
    return value


def path(value, at):
    return Path(text(value, at)).resolve()


def hex_digest(value, at, length=64):
    check(type(value) is str and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None,
          'protocol_type', at, 'lowercase hexadecimal identity required')
    return value


def read_json(file):
    check(file.is_file() and file.stat().st_size <= 8*2**20, 'protocol_file', str(file),
          'JSON file missing or exceeds 8MiB')
    value, raw = load_json(file)
    json.dumps(value, allow_nan=False)  # Also rejects 1e999 decoded as infinity.
    return value, raw


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def limits(options):
    check(type(options) is dict and not set(options)-{'engine', 'output_dir', 'timeout_s', 'max_cycles', 'memory_limit_mb'},
          'request_options', 'request.options', 'unknown execution option')
    text(options.get('engine'), 'request.options.engine')
    if options.get('output_dir') is not None:
        text(options['output_dir'], 'request.options.output_dir')
    timeout = options.get('timeout_s', 120)
    maximum = options.get('max_cycles', 1_000_000)
    memory = options.get('memory_limit_mb', 4096)
    check(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 3600,
          'request_options', 'timeout_s', 'finite timeout in (0,3600] required')
    check(type(maximum) is int and 1 <= maximum <= 1_000_000_000,
          'request_options', 'max_cycles', 'integer cycle limit required')
    check(type(memory) is int and 512 <= memory <= 16384,
          'request_options', 'memory_limit_mb', 'integer memory limit required')
    # Only output_dir and WorkloadSpec display name/weight are non-execution.
    # Defaults normalize explicitly; all three guards remain in the identity.
    return {'timeout_s': float(timeout), 'max_cycles': maximum, 'memory_limit_mb': memory}


def consume_request(payload):
    fields(payload, {'schema', 'adf', 'workload', 'mapping', 'options'}, 'request_schema', 'request')
    check(payload['schema'] == REQUEST_SCHEMA, 'request_schema', 'request.schema', 'unsupported request schema')
    guards = limits(payload['options'])
    config = compile_adf(from_dict(payload['adf']))
    workload, mapping = WorkloadSpec(**payload['workload']), MappingSpec(**payload['mapping'])
    manifest, artifact, manifest_hash, actual = load_trace(workload, mapping)
    engine, binary, libraries, engine_hash = load_engine(payload['options']['engine'])
    from arch_explorer.components.accelsim.variant import require_trace_engine
    require_trace_engine(manifest,engine)
    manifest_path = path(workload.params['accelsim']['manifest'], 'trace manifest')
    trace_path = (manifest_path.parent/manifest['kernels'][0]['file']).resolve()
    trace_hash = digest(artifact) if isinstance(artifact, bytes) else artifact.sha256
    engine_identity = {'sources': engine['sources'], 'binary_sha256': engine['binary']['sha256'],
        'libraries': [{'basename': p.name, 'sha256': row['sha256']}
                      for p, row in zip(libraries, engine['libraries'])]}
    scope = {key: actual[key] for key in ('name', 'grid', 'block', 'stream')}
    scope.update(scope=manifest['scope'], isa=manifest['isa'])
    case = digest(canonical({'schema': 'arch-explorer.accelsim-case.v1',
                             'trace_sha256': trace_hash, 'scope': scope}))
    semantic = None
    if 'flow_sha256' in actual:
        semantic_scope = dict(scope, nregs=actual['nregs'], shmem=actual['shmem'])
        semantic = digest(canonical({'schema': 'arch-explorer.accelsim-semantic-case.v1',
                                     'flow_sha256': actual['flow_sha256'], 'scope': semantic_scope}))
    execution = {'adf_sha256': config.adf_sha256, 'config_sha256': config.sha256,
        'trace_manifest_sha256': manifest_hash, 'trace_sha256': trace_hash,
        'trace_contract': manifest['schema'], 'scope': scope, 'workload': identity(workload),
        'mapping': mapping.to_dict(), 'engine_manifest_sha256': engine_hash,
        'engine': engine_identity, 'limits': guards}
    return {'payload': payload, 'config': config, 'manifest': manifest, 'actual': actual,
            'manifest_path': manifest_path, 'trace_path': trace_path,
            'engine': engine, 'binary': binary, 'libraries': libraries,
            'case_identity': case, 'semantic_identity': semantic, 'execution': execution}


def case_identity(request):
    """Derive registered identities from actual consumed bytes and scope only."""
    current = consume_request(request)
    return {key: current[key] for key in ('case_identity', 'semantic_identity')}


def frozen_rows(rows):
    check(type(rows) is list and 1 <= len(rows) <= 64, 'frozen_roles', 'frozen', 'bounded role list required')
    result = {}
    for row in rows:
        fields(row, {'role', 'path', 'sha256'}, 'frozen_roles', 'frozen row')
        role = text(row['role'], 'frozen.role')
        check(role not in result, 'frozen_roles', role, 'duplicate frozen role')
        file = path(row['path'], role)
        sha = hex_digest(row['sha256'], role+'.sha256')
        check(file.is_file() and file_digest(file) == sha, 'frozen_identity', role, 'frozen bytes missing or changed')
        result[role] = {'path': file, 'sha256': sha}
    return result


def bind(rows, role, file, expected=None):
    check(role in rows, 'frozen_roles', role, 'missing required frozen role')
    check(rows[role]['path'] == file.resolve(), 'frozen_binding', role,
          'frozen path is not the object actually consumed')
    if expected is not None:
        check(rows[role]['sha256'] == expected, 'frozen_binding', role, 'frozen digest differs from actual consumed identity')


def source_identity(rows, run):
    source, _ = read_json(rows['source-commit']['path'])
    fields(source, {'schema', 'source_sha', 'archive_sha256', 'origin'}, 'source_identity', 'source-commit')
    check(source['schema'] == SOURCE_SCHEMA, 'source_identity', 'source.schema', 'unsupported source identity schema')
    hex_digest(source['source_sha'], 'source_sha', 40)
    fields(source['origin'], {'path', 'sha256'}, 'source_identity', 'source.origin')
    bind(rows, 'source-origin', path(source['origin']['path'], 'source.origin.path'), source['origin']['sha256'])
    origin, _ = read_json(rows['source-origin']['path'])
    check(type(origin) is dict, 'source_identity', 'source-origin', 'execution source record must be an object')
    check(origin.get('source_sha') == source['source_sha'] and origin.get('archive_sha256') == source['archive_sha256']
          and origin.get('clean') is True, 'source_identity', 'source-origin', 'execution source record does not identify the archive/commit')
    check(rows['source-archive']['sha256'] == source['archive_sha256'], 'source_identity', 'source-archive', 'source archive identity differs')
    file = rows['source-archive']['path']
    check(file.stat().st_size <= 64*2**20, 'source_identity', 'source-archive', 'source archive exceeds 64MiB')
    with tarfile.open(file) as archive:
        check(archive.pax_headers.get('comment') == source['source_sha'], 'source_identity', 'source-archive', 'git archive commit label differs')
        members = archive.getmembers()
        check(len(members) <= 10000 and len({m.name for m in members}) == len(members),
              'source_identity', 'source-archive', 'duplicate/excess archive entries')
        names = {m.name for m in members}
        if SOURCE_PREFIX+'components/accelsim/__init__.py' in names:
            # A new archive contains component-owned sources and legacy shims.
            # Retained runtime sources come only from the authoritative package.
            prefix = SOURCE_PREFIX+'components/accelsim/'
            selected = set()
        else:
            prefix = SOURCE_PREFIX+'simulator/accelsim/'
            selected = {SOURCE_PREFIX+'components/accelsim.py', SOURCE_PREFIX+'archs/accelsim.py'}
        selected |= {n for n in names if n.startswith(prefix) and n.endswith('.py') and '/' not in n[len(prefix):]}
        check(selected <= names and {prefix+n for n in SOURCE_REQUIRED} <= selected and len(selected) <= 256,
              'source_identity', 'source-archive', 'source closure missing/excessive')
        hashes = {}
        for name in selected:
            member = archive.getmember(name)
            check(member.isfile() and member.size <= 2**20, 'source_identity', name, 'regular bounded adapter source required')
            hashes[name[len(SOURCE_PREFIX):]] = digest(archive.extractfile(member).read())
    check(same_json(hashes, run['adapter_sources_sha256']), 'source_identity', 'run.adapter_sources_sha256',
          'retained execution used different/incomplete adapter source bytes')
    return source['source_sha'], hashes


def read_history(file, seen=None):
    seen = set() if seen is None else seen
    check(file not in seen and len(seen) < 16, 'history', str(file), 'cyclic/excess history references')
    seen.add(file)
    data, _ = read_json(file)
    check(type(data) is dict, 'history', str(file), 'history object required')
    facts = {'cases': set(), 'semantic': set(), 'bytes': set(), 'binaries': set(), 'unresolved': False}
    if data.get('schema') == 'arch-explorer.sim27-history.v1':
        fields(data, {'schema', 'case_identities', 'closed_binary_sha256', 'policy'}, 'history', str(file))
        facts['unresolved'] = True  # Old static binaries do not prove measured scope.
        check(type(data['case_identities']) is list, 'history', 'case_identities', 'list required')
        for entry in data['case_identities']:
            text(entry, 'history.case_identity')
            if re.fullmatch('[0-9a-f]{64}:[0-9a-f]{64}', entry):
                facts['bytes'].add(entry.split(':')[0])  # Source-only manifest edits cannot evade old entries.
            else:
                hex_digest(entry, 'history.case_identity')
                facts['cases'].add(entry)
    else:
        fields(data, {'schema', 'case_identities', 'semantic_identities', 'closed_binary_sha256', 'parents', 'policy'},
               'history', str(file))
        check(data['schema'] == HISTORY_SCHEMA, 'history', 'schema', 'unsupported history schema')
        for name, destination in [('case_identities', 'cases'), ('semantic_identities', 'semantic')]:
            check(type(data[name]) is list, 'history', name, 'list required')
            for value in data[name]:
                facts[destination].add(hex_digest(value, 'history.'+name))
        check(type(data['parents']) is list and len(data['parents']) <= 16, 'history', 'parents', 'bounded parent list required')
        for parent in data['parents']:
            fields(parent, {'path', 'sha256'}, 'history', 'parent')
            p = path(parent['path'], 'history.parent')
            check(file_digest(p) == hex_digest(parent['sha256'], 'history.parent.sha256'), 'history', str(p), 'old history bytes changed')
            other = read_history(p, seen)
            for key in ('cases', 'semantic', 'bytes', 'binaries'):
                facts[key] |= other[key]
            facts['unresolved'] |= other['unresolved']
    text(data['policy'], 'history.policy')
    check(type(data['closed_binary_sha256']) is dict, 'history', 'closed_binary_sha256', 'mapping required')
    for value, origins in data['closed_binary_sha256'].items():
        facts['binaries'].add(hex_digest(value, 'history.binary'))
        check(type(origins) is list and all(type(p) is str for p in origins), 'history', value, 'origin list required')
    return facts


def validate_prediction(rows, current):
    prediction, _ = read_json(rows['prediction']['path'])
    run, _ = read_json(rows['native-run']['path'])
    source_sha, source_hashes = source_identity(rows, run)
    return validate_prediction_data(prediction, run, current, source_sha, source_hashes,
        rows['native-run']['path'].parent,
        lambda role, file, expected: bind(rows, role, file, expected))


def validate_prediction_data(prediction, run, current, source_sha, source_hashes, directory, retain):
    """Validate facts after the caller proves its complete source provenance.

    Archive review and public read-only replay share this exact numeric/input
    verifier; each supplies its own explicit original-file retention contract.
    """
    fields(prediction, {'schema', 'request', 'evaluation'}, 'prediction_schema', 'prediction')
    check(prediction['schema'] in RESULT_SCHEMAS, 'prediction_schema', 'prediction.schema', 'complete versioned result envelope required')
    claimed = consume_request(prediction['request'])
    check(same_json(claimed['execution'], current['execution']), 'prediction_identity', 'prediction.request',
          'complete prediction belongs to another request/configuration/trace/engine/guard set')
    evaluation = prediction['evaluation']
    fields(evaluation, EVALUATION_FIELDS, 'prediction_schema', 'prediction.evaluation')
    details = evaluation['details']
    check(evaluation['backend'] == 'accelsim' and evaluation['area_mm2'] is None and evaluation['cost_usd'] is None
          and evaluation['breakdown'] == [] and evaluation['utilization'] == {},
          'prediction_schema', 'prediction.evaluation', 'native backend or unknown metric scope changed')
    check(type(evaluation['warnings']) is list and all(type(v) is str for v in evaluation['warnings']),
          'prediction_schema', 'warnings', 'warning annotations must be strings')
    fields(run, RUN_FIELDS, 'native_record', 'native-run')
    check(run['schema'] == 'arch-explorer.accelsim-run.v1' and run['status'] == 'complete'
          and type(run['exit_code']) is int and run['exit_code'] == 0 and type(run['pid']) is int and run['pid'] > 0,
          'native_record', 'run.status', 'completed zero-exit native record required')
    try:
        version = producer_contract(prediction['schema'], source_hashes)
    except ValueError as error:
        raise ClosureError('prediction_version', 'prediction.schema/source', str(error)) from error
    fields(details, detail_fields(version), 'prediction_schema', 'prediction.details')
    check(type(details['capabilities']) is dict and type(details['preflight']) is dict,
          'prediction_schema', 'annotations', 'capability annotations required; not used as evidence')
    if version == 'v2':
        check(details['schema'] == DETAIL_SCHEMA, 'prediction_version',
              'prediction.details.schema', 'complete current result-details dialect required')
        check(details['component_map_schema'] == COMPONENT_MAP_SCHEMA, 'prediction_version',
              'component_map_schema', 'complete current component-map dialect required')
    if version != 'sim28':
        check(same_json(details['device_facts'], current['config'].device_facts), 'prediction_identity',
              'device_facts', 'fact report must exactly match the request audit, including null, original bytes, known/unknown, assumptions and lease/accuracy scope')
    check(same_json(details['run'], run), 'native_record', 'prediction.details.run', 'embedded run differs from frozen retained record')
    check(path(details['evidence'], 'prediction.evidence') == path(run['cwd'], 'run.cwd'),
          'native_record', 'evidence', 'recorded evidence location differs from run cwd')
    config, manifest, expected = current['config'], current['manifest'], current['actual']
    execution = current['execution']
    for key in ('engine_manifest_sha256', 'config_sha256', 'adf_sha256', 'trace_manifest_sha256'):
        check(run[key] == execution[key] and details[key] == execution[key],
              'prediction_identity', key, 'prediction/native record differs from actual consumed identity')
    check(same_json(run['engine'], current['engine']) and same_json(details['engine'], current['engine']),
          'prediction_identity', 'engine', 'engine contents or complete libraries differ')
    check(details['contract'] == REQUEST_SCHEMA and details['trace_sha256'] == execution['trace_sha256']
          and details['trace_contract'] == manifest['schema'] and same_json(details['trace_facts'], expected)
          and same_json(details['trace_source'], manifest['source']),
          'prediction_identity', 'trace/scope', 'actual trace facts, scope or source binding differs')
    try:
        expected_map = component_map(config, version)
    except ValueError as error:
        raise ClosureError('prediction_identity', 'component_map', str(error)) from error
    check(same_json(details['component_map'], expected_map), 'prediction_identity', 'component_map', 'configuration ownership or versioned metadata differs')
    provenance = [{k: row[k] for k in ('path', 'param', 'value', 'unit', 'source')} for row in expected_map]
    check(same_json(evaluation['provenance'], provenance), 'prediction_identity', 'provenance', 'ADF parameter/source provenance differs')
    normalized_guards = limits(run['limits'] | {'engine': current['payload']['options']['engine']})
    check(set(run['limits']) == {'timeout_s', 'max_cycles', 'memory_limit_mb'} and normalized_guards == execution['limits'],
          'prediction_identity', 'limits', 'native execution guards differ')
    argv = [str(current['binary']), '-config', 'gpgpusim.config', '-config', 'limits.config', '-trace', './kernelslist.g']
    environment = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'CUDA_VISIBLE_DEVICES': '',
                   'LD_LIBRARY_PATH': ':'.join(dict.fromkeys(str(p.parent) for p in current['libraries']))}
    check(same_json(run['argv'], argv) and same_json(run['environment'], environment),
          'prediction_identity', 'native invocation', 'native binary/arguments/environment differ')
    staged = 'kernel-1.'+manifest['format']
    inputs = dict(config.artifacts, **{'limits.config': '-gpgpu_max_cycle '+str(execution['limits']['max_cycles'])+'\n',
                                     'kernelslist.g': staged+'\n'})
    required_inputs = set(inputs) | {staged}
    check(type(run['files']) is dict and set(run['files']) == required_inputs,
          'native_record', 'run.files', 'incomplete/extraneous native input list')
    for name in sorted(required_inputs):
        expected_hash = execution['trace_sha256'] if name == staged else digest(inputs[name].encode())
        retain('native-input:'+name, directory/name, expected_hash)
        check(run['files'][name] == expected_hash, 'native_record', name, 'retained native input identity differs')
    for label in ('stdout', 'stderr'):
        retain('native-'+label, directory/(label+'.log'), run[label+'_sha256'])
        check((directory/(label+'.log')).stat().st_size <= 32*2**20, 'native_record', label, 'native output exceeds 32MiB')
    stdout = (directory/'stdout.log').read_text(errors='replace')
    stderr = (directory/'stderr.log').read_text(errors='replace')
    check(ACCELSIM_SHA[:7] in stdout and GPGPUSIM_SHA[:7] in stdout, 'native_record', 'stdout', 'native source build identity missing')
    clock = next(row['value'] for row in config.component_map if row['param'] == 'clock_mhz' and row['clock_domain'] == 'sm')
    try:
        stats = parse_stats(stdout, stderr, expected, clock)
    except ValueError as error:
        raise ClosureError('native_record', 'stdout/stderr', str(error)) from error
    check(same_json(run['stats'], stats) and same_json(details['native'], stats),
          'prediction_numeric', 'native statistics', 'JSON statistics disagree with independently hash-bound native output')
    latency = evaluation['latency_ns']
    check(type(latency) in (int, float) and math.isfinite(latency) and latency > 0
          and math.isclose(latency, stats['latency_ns'], rel_tol=1e-12, abs_tol=1e-9),
          'prediction_numeric', 'evaluation.latency_ns', 'top-level latency contradicts completed native cycles/clock/scope')
    return run, stats, required_inputs, source_sha, version


def _preflight(spec):
    fields(spec, PROTOCOL_FIELDS, 'protocol_schema', 'protocol')
    check(spec['schema'] == SCHEMA, 'protocol_schema', 'schema', 'versioned installable protocol required; old private drafts must be migrated')
    check(spec['purpose'] in ('software', 'hardware-accuracy') and spec['review_kind'] == 'posthoc-software-review',
          'protocol_schema', 'purpose/review_kind', 'this protocol audits existing runs; never backdates a measurement freeze')
    check(spec['compile_trace_binding'] is False and (spec['purpose'] != 'software' or spec['target'] is None),
          'protocol_schema', 'target/compile_trace_binding', 'no unvalidated attestation or ignored software target accepted')
    request_path = path(spec['request'], 'request')
    request, request_raw = read_json(request_path)
    current = consume_request(request)
    rows = frozen_rows(spec['frozen'])
    fixed = {'request': request_path, 'trace-manifest': current['manifest_path'], 'trace': current['trace_path'],
             'commands': current['manifest_path'].parent/current['manifest']['commands']['file'],
             'engine-manifest': path(request['options']['engine'], 'engine manifest'), 'engine-binary': current['binary'],
             'history': path(spec['history'], 'history')}
    fixed.update({'engine-library-'+str(i): p for i, p in enumerate(current['libraries'])})
    if current['manifest'].get('capture_binding') is not None:
        from arch_explorer.components.accelsim.capture_input import references
        fixed.update({'capture:'+role: p for role,p in references(current['manifest']['capture_binding']).items()})
    consumed_hashes = {'request': digest(request_raw), 'trace-manifest': current['execution']['trace_manifest_sha256'],
        'trace': current['execution']['trace_sha256'], 'commands': current['manifest']['commands']['sha256'],
        'engine-manifest': current['execution']['engine_manifest_sha256'],
        'engine-binary': current['engine']['binary']['sha256']}
    consumed_hashes.update({'engine-library-'+str(i): row['sha256'] for i,row in enumerate(current['engine']['libraries'])})
    required = set(fixed) | {'adf', 'config', 'prediction', 'source-commit', 'source-origin', 'source-archive',
                             'native-run', 'native-stdout', 'native-stderr'}
    check(required <= set(rows), 'frozen_roles', 'frozen', 'missing frozen roles: '+str(sorted(required-set(rows))))
    for role, file in fixed.items():
        bind(rows, role, file, consumed_hashes.get(role))
    check(rows['trace']['sha256'] == current['execution']['trace_sha256'], 'frozen_binding', 'trace', 'frozen trace differs from actual input bytes')
    frozen_adf, _ = read_json(rows['adf']['path'])
    frozen_config, _ = read_json(rows['config']['path'])
    check(same_json(frozen_adf, request['adf']), 'frozen_binding', 'adf', 'frozen ADF differs from actual request')
    check(same_json(frozen_config, current['config'].artifacts), 'frozen_binding', 'config', 'complete frozen native config differs')
    hex_digest(spec['case_identity'], 'case_identity')
    check(spec['case_identity'] == current['case_identity'], 'case_identity', 'case_identity', 'case identity must derive from actual trace bytes and execution scope')
    history = read_history(fixed['history'])
    overlap = (current['case_identity'] in history['cases'] or current['execution']['trace_sha256'] in history['bytes']
               or (current['semantic_identity'] is not None and current['semantic_identity'] in history['semantic']))
    check(not overlap, 'historical_overlap', 'history',
          'actual consumed case/bytes/known semantic flow already registered; provenance edits do not make it new')
    run, stats, native_inputs, source_sha, version = validate_prediction(rows, current)
    expected_roles = required | {'native-input:'+name for name in native_inputs}
    check(set(rows) == expected_roles, 'frozen_roles', 'frozen', 'unrelated/extraneous frozen role')
    if current['config'].device_facts is not None:
        original = current['config'].device_facts['original']
        check(file_digest(Path(original['path'])) == original['sha256'], 'frozen_identity',
              'device evidence', 'fact evidence changed during review')
    for role, row in rows.items():
        check(file_digest(row['path']) == row['sha256'], 'frozen_identity', role, 'frozen file changed during review')
    capability = assess(current['config'], current['manifest'], current['actual'], purpose=spec['purpose'], target=spec['target'],engine=request['options']['engine'])
    blockers = list(capability['blockers'])
    if spec['purpose'] == 'hardware-accuracy':
        blockers += [{'kind': 'measurement_association', 'at': 'compile_trace_binding', 'reason': 'same-device compilation/capture/measurement association has not been independently validated'},
                     {'kind': 'new_window', 'at': 'purpose', 'reason': 'no registered future hardware window in this software-review contract'}]
        if history['unresolved']:
            blockers.append({'kind': 'history_evidence', 'at': 'history', 'reason': 'legacy/static binary history has unresolved measurement/cross-binary scope'})
    return {'schema': RESULT, 'status': 'blocked' if blockers else 'passed', 'purpose': spec['purpose'],
            'review_kind': spec['review_kind'], 'blockers': blockers, 'case_identity': current['case_identity'],
            'semantic_identity': current['semantic_identity'], 'execution_identity': current['execution'],
            'execution_source_sha': source_sha, 'native': stats, 'capability': capability,
            'result_contract': version,
            'annotation_scope': 'source-bound versioned metadata; no historical clock-domain annotation is physical evidence',
            'historical_scope': 'matched_registered' if overlap else 'no_known_match; cross-binary/unknown history is not proven unseen',
            'hardware_accuracy': 'not validated', 'native_started': False, 'hardware_executed': False}


def preflight_protocol(spec):
    """Check a frozen completed prediction without runner, subprocess or writes."""
    try:
        json.dumps(spec, allow_nan=False)
        return _preflight(spec)
    except ClosureError as error:
        blocker = error.blocker
    except (ValueError, OSError, ImportError, KeyError, TypeError, tarfile.TarError, StopIteration) as error:
        blocker = {'kind': 'protocol_input', 'at': 'consumption', 'reason': str(error)}
    return {'schema': RESULT, 'status': 'blocked', 'purpose': spec.get('purpose') if type(spec) is dict else None,
            'blockers': [blocker], 'hardware_accuracy': 'not validated',
            'native_started': False, 'hardware_executed': False}


def prepare_review(request_file, prediction_file, source_origin, source_archive, history_file,
                   directory, *, purpose='software', target=None):
    """Create new review records around existing evidence, never modify origin files.

    The returned record is explicitly posthoc. Preparation alone is not a pass;
    use preflight_protocol before relying on the frozen closure.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    request_file, prediction_file, source_origin, source_archive, history_file = (
        Path(p).resolve() for p in (request_file, prediction_file, source_origin, source_archive, history_file))
    request, _ = read_json(request_file)
    current = consume_request(request)
    prediction, _ = read_json(prediction_file)
    run_path = path(prediction['evaluation']['details']['evidence'], 'prediction.evidence')/'run.json'
    run, _ = read_json(run_path)
    origin, origin_raw = read_json(source_origin)
    source = {'schema': SOURCE_SCHEMA, 'source_sha': origin['source_sha'],
              'archive_sha256': origin['archive_sha256'],
              'origin': {'path': str(source_origin), 'sha256': digest(origin_raw)}}
    artifacts = {'adf': request['adf'], 'config': current['config'].artifacts, 'source-commit': source}
    paths = {}
    for role, value in artifacts.items():
        file = directory/(role+'.json')
        check(not file.exists(), 'review_output', str(file), 'use a new review directory; never overwrite frozen records')
        file.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        paths[role] = file
    paths.update({'request': request_file, 'prediction': prediction_file, 'source-origin': source_origin,
        'source-archive': source_archive, 'history': history_file,
        'trace-manifest': current['manifest_path'], 'trace': current['trace_path'],
        'commands': current['manifest_path'].parent/current['manifest']['commands']['file'],
        'engine-manifest': path(request['options']['engine'], 'engine'), 'engine-binary': current['binary'],
        'native-run': run_path, 'native-stdout': run_path.parent/'stdout.log', 'native-stderr': run_path.parent/'stderr.log'})
    paths.update({'engine-library-'+str(i): p for i, p in enumerate(current['libraries'])})
    if current['manifest'].get('capture_binding') is not None:
        from arch_explorer.components.accelsim.capture_input import references
        paths.update({'capture:'+role: p for role,p in references(current['manifest']['capture_binding']).items()})
    paths.update({'native-input:'+name: run_path.parent/name for name in run['files']})
    spec = {'schema': SCHEMA, 'purpose': purpose, 'review_kind': 'posthoc-software-review', 'request': str(request_file),
            'frozen': [{'role': role, 'path': str(file), 'sha256': file_digest(file)} for role, file in sorted(paths.items())],
            'history': str(history_file), 'case_identity': current['case_identity'], 'target': target, 'compile_trace_binding': False}
    file = directory/'protocol.json'
    check(not file.exists(), 'review_output', str(file), 'review already exists')
    file.write_text(json.dumps(spec, indent=2, allow_nan=False)+'\n')
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', type=Path)
    args = parser.parse_args(argv)
    try:
        spec, _ = read_json(args.protocol.resolve())
        result = preflight_protocol(spec)
    except (ValueError, OSError, TypeError) as error:
        result = {'schema': RESULT, 'status': 'blocked', 'blockers': [{'kind': 'protocol_input', 'at': 'file', 'reason': str(error)}],
                  'native_started': False, 'hardware_executed': False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
