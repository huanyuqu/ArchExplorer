"""Rehashed declarations must agree with execution facts, not just each other."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate, from_dict
from arch_explorer.cli import main
from arch_explorer.simulator.accelsim import backend, runner
from arch_explorer.simulator.accelsim.example import create_example
from arch_explorer.simulator.accelsim.trace import digest, identity, load_trace


ORIGINAL_CASES = ('qa_grid', 'qa_block', 'kernel_identity', 'missing_geometry',
                  'operator_relabel', 'bool_geometry')
ADDITIONAL_CASES = ('grid_y', 'block_z', 'float_geometry', 'string_geometry',
    'empty_geometry', 'short_geometry', 'long_geometry', 'tuple_geometry',
    'missing_kernel', 'missing_block', 'missing_op', 'unknown_workload',
    'missing_trace_id', 'missing_compiler', 'unknown_mapping', 'compiler_type',
    'trace_id_type', 'kernel_type', 'kernel_grid_bool', 'kernel_block_float',
    'kernel_grid_string', 'kernel_missing_block', 'kernel_stream_bool',
    'kernel_stream_float', 'kernel_ctas_bool', 'kernel_instructions_float',
    'manifest_workload_missing_grid', 'manifest_mapping_missing_grid',
    'request_workload_missing_grid', 'request_mapping_missing_grid',
    'declared_stream', 'declared_scope')
CASES = ORIGINAL_CASES + ADDITIONAL_CASES


def make_case(directory, case):
    request = create_example(directory)
    wp, mp = request['workload']['params'], request['mapping']['params']
    ref = wp['accelsim']
    path = Path(ref['manifest'])
    manifest = json.loads(path.read_text())
    if case == 'qa_grid':
        wp['grid'] = mp['grid'] = [2, 1, 1]
    elif case == 'qa_block':
        wp['block'] = mp['block'] = [64, 1, 1]
    elif case == 'kernel_identity':
        wp['kernel'] = 'different_kernel'
    elif case == 'missing_geometry':
        del wp['grid']; del mp['grid']
    elif case == 'operator_relabel':
        wp.update(op='gemm', shape=[1024, 1024, 1024])
    elif case in {'bool_geometry', 'float_geometry', 'string_geometry', 'empty_geometry',
                  'short_geometry', 'long_geometry', 'tuple_geometry', 'grid_y'}:
        value = {'bool_geometry': [True, 1, 1], 'float_geometry': [1.0, 1, 1],
                 'string_geometry': ['1', 1, 1], 'empty_geometry': [],
                 'short_geometry': [1, 1], 'long_geometry': [1, 1, 1, 1],
                 'tuple_geometry': (1, 1, 1), 'grid_y': [1, 2, 1]}[case]
        wp['grid'] = mp['grid'] = value
    elif case == 'block_z':
        wp['block'] = mp['block'] = [32, 1, 2]
    elif case.startswith('missing_'):
        key = case.removeprefix('missing_')
        del (mp if key in {'trace_id', 'compiler'} else wp)[key]
    elif case == 'unknown_workload':
        wp['batch'] = 2
    elif case == 'unknown_mapping':
        mp['tile'] = 64
    elif case == 'compiler_type':
        mp['compiler'] = {'name': 'compiler'}
    elif case == 'trace_id_type':
        mp['trace_id'] = 1
    elif case == 'kernel_type':
        wp['kernel'] = ['synthetic_memory_chain']
    elif case == 'declared_stream':
        wp['stream'] = mp['stream'] = 1
    elif case == 'declared_scope':
        wp['scope'] = 'complete-model'
    elif case == 'renamed_control':
        raw = (path.parent / 'kernel-1.traceg').read_bytes().replace(b'synthetic_memory_chain', b'another_valid_kernel')
        (path.parent / 'kernel-1.traceg').write_bytes(raw)
        manifest['kernels'][0].update(name='another_valid_kernel', sha256=digest(raw), original_sha256=digest(raw))
        wp['kernel'] = 'another_valid_kernel'
        mp.update(trace_id='independent-trace-id', compiler='declared-provenance-only')
    elif case not in {'valid_control', 'provenance_control'} and not case.startswith(('kernel_', 'manifest_', 'request_')):
        raise AssertionError(f'unknown fixture {case}')
    if case == 'provenance_control':
        mp['compiler'] = 'different-declared-compiler'
        manifest['source']['toolchain'] = 'different-declared-toolchain'
    manifest['workload'] = identity(WorkloadSpec(**request['workload']))
    manifest['mapping'] = deepcopy(request['mapping'])
    if case.startswith('kernel_') and case not in {'kernel_identity', 'kernel_type'}:
        k = manifest['kernels'][0]
        changes = {'kernel_grid_bool': ('grid', [True, 1, 1]),
                   'kernel_block_float': ('block', [32.0, 1, 1]),
                   'kernel_grid_string': ('grid', ['1', 1, 1]),
                   'kernel_stream_bool': ('stream', False),
                   'kernel_stream_float': ('stream', 0.0),
                   'kernel_ctas_bool': ('ctas', True),
                   'kernel_instructions_float': ('thread_instructions', 128.0)}
        if case == 'kernel_missing_block':
            del k['block']
        else:
            key, value = changes[case]
            k[key] = value
    if case.startswith(('manifest_', 'request_')):
        owner, section, _, key = case.split('_')
        target = manifest[section] if owner == 'manifest' else request[section]
        del target['params'][key]
    raw = (json.dumps(manifest, indent=2) + '\n').encode()
    path.write_bytes(raw)
    ref['sha256'] = digest(raw)
    (path.parent / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    return request


class NativeBoundary(ValueError):
    pass


def instrumented(request, entry, output=None):
    calls = {'run_native': 0, 'Popen': 0}
    def native(*args, **kwargs):
        calls['run_native'] += 1
        raise NativeBoundary('native-start sentinel: no engine executed')
    def process(*args, **kwargs):
        calls['Popen'] += 1
        raise AssertionError('unexpected native subprocess')
    with ExitStack() as stack:
        stack.enter_context(patch.object(backend, 'run_native', native))
        stack.enter_context(patch.object(runner.subprocess, 'Popen', process))
        if entry == 'parser':
            load_trace(WorkloadSpec(**request['workload']), MappingSpec(**request['mapping']))
        elif entry == 'api':
            evaluate(from_dict(request['adf']), WorkloadSpec(**request['workload']),
                     MappingSpec(**request['mapping']), backend='accelsim', cost_model=False, **request['options'])
        elif entry in {'cli', 'replay'}:
            payload = request if entry == 'cli' else {'schema': backend.RESULT_SCHEMA, 'request': request, 'evaluation': {}}
            path = Path(output).with_suffix('.input.json')
            path.write_text(json.dumps(payload, indent=2) + '\n')
            code = main(['evaluate', str(path), '--output', str(output)])
            return calls, code
        else:
            raise AssertionError(entry)
    return calls, None


@pytest.mark.parametrize('case', CASES)
def test_rehashed_facts_rejected_before_native(tmp_path, case):
    request = make_case(tmp_path / case, case)
    with patch.object(backend, 'run_native') as native, patch.object(runner.subprocess, 'Popen') as process:
        with pytest.raises(ValueError):
            load_trace(WorkloadSpec(**request['workload']), MappingSpec(**request['mapping']))
        with pytest.raises(ValueError):
            evaluate(from_dict(request['adf']), WorkloadSpec(**request['workload']),
                     MappingSpec(**request['mapping']), backend='accelsim', cost_model=False)
        native.assert_not_called()
        process.assert_not_called()


@pytest.mark.parametrize('case', ORIGINAL_CASES)
@pytest.mark.parametrize('entry', ['cli', 'replay'])
def test_cli_and_result_replay_share_preflight(tmp_path, capsys, case, entry):
    request = make_case(tmp_path / case, case)
    output = tmp_path / 'must-not-exist.json'
    calls, code = instrumented(request, entry, output)
    assert code == 2 and calls == {'run_native': 0, 'Popen': 0}
    assert not output.exists()
    assert 'archexplorer:' in capsys.readouterr().err


@pytest.mark.parametrize('case', ['valid_control', 'renamed_control', 'provenance_control'])
def test_compatible_identities_reach_native(tmp_path, case):
    request = make_case(tmp_path / case, case)
    load_trace(WorkloadSpec(**request['workload']), MappingSpec(**request['mapping']))
    with pytest.raises(NativeBoundary):
        instrumented(request, 'api')


def test_clock_source_is_fixed_upstream_line35():
    from arch_explorer.adf import walk
    from arch_explorer.archs.accelsim import sm90_adf
    clocks = [p.params['clock_mhz'] for _, p in walk(sm90_adf()) if hasattr(p, 'params') and 'clock_mhz' in p.params]
    assert len(clocks) == 4
    assert all(p.source.endswith('/SM90_H100/gpgpusim.config:35') for p in clocks)
