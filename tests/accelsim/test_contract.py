"""Contract checks; true native end-to-end tests are enabled by an engine manifest."""
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path

import pytest

from arch_explorer.adf import MappingSpec, Param, SpacePoint, WorkloadSpec, evaluate, from_dict
from arch_explorer.archs.accelsim import sm90_adf
from arch_explorer.simulator.accelsim.backend import evaluate_request
from arch_explorer.simulator.accelsim.config import compile_adf
from arch_explorer.simulator.accelsim.example import create_example
from arch_explorer.simulator.accelsim.runner import load_engine
from arch_explorer.simulator.accelsim.stats import parse_stats
from arch_explorer.simulator.accelsim.trace import digest, load_trace


@pytest.fixture
def trace_request(tmp_path):
    return create_example(tmp_path)


def set_param(adf, group, point, key, value):
    node = adf.child(group).child(point) if group else adf.child(point)
    old = node.params[key]
    node.params[key] = Param(value, old.unit, 'synthetic contract test')


def specs(trace_request):
    return WorkloadSpec(**trace_request['workload']), MappingSpec(**trace_request['mapping'])


def rewrite_manifest(trace_request, update):
    ref = trace_request['workload']['params']['accelsim']
    path = Path(ref['manifest'])
    manifest = json.loads(path.read_text())
    update(manifest)
    raw = (json.dumps(manifest) + '\n').encode()
    path.write_bytes(raw)
    ref['sha256'] = digest(raw)


def test_complete_config_ownership():
    config = compile_adf(sm90_adf())
    keys = {r['native_key'] for r in config.component_map}
    for line in config.text.splitlines():
        if line.startswith('-'):
            assert line.split()[0][1:] in keys
    for line in config.interconnect_text.splitlines():
        if not line.startswith('//'):
            assert 'config_hopper_islip.icnt:' + line.split()[0] in keys
    assert all(r['source'] and r['unit'] and r['native_object'] and r['instances'] > 0 for r in config.component_map)
    assert compile_adf(from_dict(sm90_adf().to_dict())).artifacts == config.artifacts


@pytest.mark.parametrize('change,match', [
    (lambda a: setattr(a, 'repeat', 2), 'root repeat'),
    (lambda a: setattr(a.child('cluster').child('sm'), 'repeat', 2), 'point repeat'),
    (lambda a: setattr(a.child('partition').child('l2'), 'repeat', 1), 'subpartitions'),
    (lambda a: setattr(a.child('partition'), 'repeat', 39), 'partition repeat'),
    (lambda a: setattr(a.child('cluster'), 'repeat', 128), 'BookSim k'),
    (lambda a: a.elements.append(SpacePoint('extra', 'accelsim_sm')), 'requires parameter|two groups'),
    (lambda a: set_param(a, None, 'icnt', 'clock_mhz', 0), 'unsupported value'),
    (lambda a: set_param(a, None, 'icnt', 'booksim_topology', 'mesh'), 'unsupported value'),
    (lambda a: set_param(a, 'cluster', 'sm', 'gpgpu_shader_registers', 32768), 'registers_per_block'),
    (lambda a: a.child('cluster').child('sm').params.update(typo=Param(1)), 'unknown parameters'),
    (lambda a: a.child('cluster').child('sm').params.update(clock_mhz=Param(True, 'MHz', 'test')), 'expected int'),
])
def test_preflight_rejections(change, match):
    adf = sm90_adf()
    change(adf)
    with pytest.raises(ValueError, match=match):
        compile_adf(adf)


def test_count_coupling_is_consumed():
    adf = sm90_adf()
    adf.child('cluster').repeat = 128
    set_param(adf, None, 'icnt', 'booksim_k', 208)
    cfg = compile_adf(adf)
    assert '-gpgpu_n_clusters 128\n' in cfg.text
    assert 'k = 208;\n' in cfg.interconnect_text


@pytest.mark.parametrize('update,match', [
    (lambda m: m.update(isa='sm100'), 'ISA'),
    (lambda m: m.update(tracer_version=6), 'version'),
    (lambda m: m.update(extra=1), 'exactly fields'),
    (lambda m: m['source'].update(gpu='RTX PRO 5000 Blackwell'), 'source GPU'),
    (lambda m: m['mapping']['params'].update(grid=[2, 1, 1]), 'mapping identity'),
    (lambda m: m['workload']['params'].update(block=[True, 1, 1]), 'manifest.workload.params.block'),
    (lambda m: m['kernels'].append(deepcopy(m['kernels'][0])), 'one kernel'),
    (lambda m: m['kernels'][0].update(sha256='0' * 64), 'hash'),
    (lambda m: m['kernels'][0].update(file='kernel-1;touch BAD.traceg'), 'filename'),
    (lambda m: m['kernels'][0].update(thread_instructions=96), 'completion'),
    (lambda m: m['commands'].update(sha256='0' * 64), 'commands hash'),
])
def test_trace_rejections(trace_request, update, match):
    rewrite_manifest(trace_request, update)
    with pytest.raises(ValueError, match=match):
        load_trace(*specs(trace_request))


def test_trace_content_and_request_binding(trace_request):
    workload, mapping = specs(trace_request)
    assert load_trace(workload, mapping)[3]['thread_instructions'] == 128
    mapping.params['compiler'] = 'another-compiler'
    with pytest.raises(ValueError, match='mapping identity'):
        load_trace(workload, mapping)


def test_changed_shape_and_manifest_hash(trace_request):
    trace_request['workload']['params']['grid'] = [2, 1, 1]
    with pytest.raises(ValueError, match='workload/mapping'):
        load_trace(*specs(trace_request))
    trace_request['workload']['params']['accelsim']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='manifest SHA'):
        load_trace(*specs(trace_request))


def test_full_model_and_events_rejected(trace_request):
    with pytest.raises(ValueError, match='full ViT/CLIP'):
        load_trace(WorkloadSpec('vit', {'profile': 'anything'}), MappingSpec('accelsim_trace'))
    from arch_explorer.components.accelsim import AccelSimSm
    with pytest.raises(NotImplementedError, match="backend='accelsim'"):
        AccelSimSm().build(sm90_adf().child('cluster').child('sm'), None)


def test_missing_engine(trace_request):
    with pytest.raises(ImportError, match='optional native engine'):
        evaluate_request(trace_request)


EXPECTED = dict(name='example', thread_instructions=128, ctas=1)
# Unit-test transcript only; real integration tests below and delivery/ use
# independently executed native logs. The oracle exercises units, not a model.
TRANSCRIPT = '''launching kernel name: example uid: 1 cuda_stream_id: 0
kernel_name = example
kernel_launch_uid = 1
gpu_sim_cycle = 1980
gpu_tot_sim_cycle = 1980
gpu_sim_insn = 128
gpu_tot_sim_insn = 128
gpu_tot_issued_cta = 1
GPGPU-Sim uArch: clock periods: 0.00000000050505050505:0:0:0
GPGPU-Sim: *** simulation thread exiting ***
GPGPU-Sim: *** exit detected ***
'''


def test_independent_time_oracle():
    stats = parse_stats(TRANSCRIPT, '', EXPECTED, 1980)
    assert stats['latency_ns'] == float(Fraction(1980, 1_980_000_000) * 10**9) == 1000


@pytest.mark.parametrize('edit', [
    lambda s: s + 'gpu_sim_cycle = 1980\n',
    lambda s: s.replace('gpu_sim_insn = 128', 'gpu_sim_insn = 96'),
    lambda s: s.replace('gpu_tot_sim_cycle = 1980', 'gpu_tot_sim_cycle = 3960'),
    lambda s: s.replace('kernel_name = example', 'kernel_name = wrong'),
    lambda s: s.replace('kernel_launch_uid = 1', 'kernel_launch_uid = 2'),
    lambda s: s.replace('GPGPU-Sim: *** exit detected ***', ''),
    lambda s: s + 'GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **\n',
    lambda s: s + 'deadlock detected\n',
])
def test_incomplete_statistics_rejected(edit):
    with pytest.raises(ValueError):
        parse_stats(edit(TRANSCRIPT), '', EXPECTED, 1980)


ENGINE = os.environ.get('ARCH_EXPLORER_ACCELSIM_ENGINE')


@pytest.mark.skipif(not ENGINE, reason='optional pinned native engine not configured')
def test_real_native_public_api_and_replay(trace_request):
    trace_request['options']['engine'] = ENGINE
    first = evaluate_request(trace_request)
    again = evaluate_request(first)
    assert first['evaluation']['details']['native'] == again['evaluation']['details']['native']
    assert first['evaluation']['area_mm2'] is None and first['evaluation']['cost_usd'] is None
    # The default physical estimator must also leave native-only components unknown.
    result = evaluate(from_dict(trace_request['adf']), *specs(trace_request), backend='accelsim', **trace_request['options'])
    assert result.area_mm2 is None and result.cost_usd is None
    assert result.breakdown == [] and result.utilization == {}


@pytest.mark.skipif(not ENGINE, reason='optional pinned native engine not configured')
@pytest.mark.parametrize('options,match', [({'max_cycles': 1}, 'truncated'),
    ({'timeout_s': 0.001}, 'timeout'), ({'timeout_s': True}, 'timeout_s'),
    ({'memory_limit_mb': 2}, 'memory_limit_mb'), ({'max_cycles': False}, 'max_cycles')])
def test_real_native_failures(trace_request, options, match):
    trace_request['options'].update(engine=ENGINE, **options)
    with pytest.raises(ValueError, match=match):
        evaluate_request(trace_request)


@pytest.mark.skipif(not ENGINE, reason='optional pinned native engine not configured')
def test_engine_tamper(trace_request, tmp_path):
    manifest = json.loads(Path(ENGINE).read_text())
    manifest['binary']['sha256'] = '0' * 64
    other = tmp_path / 'engine-bad.json'
    other.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='binary SHA-256 mismatch'):
        load_engine(other)


@pytest.mark.skipif(not ENGINE, reason='optional pinned native engine not configured')
def test_nonzero_native_exit_is_retained(trace_request, tmp_path):
    # A real nonzero process verifies lifecycle failure handling. It cannot
    # yield a successful result because native completion/build checks follow.
    manifest = json.loads(Path(ENGINE).read_text())
    binary = Path('/bin/false').resolve()
    manifest['binary'] = {'path': str(binary), 'sha256': digest(binary.read_bytes())}
    other = tmp_path / 'nonzero-engine.json'
    other.write_text(json.dumps(manifest))
    trace_request['options']['engine'] = str(other)
    with pytest.raises(ValueError, match='native_failed, exit=1'):
        evaluate_request(trace_request)
    run = json.loads(next((tmp_path / 'runs').glob('*/run.json')).read_text())
    assert run['exit_code'] == 1 and run['status'] == 'native_failed'
    assert run['adapter_sources_sha256']


def test_printed_native_option_help_is_not_a_failure():
    help_text = '-gpgpu_deadlock_detect 1 # Stop simulation on deadlock\n-gpgpu_max_cycle 1000000 # maximum cycles\n'
    assert parse_stats(help_text + TRANSCRIPT, '', EXPECTED, 1980)['latency_ns'] == 1000
