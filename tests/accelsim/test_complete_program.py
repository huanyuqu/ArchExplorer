"""Complete public program contracts; these CPU checks never launch native."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
from arch_explorer.simulator.accelsim import program
from arch_explorer.simulator.accelsim.program_stats import parse_program_stats
from arch_explorer.workload.reference_execution import execute_reference


@pytest.mark.parametrize('module', (
    'components.gpu_local', 'components.gpu_kernel_service', 'devices.gpus',
    'devices.gpu_service_presets', 'mapping.kernel_point', 'mapping.service_bound',
    'mapping.warp_copy', 'workload.gpu_kernels', 'workload.gpu_validation',
))
def test_removed_gpu_imports_are_unavailable(module):
    with pytest.raises(ModuleNotFoundError, match='arch_explorer'):
        importlib.import_module('arch_explorer.' + module)


def load(path):
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    requests = {}
    root = tmp_path_factory.mktemp('complete-program')
    for model in ('vit', 'clip'):
        specs = program.demo_specs(model)
        for mapping in specs['mappings']:
            threads = mapping['params']['threads_per_block']
            requests[model, threads] = program.prepare_request(
                from_dict(specs['adf']), WorkloadSpec(**specs['workload']), MappingSpec(**mapping),
                output_dir=root / (model + str(threads)))
    return requests


@pytest.mark.parametrize('model', ('vit', 'clip'))
def test_mapping_changes_real_geometry_without_changing_complete_outputs(prepared, model):
    a, b = [load(Path(prepared[model, n]['program']['path'])) for n in (32, 64)]
    assert a['reference']['complete'] and b['reference']['complete']
    assert a['reference']['missing_operators'] == b['reference']['missing_operators'] == []
    assert a['outputs'] == b['outputs']
    assert a['reference']['input_tensors'] == b['reference']['input_tensors']
    assert [(v['name'], v['sha256']) for v in a['buffers'] if v['kind'] != 'activation'] == [
        (v['name'], v['sha256']) for v in b['buffers'] if v['kind'] != 'activation']
    assert [v['operator'] for v in a['kernels']] == [v['operator'] for v in b['kernels']]
    assert any(x['facts']['ctas'] != y['facts']['ctas'] for x, y in zip(a['kernels'], b['kernels']))
    assert prepared[model, 32]['program']['sha256'] != prepared[model, 64]['program']['sha256']
    for doc in (a, b):
        assert set(doc['reference']['outputs']) == set(doc['outputs'])
        assert all(v['initialized_after_program'] for v in doc['buffers'])
        assert all(v['op'] == 'view' or v.get('kernel') for v in doc['operators'])


@pytest.mark.parametrize('field', ('mapping', 'shape', 'adf', 'unknown-options'))
def test_public_spec_mismatch_is_rejected_before_native(prepared, monkeypatch, field):
    request = deepcopy(prepared['vit', 32])
    if field == 'mapping':
        request['mapping']['params']['threads_per_block'] = 64
    elif field == 'shape':
        request['workload']['params']['batch'] = 2
    elif field == 'adf':
        request['adf']['name'] = 'different public ADF identity'
    else:
        request['options']['hidden_shape'] = 7
    calls = []
    monkeypatch.setattr(program, 'run_native', lambda *a, **kw: calls.append(True))
    assert program.assess_request(request)['status'] == 'unsupported'
    with pytest.raises(ValueError):
        program.evaluate_request(request)
    assert not calls


@pytest.mark.parametrize('change', ('missing-kernel', 'missing-input', 'unknown-isa', 'incomplete-graph'))
def test_changed_artifacts_cannot_complete_even_after_manifest_rehash(prepared, tmp_path, monkeypatch, change):
    request = deepcopy(prepared['vit', 32])
    original = Path(request['program']['path'])
    shutil.copytree(original.parent / 'program', tmp_path / 'program')
    doc = load(original)
    if change == 'missing-kernel':
        (tmp_path / 'program' / doc['kernels'][0]['file']).unlink()
    elif change == 'missing-input':
        item = next(v for v in doc['buffers'] if 'file' in v)
        (tmp_path / 'program' / item['file']).unlink()
    elif change == 'unknown-isa':
        item = doc['kernels'][0]
        path = tmp_path / 'program' / item['file']
        raw = path.read_bytes().replace(b'EXIT', b'UNKNOWN_OPCODE')
        assert raw != path.read_bytes()
        path.write_bytes(raw)
        item['sha256'] = program.sha(raw)
    else:
        doc['operators'].pop()
    manifest = tmp_path / 'PROGRAM.json'
    manifest.write_bytes(program.canonical(doc))
    request['program'] = {'path': str(manifest), 'sha256': program.sha(manifest.read_bytes())}
    calls = []
    monkeypatch.setattr(program, 'run_native', lambda *a, **kw: calls.append(True))
    assert program.assess_request(request)['status'] == 'unsupported'
    with pytest.raises(ValueError):
        program.evaluate_request(request)
    assert not calls


def test_absent_engine_is_explicitly_unsupported(prepared):
    request = prepared['vit', 32]
    assessment = program.assess_request(request)
    assert assessment['status'] == 'unsupported'
    assert any(v['kind'] == 'engine' for v in assessment['gaps'])
    with pytest.raises(ImportError, match='optional native engine'):
        program.evaluate_request(request)
    assert not Path(request['options']['output_dir']).exists()


def test_supported_numpy_version_change_alone_preserves_program_validation(tmp_path, monkeypatch):
    """Producer/consumer dependency versions may differ within declared support."""
    specs = program.demo_specs('vit')
    with monkeypatch.context() as producer:
        # Change only recorded provenance; arithmetic, specs and source remain
        # identical. This isolates the independently reproduced 2.5.4 -> 2.2.6
        # replay failure without requiring a second NumPy in the test process.
        producer.setattr(np, '__version__', '2.5.4' if np.__version__ != '2.5.4' else '2.2.6')
        request = program.prepare_request(from_dict(specs['adf']), WorkloadSpec(**specs['workload']),
            MappingSpec(**specs['mappings'][0]), output_dir=tmp_path / 'producer')
    assessment = program.assess_request(request)
    # An absent engine is expected here; a changed dependency version label
    # alone must not masquerade as changed program semantics.
    assert assessment['gaps'] and all(gap['kind'] == 'engine' for gap in assessment['gaps'])


def test_new_clip_shape_preserves_dual_tower_similarity_and_first_eot():
    workload = WorkloadSpec('clip', {'profile': 'synthetic', 'dtype': 'fp32',
        'image_batch': 2, 'text_batch': 3, 'config': {'layers': 1, 'text_layers': 1, 'vocab': 1}})
    result = execute_reference(workload)
    tensors, outputs = result['tensors'], result['outputs']
    np.testing.assert_array_equal(tensors['token_ids'], 0)
    np.testing.assert_array_equal(tensors['text.eot_index'], np.zeros(3, dtype=np.int32))
    image, text = tensors['image_features'], tensors['text_features']
    np.testing.assert_allclose(np.linalg.norm(image, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(text, axis=1), 1, atol=1e-6)
    manual = np.exp(tensors['logit_scale']) * image @ text.T
    assert outputs['logits_per_image'].shape == (2, 3)
    assert outputs['logits_per_text'].shape == (3, 2)
    np.testing.assert_allclose(outputs['logits_per_image'], manual, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(outputs['logits_per_text'], outputs['logits_per_image'].T)
    mask = tensors['text.causal']
    assert np.all(np.isneginf(mask[np.triu_indices(mask.shape[0], 1)]))
    assert np.all(mask[np.tril_indices(mask.shape[0])] == 0)


def counter_fixture():
    expected = [{'name': 'one', 'thread_instructions': 64, 'ctas': 1, 'warp_instructions': 2},
                {'name': 'two', 'thread_instructions': 96, 'ctas': 2, 'warp_instructions': 3}]
    lines = ['GPGPU-Sim uArch: clock periods: 1e-09:']
    cycles, insns, ctas, warps = 0, 0, 0, 0
    for i, (e, duration) in enumerate(zip(expected, (11, 19)), 1):
        cycles += duration; insns += e['thread_instructions']; ctas += e['ctas']; warps += e['warp_instructions']
        lines.extend([f"launching kernel name: {e['name']} uid: {i} cuda_stream_id: 0",
            f"kernel_name = {e['name']}", f'kernel_launch_uid = {i}', f'gpu_sim_cycle = {duration}',
            f'gpu_tot_sim_cycle = {cycles}', f"gpu_sim_insn = {e['thread_instructions']}",
            f'gpu_tot_sim_insn = {insns}', f'gpu_tot_issued_cta = {ctas}', f'gpgpu_n_tot_w_icount = {warps}'])
    lines.extend(['GPGPU-Sim: *** exit detected ***', 'GPGPU-Sim: *** simulation thread exiting ***'])
    return '\n'.join(lines), expected


def test_program_latency_uses_final_cumulative_counter_once():
    raw, expected = counter_fixture()
    stats = parse_program_stats(raw, '', expected, 1000)
    assert stats['latency_ns'] == 30
    assert stats['gpu_tot_sim_cycle'] == 30
    assert stats['gpu_tot_issued_cta'] == 3


@pytest.mark.parametrize('change', ('limit', 'missing-exit', 'stream', 'retired', 'cumulative', 'clock'))
def test_incomplete_or_inconsistent_native_logs_are_rejected(change):
    raw, expected = counter_fixture()
    replacements = {'missing-exit': ('GPGPU-Sim: *** exit detected ***', ''),
        'stream': ('cuda_stream_id: 0', 'cuda_stream_id: 1'),
        'retired': ('gpu_sim_insn = 96', 'gpu_sim_insn = 95'),
        'cumulative': ('gpu_tot_sim_cycle = 30', 'gpu_tot_sim_cycle = 41'),
        'clock': ('clock periods: 1e-09:', 'clock periods: 2e-09:')}
    if change == 'limit':
        raw += '\nGPGPU-Sim break due to reaching cycle limit\n'
    else:
        raw = raw.replace(*replacements[change])
    with pytest.raises(ValueError):
        parse_program_stats(raw, '', expected, 1000)
