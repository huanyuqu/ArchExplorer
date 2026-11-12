"""Real request eligibility, execution and replay at the sampling boundary."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from arch_explorer.adf import MAPPING_TEMPLATES, MappingSpec, WorkloadSpec, from_dict
from arch_explorer.adf.registry import Registry, MappingTemplate
from arch_explorer.agent_env import ExplorationEnv, replay_trace
from arch_explorer.agent_env.environment import load_registrations
from arch_explorer.agent_env.errors import EnvError
from arch_explorer.agent_env.sampling import mapping_candidates, check_mapping
from non_gpu_fixture import shared_adf
from non_gpu_fixture import shared_configuration
from arch_explorer.mapping.multi_device import MultiDevice
from tools.generate_multi_device_execution import configuration

ROOT = Path(__file__).resolve().parents[2]
MODES = ('none', 'mapping', 'parameters_and_mapping')


@pytest.fixture(autouse=True)
def fake_registry():
    # Override the directory's fake fixture; no executor or registry is replaced.
    load_registrations({})
    assert 'multi_device' in MAPPING_TEMPLATES.names()


def ok(response):
    assert 'error' not in response, response
    return response['result']


def sample(adf, workload, **kwargs):
    return mapping_candidates(adf, workload, MAPPING_TEMPLATES, 8, random.Random(17), **kwargs)


def fixture():
    cfg = configuration('operator', tp=True)
    return from_dict(cfg['seed_adf']), WorkloadSpec(**cfg['workloads'][0]), cfg['mapping_selectors']['multi_device']


def test_implicit_registration_skips_missing_request_but_keeps_real_templates(monkeypatch):
    calls = []
    original = MultiDevice.request_space

    def observed(self, adf, workload, params):
        calls.append(deepcopy(params))
        return original(self, adf, workload, params)

    monkeypatch.setattr(MultiDevice, 'request_space', observed)
    adf, workload, params = fixture()
    candidates = sample(shared_adf(1), workload)
    assert candidates and all(c.template != 'multi_device' for c in candidates)
    assert calls == []
    for candidate in candidates:
        check_mapping(shared_adf(1), workload, candidate, MAPPING_TEMPLATES)
    # Supplying a request without a names filter still makes it eligible.
    candidates = sample(adf, workload, selectors={'multi_device': params})
    assert calls == [params]
    assert 'multi_device' in MAPPING_TEMPLATES.names()


@pytest.mark.parametrize('names', [['multi_device'], ['gsm_tiled', 'multi_device']])
def test_explicit_selection_without_request_does_not_fall_back(names):
    _, workload, _ = fixture()
    with pytest.raises(EnvError, match='explicit structured request') as error:
        sample(shared_adf(1), workload, names=names)
    assert error.value.code == 'invalid_mapping_space'


def test_no_executable_template_is_not_empty_success():
    registry = Registry('mapping template', MappingTemplate)
    registry.register(MultiDevice)  # Separate caller-owned catalog of the real template.
    _, workload, _ = fixture()
    with pytest.raises(EnvError) as error:
        mapping_candidates(shared_adf(1), workload, registry, 1, random.Random(17))
    assert error.value.code == 'no_mapping'
    assert 'gsm_tiled' in MAPPING_TEMPLATES.names()  # Global registration is untouched.


@pytest.mark.parametrize('names', [None, ['multi_device']])
def test_complete_request_is_one_correlated_candidate(names):
    adf, workload, params = fixture()
    # Requesting a single template guarantees inclusion, without splitting any fields.
    candidates = sample(adf, workload, names=names, selectors={'multi_device': params})
    if names:
        assert [m.to_dict() for m in candidates] == [MappingSpec('multi_device', params).to_dict()]
    for candidate in candidates:
        if candidate.template == 'multi_device':
            assert candidate.params == params
            check_mapping(adf, workload, candidate, MAPPING_TEMPLATES)


@pytest.mark.parametrize('params', [None, {}, {'typo': 1}])
@pytest.mark.parametrize('names', [None, ['gsm_tiled', 'multi_device']])
def test_explicit_bad_request_is_never_treated_as_absent(params, names):
    adf, workload, _ = fixture()
    with pytest.raises(EnvError) as error:
        sample(adf, workload, names=names, selectors={'multi_device': params})
    assert error.value.code == 'invalid_mapping_space'
    with pytest.raises((EnvError, ValueError)):
        check_mapping(adf, workload, MappingSpec('multi_device', params), MAPPING_TEMPLATES)


@pytest.mark.parametrize('change', [
    lambda p: p['shards'][0].update(size=6, ranges=[{'rank': 0, 'start': 0, 'stop': 3},
                                                {'rank': 1, 'start': 3, 'stop': 6}]),
    lambda p: p.update(topology_edges=[]),
    lambda p: p['collectives'][0].update(algorithm='not_implemented'),
    lambda p: p['placements']['matmul']['mapping']['params'].update(tile_m=3),
])
def test_sampling_preserves_shape_route_algorithm_and_local_mapping_validation(change):
    adf, workload, params = fixture()
    change(params)
    with pytest.raises((EnvError, ValueError, NotImplementedError)):
        sample(adf, workload, names=['multi_device'], selectors={'multi_device': params})
    with pytest.raises((EnvError, ValueError, NotImplementedError)):
        check_mapping(adf, workload, MappingSpec('multi_device', params), MAPPING_TEMPLATES)
    with pytest.raises((ValueError, NotImplementedError)):
        MultiDevice().request_space(adf, workload, params)


def replay_winner(path, tmp_path):
    output = tmp_path / 'cli-replayed.json'
    run = subprocess.run([sys.executable, '-m', 'arch_explorer', 'evaluate', str(path),
                          '--output', str(output)], capture_output=True, text=True, cwd=ROOT, timeout=60)
    (tmp_path / 'cli-stdout.txt').write_text(run.stdout)
    (tmp_path / 'cli-stderr.txt').write_text(run.stderr)
    assert run.returncode == 0, run.stderr
    assert json.loads(path.read_text())['best'] == json.loads(output.read_text())['best']


@pytest.mark.parametrize('preset', ['shared', 'dmc'])
@pytest.mark.parametrize('mode', MODES)
def test_public_defaults_execute_all_modes_and_replay_entire_winner(preset, mode, tmp_path, monkeypatch):
    cfg = (shared_configuration() if preset == 'shared' else
           json.loads((ROOT / 'examples/adf-dmc.json').read_text()))
    if preset == 'shared':
        cfg.update(mapping_samples=4, parameter_samples=2)
    cfg['trace_path'] = str(tmp_path / 'tools.jsonl')
    env = ExplorationEnv(cfg)
    assert 'multi_device' not in ok(env.describe_environment())['mapping_spaces'][0]
    assert env.used == 0 and not env.attempts
    from arch_explorer.agent_env import environment
    discover, check = environment.mapping_candidates, environment.check_mapping
    reservations, no_mapping = [], []

    def observed_discovery(*args, **kwargs):
        charged = env.used
        try:
            return discover(*args, **kwargs)
        except EnvError as error:
            if error.code == 'no_mapping':
                assert env.used == charged  # No concrete attempt was reserved.
                no_mapping.append(error.to_dict())
            raise

    def observed_check(adf, workload, mapping, registry):
        # Observe the validation boundary, not just the final counter: every
        # concrete attempt must already be charged, even if validation fails.
        assert env.used == len(reservations) + 1
        reservations.append(mapping.to_dict())
        return check(adf, workload, mapping, registry)

    with monkeypatch.context() as observed:
        observed.setattr(environment, 'mapping_candidates', observed_discovery)
        observed.setattr(environment, 'check_mapping', observed_check)
        result = ok(env.evaluate(optimize=mode))
    if mode != 'parameters_and_mapping':
        assert env.used == {'none': 1, 'mapping': 4}[mode]
    else:
        assert 0 < env.used <= min(cfg['max_evaluations'],
                                  cfg['parameter_samples'] * cfg['mapping_samples'])
        assert any('score' in trial for trial in result['trials'])
        assert no_mapping == [trial['cause'] for trial in result['trials']
                              if trial.get('cause', {}).get('code') == 'no_mapping']
        assert len(result['trials']) <= cfg['parameter_samples']
    assert len(env.attempts) == env.used
    assert [a['index'] for a in env.attempts] == list(range(env.used))
    assert [a['mapping'] for a in env.attempts] == reservations
    assert result['evaluations_used'] == env.used
    assert result['evaluations_remaining'] == env.remaining == cfg['max_evaluations'] - env.used
    assert all(('evaluation' in a) if a['valid'] else ('error' in a) for a in env.attempts)
    assert all(a['mapping']['template'] != 'multi_device' for a in env.attempts)
    winner = tmp_path / 'winner.json'
    ok(env.submit(path=str(winner)))
    replay_winner(winner, tmp_path)
    restored = replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines())
    assert restored.get_state() == env.get_state()


@pytest.mark.parametrize('failure', ['no_mapping', 'parameters', 'backend'])
def test_reserved_validation_and_backend_failures_are_charged(failure, monkeypatch):
    from arch_explorer.agent_env import environment
    cfg = shared_configuration()
    if failure == 'no_mapping':
        # This registered template cannot map the GSM fixture. No catalog or
        # executor is replaced to manufacture the discovery failure.
        cfg['mapping_templates'] = ['pim_offload']
    if failure == 'backend':
        # A real backend rejection, after a valid mapping passed validation.
        cfg['backend_options'] = {'scheduler': 'invalid-for-accounting-test'}
    env = ExplorationEnv(cfg)
    params = dict(tile_m=64, tile_n=64, rows_per_cta=4)
    if failure == 'parameters':
        params['tile_m'] = 3  # Rejected by the real mapping validator.
    calls = []
    backend = environment.backend_evaluate

    def observed_backend(*args, **kwargs):
        assert env.used == 1 and not env.attempts
        calls.append(True)
        return backend(*args, **kwargs)

    monkeypatch.setattr(environment, 'backend_evaluate', observed_backend)
    response = (env.evaluate() if failure == 'no_mapping' else
                env.evaluate(mappings=[MappingSpec('gsm_tiled', params).to_dict()]))
    assert response['error']['code'] == 'evaluation_failed'
    if failure == 'no_mapping':
        trials = response['error']['details']['trials']
        assert trials and all(t['cause']['code'] == 'no_mapping' for t in trials)
        assert env.used == 0 and not env.attempts and not calls
        assert env.remaining == cfg['max_evaluations']
        assert env.best_candidate is None and 'error' in env.submit()
        return
    assert env.used == len(env.attempts) == 1
    assert env.remaining == cfg['max_evaluations'] - 1
    assert env.attempts[0]['index'] == 0 and env.attempts[0]['valid'] is False
    assert env.attempts[0]['error']
    assert len(calls) == (1 if failure == 'backend' else 0)
    assert env.best_candidate is None
    assert 'error' in env.submit()


@pytest.mark.parametrize('mode', MODES)
def test_each_workload_draft_survives_all_optimization_modes(mode, tmp_path):
    cfg = configuration('operator', tp=True)
    cfg.update(parameter_samples=2, max_evaluations=8, trace_path=str(tmp_path / 'tools.jsonl'))
    cfg['workloads'].append(deepcopy(cfg['workloads'][0]))
    cfg['workloads'][1]['params']['k'] = 7
    first = deepcopy(cfg['mapping_selectors']['multi_device'])
    first['placements']['matmul']['mapping']['params']['tile_n'] = 128
    second = deepcopy(first)
    for shard in second['shards']:
        shard.update(size=7, ranges=[{'rank': 0, 'start': 0, 'stop': 3},
                                     {'rank': 1, 'start': 3, 'stop': 7}])
    mappings = [MappingSpec('multi_device', p).to_dict() for p in (first, second)]
    env = ExplorationEnv(cfg)
    ok(env.set_mappings(mappings=mappings))
    ok(env.checkpoint(name='drafts'))
    ok(env.describe_environment())
    assert env.used == 0
    result = ok(env.evaluate(optimize=mode))
    assert result['mappings'] == mappings
    assert env.used == (4 if mode == 'parameters_and_mapping' else 2)
    assert all(a['mapping'] == mappings[a['workload']] for a in env.attempts)
    assert env.config['mapping_selectors'] == cfg['mapping_selectors']
    charged = env.used
    ok(env.restore(name='drafts'))
    assert env.used == charged and ok(env.get_state())['mappings'] == mappings
    ok(env.describe_environment())
    winner = tmp_path / 'winner.json'
    ok(env.submit(path=str(winner)))
    replay_winner(winner, tmp_path)
    restored = replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines())
    assert restored.get_state() == env.get_state()


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('present', [False, True])
def test_explicit_missing_or_empty_structure_rejects_before_budget(mode, present):
    cfg = configuration('operator')
    cfg['mapping_selectors'] = {'multi_device': {}} if present else {}
    env = ExplorationEnv(cfg)
    assert 'error' in env.evaluate(optimize=mode)
    assert env.used == 0 and not env.attempts and env.best_candidate is None
    assert 'error' in env.submit()


def test_none_selector_rejected_by_config_preflight():
    cfg = configuration('operator')
    cfg['mapping_selectors']['multi_device'] = None
    with pytest.raises(EnvError, match='mapping_selectors'):
        ExplorationEnv(cfg)
