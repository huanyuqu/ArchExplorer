"""Discovery follows real per-workload requests; no executor is substituted."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from arch_explorer.adf import MAPPING_TEMPLATES
from arch_explorer.agent_env import ExplorationEnv, replay_trace
from arch_explorer.agent_env.environment import load_registrations
from non_gpu_fixture import shared_adf
from tools.generate_multi_device import configuration as entry_configuration
from tools.generate_multi_device_execution import configuration
from tools.generate_pi05 import configuration as pi05_configuration


@pytest.fixture(autouse=True)
def fake_registry():
    # Override the directory fixture with actual production registrations.
    load_registrations({})
    assert 'multi_device' in MAPPING_TEMPLATES.names()


def ok(response):
    assert 'error' not in response, response
    return response['result']


def draft(params, template='multi_device'):
    return {'template': template, 'params': deepcopy(params)}


def assert_request(description, index, params):
    space = description['mapping_spaces'][index]['multi_device']
    expected = {key: [value] for key, value in params.items()}
    assert space['contract']['branches'] == [expected]
    assert space['sampling_spaces'] == [expected]


def assert_unmetered(env):
    assert env.used == 0 and env.attempts == [] and env.best_candidate is None


def test_entry_draft_without_config_selection_evaluates_and_replays(tmp_path):
    workload = configuration('operator')['workloads'][0]
    cfg = entry_configuration(workload=workload)
    params = cfg.pop('mapping_selectors')['multi_device']
    cfg.pop('mapping_templates')
    cfg['trace_path'] = str(tmp_path / 'tools.jsonl')
    env = ExplorationEnv(cfg)
    normalized_config = deepcopy(env.config)
    ok(env.set_mappings(mappings=[draft(params)]))
    description = ok(env.describe_environment())
    assert_request(description, 0, params)
    assert ok(env.get_state())['mappings'] == [draft(params)]
    assert env.config == normalized_config
    assert_unmetered(env)
    result = ok(env.evaluate())
    assert result['mappings'] == [draft(params)]
    assert result['workloads'][0]['evaluation']['latency_ns'] > 0
    assert env.used == 1 and env.attempts[0]['valid']
    winner = tmp_path / 'winner.json'
    ok(env.submit(path=str(winner)))
    restored = replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines())
    assert restored.get_state() == env.get_state()
    replayed = tmp_path / 'evaluated.json'
    run = subprocess.run([sys.executable, '-m', 'arch_explorer', 'evaluate', str(winner),
                          '--output', str(replayed)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert json.loads(winner.read_text())['best'] == json.loads(replayed.read_text())['best']


def test_different_workload_shapes_use_their_own_drafts_for_both_spaces():
    cfg = configuration('operator', tp=True)
    initial = deepcopy(cfg['mapping_selectors']['multi_device'])
    cfg['workloads'].append(deepcopy(cfg['workloads'][0]))
    cfg['workloads'][1]['params']['k'] = 7
    first, second = deepcopy(initial), deepcopy(initial)
    first['placements']['matmul']['mapping']['params']['tile_n'] = 128
    for shard in second['shards']:
        shard.update(size=7, ranges=[{'rank': 0, 'start': 0, 'stop': 3},
                                     {'rank': 1, 'start': 3, 'stop': 7}])
    env = ExplorationEnv(cfg)
    mappings = [draft(first), draft(second)]
    ok(env.set_mappings(mappings=mappings))
    description = ok(env.describe_environment())
    assert_request(description, 0, first)
    assert_request(description, 1, second)
    assert ok(env.get_state())['mappings'] == mappings
    assert env.config['mapping_selectors']['multi_device'] == initial
    assert_unmetered(env)
    result = ok(env.evaluate(optimize='mapping'))
    assert result['mappings'] == mappings
    assert env.used == 2 and all(attempt['valid'] for attempt in env.attempts)


def test_restore_undo_and_topology_edit_keep_discovery_on_current_draft(tmp_path):
    cfg = configuration('operator', tp=True)
    cfg['trace_path'] = str(tmp_path / 'restore.jsonl')
    env = ExplorationEnv(cfg)
    normalized_config = deepcopy(env.config)
    initial = deepcopy(cfg['mapping_selectors']['multi_device'])
    assert_request(ok(env.describe_environment()), 0, initial)
    first = deepcopy(initial)
    first['placements']['matmul']['mapping']['params']['tile_n'] = 128
    ok(env.set_mappings(mappings=[draft(first)]))
    ok(env.checkpoint(name='draft'))
    second = deepcopy(first)
    second['placements']['matmul']['mapping']['params']['tile_m'] = 128
    ok(env.set_mappings(mappings=[draft(second)]))
    assert_request(ok(env.describe_environment()), 0, second)
    ok(env.undo())
    assert_request(ok(env.describe_environment()), 0, first)
    ok(env.set_parameters(parameters=[{'path': 'system/link0_1', 'param': 'bandwidth', 'value': 50}]))
    assert_request(ok(env.describe_environment()), 0, first)
    state = ok(env.apply_edits(edits=[{'op': 'delete', 'component': 'system/link0_1'}]))
    assert state['mappings'] == [draft(first)]
    assert not state['multi_device']['requests'][0]['valid_entry']
    assert 'error' in env.describe_environment()
    assert env.evaluate()['error']['code'] == 'invalid_mapping'
    ok(env.restore(name='draft'))
    assert_request(ok(env.describe_environment()), 0, first)
    assert ok(env.get_state())['mappings'] == [draft(first)]
    assert env.config['mapping_selectors']['multi_device'] == initial
    assert_unmetered(env)
    restored = replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines())
    assert restored.get_state() == env.get_state()


@pytest.mark.parametrize('selected_only', [False, True])
def test_no_explicit_multi_device_request_is_not_an_executable_space(selected_only):
    cfg = {'seed_adf': shared_adf(1).to_dict(),
           'workloads': configuration('operator')['workloads'], 'backend': 'events'}
    if selected_only:
        cfg['mapping_templates'] = ['multi_device']
    env = ExplorationEnv(cfg)
    normalized_config = deepcopy(env.config)
    description = ok(env.describe_environment())
    assert 'multi_device' in description['mapping_templates']
    assert 'multi_device' not in description['mapping_spaces'][0]
    if not selected_only:
        assert description['mapping_spaces'][0]['gsm_tiled']['sampling_spaces']
    assert_unmetered(env)


@pytest.mark.parametrize("selected_gsm", [False, True])
def test_flat_draft_overrides_its_selector_and_stays_discoverable(selected_gsm):
    cfg = {'seed_adf': shared_adf(1).to_dict(),
           'workloads': configuration('operator')['workloads'], 'backend': 'events',
           'mapping_templates': ['tensor_parallel'],
           'mapping_selectors': {'tensor_parallel': {'tp': 1}}}
    if selected_gsm:
        cfg['mapping_templates'].append('gsm_tiled')
        cfg['mapping_selectors']['gsm_tiled'] = {'tile_m': 64}
    env = ExplorationEnv(cfg)
    normalized_config = deepcopy(env.config)
    before = ok(env.describe_environment())['mapping_spaces'][0]['tensor_parallel']
    params = {'tile_m': 128, 'tile_n': 64, 'rows_per_cta': 4}
    ok(env.set_mappings(mappings=[draft(params, 'gsm_tiled')]))
    description = ok(env.describe_environment())
    assert description['mapping_spaces'][0]['gsm_tiled']['sampling_spaces'] == [
        {key: [value] for key, value in params.items()}]
    assert description['mapping_spaces'][0]['tensor_parallel'] == before
    assert env.config == normalized_config
    assert_unmetered(env)


@pytest.mark.parametrize('change, message', [
    (lambda p: p['shards'][0].update(size=6, ranges=[{'rank': 0, 'start': 0, 'stop': 3},
                                                  {'rank': 1, 'start': 3, 'stop': 6}]), 'shape'),
    (lambda p: p.update(topology_edges=[]), 'route'),
    (lambda p: p['placements']['matmul']['mapping']['params'].update(tile_m=3), 'tile_m'),
    (lambda p: p.update(routing='unimplemented'), 'routing'),
])
def test_invalid_explicit_physical_draft_never_falls_back(change, message):
    cfg = configuration('operator', tp=True)
    env = ExplorationEnv(cfg)
    normalized_config = deepcopy(env.config)
    assert_request(ok(env.describe_environment()), 0, cfg['mapping_selectors']['multi_device'])
    params = deepcopy(cfg['mapping_selectors']['multi_device'])
    change(params)
    ok(env.set_mappings(mappings=[draft(params)]))  # entry schema valid, physical plan invalid
    error = env.describe_environment()['error']
    assert message in error['message'], error
    assert ok(env.get_state())['mappings'] == [draft(params)]
    assert env.config == normalized_config
    assert_unmetered(env)


def test_explicit_empty_selector_and_malformed_draft_are_not_absence():
    cfg = configuration('operator')
    cfg['mapping_selectors']['multi_device'] = {}
    env = ExplorationEnv(cfg)
    assert 'expected exactly' in env.describe_environment()['error']['message']
    assert env.set_mappings(mappings=[draft({})])['error']['code'] == 'invalid_mapping'
    assert_unmetered(env)


def test_pi05_support_domain_preserves_other_draft_and_partial_suite_budget():
    cfg = configuration('operator')
    cfg['workloads'].extend(pi05_configuration()['workloads'])
    env = ExplorationEnv(cfg)
    params = cfg['mapping_selectors']['multi_device']
    mappings = [draft(params), draft({'tile_m': 64, 'tile_n': 64, 'rows_per_cta': 1}, 'gsm_tiled')]
    before = ok(env.get_state())
    assert env.set_mappings(mappings=mappings)['error']['code'] == 'invalid_mapping'
    assert ok(env.get_state()) == before
    mappings = [draft(params), draft(params)]
    ok(env.set_mappings(mappings=mappings))
    description = ok(env.describe_environment())
    assert_request(description, 0, params)
    assert description['mapping_spaces'][1] == {}
    assert description['model_specs'][0]['executor_registered']
    ok(env.checkpoint(name='unsupported-domain'))
    ok(env.restore(name='unsupported-domain'))
    assert ok(env.get_state())['mappings'] == mappings
    error = env.evaluate()['error']
    assert error['code'] == 'evaluation_failed'
    assert error['details']['trials'][0]['cause']['code'] == 'evaluation_failed'
    # Explicit drafts reach evaluation: one success and one charged applicability failure.
    assert env.used == 2 and len(env.attempts) == 2
    assert [a['workload'] for a in env.attempts] == [0, 1]
    assert 'error' not in env.attempts[0]
    assert env.attempts[1]['error'] == 'multi_device does not apply'
    assert env.best_candidate is None
    assert 'error' in env.submit()
    ok(env.restore(name='unsupported-domain'))
    assert env.used == 2 and len(env.attempts) == 2
