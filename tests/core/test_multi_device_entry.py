"""Registered multi-device entry, execution/rejection and real transport contracts."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from arch_explorer.adf import COMPONENTS, MAPPING_TEMPLATES, MappingSpec, from_dict
from arch_explorer.adf.multi_device import validate_request, request_space
from arch_explorer.adf.registry import mapping_spaces, validate_mapping_params
from arch_explorer.agent_env import ExplorationEnv, replay_trace
from arch_explorer.agent_env.environment import load_registrations
from arch_explorer.agent_env.sampling import parameter_variant
from arch_explorer.archs.adf_seeds import dmc_adf
from non_gpu_fixture import shared_adf
from arch_explorer.devices.topology import compose, discover, instance_inventory, link_resource
from arch_explorer.search.candidates import proposals
from tools.generate_multi_device import configuration
from tools.export_source import export

ROOT = Path(__file__).resolve().parents[2]


def test_generator_is_exported_and_reproduces_explicit_configurations(tmp_path):
    exported = tmp_path / 'export'
    manifest = export(exported, root=ROOT)
    assert 'tools/generate_multi_device.py' in manifest
    assert not (exported / 'examples/multi_device/generate.py').exists()
    expected = {
        'two-device.json': configuration(),
        'four-device.json': configuration(('alpha', 'beta', 'gamma', 'delta'),
            {'kind': 'clip', 'params': {'profile': 'clip_vit_b32', 'dtype': 'fp16',
                'image_batch': 1, 'text_batch': 1, 'iterations': 1}}),
    }
    for name, config in expected.items():
        assert not (exported / 'examples/multi_device' / name).exists()
    generated = tmp_path / 'generated'
    process = subprocess.run([sys.executable, str(exported / 'tools/generate_multi_device.py'), str(generated)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    for name, config in expected.items():
        assert json.loads((generated / name).read_text()) == config


@pytest.fixture(autouse=True)
def fake_registry():
    # Shadow the directory's fake fixture; keep every production registration.
    load_registrations({})
    assert 'tensor_parallel' in MAPPING_TEMPLATES.names()
    assert 'multi_device' in MAPPING_TEMPLATES.names()


def ok(response):
    assert 'error' not in response, response
    return response['result']


def small_configuration(names=('north', 'south')):
    # Wildcard placement is a complete replica per rank, not a TP speedup.
    return configuration(names, {'kind': 'operator', 'params': {
        'op': 'matmul', 'm': 4, 'n': 4, 'k': 5, 'dtype': 'fp16'}})


def make_env(tmp_path, names=('north', 'south')):
    cfg = small_configuration(names)
    cfg['trace_path'] = str(tmp_path / 'wire.jsonl')
    return ExplorationEnv(cfg)


@pytest.mark.parametrize('count', [1, 2, 3, 4, 5])
def test_arbitrary_counts_names_roles_and_roundtrip(count, tmp_path):
    env = make_env(tmp_path, tuple(f'unclassified_{i}' for i in range(count)))
    state = ok(env.get_state())
    assert from_dict(state['adf']).to_dict() == state['adf']
    assert len(state['mappings'][0]['params']['device_paths']) == count
    assert state['multi_device']['requests'] == [{'valid_entry': True}]
    assert state['multi_device']['executor_registered']
    discovery = ok(env.describe_environment())['mapping_spaces'][0]['multi_device']
    assert discovery['sampling_spaces']
    assert discovery['contract']['branches']
    result = ok(env.evaluate())
    assert env.used == 1 and len(env.attempts) == 1
    evaluation = env.best_candidate['workloads'][0]['evaluation']
    if count > 1:
        ledger = evaluation['details']['multi_device']
        assert sorted(op['rank'] for op in ledger['operators']) == list(range(count))
        assert ledger['messages'] == []  # Independent replicas need no fabricated exchange.
    assert 'multi_device' in MAPPING_TEMPLATES.names()


def test_heterogeneous_repeated_instances_are_not_family_dispatched():
    device = shared_adf(1)
    device.repeat = 3
    adf = compose('rack', {'arbitrary': device, 'other': dmc_adf(1)})
    inv = instance_inventory(adf)
    assert inv['rack/arbitrary[2]']['type_path'] == 'rack/arbitrary'
    assert inv['rack/other']['roles'] == ['compute', 'memory']
    assert device.name == 'chip' and device.repeat == 3
    params = configuration()['mapping_selectors']['multi_device']
    params.update(device_paths=['rack/other', 'rack/arbitrary[2]'], topology_edges=[])
    assert validate_request(adf, params)['device_paths'][0] == 'rack/other'
    with pytest.raises(ValueError, match='discovery exceeds'):
        instance_inventory(adf, max_instances=2)
    adf.repeat = 2
    with pytest.raises(ValueError, match='repeat=1'):
        instance_inventory(adf)


def test_sparse_directed_links_cost_inventory_and_parameter_sampling(tmp_path):
    env = make_env(tmp_path, ('alpha', 'beta', 'gamma', 'delta'))
    state = ok(env.get_state())
    params = state['mappings'][0]['params']
    assert len(params['topology_edges']) == 6  # sparse chain, not complete 12-edge graph
    assert not any(e['src'].endswith('alpha') and e['dst'].endswith('delta') for e in params['topology_edges'])
    networks = {r['path']: r for r in state['multi_device']['inventory']['networks']}
    assert networks['system/edge_0_1']['parameters']['bandwidth']['value'] == 25
    assert networks['system/edge_1_0']['parameters']['bandwidth']['value'] == 7
    costs = state['costs']
    rows = {row['path']: row for row in costs['by_path']}
    for name in ('alpha', 'beta', 'gamma', 'delta'):
        assert rows[f'system/{name}/core/array']['instances'] == 1
        assert rows[f'system/{name}/core/vector']['instances'] == 1
    for edge in params['topology_edges']:
        assert rows[edge['network']]['instances'] == 1
    # Existing independent cost model accounts for all inventory, preserving unknowns.
    assert costs['unsupported'] and costs['memory']['die_area_mm2'] is None
    two = ok(ExplorationEnv(configuration()).get_state())['costs']
    assert costs['by_category']['arithmetic'] == pytest.approx(2 * two['by_category']['arithmetic'])
    assert all(rows[e['network']]['area_mm2'] > 0 for e in params['topology_edges'])
    sampled = parameter_variant(env.adf, COMPONENTS, random.Random(7), .25, domains=env.parameter_domains)
    assert sampled.child('edge_0_1').params['bandwidth'].value in (7, 25, 50)
    assert sampled.child('edge_1_0').params['bandwidth'].value == 7  # fixed per component declaration
    catalog = ok(env.describe_environment())['components']
    edits = proposals(state, catalog)
    assert any(isinstance(e, dict) and e.get('op') == 'create' and e.get('spec', {}).get('name') == 'alpha_copy' for e in edits)


def test_unknown_cost_in_one_device_cannot_meet_cluster_budget():
    cfg = configuration()
    # GDDR interface area is modeled; its memory price is absent by default.
    hbm = cfg['seed_adf']['elements'][0]['elements'][-1]
    assert hbm['component'] == 'dram'
    hbm['params']['technology']['value'] = 'gddr'
    state = ok(ExplorationEnv(cfg).get_state())
    assert state['costs']['area_mm2'] is not None
    assert state['costs']['cost_usd'] is None
    assert state['latest_evaluation'] is None


def partition_request():
    params = configuration()['mapping_selectors']['multi_device']
    params['placements'] = {'projection': params['placements']['*']}
    params['shards'] = [{'tensor': 'activation', 'axis': 1, 'size': 5,
                         'ranges': [{'rank': 0, 'start': 0, 'stop': 3}, {'rank': 1, 'start': 3, 'stop': 5}]}]
    params['collectives'] = [{'after': 'projection', 'tensor': 'activation', 'operation': 'allreduce',
                              'algorithm': 'future_algorithm', 'ranks': [1, 0], 'dtype': 'fp16', 'elements': 5}]
    return params


def test_correlated_structured_choices_and_unequal_tail():
    adf = from_dict(configuration()['seed_adf'])
    params = partition_request()
    space = request_space(adf, params)
    branches = mapping_spaces(space, params)
    assert len(branches) == 1 and all(len(v) == 1 for v in branches[0].values())
    assert validate_mapping_params(space, params, complete=True) == params
    assert MappingSpec(**json.loads(json.dumps(MappingSpec('multi_device', params).to_dict()))).params == params
    changed = deepcopy(params)
    changed['device_paths'].reverse()
    assert validate_request(adf, changed)['device_paths'] != params['device_paths']
    # No ring/tree whitelist here: algorithm support belongs to the backend.
    assert validate_request(adf, params)['collectives'][0]['algorithm'] == 'future_algorithm'


@pytest.mark.parametrize('change', [
    lambda p: p.update(typo=1),
    lambda p: p.update(device_paths=['system/north', 'system/north']),
    lambda p: p['device_paths'].__setitem__(1, 'system/missing'),
    lambda p: p['placements']['projection'].update(ranks=[True]),
    lambda p: p['placements']['projection'].update(ranks=[2]),
    lambda p: p['shards'][0]['ranges'][1].update(start=2),
    lambda p: p['shards'][0]['ranges'][1].update(start=4),
    lambda p: p['shards'][0].update(size=6),
    lambda p: p['collectives'][0].update(elements='10B'),
    lambda p: p['topology_edges'][0].update(network='system/north/dram'),
    lambda p: p['topology_edges'][0].update(dst='system/nope'),
    lambda p: p['topology_edges'].append(deepcopy(p['topology_edges'][0])),
])
def test_malformed_is_not_unsupported(change, tmp_path):
    env = make_env(tmp_path)
    params = partition_request()
    change(params)
    result = env.evaluate(mappings=[{'template': 'multi_device', 'params': params}])
    assert result['error']['code'] == 'invalid_mapping', result
    assert env.used == 0 and env.best_candidate is None


def test_edit_draft_checkpoint_and_trace_replay(tmp_path):
    env = make_env(tmp_path)
    initial = ok(env.get_state())
    draft = initial['mappings']
    draft[0]['params']['device_paths'].reverse()
    ok(env.set_mappings(mappings=draft))
    ok(env.checkpoint(name='ready'))
    ok(env.set_parameters(parameters=[{'path': 'system/edge_0_1', 'param': 'bandwidth', 'value': 50}]))
    # Changes to topology do not discard the draft or infer replacement paths.
    state = ok(env.apply_edits(edits=[{'op': 'delete', 'component': 'system/edge_0_1'}]))
    assert state['mappings'] == draft
    assert not state['multi_device']['requests'][0]['valid_entry']
    assert env.evaluate()['error']['code'] == 'invalid_mapping'
    state = ok(env.restore(name='ready'))
    assert state['adf'] == initial['adf'] and state['mappings'] == draft
    ok(env.undo())
    assert ok(env.get_state())['mappings'] == draft
    ok(env.restore(name='ready'))
    ok(env.evaluate(optimize='mapping'))
    assert env.used == 1 and len(env.attempts) == 1
    winner = ok(env.submit(selection='best'))
    assert winner['best']['workloads'][0]['evaluation']['latency_ns'] > 0
    replayed = replay_trace([json.loads(line) for line in env.trace_path.read_text().splitlines()])
    assert ok(replayed.get_state()) == ok(env.get_state())


def test_cli_real_transport_and_replay(tmp_path):
    cfg = small_configuration()
    cfg['trace_path'] = str(tmp_path/'trace.jsonl')
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    calls = [{'id': i, 'tool': name} for i, name in enumerate([
        'describe_environment', 'checkpoint', 'evaluate', 'get_state'])]
    run = subprocess.run([sys.executable, '-m', 'arch_explorer', 'env', 'serve', '--config', str(path)],
                         input='\n'.join(map(json.dumps, calls))+'\n', cwd=tmp_path,
                         text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    rows = list(map(json.loads, run.stdout.splitlines()))
    assert 'error' not in rows[2], rows[2]
    assert rows[-1]['result']['evaluations_used'] == 1
    output = tmp_path/'state.json'
    run = subprocess.run([sys.executable, '-m', 'arch_explorer', 'env', 'replay', str(cfg['trace_path']),
                          '--output', str(output)], cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert json.loads(output.read_text()) == rows[-1]['result']
    cfg.pop('trace_path')
    path.write_text(json.dumps(cfg))
    run = subprocess.run([sys.executable, '-m', 'arch_explorer', 'search', '--config', str(path),
                          '--output', str(tmp_path/'winner.json')], cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    winner = json.loads((tmp_path/'winner.json').read_text())
    replay = tmp_path/'winner-replay.json'
    run = subprocess.run([sys.executable, '-m', 'arch_explorer', 'evaluate', str(tmp_path/'winner.json'),
                          '--output', str(replay)], cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert json.loads(replay.read_text())['best'] == winner['best']


def test_registered_draft_is_discovered_without_config_selection(tmp_path):
    cfg = small_configuration()
    params = cfg.pop('mapping_selectors')['multi_device']
    cfg.pop('mapping_templates')
    env = ExplorationEnv(cfg)
    ok(env.set_mappings(mappings=[{'template': 'multi_device', 'params': params}]))
    contract = ok(env.describe_environment())['mapping_spaces'][0]['multi_device']
    assert contract['sampling_spaces']
    ok(env.evaluate())
    assert env.used == 1


def test_entry_valid_future_algorithm_is_rejected_by_real_backend(tmp_path):
    cfg = small_configuration()
    cfg['workloads'] = [{'kind': 'operator', 'params': {'op': 'allreduce', 'bytes': 20, 'dtype': 'fp32'}}]
    params = cfg['mapping_selectors']['multi_device']
    params['collectives'] = [{'after': 'allreduce', 'tensor': 'out', 'operation': 'allreduce',
        'algorithm': 'future_algorithm', 'ranks': [0, 1], 'dtype': 'fp32', 'elements': 5}]
    assert validate_request(from_dict(cfg['seed_adf']), params)['collectives'][0]['algorithm'] == 'future_algorithm'
    env = ExplorationEnv(cfg)
    rejected = env.evaluate()
    assert 'error' in rejected and env.best_candidate is None
    assert 'collective algorithm not implemented' in json.dumps(rejected)
    assert env.used == 0 and env.attempts == []  # request_space rejects before backend execution.


def test_attempted_capacity_failure_consumes_budget():
    cfg = small_configuration()
    # All working sets fit individually; live buffers overflow only during scheduling.
    cfg['workloads'] = [{'kind': 'operator', 'params': {'op': 'allreduce', 'bytes': 64, 'dtype': 'fp32'}}]
    params = cfg['mapping_selectors']['multi_device']
    params['collectives'] = [{'after': 'allreduce', 'tensor': 'out', 'operation': 'allreduce',
        'algorithm': 'tree', 'ranks': [0, 1], 'dtype': 'fp32', 'elements': 16}]
    for device in cfg['seed_adf']['elements']:
        device['elements'][-1]['params']['capacity']['value'] = 128
    env = ExplorationEnv(cfg)
    rejected = env.evaluate()
    assert 'error' in rejected and env.best_candidate is None
    assert 'capacity exceeded' in json.dumps(rejected)
    assert env.used == 1 and len(env.attempts) == 1


def test_pi05_rejected_in_unsupported_multi_device_domain():
    cfg = small_configuration()
    cfg['workloads'] = [{'kind': 'pi05', 'params': {'num_steps': 2, 'action_horizon': 7}}]
    env = ExplorationEnv(cfg)
    assert ok(env.get_state())['multi_device']['executor_registered']
    response = env.evaluate()
    assert response['error']['code'] == 'evaluation_failed'
    assert response['error']['details']['trials'][0]['cause']['code'] == 'invalid_mapping_space'
    assert 'does not apply' in response['error']['details']['trials'][0]['error']
    assert env.used == 0 and env.attempts == [] and env.best_candidate is None
