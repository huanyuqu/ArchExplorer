"""Real budget, feasible selection and subprocess replay, without small-model aliases."""
import importlib.util
import json
from pathlib import Path

import pytest
from non_gpu_fixture import shared_adf
from arch_explorer.adf import find

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('baseline_runner', ROOT/'tools/run_model_baseline.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def fake_registry():
    # Override the parent fake fixture: exercise real components and backend.
    from arch_explorer.agent_env.environment import load_registrations
    load_registrations({})


def scenario():
    # A retained complete standard ViT graph on synthetic non-GPU vector/memory
    # resources. Historical GPU example files are no longer public inputs.
    adf = shared_adf(1)
    find(adf, 'chip/dram').name = 'hbm'
    mapping = {'template': 'gsm_tiled', 'params': {'tile_m': 64, 'tile_n': 64, 'rows_per_cta': 4}}
    return {'source_baseline': 'explicit non-GPU complete-model test scenario',
        'config': {'seed_adf': adf.to_dict(), 'workloads': [{'kind': 'vit',
            'params': {'profile': 'vit_b16_224', 'dtype': 'fp16'}, 'weight': 1}],
            'backend': 'events', 'mapping_templates': ['gsm_tiled'], 'max_evaluations': 12,
            'parameter_samples': 1, 'mapping_samples': 1, 'random_seed': 17,
            'parameter_domains': [
                {'path': 'chip/core/vector', 'param': 'clock_ghz', 'values': [1.0, 1.2], 'source': 'explicit CPU test domain'},
                {'path': 'chip/hbm', 'param': 'capacity', 'values': [2**20, 2**30], 'source': 'explicit capacity test domain'}],
            'area_budget_mm2': 500, 'cost_budget_usd': 1000, 'cost_model': {}},
        'initial_mappings': [mapping], 'generic_steps': 0,
        'limits': {'max_point_instances': 128, 'max_capacity_bytes': 2**34, 'required_paths': ['chip/hbm']},
        'experiments': [
            {'label': 'generic topology', 'calls': [{'tool': 'apply_edits', 'args': {'edits': [
                {'op': 'insert', 'parent': 'chip', 'children': ['chip/core'], 'name': 'partition'}]}}]},
            {'label': 'clock', 'calls': [{'tool': 'set_parameters', 'args': {'parameters': [
                {'path': 'chip/core/vector', 'param': 'clock_ghz', 'value': 1.2, 'source': 'test sensitivity'}]}}]},
            {'label': 'mapping', 'calls': [{'tool': 'set_mappings', 'args': {'mappings': [
                {'template': 'gsm_tiled', 'params': {'tile_m': 128, 'tile_n': 128, 'rows_per_cta': 4}}]}}]},
            {'label': 'force actual capacity failure', 'calls': [{'tool': 'set_parameters', 'args': {'parameters': [
                {'path': 'chip/hbm', 'param': 'capacity', 'value': 2**20, 'source': 'capacity rejection'}]}}]}]}


def test_complete_vit_budget_failure_and_feasible_best_replay(tmp_path):
    cfg = scenario()
    cfg['experiments'] = [cfg['experiments'][-1]]  # Keep the full ViT; force capacity failure.
    cfg['generic_steps'] = 0
    output = tmp_path/'run'
    result = runner.run(cfg, output, max_evaluations=2)
    assert result['backend_attempts'] == 2 and result['failed_backend_attempts'] == 1
    assert result['status'] == 'feasible_winner'
    assert all(r['entire_object_equal'] for r in result['replay'])
    best = json.loads((output/'best.json').read_text())['best']
    assert best['workloads'][0]['workload']['params']['profile'] == 'vit_b16_224'
    assert best['score']['feasible']
    assert json.loads((output/'state.json').read_text())['evaluations_used'] == 2


@pytest.mark.parametrize('unknown', [False, True])
def test_hard_budget_no_feasible_winner_preserves_null_and_replays(tmp_path, unknown):
    cfg = scenario()
    if unknown:
        cfg['config']['cost_model']['overrides'] = {'chip/hbm':{'dram_usd_gib':
            {'value':-1, 'unit':'USD/GiB', 'source':'intentionally unavailable price'}}}
    else:
        cfg['config']['area_budget_mm2'] = 1
    cfg['experiments'] = []
    cfg['generic_steps'] = 0
    output = tmp_path/'run'
    result = runner.run(cfg, output, max_evaluations=1)
    assert result['status'] == 'no_feasible_winner'
    assert result['backend_attempts'] == 1
    assert not (output/'best.json').exists()
    assert result['replay'][0]['entire_object_equal']
    row = json.loads((output/'trajectory.json').read_text())[0]
    if unknown:
        assert row['costs']['cost_usd'] is None and row['score']['objective'] is None
        assert row['score']['unavailable_constraints'] == ['cost_usd']
    assert not row['score']['feasible']


def test_provenance_only_duplicate_and_resource_rejection_do_not_spend(tmp_path):
    cfg = scenario()
    cfg['experiments'] = [
        {'label':'same values new source','calls':[{'tool':'set_parameters','args':{'parameters':[
            {'path':'chip/core/vector','param':'clock_ghz','value':1.0,'source':'different citation'}]}}]},
        {'label':'required storage removed','calls':[{'tool':'apply_edits','args':{'edits':[
            {'op':'delete','component':'chip/hbm'}]}}]},
    ]
    cfg['generic_steps'] = 0
    output = tmp_path/'run'
    result = runner.run(cfg, output, max_evaluations=2)
    assert result['backend_attempts'] == 1
    assert result['duplicate_proposals'] == 1 and result['resource_rejections'] == 1
    assert all(r['entire_object_equal'] for r in result['replay'])




def test_atomic_topology_and_clock_preserve_mapping_until_explicit_change(tmp_path):
    cfg = scenario()
    cfg['experiments'] = cfg['experiments'][:3]
    cfg['generic_steps'] = 0
    output = tmp_path/'run'
    result = runner.run(cfg, output, max_evaluations=4)
    assert result['backend_attempts'] == 4 and result['failed_backend_attempts'] == 0
    rows = json.loads((output/'trajectory.json').read_text())
    assert rows[1]['structure_changed'] and not rows[1]['mapping_changed']
    assert rows[2]['adf_values_changed'] and not rows[2]['structure_changed'] and not rows[2]['mapping_changed']
    assert rows[3]['mapping_changed'] and not rows[3]['adf_values_changed']
    assert rows[0]['mappings'] == rows[1]['mappings'] == rows[2]['mappings']
    assert all(r['entire_object_equal'] for r in result['replay'])
