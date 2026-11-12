"""End-to-end conditional mapping discovery, environment winner and replay."""
import json

import pytest
from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.agent_env.environment import replay_result
from arch_explorer.devices.pim import neupims
from arch_explorer.search.nested import run_search


def config(memory):
    model = dict(name='synthetic',hidden=128,heads=1,kv_heads=1,head_dim=128,intermediate=128,
                 layers=1,vocab=128,gated=False,norm='layernorm',activation='gelu',source='synthetic test')
    return {'seed_adf':neupims(64).to_dict(), 'workloads':[{'kind':'transformer', 'params':dict(
        model=model,phase='decode',batch=2,seq_len=32,layers=1)}], 'backend':'events',
        'max_evaluations':1, 'parameter_samples':1, 'mapping_samples':1, 'cost_model':{},
        'mapping_templates':['pim_offload'], 'mapping_selectors':{'pim_offload':{
            'npu_execution':'instructions','memory_execution':memory}},
        'search':{'max_steps':1,'proposals':[]}, 'random_seed':17}


def unwrap(value):
    assert 'error' not in value, value
    return value['result']


@pytest.mark.parametrize('memory', ['aggregate', 'transactions'])
def test_full_public_contract_to_env_submit_and_replay(memory):
    conf = config(memory)
    env = ExplorationEnv(conf)
    description = unwrap(env.describe_environment())
    entry = description['mapping_spaces'][0]['pim_offload']
    assert entry['contract']['kind'] == 'conditional'
    product = entry['sampling_spaces'][0]
    mapping = MappingSpec('pim_offload', {k:v[0] for k,v in product.items()})
    assert not any(k.startswith('inner.') for k in mapping.params)
    outcome = unwrap(env.evaluate(optimize='none', mappings=[mapping.to_dict()]))
    assert outcome['evaluations_used'] == 1
    winner = unwrap(env.submit(selection='current'))
    replayed = replay_result(json.loads(json.dumps(winner)))
    assert winner['best'] == replayed['best']
    direct = evaluate(neupims(64), WorkloadSpec(**conf['workloads'][0]),
                      MappingSpec('pim_offload',conf['mapping_selectors']['pim_offload']),cost_model=False)
    assert direct.latency_ns == winner['best']['workloads'][0]['evaluation']['latency_ns']


@pytest.mark.parametrize('memory', ['aggregate','transactions'])
def test_opt_in_search_emits_replayable_complete_winner(memory):
    winner = run_search(config(memory),policy='sa')
    assert winner['evaluations_used'] == 1
    assert winner['best']['mappings'][0]['params']['memory_execution'] == memory
    assert winner['best'] == replay_result(winner)['best']


@pytest.mark.parametrize('params', [
    {'npu_execution':'instructions','inner.tile_n':64},
    {'memory_execution':'transactions'},
    {'npu_execution':'instructions','sub_batches':2},
    {'npu_execution':'instructions','inner':'dmc_partition'},
    {'npu_execution':'instructions','typo':1},
])
def test_direct_api_also_rejects_invalid_or_inactive_parameters(params):
    conf = config('aggregate')
    with pytest.raises(ValueError,match='Invalid mapping'):
        evaluate(neupims(64),WorkloadSpec(**conf['workloads'][0]),MappingSpec('pim_offload',params),cost_model=False)


def test_direct_iterations_override_rejected():
    conf = config('aggregate')
    with pytest.raises(ValueError, match='one iteration'):
        evaluate(neupims(64),WorkloadSpec(**conf['workloads'][0]),MappingSpec('pim_offload',{
            'npu_execution':'instructions'}),cost_model=False, iterations=2)
