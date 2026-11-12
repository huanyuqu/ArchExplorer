"""Physical resource limits survive hierarchy, edits, real evaluation and replay."""
from copy import deepcopy
import json
from pathlib import Path
import random
import pytest
from arch_explorer.agent_env import ExplorationEnv, replay_trace
from arch_explorer.agent_env.environment import load_registrations, replay_result
from arch_explorer.search.candidates import proposals
from arch_explorer.search.nested import run_search
from arch_explorer.adf import from_dict
from arch_explorer.structure.resources import inventory
from arch_explorer.archs.adf_seeds import memory, bus
from non_gpu_fixture import shared_adf

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(autouse=True)
def fake_registry():
    load_registrations({})

def config(tmp_path):
    adf = shared_adf(4)
    adf.elements[2].name = 'hbm'
    adf.elements[0].elements.append(memory('sram', capacity=2**20))
    adf.elements[0].networks.append(bus())
    c = {'seed_adf': adf.to_dict(), 'backend': 'events', 'max_evaluations': 12,
         'parameter_samples': 1, 'mapping_samples': 1, 'random_seed': 17,
         'area_budget_mm2': 150, 'cost_budget_usd': 500, 'area_penalty': 0, 'cost_penalty': 0,
         'cost_model': {'basis': 'raw'}, 'resource_constraints': {
             'totals': [
                 {'select': {}, 'unit': 'count', 'max': 16},
                 {'select': {'role': 'compute'}, 'unit': 'count', 'min': 4, 'max': 4},
                 {'select': {'role': 'memory'}, 'unit': 'count', 'min': 6, 'max': 6},
                 {'select': {'role': 'network'}, 'unit': 'count', 'min': 5, 'max': 5},
                 {'select': {'role': 'memory'}, 'param': 'capacity', 'unit': 'B', 'min': 1094713344, 'max': 1094713344},
                 {'select': {'role': 'memory'}, 'param': 'bandwidth', 'unit': 'B/ns', 'max': 768.0},
                 {'select': {'role': 'network'}, 'param': 'bandwidth', 'unit': 'B/ns', 'max': 320.0},
                 {'select': {'role': 'network'}, 'param': 'link_bandwidth', 'unit': 'B/ns', 'max': 64.0}],
             'parameters': [
                 {'select': {}, 'param': 'clock_ghz', 'unit': 'GHz', 'value': 1.0},
                 {'select': {'component': 'vector_unit'}, 'param': 'lanes', 'unit': 'count', 'value': 16},
                 {'select': {'component': 'vector_unit'}, 'param': 'sfu_lanes', 'unit': 'count', 'value': 4},
                 {'select': {'role': 'memory'}, 'param': 'latency', 'unit': 'ns', 'min': 1.0},
                 {'select': {'role': 'network'}, 'param': 'latency', 'unit': 'ns', 'min': 1.0},
                 {'select': {'role': 'network'}, 'param': 'hop_latency', 'unit': 'ns', 'min': 1.0}]}}
    c['workloads']=[{'kind':'operator','params':{'op':'matmul','batch':1,'m':8,'k':5,'n':8,'dtype':'fp16','iterations':1}}]
    c['trace_path']=str(tmp_path/'trace.jsonl')
    return c

def ok(r):
    assert 'error' not in r,r
    return r['result']

def test_rename_nested_repeat_fixed_units_and_atomic_budget(tmp_path):
    e=ExplorationEnv(config(tmp_path));before=ok(e.get_state())
    core=deepcopy(before['adf']['elements'][0]);core['name']='renamed';core['repeat']*=2
    bad=e.apply_edits(edits=[{'op':'delete','component':'chip/core'}, {'op':'create','container':'chip','spec':core}])
    assert bad['error']['code']=='resource_constraint'
    assert ok(e.get_state())==before and e.used==0
    core['repeat']//=2
    ok(e.apply_edits(edits=[{'op':'delete','component':'chip/core'}, {'op':'create','container':'chip','spec':core}]))
    nested=ok(e.get_state());sm=deepcopy(core['elements'][0]);sm['params']['clock_ghz']['value']=2.0
    bad=e.apply_edits(edits=[{'op':'delete','component':'chip/renamed/vector'}, {'op':'create','container':'chip/renamed','spec':sm}])
    assert bad['error']['code']=='resource_constraint' and ok(e.get_state())==nested
    assert sum(r['instances'] for r in inventory(e.adf,e.components) if r['role']=='compute')==4
    c=config(tmp_path/'units');c['resource_constraints']['totals'][4]['unit']='KiB'
    from arch_explorer.agent_env.errors import EnvError
    with pytest.raises(EnvError, match='parameter/unit'):
        ExplorationEnv(c)


def test_unknown_hard_cost_stays_null_and_does_not_call_backend(tmp_path):
    c=config(tmp_path);c['cost_model']['overrides']={'chip/hbm':{'dram_usd_gib':{'value':-1,'unit':'USD/GiB','source':'unknown price probe'}}}
    e=ExplorationEnv(c)
    assert ok(e.get_state())['costs']['cost_usd'] is None
    assert e.evaluate()['error']['code']=='resource_constraint' and e.used==0


def test_no_candidates_is_finite_and_auditable(tmp_path):
    e=ExplorationEnv(config(tmp_path));state=ok(e.get_state());state['resource_constraints']['totals'][0]['max']=0
    audit=[]
    assert proposals(state,ok(e.describe_environment())['components'],max_attempts=7,rng=random.Random(17),audit=audit)==[]
    assert len(audit)==7 and e.used==0


@pytest.mark.parametrize('policy',['sa','greedy','random'])
def test_shared_search_constraint_replay_and_budget(policy,tmp_path):
    c=config(tmp_path);c.update(max_evaluations=4,parameter_samples=3)
    c['search']={'max_steps':4,'max_proposals':20}
    e=ExplorationEnv(c);result=run_search(env=e,policy=policy)
    assert e.used<=4 and result['resource_constraints']==c['resource_constraints']
    assert len(result['search']['proposal_audit'])<=20
    assert replay_result(result)['best']==result['best']
    state=ok(e.get_state())
    replayed=replay_trace([json.loads(s) for s in Path(c['trace_path']).read_text().splitlines()])
    assert ok(replayed.get_state())==state


def test_backend_failure_charges_and_checkpoint_does_not_refund(tmp_path):
    c=config(tmp_path)
    # Smaller input storage is valid under a deliberately open resource envelope,
    # but the actual backend must reject insufficient physical capacity.
    c['resource_constraints']={'parameters':c['resource_constraints']['parameters'][:1]}
    c['seed_adf']['elements'][2]['params']['capacity']['value']=1
    e=ExplorationEnv(c);ok(e.checkpoint(name='start'))
    assert 'error' in e.evaluate()
    assert e.used==1 and len(e.attempts)==1 and not e.attempts[0]['valid']
    ok(e.restore(name='start'));assert e.used==1


def test_explicit_multi_device_request_remains_correlated(tmp_path):
    from tools.generate_multi_device_execution import configuration
    c=configuration('operator',tp=True);c['resource_constraints']={};c['trace_path']=str(tmp_path/'multi.jsonl')
    e=ExplorationEnv(c);r=ok(e.evaluate(optimize='mapping'))
    assert r['mappings'][0]['template']=='multi_device'
    assert r['mappings'][0]['params']['topology_edges']==c['mapping_selectors']['multi_device']['topology_edges']
    assert replay_result(ok(e.submit()))['best']==ok(e.submit())['best']


def test_extensive_resource_exchange_preserves_physical_total(tmp_path):
    from arch_explorer.search.candidates import _exchanges
    from arch_explorer.structure import apply_edit
    c=config(tmp_path);c.pop('resource_constraints')
    core=c['seed_adf']['elements'][0]
    extra=deepcopy(core['elements'][1]);extra['name']='extra_sram';extra['repeat']=2
    core['elements'].append(extra)
    e=ExplorationEnv(c);before=inventory(e.adf,e.components)
    total=lambda rows:sum(r['instances']*r['params']['capacity']['value'] for r in rows if r['component']=='sram')
    edits=[x for x in _exchanges(ok(e.get_state()),ok(e.describe_environment())['components'])
           if x[0]['component']=='chip/core']
    assert edits
    for group in edits:
        a=e.adf
        for edit in group:
            a=apply_edit(a,edit).adf
        assert total(inventory(a,e.components))==total(before)
        assert a.to_dict()!=e.adf.to_dict()


@pytest.mark.parametrize('selector,param',[({'component':'misspelled_sm'},'clock_ghz'),({},'clok_ghz')])
def test_typo_in_constraint_does_not_silently_disable_it(selector,param,tmp_path):
    from arch_explorer.agent_env.errors import EnvError
    c=config(tmp_path)
    c['resource_constraints']={'parameters':[{'select':selector,'param':param,'unit':'GHz','value':1.0}]}
    with pytest.raises(EnvError):
        ExplorationEnv(c)


def test_fixed_numeric_quantity_accepts_equivalent_json_integer(tmp_path):
    c=config(tmp_path)
    c['seed_adf']['elements'][0]['elements'][0]['params']['clock_ghz']['value']=1
    e=ExplorationEnv(c)
    assert ok(e.evaluate())['score']['feasible'] and e.used==1
