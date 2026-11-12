"""Generic ADF editing retains CGRA physical and public replay contracts."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.agent_env.environment import load_registrations
from arch_explorer.devices import get
from arch_explorer.search import run_search

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(autouse=True)
def fake_registry():
    load_registrations({})

def config(suffix='cgra'):
    c=json.loads((ROOT/f'examples/adf-{suffix}.json').read_text())
    c.update(parameter_samples=1,mapping_samples=1,max_evaluations=4)
    c['search']={'max_steps':1,'proposals':[]}
    return c

def ok(response):
    assert 'error' not in response,response
    return response['result']

@pytest.mark.parametrize('preset',['vectorcgra_4x4','vectorcgra_4x4_diag'])
@pytest.mark.parametrize('params',[{}, {'words_per_bank':128}, {'clock_ghz':2}])
def test_seed_and_real_evaluation(preset,params):
    c=config();c['seed_adf']={'preset':preset,'parameters':params}
    env=ExplorationEnv(c)
    assert env.adf.to_dict()==get(preset,**params).to_dict()
    r=ok(env.evaluate(optimize='mapping'))
    assert r['score']['metrics']=={'area_mm2':None,'cost_usd':None}
    assert env.used==1

def test_capacity_and_unknown_budget_remain_failures():
    c=config();c['workloads'][0]['params'].update(m=6,k=6,n=6)
    env=ExplorationEnv(c);r=env.evaluate(optimize='mapping')
    assert 'error' in r and 'resident buffers need 108 words, available 64' in json.dumps(r)
    assert env.used==1 and env.best_candidate is None
    c=config();c.update(area_budget_mm2=100,cost_budget_usd=100)
    r=ok(ExplorationEnv(c).evaluate())
    assert not r['score']['feasible'] and r['score']['objective'] is None
    assert set(r['score']['unavailable_constraints'])=={'area_mm2','cost_usd'}

def test_generic_edit_checkpoint_restore_and_physical_rejection(tmp_path):
    c=config();c['trace_path']=str(tmp_path/'trace.jsonl')
    env=ExplorationEnv(c);base=ok(env.evaluate(optimize='mapping'))
    ok(env.checkpoint(name='seed'))
    mesh=env.adf.child('array').networks[0]
    spec=mesh.to_dict();spec['params']['diagonal']['value']=not mesh.value('diagonal')
    edits=[{'op':'delete','component':'vectorcgra/array/'+mesh.name},
           {'op':'create','container':'vectorcgra/array','spec':spec}]
    ok(env.apply_edits(edits=edits));other=ok(env.evaluate(optimize='mapping'))
    assert other['workloads'][0]['evaluation']['details']['cycles']['end_to_end']>0
    used=env.used;ok(env.restore(name='seed'));assert env.used==used
    assert ok(env.get_state())['fingerprint']==base['fingerprint']
    # Graph edit reachability is independent of physical model applicability.
    ok(env.apply_edits(edits=[{'op':'delete','component':'vectorcgra/array/pe'}]))
    assert 'error' in env.evaluate(optimize='mapping')

@pytest.mark.parametrize('suffix',['cgra','cgra-diag'])
@pytest.mark.parametrize('policy',['sa','greedy','random','scripted'])
def test_explicit_restricted_experiment_and_cli_replay(suffix,policy,tmp_path):
    c=config(suffix);cfg=tmp_path/'c.json';cfg.write_text(json.dumps(c))
    winner=tmp_path/'winner.json';again=tmp_path/'again.json'
    cmd=['agent','run'] if policy=='scripted' else ['search']
    def cli(args):
        return subprocess.run([sys.executable,'-m','arch_explorer',*map(str,args)],cwd=ROOT,
                              capture_output=True,text=True,timeout=90)
    r=cli(cmd+['--config',cfg,'--policy',policy,'--output',winner]);assert r.returncode==0,r.stderr
    r=cli(['evaluate',winner,'--output',again]);assert r.returncode==0,r.stderr
    a=json.loads(winner.read_text());assert a['best']==json.loads(again.read_text())['best']
    assert 0<a['evaluations_used']<=4
    if policy!='scripted':assert a['search']['trajectory']==[]
