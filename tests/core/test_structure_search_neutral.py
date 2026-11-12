"""Behavioral counterexamples to seed names, family gates and CGRA locks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from arch_explorer.adf import SpacePoint, fingerprint, from_dict, walk
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.search import proposals
from arch_explorer.structure import apply_edit

ROOT = Path(__file__).resolve().parents[2]


def test_catalog_can_introduce_absent_component_and_new_tree(config):
    env = ExplorationEnv(config)
    root = env.adf
    assert not any(isinstance(n, SpacePoint) and n.component == 'systolic_array' for _, n in walk(root))
    choices = proposals(env.get_state()['result'], env.describe_environment()['result']['components'])
    create = next(p for p in choices if isinstance(p, dict) and p['op'] == 'create'
                  and p.get('spec', {}).get('component') == 'systolic_array')
    changed = apply_edit(root, create).adf
    assert fingerprint(changed) != fingerprint(root)
    assert any(isinstance(n, SpacePoint) and n.component == 'systolic_array' for _, n in walk(changed))
    assert {p['op'] for p in choices if isinstance(p, dict)} >= {'split','insert','create','delete','remove'}


def test_renaming_nodes_preserves_generated_edit_sequence(config):
    env = ExplorationEnv(config)
    catalog = env.describe_environment()['result']['components']
    root = deepcopy(env.adf)
    # Preserve relative ordering but replace every architecture label.
    for _, node in list(walk(root)):
        node.name = 'renamed_' + node.name
    first = proposals({'adf':env.adf.to_dict()}, catalog)
    second = proposals({'adf':root.to_dict()}, catalog)
    def shape(value):
        if isinstance(value, dict):
            return {k:shape(v) for k,v in value.items() if k not in {'name','point','container','parent','component','level','children','points'}}
        if isinstance(value, list): return [shape(x) for x in value]
        return value
    assert shape(first) == shape(second)
    assert len(first) == len(second)
    assert all(not isinstance(p, dict) or not p['op'].endswith('_transform') for p in first)


def test_different_tree_exposes_generic_moves_without_family_label(config):
    env = ExplorationEnv(config)
    before = env.adf.to_dict()
    assert 'result' in env.apply_edits(edits=[dict(op='insert', parent='chip', children=['chip/core'],
                                                 name='user_partition', group_size=2)])
    assert env.adf.to_dict() != before
    choices = proposals(env.get_state()['result'], env.describe_environment()['result']['components'])
    move = next(p for p in choices if isinstance(p, dict) and p.get('container') == 'chip/user_partition'
                and p.get('component') == 'chip/l2')
    # Parent multiplicity differs; graph validation must reject without silently duplicating storage.
    response = env.apply_edits(edits=[move])
    assert response['error']['code'] == 'repeat_mismatch'
    assert any(isinstance(p,dict) and p.get('spec',{}).get('component') == 'systolic_array'
               and p.get('container') == 'chip/user_partition' for p in choices)


def test_real_cgra_generic_edits_domains_and_physical_rejection(tmp_path):
    script = r'''
import json
from pathlib import Path
from arch_explorer.adf import walk, SpacePoint
from arch_explorer.agent_env import ExplorationEnv, replay_result
from arch_explorer.devices import get
from arch_explorer.search import proposals
adf = get('vectorcgra_4x4', words_per_bank=32)
for _, n in list(walk(adf)): n.name = 'user_' + n.name
config = dict(seed_adf=adf.to_dict(), workloads=[dict(kind='operator',params=dict(op='matmul',m=3,k=2,n=3,dtype='uint32'))],
              backend='events',mapping_templates=['cgra_spatial'],max_evaluations=8,parameter_samples=2,mapping_samples=1,
              parameter_domains=[dict(path='user_vectorcgra/user_bank',param='capacity',values=[128,256],source='explicit synthetic domain'),
                                 dict(path='user_vectorcgra/user_array/user_mesh',param='link_bandwidth',values=[4,8],source='explicit synthetic domain')])
env = ExplorationEnv(config)
base = env.evaluate(optimize='mapping'); assert 'result' in base, base
choices = proposals(env.get_state()['result'],env.describe_environment()['result']['components'])
assert any(isinstance(p,dict) and p['op']=='create' for p in choices)
change = next(p for p in choices if isinstance(p,list) and p[-1]['spec'].get('component')=='mesh2d'
              and p[-1]['spec']['params']['diagonal']['value'] is True)
assert 'result' in env.apply_edits(edits=change)
assert 'result' in env.evaluate(optimize='mapping')
assert 'result' in env.set_parameters(parameters=[dict(path='user_vectorcgra/user_bank',param='capacity',value=256)])
assert 'result' in env.evaluate(optimize='parameters_and_mapping')
output = env.submit()['result']; assert replay_result(output)['best']==output['best']
# A syntactically valid edit is reachable, but its actual physical configuration is unsupported.
assert 'result' in env.apply_edits(edits=[dict(op='delete',component='user_vectorcgra/user_array/user_pe')])
before=env.used
error=env.evaluate(optimize='mapping'); assert 'error' in error,error
# No-mapping rejection occurs before backend; a bad linked clock below does call the backend.
assert 'result' in env.undo()
bad = env.set_parameters(parameters=[dict(path='user_vectorcgra/user_array/user_mesh',param='link_bandwidth',value=8)])
assert 'result' in bad, bad
before=env.used
error=env.evaluate(optimize='mapping'); assert 'error' in error,error
assert env.used==before+1
assert 'one uint32 per tick' in json.dumps(error), error
print(json.dumps(dict(evaluations_used=env.used, proposal_count=len(choices), physical_error=error)))
'''
    result = subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=90)
    assert result.returncode == 0, result.stderr
    (tmp_path/'cgra-evidence.json').write_text(result.stdout)
