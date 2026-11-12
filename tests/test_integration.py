"""Public ADF search/evaluation integration; retired grammar contracts removed."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.agent_env.environment import load_registrations
from arch_explorer.search import run_search
from arch_explorer.exploration import explore, replay

ROOT = Path(__file__).resolve().parents[1]


def config():
    load_registrations({})
    c = json.loads((ROOT/'examples/reusable_detail/demand.json').read_text())
    c.update(max_evaluations=6, parameter_samples=1, mapping_samples=1)
    c['search'] = {'max_steps':2, 'proposals':[]}
    return c


def test_single_public_search_and_complete_replay():
    assert explore is run_search
    result = explore(config())
    assert result['best'] and result['evaluations_used'] <= 6
    assert replay(json.loads(json.dumps(result)))['best'] == result['best']


def test_invalid_schema_and_budget_are_not_compatibility_routes():
    for value in ({}, {'architectures':[], 'workload':{'m':4}}, {'seed_adf':{}}):
        with pytest.raises(ValueError):
            run_search(value)
    c = config()
    c['area_budget_mm2'] = {'value':100, 'unit':'mm2eq'}
    with pytest.raises(ValueError):
        run_search(c)


def test_cli_from_other_cwd_and_removed_command(tmp_path):
    c = config()
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps(c))
    # Keep the source package explicit for source-tree pytest. Installed-wheel
    # acceptance separately uses -I without this path.
    wrapper = 'import sys;sys.path.insert(0,sys.argv.pop(1));from arch_explorer.cli import main;raise SystemExit(main())'
    def cli(*args):
        return subprocess.run([sys.executable,'-c',wrapper,str(ROOT),*map(str,args)],
                              cwd=tmp_path,capture_output=True,text=True,timeout=90)
    winner=tmp_path/'winner.json'; again=tmp_path/'again.json'
    assert cli('search','--config',cfg,'--output',winner).returncode==0
    assert cli('evaluate',winner,'--output',again).returncode==0
    assert json.loads(winner.read_text())['best']==json.loads(again.read_text())['best']
    removed=cli('explore',cfg)
    assert removed.returncode==2 and 'invalid choice' in removed.stderr


def test_exploration_cost_configuration_is_an_object():
    c=config();c['cost_model']=False
    with pytest.raises(ValueError,match='cost_model must be an object'):
        run_search(c)
