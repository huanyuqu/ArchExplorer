"""Public ADF entry regression, including real subprocesses without test registries."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from arch_explorer.agent_env import EnvError, ExplorationEnv
from arch_explorer.exploration import explore
from arch_explorer.search import run_search

ROOT = Path(__file__).resolve().parents[2]


def run(*args, input=None):
    return subprocess.run([sys.executable, '-m', 'arch_explorer', *map(str, args)],
                          cwd=ROOT, input=input, text=True, capture_output=True, timeout=90)


@pytest.mark.parametrize('change', [
    {'anything': True}, {'search': {'family_quota': {'x': 2}}},
    {'agent': {'provider': 'remote'}}, {'mapping_templates': []}, {'workloads': {}},
])
def test_unified_configuration_rejects_unknown_or_malformed(config, change):
    config.update(change)
    with pytest.raises(EnvError, match='fields|list'):
        ExplorationEnv(config)
    with pytest.raises(EnvError):
        run_search(config)
    assert explore is run_search


def test_imports_and_real_search_do_not_load_removed_pipeline(tmp_path):
    script = r'''
import importlib.abc, json, sys
class Forbid(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        forbidden = ('arch_explorer.metadse_sim', 'arch_explorer.grammar',
                     'arch_explorer.runtime', 'arch_explorer.hardware',
                     'arch_explorer.search.evolutionary', 'arch_explorer.search.llm_guided')
        if any(fullname == p or fullname.startswith(p + '.') for p in forbidden):
            raise AssertionError('unwanted dependency: ' + fullname)
sys.meta_path.insert(0, Forbid())
from arch_explorer.exploration import explore, replay
from arch_explorer import search
from arch_explorer.cli import main
assert explore is search.run_search
assert main(['search', '--config', 'examples/reusable_detail/demand.json', '--output', sys.argv[1]]) == 0
assert main(['evaluate', sys.argv[1], '--output', sys.argv[2]]) == 0
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path/'result.json'), str(tmp_path/'replay.json')],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    first = json.loads((tmp_path/'result.json').read_text())
    second = json.loads((tmp_path/'replay.json').read_text())
    assert first['best'] == second['best']
    assert first['backend_options']['scheduler'] == 'conservative'


def test_real_jsonlines_and_scripted_submit_replay(tmp_path):
    config = json.loads((ROOT/'examples/reusable_detail/demand.json').read_text())
    config['trace_path'] = str(tmp_path/'env.jsonl')
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    frames = [dict(id=1, tool='describe_environment'), dict(id=2, tool='evaluate', args={'optimize':'mapping'}),
              dict(id=3, tool='submit', args={'path':str(tmp_path/'result.json')})]
    response = run('env', 'serve', '--config', path, input='\n'.join(json.dumps(x) for x in frames)+'\n')
    assert response.returncode == 0, response.stderr
    rows = [json.loads(line) for line in response.stdout.splitlines()]
    assert all('result' in row for row in rows), rows
    assert rows[0]['result']['search_space']['kind'] == 'component_graph'
    assert run('evaluate', tmp_path/'result.json', '--output', tmp_path/'replay.json').returncode == 0
    assert json.loads((tmp_path/'replay.json').read_text())['best'] == rows[-1]['result']['best']
    config.pop('trace_path')
    path.write_text(json.dumps(config))
    scripted = run('agent', 'run', '--config', path, '--max-steps', 1, '--output', tmp_path/'scripted.json')
    assert scripted.returncode == 0, scripted.stderr
    assert json.loads((tmp_path/'scripted.json').read_text())['agent']['policy'] == 'scripted'


def test_invalid_cli_and_unavailable_cost_are_errors(tmp_path):
    path = tmp_path/'invalid.json'
    path.write_text('{}')
    for args in [('search','--config',path), ('evaluate',path), ('env','serve','--config',path),
                 ('mcp','--config',path), ('agent','run','--config',path)]:
        result = run(*args)
        assert result.returncode == 2 and 'Traceback' not in result.stderr
    config = json.loads((ROOT/'examples/adf-cgra.json').read_text())
    config.update(area_budget_mm2=100, mapping_samples=1, max_evaluations=1)
    path.write_text(json.dumps(config))
    result = run('search','--config',path)
    assert result.returncode == 2 and 'unavailable' in result.stderr.lower()


@pytest.mark.parametrize('value', ['100 mm2', {'value': 100, 'unit': 'mm2eq'}, -1, True])
def test_budget_requires_positive_metric_value(config, value):
    config['area_budget_mm2'] = value
    with pytest.raises(EnvError, match='finite'):
        ExplorationEnv(config)
