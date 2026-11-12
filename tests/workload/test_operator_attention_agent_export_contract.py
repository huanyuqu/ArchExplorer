"""Targeted CORE df1435c Agent sealing/strict-reader validation, one CPU run."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_operator_mapping_attention_contract import no_backend,files
from test_operator_mapping_extension_contract import reseal


@pytest.fixture(scope='module')
def exported_agent(tmp_path_factory):
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.agent_env.driver import run_agent
    from arch_explorer.archs.attention import attention_search_config
    directory=tmp_path_factory.mktemp('agent-export')
    config=attention_search_config()
    config.update(trace_path=str(directory/'tools.jsonl'),max_evaluations=1,
                  parameter_samples=1,mapping_samples=1,parameter_domains=[])
    config['mapping_selectors']['compile_graph_v2']=config['mapping_selectors']['compile_graph_v2'][:1]
    config['agent']=dict(max_steps=2,max_tool_calls=2,log_path=str(directory/'agent.jsonl'),script=[
        {'tool':'evaluate','args':{'optimize':'parameters_and_mapping'}},
        {'tool':'submit','args':{'selection':'best'}}])
    (directory/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    env=ExplorationEnv(config)
    payload=run_agent(config,env=env)
    assert env.used==payload['evaluations_used']==1
    assert payload['schema']=='archexplorer.compilation-search-result.v1'
    assert payload['agent']['policy']=='scripted'
    assert payload['agent']['steps']==payload['agent']['tool_calls']==2
    assert payload['agent']['stop_reason']=='submitted'
    # Independently recompute the complete exported envelope, including Agent metadata.
    unsigned={k:v for k,v in payload.items() if k!='binding'}
    encoded=json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    assert payload['binding']=={'sha256':hashlib.sha256(encoded).hexdigest()}
    path=directory/'FINAL-AGENT.json'
    path.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    return directory,path,payload,env


def test_scripted_agent_export_has_valid_full_binding_and_strict_readonly_replay(exported_agent,monkeypatch):
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.exploration import replay
    from arch_explorer.simulator import api
    directory,path,payload,env=exported_agent
    before=files(directory)
    def forbidden(*args,**kwargs):raise AssertionError('Agent export reader created new execution state')
    with no_backend(monkeypatch),monkeypatch.context() as patch:
        patch.setattr(ExplorationEnv,'__init__',forbidden)
        patch.setattr(api,'prepare_compiled',forbidden)
        assert replay(payload)==payload
        assert files(directory)==before
        response=env.call('evaluate_request',{'payload':payload})
        assert 'error' not in response and response['result']==payload
    # Tool audit logging may append a trace row; prepared/exported artifacts remain readonly.
    audited={'tools.jsonl','agent.jsonl'}
    assert {k:v for k,v in files(directory).items() if k not in audited}=={k:v for k,v in before.items() if k not in audited}
    assert env.used==1
    assert json.loads(path.read_text())==payload


@pytest.mark.parametrize('mutation',['unsealed-agent','unknown-root','unknown-agent-field','agent-type','bool-count'])
def test_agent_reader_rejects_unsealed_and_rehashed_invalid_metadata(exported_agent,monkeypatch,mutation):
    from arch_explorer.agent_env import EnvError
    from arch_explorer.exploration import replay
    directory,_,payload,env=exported_agent
    changed=deepcopy(payload)
    if mutation=='unsealed-agent':changed['agent']['stop_reason']='step_limit'
    elif mutation=='unknown-root':changed['unexpected_agent_field']={}
    elif mutation=='unknown-agent-field':changed['agent']['unexpected']=True
    elif mutation=='agent-type':changed['agent']=[]
    else:changed['agent'].update(steps=True,tool_calls=True)
    if mutation!='unsealed-agent':changed=reseal(changed)
    before=files(directory)
    with no_backend(monkeypatch),pytest.raises(EnvError):
        replay(changed)
    assert env.used==1 and files(directory)==before


def test_new_process_console_reader_accepts_agent_export_without_reexecution(exported_agent,tmp_path):
    directory,path,payload,env=exported_agent
    before=files(directory)
    environment={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    environment.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    executable=str(Path(sys.executable).with_name('archexplorer'))
    direct=subprocess.run([executable,'evaluate',str(path)],cwd=tmp_path,env=environment,
        capture_output=True,text=True,timeout=60)
    assert direct.returncode==0,direct.stderr
    assert json.loads(direct.stdout)==payload
    code=r'''
import json,sys
from pathlib import Path
import numpy,pytest
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.simulator.events import engine
from arch_explorer.simulator import api
from arch_explorer.cli import main
def forbidden(*args,**kwargs):raise AssertionError('readonly CLI reader executed')
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(ExplorationEnv,'__init__',forbidden)
    patch.setattr(api,'prepare_compiled',forbidden)
    patch.setattr(engine,'simulate',forbidden)
    patch.setattr(engine.ConsistentScheduler,'run',forbidden)
    assert main(['evaluate',sys.argv[1]])==0
'''
    guarded=subprocess.run([sys.executable,'-c',code,str(path)],cwd=tmp_path,env=environment,
        capture_output=True,text=True,timeout=60)
    assert guarded.returncode==0,guarded.stderr
    assert json.loads(guarded.stdout)==payload
    assert env.used==1 and files(directory)==before
