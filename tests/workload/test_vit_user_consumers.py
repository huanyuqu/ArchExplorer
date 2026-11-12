"""Installed CORE30 public API/CLI/offline Agent/search and immutable replay."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import files,no_backend
from view_storage_search_consumer import validate_saved
from vit_contract_cases import KEYS,REJECT,frozen_inputs,vit_api


def core():
    module=importlib.import_module('arch_explorer.archs.vit_compilation')
    for name in ('vit_inputs','vit_workload','vit_scene','vit_search_config','vit_agent_config'):
        assert callable(getattr(module,name))
    return module


def options(key):
    p=deepcopy(frozen_inputs()['cases'][key]['workload']['params'])
    result={k:p[k] for k in ('layers','dtype','batch','image_size','patch','width','heads','classes','inputs')}
    result['mlp']=p['mlp_hidden']
    return result


@pytest.mark.parametrize('key',['regular-2-fp32','boundary-4-bf16'])
def test_CORE_data_scene_search_Agent_forward_the_exact_public_contract(vit_api,monkeypatch,key):
    w,s=vit_api;c=core();kwargs=options(key);before=deepcopy(kwargs)
    with no_backend(monkeypatch):
        spec=c.vit_workload(**kwargs)
        assert spec.params['inputs']==kwargs['inputs'] and spec.params['mlp_hidden']==kwargs['mlp']
        graph=w.workload_from_spec(spec)
        for candidate in (0,1):
            adf,same,mapping=c.vit_scene(**kwargs,candidate=candidate,layout='views')
            assert same.to_dict()==spec.to_dict()
            assert set(mapping.params['compute'])=={op.name for op in graph.operators}
            assert len(set(mapping.params['compute'].values()))==2
            config=c.vit_search_config(**kwargs,candidate=candidate,layout='views',compare_layouts=True)
            requests=config['mapping_selectors']['compile_graph_v3']
            assert config['max_evaluations']==12 and config['mapping_samples']==len(requests)==4
            assert requests[0]==mapping.params
            for params in requests:
                assert set(params)=={'compute','memory','rules','operator_rules'}
            single=c.vit_search_config(**kwargs,candidate=candidate,layout='views',compare_layouts=False)
            assert single['max_evaluations']==6 and len(single['mapping_selectors']['compile_graph_v3'])==2
            agent=c.vit_agent_config(**kwargs,candidate=candidate,layout='views')
            assert agent['max_evaluations']==1 and 'search' not in agent
            assert [row['tool'] for row in agent['agent']['script']]==['operator_mappings','evaluate','submit']
    assert kwargs==before


def fresh_readonly(path, tmp_path, expected='search'):
    before=files(path.parent)
    script=r'''
import builtins,io,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import pytest
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.simulator import api as s
from arch_explorer.exploration import replay
from test_operator_mapping_attention_contract import no_backend
p=Path(sys.argv[2]);raw=p.read_bytes();value=json.loads(raw)
def forbidden(*args,**kwargs):raise AssertionError('readonly constructed execution state')
open_original=builtins.open;io_open_original=io.open
def read_only_open(original):
    def opened(file,mode='r',*args,**kwargs):
        if any(flag in mode for flag in 'wax+'):raise AssertionError('readonly wrote a file')
        return original(file,mode,*args,**kwargs)
    return opened
with pytest.MonkeyPatch.context() as patch,no_backend(patch):
    patch.setattr(ExplorationEnv,'__init__',forbidden)
    patch.setattr(s,'prepare_compiled',forbidden)
    patch.setattr(builtins,'open',read_only_open(open_original))
    patch.setattr(io,'open',read_only_open(io_open_original))
    assert (s.evaluate_request(value) if sys.argv[3]=='compiled' else replay(value))==value
assert p.read_bytes()==raw
print('fresh-ViT-readonly')
'''
    answer=command([sys.executable,'-c',script,str(Path(__file__).parent),str(path),expected],tmp_path,timeout=540)
    assert answer.stdout.strip()=='fresh-ViT-readonly'
    assert files(path.parent)==before


def test_real_four_candidate_search_has_charged_failures_and_an_actual_winner(tmp_path):
    consumer=Path(__file__).with_name('vit_search_consumer.py')
    root=tmp_path/'search'
    command([sys.executable,str(consumer),str(root)],tmp_path,timeout=540)
    best=validate_saved(root)
    assert best['best']['workloads'][0]['workload']['kind']=='vit_classifier'
    fresh_readonly(root/'BEST.json',tmp_path)


def test_real_offline_Agent_uses_one_execution_and_fresh_readonly(vit_api,tmp_path):
    from arch_explorer.agent_env.driver import run_agent
    from arch_explorer.agent_env import ExplorationEnv
    c=core();config=c.vit_agent_config(layers=2,dtype='fp16',layout='views')
    config['trace_path']=str(tmp_path/'agent-tools.jsonl')
    env=ExplorationEnv(config);result=run_agent(env=env,policy='scripted')
    assert result['evaluations_used']==env.used==1
    assert result['best']['workloads'][0]['workload']['kind']=='vit_classifier'
    row=result['best']['workloads'][0]
    assert row['mapping']==row['compiled_result']['request']['compilation']['source']['mapping']
    path=tmp_path/'AGENT-RESULT.json';path.write_text(json.dumps(result,indent=2)+'\n')
    fresh_readonly(path,tmp_path)


def test_exported_ViT_CLI_example_really_compiles_prepares_and_executes(vit_api,tmp_path):
    export=Path(os.environ['VAL59_SOURCE_EXPORT'])
    example=export/'examples/vit_compilation'
    target=tmp_path/'example';shutil.copytree(example,target)
    source=tmp_path/'source.json';search=tmp_path/'search.json';agent=tmp_path/'agent.json'
    command([sys.executable,str(target/'scene.py'),'--dtype','bf16','--layout','views',
        '--output',str(source),'--search-config',str(search),'--agent-config',str(agent)],tmp_path)
    cli=str(Path(sys.executable).with_name('archexplorer'))
    compiled=tmp_path/'compiled.json';result=tmp_path/'RESULT.json'
    command([cli,'compile',str(source),'--output',str(compiled)],tmp_path,timeout=540)
    request=command([cli,'prepare',str(compiled),'--options',str(target/'provider-options.json'),
                     '--output',str(tmp_path/'new-request')],tmp_path,timeout=540)
    path=tmp_path/'request.json';path.write_text(request.stdout)
    command([cli,'evaluate',str(path),'--output',str(result)],tmp_path,timeout=540)
    saved=json.loads(result.read_text())
    assert saved['request']['compilation']['source']['workload']['kind']=='vit_classifier'
    assert saved['evaluation']['latency_ns']>0
    fresh_readonly(result,tmp_path,'compiled')


def test_explicit_rerun_reuses_completed_search_without_new_search(tmp_path):
    from arch_explorer.exploration import rerun
    path=Path(os.environ['VAL59_SEARCH_RESULT']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL59_SEARCH_RESULT_SHA256']
    original=json.loads(raw);before=files(path.parent)
    directory=tmp_path/'new-rerun'
    result=rerun(original,output_dir=directory)
    assert result['evaluations_used']==1
    assert result['best']['workloads'][0]['workload']['kind']=='vit_classifier'
    old=original['best']['workloads'][0]['compiled_result']['request']
    new=result['best']['workloads'][0]['compiled_result']['request']
    assert new!=old and directory.is_dir() and files(directory)
    assert str(directory) in json.dumps(new) and str(directory) not in json.dumps(old)
    assert files(path.parent)==before
