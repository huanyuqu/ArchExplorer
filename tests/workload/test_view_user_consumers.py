"""Frozen CORE29 helpers feed real CLI/Agent/search and strict saved readers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from test_multilayer_user_consumers import expected_chain
from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import files,no_backend
from test_precision_transformer_contract import close_values
from view_storage_contract_cases import view_api


@pytest.mark.parametrize('candidate',[0,1])
def test_CORE_scene_forwards_public_layout_choices_without_changing_semantic_inputs(view_api,monkeypatch,candidate):
    from arch_explorer.archs.multilayer import encoder_inputs,encoder_scene
    w,_=view_api;inputs=encoder_inputs(layers=2,input_case='boundary');before=deepcopy(inputs)
    with no_backend(monkeypatch):
        adf,spec,materialized=encoder_scene(inputs=inputs,dtype='bf16',candidate=candidate,layout='materialized')
        other,same,views=encoder_scene(inputs=inputs,dtype='bf16',candidate=candidate,layout='views')
        public=w.layout_candidates(spec,materialized)
        assert views.to_dict()==public['views'].to_dict()
        assert materialized.to_dict()==public['materialized'].to_dict()
        assert same.to_dict()==spec.to_dict() and other.to_dict()==adf.to_dict()
        raw=w.execute_reference(spec)
    assert inputs==before and spec.params['inputs']==before
    close_values(raw['outputs']['Y'],expected_chain(spec)['output'],'bf16')


def fresh_reader(path,cwd):
    code=r'''
import io,json,sys,numpy,pytest
from pathlib import Path
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.simulator import api
from arch_explorer.simulator.events import engine
from arch_explorer.exploration import replay
from arch_explorer.cli import main
payload=json.loads(Path(sys.argv[1]).read_text())
def forbidden(*a,**kw):raise AssertionError('readonly view result executed')
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(ExplorationEnv,'__init__',forbidden)
    patch.setattr(api,'prepare_compiled',forbidden)
    patch.setattr(engine,'simulate',forbidden)
    patch.setattr(engine.ConsistentScheduler,'run',forbidden)
    if payload['schema']=='arch-explorer.compiled-result.v3':assert api.evaluate_request(payload)==payload
    else:assert replay(payload)==payload
    old=sys.stdout;stream=io.StringIO()
    try:
        sys.stdout=stream
        assert main(['evaluate',sys.argv[1]])==0
    finally:sys.stdout=old
    assert json.loads(stream.getvalue())==payload
print('fresh-view-readonly')
'''
    assert command([sys.executable,'-c',code,str(path)],cwd,timeout=600).stdout.strip()=='fresh-view-readonly'


def test_comparison_search_budget_failure_winner_readonly_and_explicit_rerun(view_api,monkeypatch,tmp_path):
    from arch_explorer.exploration import replay,rerun
    from view_storage_search_consumer import validate_saved
    retained=os.environ.get('VAL58_RETAINED_SEARCH_DIRECTORY')
    root=Path(retained).resolve() if retained else tmp_path/'search'
    if not retained:command([sys.executable,str(Path(__file__).with_name('view_storage_search_consumer.py')),str(root)],tmp_path,timeout=1200)
    before=files(root);result=validate_saved(root)
    with no_backend(monkeypatch):assert replay(result)==result
    fresh_reader(root/'BEST.json',tmp_path);assert files(root)==before
    renewed=rerun(result,output_dir=tmp_path/'rerun')
    assert renewed!=result and renewed['evaluations_used']==len(result['best']['workloads'])
    assert [r['evaluation'] for r in renewed['best']['workloads']]==[r['evaluation'] for r in result['best']['workloads']]
    assert files(root)==before


def test_view_Agent_discovery_exact_mapping_strict_export_and_one_execution_budget(view_api,monkeypatch,tmp_path):
    from arch_explorer.archs.multilayer import encoder_agent_config
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.agent_env.driver import run_agent
    from arch_explorer.exploration import replay
    config=encoder_agent_config(layers=2,dtype='fp16',candidate=1,layout='views')
    config['trace_path']=str(tmp_path/'tools.jsonl');config['agent']['log_path']=str(tmp_path/'agent.jsonl')
    env=ExplorationEnv(config);result=run_agent(config,env=env)
    path=tmp_path/'AGENT.json';path.write_text(json.dumps(result,indent=2)+'\n')
    assert result['evaluations_used']==env.used==1
    assert result['agent']['tool_calls']==result['agent']['steps']==3
    row=result['best']['workloads'][0];c=row['compiled_result']['request']['compilation']
    assert row['mapping']==c['source']['mapping']
    assert any(b['id']=='transpose.view' for b in row['mapping']['params']['operator_rules'].values())
    assert any(t['alias_of'] is not None for t in c['program']['tensors'])
    with no_backend(monkeypatch):
        assert replay(result)==result
        assert env.call('evaluate_request',{'payload':result})['result']==result
    assert env.used==1
    assert env.call('evaluate',{'optimize':'none'})['error']['code']=='evaluation_budget_exhausted'
    fresh_reader(path,tmp_path)


def test_documented_view_CLI_example_compile_execute_Agent_and_fresh_readonly(view_api,tmp_path):
    export=Path(os.environ['VAL58_SOURCE_EXPORT']).resolve();assert (export/'EXPORT_MANIFEST.json').is_file()
    source=tmp_path/'SOURCE.json';search=tmp_path/'SEARCH.json';agent=tmp_path/'AGENT.json'
    executable=str(Path(sys.executable).with_name('archexplorer'));compiled=tmp_path/'COMPILED.json'
    retained=os.environ.get('VAL58_RETAINED_CLI_DIRECTORY')
    if retained:
        original=Path(retained).resolve();before=files(original)
        compiled=original/'COMPILED.json';search=original/'SEARCH.json';saved=original/'RESULT.json'
        config=json.loads((original/'AGENT.json').read_text())
        config['trace_path']=str(tmp_path/'retained-cli-tools.jsonl')
        config['agent']['log_path']=str(tmp_path/'retained-cli-agent.jsonl')
        agent.write_text(json.dumps(config))
        (tmp_path/'RETAINED-CLI.json').write_text(json.dumps(dict(original=str(original),files=before,
            repeated_compile=False,repeated_prepare=False,repeated_component_execution=False),indent=2)+'\n')
    else:
        command([sys.executable,str(export/'examples/multilayer_encoder/scene.py'),'--layers','2','--dtype','bf16',
            '--layout','views','--single-layout-search','--roofline','--candidate','1','--output',str(source),
            '--search-config',str(search),'--agent-config',str(agent)],tmp_path)
        command([executable,'compile',str(source),'--output',str(compiled)],tmp_path,timeout=600)
        prepared=command([executable,'prepare',str(compiled),'--options',
            str(export/'examples/multilayer_encoder/provider-options.json'),'--output',str(tmp_path/'run')],tmp_path,timeout=600)
        request=tmp_path/'REQUEST.json';request.write_text(prepared.stdout)
        result=json.loads(command([executable,'evaluate',str(request)],tmp_path,timeout=600).stdout)
        saved=tmp_path/'RESULT.json';saved.write_text(json.dumps(result))
    assert json.loads(search.read_text())['max_evaluations']==6
    c=json.loads(compiled.read_text());assert any(t['alias_of'] for t in c['program']['tensors'])
    fresh_reader(saved,tmp_path)
    output=tmp_path/'CLI-AGENT.json'
    command([executable,'agent','run','--config',str(agent),'--output',str(output)],tmp_path,timeout=600)
    assert json.loads(output.read_text())['evaluations_used']==1
    fresh_reader(output,tmp_path)
    if retained:assert files(original)==before
