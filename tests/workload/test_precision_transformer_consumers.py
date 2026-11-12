"""Installed CORE27 API/CLI/Agent/search consumers, not source-only success."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from precision_transformer_expectations import frozen_block_reference,store
from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import no_backend,files
from test_operator_mapping_extension_contract import reseal
from test_precision_transformer_contract import precision_api,close_values


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('case',['regular','boundary','constant'])
def test_public_scene_inputs_and_raw_block_reference(precision_api,monkeypatch,dtype,case):
    from arch_explorer.archs.transformer import block_inputs,block_scene,block_workload
    w,_=precision_api
    with no_backend(monkeypatch):
        inputs=block_inputs(input_case=case)
        assert inputs==block_inputs(input_case=case)
        copied=deepcopy(inputs)
        spec=block_workload(dtype=dtype,inputs=inputs)
        inputs['X'][0][0][0]+=1
        assert spec.params['inputs']==copied
        adf,scene,mapping=block_scene(dtype=dtype,input_case=case)
        assert scene.kind=='transformer_block' and scene.params['inputs']==copied
        assert set(mapping.params)=={'compute','memory','rules'}
        assert all('threads' not in key for key in mapping.params)
        expected=frozen_block_reference(scene.params)
        reference=w.execute_reference(scene)
    assert reference['outputs']['Y'].shape==(2,3,8)
    close_values(reference['outputs']['Y'],expected['output'],dtype)
    if case=='boundary' and dtype!='fp32':
        assert any(store(value,dtype)!=value for batch in copied['X'] for row in batch for value in row)


def test_existing_search_observes_block_mapping_parameter_failures_and_fresh_replay(
        precision_api,monkeypatch,tmp_path):
    from arch_explorer.exploration import replay,rerun
    observer=Path(__file__).with_name('precision_transformer_search_consumer.py')
    directory=tmp_path/'bf16-search'
    command([sys.executable,str(observer),str(directory),'bf16'],tmp_path)
    payload=json.loads((directory/'BEST.json').read_text())
    assert payload['evaluations_used']<=6
    before=files(directory)
    code=r'''
from pathlib import Path
import io,json,sys,numpy,pytest
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.simulator import api
from arch_explorer.simulator.events import engine
from arch_explorer.exploration import replay
def forbidden(*args,**kwargs):raise AssertionError('readonly block replay executed')
payload=json.loads(Path(sys.argv[1]).read_text())
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(ExplorationEnv,'__init__',forbidden)
    patch.setattr(api,'prepare_compiled',forbidden)
    patch.setattr(engine,'simulate',forbidden)
    patch.setattr(engine.ConsistentScheduler,'run',forbidden)
    assert replay(payload)==payload
    from arch_explorer.cli import main
    stream=io.StringIO()
    original=sys.stdout
    try:
        sys.stdout=stream
        assert main(['evaluate',sys.argv[1]])==0
    finally:sys.stdout=original
    assert json.loads(stream.getvalue())==payload
print('block-fresh-readonly')
'''
    assert command([sys.executable,'-c',code,str(directory/'BEST.json')],tmp_path).stdout.strip()=='block-fresh-readonly'
    with no_backend(monkeypatch):
        assert replay(payload)==payload
        for change in ('objective','dtype','evaluation','effective_mapping'):
            bad=deepcopy(payload)
            row=bad['best']['workloads'][0]
            if change=='objective':bad['best']['score']['objective']+=1
            elif change=='dtype':row['compiled_result']['request']['compilation']['source']['workload']['params']['dtype']='fp32'
            elif change=='evaluation':row['evaluation']['latency_ns']+=1
            else:row['mapping']['params']['rules']['matmul']['options']['tiles']['k']+=1
            with pytest.raises((ValueError,TypeError,KeyError)):
                replay(reseal(bad))
    assert files(directory)==before
    # Explicit rerun is visibly different and charges a fresh bounded execution.
    renewed=rerun(payload,output_dir=tmp_path/'explicit-rerun')
    assert renewed!=payload and renewed['evaluations_used']==len(payload['best']['workloads'])
    assert [r['evaluation'] for r in renewed['best']['workloads']]==[r['evaluation'] for r in payload['best']['workloads']]
    assert files(directory)==before


def test_default_block_agent_exports_sealed_result_and_charges_only_execution(
        precision_api,monkeypatch,tmp_path):
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.agent_env.driver import run_agent
    from arch_explorer.archs.transformer import block_agent_config
    from arch_explorer.exploration import replay
    config=block_agent_config(dtype='fp16',input_case='boundary')
    config['trace_path']=str(tmp_path/'tools.jsonl')
    config['agent']['log_path']=str(tmp_path/'agent.jsonl')
    assert config['max_evaluations']==1 and len(config['agent']['script'])==3
    (tmp_path/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    env=ExplorationEnv(config)
    payload=run_agent(config,env=env)
    assert payload['schema']=='archexplorer.compilation-search-result.v1'
    assert payload['evaluations_used']==env.used==1
    assert payload['agent']['steps']==payload['agent']['tool_calls']==3
    body={k:v for k,v in payload.items() if k!='binding'}
    encoded=json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    assert payload['binding']=={'sha256':hashlib.sha256(encoded).hexdigest()}
    target=tmp_path/'AGENT-RESULT.json';target.write_text(json.dumps(payload,indent=2)+'\n')
    before=files(tmp_path)
    with no_backend(monkeypatch):
        assert replay(payload)==payload
        result=env.call('evaluate_request',{'payload':payload})
        assert 'error' not in result and result['result']==payload
    assert env.used==1
    audit={'tools.jsonl','agent.jsonl'}
    assert {k:v for k,v in files(tmp_path).items() if k not in audit}=={k:v for k,v in before.items() if k not in audit}
    executable=str(Path(sys.executable).with_name('archexplorer'))
    assert json.loads(command([executable,'evaluate',str(target)],tmp_path).stdout)==payload
    assert env.call('evaluate',{'optimize':'none'})['error']['code']=='evaluation_budget_exhausted'


def test_exported_documentation_scene_commands_use_installed_main(precision_api,tmp_path):
    # The runner supplies the exact lead source-export after verifying its
    # manifest. Never silently fall back to a mutable development checkout.
    export=Path(os.environ['VAL56_SOURCE_EXPORT']).resolve()
    assert (export/'EXPORT_MANIFEST.json').is_file()
    script=export/'examples/transformer_block/scene.py'
    assert script.is_file()
    specs=tmp_path/'SPECS.json';search=tmp_path/'SEARCH.json';agent=tmp_path/'AGENT.json'
    command([sys.executable,str(script),'--dtype','fp16','--input-case','boundary',
             '--output',str(specs),'--search-config',str(search),'--agent-config',str(agent)],tmp_path)
    document=json.loads(specs.read_text())
    assert document['workload']['kind']=='transformer_block'
    assert document['workload']['params']['dtype']=='fp16'
    executable=str(Path(sys.executable).with_name('archexplorer'))
    compilation=tmp_path/'COMPILATION.json'
    command([executable,'compile',str(specs),'--output',str(compilation)],tmp_path)
    request=json.loads(command([executable,'prepare',str(compilation),'--output',str(tmp_path/'run')],tmp_path).stdout)
    request_file=tmp_path/'REQUEST.json';request_file.write_text(json.dumps(request,indent=2))
    saved=json.loads(command([executable,'evaluate',str(request_file)],tmp_path).stdout)
    saved_file=tmp_path/'RESULT.json';saved_file.write_text(json.dumps(saved,indent=2))
    assert json.loads(command([executable,'evaluate',str(saved_file)],tmp_path).stdout)==saved
    agent_result=tmp_path/'CLI-AGENT-RESULT.json'
    command([executable,'agent','run','--config',str(agent),'--output',str(agent_result)],tmp_path)
    payload=json.loads(agent_result.read_text())
    assert payload['evaluations_used']==1 and payload['agent']['stop_reason']=='submitted'
    assert json.loads(command([executable,'evaluate',str(agent_result)],tmp_path).stdout)==payload
