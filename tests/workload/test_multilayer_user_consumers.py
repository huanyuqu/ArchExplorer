"""CORE28 actual installed API/CLI/Agent/search consumers, candidate-gated."""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import pytest

from multilayer_contract_cases import composition_api
from multilayer_expectations import reference_for
from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import no_backend,files
from test_operator_mapping_extension_contract import reseal
from test_precision_transformer_contract import close_values


def expected_chain(spec):
    p=spec.params
    return reference_for(dict(config={k:p[k] for k in
        ('batch','sequence','hidden','heads','mlp_hidden','epsilon')},dtype=p['dtype'],X=p['inputs']['X'],
        instances=[dict(id='layer'+str(i),parameters=value) for i,value in enumerate(p['inputs']['layers'])]))


@pytest.mark.parametrize('layers',[2,4])
@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_core_two_candidate_scenes_have_distinct_real_instance_placement_and_tiles(
        composition_api,monkeypatch,layers,dtype):
    from arch_explorer.archs.multilayer import encoder_inputs,encoder_scene
    w,_=composition_api;inputs=encoder_inputs(layers=layers,input_case='boundary');before=deepcopy(inputs)
    mappings=[]
    for candidate in (0,1):
        with no_backend(monkeypatch):
            adf,spec,mapping=encoder_scene(layers=layers,dtype=dtype,inputs=inputs,candidate=candidate)
            graph=w.workload_from_spec(spec)
            raw=w.execute_reference(spec)
        assert inputs==before and spec.params['inputs']==before
        assert mapping.template=='compile_graph_v3'
        assert set(mapping.params)=={'compute','memory','rules','operator_rules'}
        assert set(mapping.params['compute'])=={op.name for op in graph.operators}
        operators={op.name:op for op in graph.operators}
        for index,row in enumerate(graph.meta['composition']['instances']):
            position=(index+candidate)%2
            expected_path='board/core/'+('sm_peer' if position else 'sm')
            tiles=[dict(m=2,k=3,n=5),dict(m=3,k=5,n=3)][position]
            for name in row['operator_names'].values():
                assert mapping.params['compute'][name]==expected_path
                if operators[name].op=='matmul':assert mapping.params['operator_rules'][name]['options']['tiles']==tiles
        close_values(raw['outputs']['Y'],expected_chain(spec)['output'],dtype)
        mappings.append(mapping.to_dict())
    assert mappings[0]!=mappings[1]
    inputs['layers'][0]['b2'][0]+=1
    assert spec.params['inputs']==before


def test_existing_search_budget_instance_winner_fresh_readonly_and_explicit_rerun(
        composition_api,monkeypatch,tmp_path):
    from arch_explorer.exploration import replay,rerun
    observer=Path(__file__).with_name('multilayer_search_consumer.py')
    directory=tmp_path/'search'
    retained=os.environ.get('VAL57_RETAINED_SEARCH_DIRECTORY')
    if retained:
        from multilayer_search_consumer import validate_saved_search
        directory=Path(retained).resolve()
        original=files(directory)
        validate_saved_search(directory,observation_dir=tmp_path)
        assert files(directory)==original
        (tmp_path/'RETAINED-SEARCH.json').write_text(json.dumps(dict(
            path=str(directory),original_files=original,new_search_execution=False),indent=2))
    else:command([sys.executable,str(observer),str(directory)],tmp_path,timeout=600)
    payload=json.loads((directory/'BEST.json').read_text());before=files(directory)
    code=r'''
import io,json,sys,numpy,pytest
from pathlib import Path
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.simulator import api
from arch_explorer.simulator.events import engine
from arch_explorer.exploration import replay
from arch_explorer.cli import main
payload=json.loads(Path(sys.argv[1]).read_text())
def forbidden(*args,**kwargs):raise AssertionError('readonly multi-layer result executed')
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(ExplorationEnv,'__init__',forbidden)
    patch.setattr(api,'prepare_compiled',forbidden)
    patch.setattr(engine,'simulate',forbidden)
    patch.setattr(engine.ConsistentScheduler,'run',forbidden)
    assert replay(payload)==payload
    original=sys.stdout;stream=io.StringIO()
    try:
        sys.stdout=stream
        assert main(['evaluate',sys.argv[1]])==0
    finally:sys.stdout=original
    assert json.loads(stream.getvalue())==payload
print('fresh-multilayer-readonly')
'''
    assert command([sys.executable,'-c',code,str(directory/'BEST.json')],tmp_path,timeout=600).stdout.strip()=='fresh-multilayer-readonly'
    with no_backend(monkeypatch):
        assert replay(payload)==payload
        for mutation in ('objective','instance-rule','evaluation'):
            bad=deepcopy(payload)
            if mutation=='objective':bad['best']['score']['objective']+=1
            elif mutation=='evaluation':bad['best']['workloads'][0]['evaluation']['latency_ns']+=1
            else:
                params=bad['best']['workloads'][0]['mapping']['params']['operator_rules']
                name=next(key for key,row in params.items() if row['id']=='matmul.tiled')
                params[name]['options']['tiles']['k']+=1
            with pytest.raises((ValueError,TypeError,KeyError)):replay(reseal(bad))
    assert files(directory)==before
    renewed=rerun(payload,output_dir=tmp_path/'rerun')
    assert renewed!=payload and renewed['evaluations_used']==len(payload['best']['workloads'])
    assert [row['evaluation'] for row in renewed['best']['workloads']]==[row['evaluation'] for row in payload['best']['workloads']]
    assert files(directory)==before


def test_offline_agent_preserves_full_mapping_and_only_charges_execution(
        composition_api,monkeypatch,tmp_path):
    from arch_explorer.archs.multilayer import encoder_agent_config
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.agent_env.driver import run_agent
    from arch_explorer.exploration import replay
    config=encoder_agent_config(layers=2,dtype='fp16',input_case='boundary',candidate=1)
    config['trace_path']=str(tmp_path/'tools.jsonl');config['agent']['log_path']=str(tmp_path/'agent.jsonl')
    assert config['max_evaluations']==1 and len(config['agent']['script'])==3
    env=ExplorationEnv(config);result=run_agent(config,env=env)
    assert result['evaluations_used']==env.used==1
    assert result['agent']['steps']==result['agent']['tool_calls']==3
    body={key:value for key,value in result.items() if key!='binding'}
    encoded=json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    assert result['binding']=={'sha256':hashlib.sha256(encoded).hexdigest()}
    row=result['best']['workloads'][0]
    assert row['mapping']==row['compiled_result']['request']['compilation']['source']['mapping']
    assert row['mapping']['params']['operator_rules']
    with no_backend(monkeypatch):
        assert replay(result)==result
        read=env.call('evaluate_request',{'payload':result})
        assert 'error' not in read and read['result']==result
    assert env.used==1
    assert env.call('evaluate',{'optimize':'none'})['error']['code']=='evaluation_budget_exhausted'
    (tmp_path/'AGENT-RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


def test_documented_cli_scene_compile_prepare_execute_and_agent(composition_api,tmp_path):
    export=Path(os.environ['VAL57_SOURCE_EXPORT']).resolve()
    assert (export/'EXPORT_MANIFEST.json').is_file()
    specs=tmp_path/'SOURCE.json';search=tmp_path/'SEARCH.json';agent=tmp_path/'AGENT.json'
    script=export/'examples/multilayer_encoder/scene.py'
    command([sys.executable,str(script),'--layers','2','--dtype','bf16','--input-case','boundary',
        '--candidate','1','--roofline','--output',str(specs),'--search-config',str(search),
        '--agent-config',str(agent)],tmp_path)
    document=json.loads(specs.read_text())
    assert document['workload']['kind']=='transformer_encoder'
    assert document['mapping']['template']=='compile_graph_v3'
    executable=str(Path(sys.executable).with_name('archexplorer'))
    compilation=tmp_path/'COMPILATION.json'
    command([executable,'compile',str(specs),'--output',str(compilation)],tmp_path,timeout=600)
    options=export/'examples/multilayer_encoder/provider-options.json'
    request=json.loads(command([executable,'prepare',str(compilation),'--options',str(options),
        '--output',str(tmp_path/'run')],tmp_path,timeout=600).stdout)
    path=tmp_path/'REQUEST.json';path.write_text(json.dumps(request))
    result=json.loads(command([executable,'evaluate',str(path)],tmp_path,timeout=600).stdout)
    saved=tmp_path/'RESULT.json';saved.write_text(json.dumps(result))
    assert json.loads(command([executable,'evaluate',str(saved)],tmp_path,timeout=600).stdout)==result
    agent_result=tmp_path/'CLI-AGENT.json'
    command([executable,'agent','run','--config',str(agent),'--output',str(agent_result)],tmp_path,timeout=600)
    payload=json.loads(agent_result.read_text());assert payload['evaluations_used']==1
    assert json.loads(command([executable,'evaluate',str(agent_result)],tmp_path,timeout=600).stdout)==payload
