"""Candidate-only API/CLI/Agent/search/winner/fresh/rerun CORE32 consumers."""
from copy import deepcopy
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import pytest
from mla_absorbed_expectations import GEOMETRY,root_shapes
from mla_contract_cases import REJECT,compare_stages,frozen_inputs,mla_api
from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import files,no_backend
from test_vit_user_consumers import fresh_readonly


def core():
    module=importlib.import_module('arch_explorer.archs.mla')
    for name in ('mla_inputs','mla_workload','mla_scene','mla_search_config','mla_agent_config'):
        assert callable(getattr(module,name)),name
    return module


def test_CORE32_exact_roots_placements_replicas_and_complete_hardware(mla_api,monkeypatch):
    w,s=mla_api;c=core();params=deepcopy(frozen_inputs()['workload']['params']);before=deepcopy(params)
    with no_backend(monkeypatch):
        spec=c.mla_workload(**params);assert spec.params==params
        graph=w.workload_from_spec(spec)
        roots=c.mla_inputs(**{k:params[k] for k in GEOMETRY})
        assert set(roots)==set(root_shapes(params))
        generated=deepcopy(params);generated['inputs']=roots
        assert len(generated['inputs']['q_gamma'])==params['query_rank']
        assert all(math.isfinite(value) for value in generated['inputs']['q_gamma'])
        for shared in (True,False):
            for candidate in (0,1):
                adf,same,mapping=c.mla_scene(**params,shared=shared,candidate=candidate,layout='views')
                assert same.params==params and mapping.template=='compile_graph_v4'
                operators={op.name:op for op in graph.operators}
                assert set(mapping.params['compute'])==set(mapping.params['memory'])==set(operators)
                assert set(mapping.params['storage'])==set(params['inputs'])
                selected=set()
                for replica in mapping.params['replicas']:
                    source=replica['source']
                    actual_readers={op.name for op in graph.operators if source in op.inputs}
                    readers=[name for dest in replica['destinations'] for name in dest['consumers']]
                    assert set(readers)==actual_readers and len(readers)==len(set(readers))
                    selected.add(source)
                assert selected=={'latent_kv','pos_key'}
        config=c.mla_search_config(**params)
        assert config['max_evaluations']==12 and config['mapping_samples']==4
        requests=config['mapping_selectors']['compile_graph_v4']
        assert len(requests)==4 and len({json.dumps(r,sort_keys=True) for r in requests})==4
        agent=c.mla_agent_config(**params,candidate=0,layout='views')
        assert agent['max_evaluations']==1
        assert [row['tool'] for row in agent['agent']['script']]==['operator_mappings','evaluate','submit']
        for hardware in ({'groups':2},{'groups':2,'tiles_per_group':2,'capacity':32768,
                          'link_bandwidth':16,'port_bandwidth':16,'hop_latency':1,'shared':True}):
            with pytest.raises(REJECT):c.mla_scene(**params,hardware=hardware)
    assert params==before
    # Actual CPU reference comparison remains separate from component timings.
    compare_stages(w.execute_reference(c.mla_workload(**generated)),generated)


def test_CORE32_inputs_are_copied_without_filling_or_dtype_cast(mla_api,monkeypatch):
    w,s=mla_api;c=core();params=deepcopy(frozen_inputs()['workload']['params'])
    with no_backend(monkeypatch):
        spec=c.mla_workload(**params);params['inputs']['X'][0][0]+=1
        assert spec.params['inputs']!=params['inputs']
        bad=deepcopy(spec.params);del bad['inputs']['head0.WVO']
        with pytest.raises(REJECT):c.mla_workload(**bad)
        for dtype in ('fp16','bf16'):
            with pytest.raises(REJECT):c.mla_workload(**dict(spec.params,dtype=dtype))


def test_MLA_failed_request_consumes_budget_and_exhaustion_refuses_engine(mla_api,monkeypatch,tmp_path):
    from arch_explorer.agent_env import ExplorationEnv
    c=core();_,_,mapping=c.mla_scene(candidate=0,layout='views')
    config=c.mla_search_config(candidate=0,layout='views')
    config.update(max_evaluations=1,trace_path=str(tmp_path/'failed-budget.jsonl'))
    env=ExplorationEnv(config)
    with no_backend(monkeypatch):
        compiled=env.call('compile_workload',{'mapping':mapping.to_dict(),'workload_index':0})
        assert 'error' not in compiled
        request=env.call('prepare_compiled',{'compilation':compiled['result'],
            'output_dir':str(tmp_path/'request'),'options':{'trace':True}})
        assert 'error' not in request and env.used==0
        bad=deepcopy(request['result']);bad['binding']['sha256']='0'*64
        refused=env.call('evaluate_request',{'payload':bad})
        assert 'error' in refused and env.used==1 and env.remaining==0
        exhausted=env.call('evaluate_request',{'payload':request['result']})
        assert exhausted['error']['code']=='evaluation_budget_exhausted' and env.used==1


def test_installed_MLA_public_search_records_actual_failures_and_executed_winner(tmp_path):
    directory=tmp_path/'search'
    command([sys.executable,str(Path(__file__).with_name('mla_search_consumer.py')),str(directory)],tmp_path,timeout=120)
    best=json.loads((directory/'BEST.json').read_text())
    assert best['best']['workloads'][0]['workload']['kind']=='mla_decode'
    fresh_readonly(directory/'BEST.json',tmp_path)


def test_offline_MLA_Agent_executes_one_exact_mapping_and_submits(mla_api,tmp_path):
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.agent_env.driver import run_agent
    c=core();config=c.mla_agent_config(candidate=0,layout='views')
    config['trace_path']=str(tmp_path/'agent-tools.jsonl')
    env=ExplorationEnv(config);result=run_agent(env=env,policy='scripted')
    assert result['evaluations_used']==env.used==1
    row=result['best']['workloads'][0]
    compiled=row['compiled_result']['request']['compilation']
    assert row['mapping']==compiled['source']['mapping']
    assert compiled['program']['meta']['outputs']==['Y']
    path=tmp_path/'AGENT-RESULT.json';path.write_text(json.dumps(result,indent=2)+'\n')
    fresh_readonly(path,tmp_path)


def test_exported_MLA_example_real_CLI_compile_prepare_evaluate_and_fresh(mla_api,tmp_path):
    export=Path(os.environ['VAL61_SOURCE_EXPORT']);example=tmp_path/'example'
    shutil.copytree(export/'examples/workload_scenarios',example)
    output=tmp_path/'generated'
    command([sys.executable,str(example/'mla.py'),'--sharing','grouped','--multicast','on','--output',str(output)],tmp_path,timeout=120)
    scene=json.loads((output/'SCENE.json').read_text())
    assert scene['workload']['kind']=='mla_decode'
    cli=str(Path(sys.executable).with_name('archexplorer'))
    compiled=tmp_path/'compiled.json'
    command([cli,'compile',str(output/'SCENE.json'),'--output',str(compiled)],tmp_path,timeout=120)
    prepared=command([cli,'prepare',str(compiled),'--output',str(tmp_path/'new-request')],tmp_path,timeout=120)
    request=tmp_path/'request.json';request.write_text(prepared.stdout)
    result=tmp_path/'RESULT.json'
    command([cli,'evaluate',str(request),'--output',str(result)],tmp_path,timeout=120)
    saved=json.loads(result.read_text());assert saved['schema']=='arch-explorer.compiled-result.v3'
    assert saved['evaluation']['latency_ns']>0
    probe=Path(__file__).with_name('mla_readonly_probe.py')
    command([sys.executable,str(probe),str(result),str(tmp_path/'READONLY.json')],tmp_path,timeout=120)
    facts=json.loads((tmp_path/'READONLY.json').read_text())
    assert facts['prepare_engine_solver_native_writes_forbidden'] and facts['unchanged']


def test_explicit_MLA_rerun_is_a_new_directory_and_preserves_original(tmp_path):
    from arch_explorer.exploration import rerun
    import hashlib
    path=Path(os.environ['VAL61_SEARCH_RESULT']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL61_SEARCH_RESULT_SHA256']
    old=json.loads(raw);before=files(path.parent);directory=tmp_path/'explicit-rerun'
    new=rerun(old,output_dir=directory)
    assert new['evaluations_used']==1 and files(directory)
    request=new['best']['workloads'][0]['compiled_result']['request']
    assert str(directory) in json.dumps(request) and str(directory) not in raw.decode()
    assert files(path.parent)==before


def test_saved_MLA_sources_and_storage_reject_resealed_changes(mla_api,monkeypatch):
    from test_operator_mapping_extension_contract import reseal
    w,s=mla_api
    path=Path(os.environ['VAL61_PHYSICAL_FACTS']).with_name('RESULT.json')
    saved=json.loads(path.read_text())
    with no_backend(monkeypatch):
        assert s.evaluate_request(saved)==saved
        for field in ('source-file','physical-location'):
            bad=deepcopy(saved['request']['compilation'])
            if field=='source-file':
                key='mapping/storage_plan.py'
                assert key in bad['sources']['files']
                bad['sources']['files'][key]='0'*64
            else:
                bad['program']['meta']['storage_plan']['locations']['latent_kv']='missing/physical/root'
            bad=reseal(bad)
            with pytest.raises(REJECT):w.validate_compilation(bad)
            request=deepcopy(saved['request']);request['compilation']=bad
            with pytest.raises(REJECT):s.evaluate_request(reseal(request))


def test_installed_unknown_FP32_extension_after_MLA_keeps_two_input_reads(mla_api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec,MappingSpec
    from ae_operator_mappings import semantics,mapping as extension
    from test_operator_mapping_attention_contract import node_spec,primitive_scene
    from precision_transformer_expectations import f32,store
    from mla_absorbed_expectations import absorbed_decode
    w,s=mla_api
    assert 'site-packages' in semantics.__file__
    saved=json.loads(Path(os.environ['VAL61_PHYSICAL_FACTS']).with_name('RESULT.json').read_text())
    params=saved['request']['compilation']['source']['workload']['params']
    with no_backend(monkeypatch):
        actual=w.execute_reference(WorkloadSpec('mla_decode',params))['outputs']['Y']
        assert actual.reshape(-1).tolist()==pytest.approx(absorbed_decode(params)['output'][0],abs=2e-6,rel=2e-6)
        shape=actual.shape;values=actual.tolist()
        spec=node_spec(semantics.KIND,[shape],shape,{},[values]);adf,spec,seed=primitive_scene(spec,'events')
        rule=w.MappingRuleBinding(extension.MATERIALIZED.id,'1').to_dict()
        mapping=MappingSpec('compile_graph_v3',dict(seed.params,rules={semantics.KIND:rule},operator_rules={}))
        compiled=w.compile_workload(adf,spec,mapping)
        output=w.execute_reference(WorkloadSpec('graph',compiled['program']))['outputs']['y']
        assert output.tolist()==[[store(f32(v*v),'fp32') for v in row] for row in values]
        assert sum(row['access']['read_bytes'] for row in compiled['plan']['steps'])==actual.size*8
        assert sum(row['access']['write_bytes'] for row in compiled['plan']['steps'])==actual.size*4
        request=s.prepare_compiled(compiled,output_dir=tmp_path/'square',options={'trace':True})
        for dtype in ('fp16','bf16'):
            bad=deepcopy(spec.params)
            for tensor in bad['tensors']:tensor['dtype']=dtype
            with pytest.raises(REJECT):w.compile_workload(adf,WorkloadSpec('graph',bad),mapping)
    result=s.evaluate_request(request)
    with no_backend(monkeypatch):assert s.evaluate_request(result)==result


@pytest.mark.parametrize('model,hidden,heads,kv_heads',[
    ('llama2-7b',4096,32,32),('llama2-70b',8192,64,8),
    ('llama3-70b',8192,64,8),('qwen-72b',8192,64,64)])
@pytest.mark.parametrize('phase',['prefill','decode'])
def test_CORE32_LLM_wrapper_keeps_explicit_shapes_without_weights(mla_api,monkeypatch,model,hidden,heads,kv_heads,phase):
    from arch_explorer.archs.llm_scenarios import llm_workload
    w,s=mla_api
    with no_backend(monkeypatch):
        spec=llm_workload(model=model,phase=phase,batch=2,seq_len=3,layers=1,dtype='fp16',iterations=1)
        graph=w.workload_from_spec(spec)
        assert spec.kind=='transformer' and graph.meta['model']['source']
        assert graph.tensors['x0'].shape==(6 if phase=='prefill' else 2,hidden)
        assert graph.tensors['l0.w_qkv'].shape==(hidden,(heads+2*kv_heads)*128)
        assert graph.meta['layers']==1 and graph.meta['seq_len']==3
        assert 'inputs' not in graph.meta
        with pytest.raises(ValueError):
            llm_workload(model=model,phase=phase,batch=True,seq_len=3,layers=1,dtype='fp16',iterations=1)
