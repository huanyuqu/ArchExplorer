"""Deferred CORE33 API/CLI/Env/Agent/search/fresh/rerun, only sole installation."""
from copy import deepcopy
import hashlib,importlib,json,os,shutil,sys
from pathlib import Path
import pytest
from onnx_contract_cases import formal_spec,formal_input
from test_onnx_import_contract import onnx_api
from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import no_backend,files
from test_operator_mapping_extension_contract import reseal

def core():return importlib.import_module('arch_explorer.archs.onnx_compilation')

def fresh(path,directory):
    receipt=Path(directory)/'FRESH.json'
    command([sys.executable,str(Path(__file__).with_name('onnx_readonly_probe.py')),str(path),str(receipt)],directory)
    facts=json.loads(receipt.read_text())
    assert facts['unchanged'] and facts['prepare_engine_solver_native_network_writes_forbidden']
    assert facts['positive_controls']
    return facts

@pytest.mark.parametrize('key',['mlp','attention'])
def test_CORE33_spec_copy_all_operators_two_mappings_and_layouts(onnx_api,monkeypatch,key):
    module,w,s=onnx_api;c=core();path,options,_=formal_input(key)
    before=deepcopy(options)
    with no_backend(monkeypatch):
        spec=c.onnx_workload(path,**options);assert options==before
        graph=w.workload_from_spec(spec)
        for index in (0,1):
            for policy in ('materialized','views'):
                adf,same,mapping=c.onnx_scene(spec,candidate=index,layout=policy)
                assert same.to_dict()==spec.to_dict()
                assert set(mapping.params['compute'])=={op.name for op in graph.operators}
                assert len(set(mapping.params['compute'].values()))==2
        config=c.onnx_search_config(spec)
        assert config['max_evaluations']==12 and config['mapping_samples']==4
        selectors=config['mapping_selectors']['compile_graph_v3']
        assert len(selectors)==len({json.dumps(v,sort_keys=True) for v in selectors})==4
        agent=c.onnx_agent_config(spec,candidate=0,layout='views')
        assert agent['max_evaluations']==1
        assert [v['tool'] for v in agent['agent']['script']]==['operator_mappings','evaluate','submit']
        # A caller mutation must not rewrite the retained model roots.
        options['inputs'].clear();assert spec.params['inputs']

def test_Env_failure_cost_and_exhaustion_before_execution(onnx_api,monkeypatch,tmp_path):
    from arch_explorer.agent_env import ExplorationEnv
    spec,_=formal_spec('mlp');c=core();_,_,mapping=c.onnx_scene(spec,candidate=0,layout='views')
    config=c.onnx_search_config(spec,candidate=0,layout='views')
    config.update(max_evaluations=1,trace_path=str(tmp_path/'budget.jsonl'));env=ExplorationEnv(config)
    with no_backend(monkeypatch):
        compiled=env.call('compile_workload',{'mapping':mapping.to_dict(),'workload_index':0});assert 'error' not in compiled
        request=env.call('prepare_compiled',{'compilation':compiled['result'],'output_dir':str(tmp_path/'prepare'),'options':{'trace':True}})
        assert 'error' not in request and env.used==0
        bad=deepcopy(request['result']);bad['binding']['sha256']='0'*64
        rejected=env.call('evaluate_request',{'payload':bad});assert 'error' in rejected and env.used==1
        exhausted=env.call('evaluate_request',{'payload':request['result']})
        assert exhausted['error']['code']=='evaluation_budget_exhausted' and env.remaining==0

@pytest.mark.parametrize('key',['mlp','attention'])
def test_public_search_executes_winner_records_failure_and_fresh_replays(tmp_path,key):
    directory=tmp_path/'search'
    command([sys.executable,str(Path(__file__).with_name('onnx_search_consumer.py')),str(directory),key],tmp_path)
    saved=json.loads((directory/'BEST.json').read_text());assert saved['best']['workloads'][0]['workload']['kind']=='onnx'
    fresh(directory/'BEST.json',tmp_path)

def test_offline_Agent_one_actual_evaluation_then_fresh(onnx_api,tmp_path):
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.agent_env.driver import run_agent
    spec,_=formal_spec('attention');config=core().onnx_agent_config(spec,candidate=1,layout='views')
    config['trace_path']=str(tmp_path/'agent.jsonl');env=ExplorationEnv(config)
    result=run_agent(env=env,policy='scripted');assert result['evaluations_used']==env.used==1
    item=result['best']['workloads'][0]
    assert item['workload']['kind']=='onnx' and item['mapping']==item['compiled_result']['request']['compilation']['source']['mapping']
    path=tmp_path/'AGENT.json';path.write_text(json.dumps(result,indent=2)+'\n');fresh(path,tmp_path)

def test_exported_example_import_compile_prepare_evaluate_from_other_cwd(onnx_api,tmp_path):
    export=Path(os.environ['VAL62_SOURCE_EXPORT']);example=tmp_path/'example'
    shutil.copytree(export/'examples/onnx_import',example)
    generated=tmp_path/'models'
    command([sys.executable,str(example/'models.py'),'--output',str(generated),'--external-data','--symbolic-batch'],tmp_path)
    model=generated/'mlp/model.onnx';values=generated/'mlp/INPUTS.json';shapes=generated/'mlp/INPUT-SHAPES.json'
    cli=str(Path(sys.executable).with_name('archexplorer'));workload=tmp_path/'WORKLOAD.json'
    args=[str(model),'--inputs',str(values),'--input-shapes',str(shapes),'--external-root',str(model.parent)]
    command([cli,'import-onnx',*args,'--output',str(workload)],tmp_path)
    assert json.loads(workload.read_text())['kind']=='onnx'
    emitted=tmp_path/'emitted';command([sys.executable,str(example/'emit.py'),*args,'--output',str(emitted)],tmp_path)
    assert json.loads((emitted/'WORKLOAD.json').read_text())==json.loads(workload.read_text())
    compiled=tmp_path/'COMPILED.json';command([cli,'compile',str(emitted/'SCENE.json'),'--output',str(compiled)],tmp_path)
    prepared=command([cli,'prepare',str(compiled),'--output',str(tmp_path/'prepare')],tmp_path)
    request=tmp_path/'REQUEST.json';request.write_text(prepared.stdout)
    result=tmp_path/'RESULT.json';command([cli,'evaluate',str(request),'--output',str(result)],tmp_path)
    assert json.loads(result.read_text())['evaluation']['latency_ns']>0
    fresh(result,tmp_path)

@pytest.mark.parametrize('target',['source-file','model-digest','external-bytes','root','mapped-operator','mapping-tiles'])
def test_saved_sources_roots_plan_and_mapping_reseal_are_refused(onnx_api,monkeypatch,tmp_path,target):
    module,w,s=onnx_api;path=Path(os.environ['VAL62_COMPILED_RESULT']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL62_COMPILED_RESULT_SHA256']
    saved=json.loads(raw)
    diagnostics=[]
    mismatch='compilation source, mapping, adapter, options, component or product differs'
    expected_errors={'source-file':mismatch,'model-digest':'ONNX source digest/encoding mismatch',
        'external-bytes':'ONNX source digest/encoding mismatch',
        'root':'ONNX root X .*finite FP32 numbers','mapped-operator':mismatch,'mapping-tiles':mismatch}
    with no_backend(monkeypatch):
        assert s.evaluate_request(saved)==saved
        for field in (target,):
            bad=deepcopy(saved['request']['compilation'])
            if field=='source-file':
                key=next(k for k in bad['sources']['files'] if 'onnx' in k)
                bad['sources']['files'][key]='0'*64
            elif field=='model-digest':bad['source']['workload']['params']['source']['model']['sha256']='0'*64
            elif field=='external-bytes':
                import base64
                entry=next(iter(bad['source']['workload']['params']['source']['external'].values()))
                value=bytearray(base64.b64decode(entry['base64']));value[-1]^=1
                entry['base64']=base64.b64encode(value).decode()
            elif field=='root':
                import re
                values=bad['source']['workload']['params']['inputs'];name=next(iter(values));values[name]=[0.]
                expected_errors[field]='ONNX root '+re.escape(name)+' .*finite FP32 numbers'
            elif field=='mapping-tiles':
                rules=bad['source']['mapping']['params']['operator_rules']
                selected=next(v for v in rules.values() if v['id']=='matmul.tiled')
                selected['options']['tiles']['m']+=1
            else:bad['program']['operators'][0]['attrs']['VAL62_invalid']=True
            with pytest.raises(ValueError,match=expected_errors[field]) as compile_error:
                w.validate_compilation(reseal(bad))
            request=deepcopy(saved['request']);request['compilation']=reseal(bad)
            with pytest.raises(ValueError,match=expected_errors[field]) as request_error:
                s.evaluate_request(reseal(request))
            diagnostics.append(dict(field=field,type='ValueError',compile=str(compile_error.value),request=str(request_error.value)))
    (tmp_path/'TAMPER.json').write_text(json.dumps(diagnostics,indent=2)+'\n')

def test_explicit_rerun_preserves_original_search_bytes_and_writes_new_directory(tmp_path):
    from arch_explorer.exploration import rerun
    path=Path(os.environ['VAL62_SEARCH_RESULT']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL62_SEARCH_RESULT_SHA256']
    before=files(path.parent);output=tmp_path/'rerun';result=rerun(json.loads(raw),output_dir=output)
    assert result['evaluations_used']==1 and files(output)
    assert str(output) in json.dumps(result['best']['workloads'][0]['compiled_result']['request'])
    assert files(path.parent)==before and path.read_bytes()==raw

def test_optional_absence_isolated_process_keeps_core_working(tmp_path):
    path,_,_=formal_input('mlp');receipt=tmp_path/'OPTIONAL.json'
    command([sys.executable,str(Path(__file__).with_name('onnx_optional_probe.py')),str(path),str(receipt)],tmp_path)
    result=json.loads(receipt.read_text());assert result['core_import'] and not result['onnx_loaded']

def test_resealed_outer_result_value_has_precise_readonly_refusal(onnx_api,monkeypatch,tmp_path):
    _,_,s=onnx_api;path=Path(os.environ['VAL62_COMPILED_RESULT']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL62_COMPILED_RESULT_SHA256']
    original=json.loads(raw);changed=deepcopy(original)
    changed['evaluation']['latency_ns']+=1
    with no_backend(monkeypatch):
        assert s.evaluate_request(original)==original
        with pytest.raises(ValueError,match='compiled result evaluation differs') as error:
            s.evaluate_request(reseal(changed))
    (tmp_path/'RESULT-REFUSAL.json').write_text(json.dumps(dict(type='ValueError',diagnostic=str(error.value),
        original_accepted=True,resealed_value_refused=True,execution_guarded=True),indent=2)+'\n')
