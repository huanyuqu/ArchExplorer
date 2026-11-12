"""Frozen CORE31 API, real installed CLI/Agent/search and immutable replay."""
from copy import deepcopy
import hashlib,importlib,json,os,shutil,subprocess,sys
from pathlib import Path
import pytest
from clip_contract_cases import BLOCK,GEOMETRY,REJECT,clip_api,frozen_inputs,shapes
from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import files,no_backend
from test_vit_user_consumers import fresh_readonly
from view_storage_search_consumer import validate_saved

def core():
    module=importlib.import_module('arch_explorer.archs.clip_compilation')
    for n in ('clip_inputs','clip_workload','clip_scene','clip_search_config','clip_agent_config'):assert callable(getattr(module,n))
    return module

def options(key):
    p=deepcopy(frozen_inputs()['cases'][key]['workload']['params'])
    result={n:v for n,v in p.items() if n not in ('name','mlp_hidden')};result['mlp']=p['mlp_hidden']
    return result

@pytest.mark.parametrize('key',['regular-2-fp32','boundary-4-bf16'])
def test_CORE31_root_generation_scene_search_Agent_preserve_public_contract(clip_api,monkeypatch,key):
    w,s=clip_api;c=core();kwargs=options(key);before=deepcopy(kwargs)
    with no_backend(monkeypatch):
        spec=c.clip_workload(**kwargs);p=frozen_inputs()['cases'][key]['workload']['params']
        assert spec.params['inputs']==kwargs['inputs'] and spec.params['mlp_hidden']==kwargs['mlp']
        generated=c.clip_inputs(**{n:v for n,v in kwargs.items() if n not in ('dtype','inputs')})
        assert set(generated)==set(shapes(p))
        for row in generated['token_ids']:
            assert all(type(v) is int and 0<=v<p['vocab'] for v in row)
        assert generated['logit_scale']==[.125]
        assert generated['vision.block0.Wq']!=generated['vision.block1.Wq']
        assert generated['text.block0.Wq']!=generated['text.block1.Wq']
        graph=w.workload_from_spec(spec);operators={o.name:o for o in graph.operators}
        for index in (0,1):
            adf,same,mapping=c.clip_scene(**kwargs,candidate=index,layout='views')
            assert same.to_dict()==spec.to_dict()
            assert set(mapping.params['compute'])==set(operators) and len(set(mapping.params['compute'].values()))==2
            for op in graph.operators:
                if op.op=='matmul':
                    binding=mapping.params['operator_rules'][op.name]
                    assert binding['id']=='matmul.tiled' and tuple(binding['options']['tiles'][n] for n in ('m','k','n')) in ((8,8,8),(6,7,10))
            config=c.clip_search_config(**kwargs,candidate=index,layout='views',compare_layouts=True)
            requests=config['mapping_selectors']['compile_graph_v3']
            assert config['max_evaluations']==12 and config['mapping_samples']==len(requests)==4
            assert requests[0]==mapping.params
            assert len({json.dumps(row,sort_keys=True) for row in requests})==4
            assert all(set(row)=={'compute','memory','rules','operator_rules'} for row in requests)
            single=c.clip_search_config(**kwargs,candidate=index,layout='views',compare_layouts=False)
            assert single['max_evaluations']==6 and len(single['mapping_selectors']['compile_graph_v3'])==2
            agent=c.clip_agent_config(**kwargs,candidate=index,layout='views')
            assert agent['max_evaluations']==1 and 'search' not in agent
            assert [row['tool'] for row in agent['agent']['script']]==['operator_mappings','evaluate','submit']
            assert agent['agent']['script'][1]['args']['mappings'][0]==mapping.to_dict()
    assert kwargs==before

def test_explicit_root_values_are_copied_without_filling_or_casting(clip_api,monkeypatch):
    w,s=clip_api;c=core();kwargs=options('regular-2-fp16');before=deepcopy(kwargs)
    with no_backend(monkeypatch):
        spec=c.clip_workload(**kwargs);kwargs['inputs']['token_ids'][0][0]+=1
        assert spec.params['inputs']==before['inputs']
        for failure in ('token','missing','extra'):
            bad=deepcopy(before)
            if failure=='token':bad['inputs']['token_ids'][0][0]=1.
            elif failure=='missing':del bad['inputs']['text.block1.Wq']
            else:bad['inputs']['precomputed_lookup']=[0.]
            with pytest.raises(REJECT):w.workload_from_spec(c.clip_workload(**bad))

def test_shared_helper_default_tiles_remain_legacy_and_new_proposals_are_explicit(clip_api,monkeypatch):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.archs.compiled_configs import per_operator_mappings
    w,s=clip_api;g=w.Workload('generic-tiles-regression')
    for n,shape in [('X',(1,1,2)),('W',(2,2)),('Y',(1,1,2)),('Z',(1,1,2))]:g.tensor(n,shape)
    for name,x,y in [('first','X','Y'),('second','Y','Z')]:g.add(name,'matmul',[x,'W'],[y],batch=1,m=1,k=2,n=2,shared_b=True)
    g.meta['outputs']=['Z'];spec=WorkloadSpec('graph',g.to_dict());paths=('board/core/a','board/core/b')
    with no_backend(monkeypatch):
        legacy=per_operator_mappings(spec,paths,'board/memory')
        explicit=per_operator_mappings(spec,paths,'board/memory',matmul_tiles=((2,3,5),(3,5,3)))
        assert [m.to_dict() for m in legacy]==[m.to_dict() for m in explicit]
        custom=per_operator_mappings(spec,paths,'board/memory',matmul_tiles=((8,8,8),(6,7,10)))
        for i,(old,new) in enumerate(zip(legacy,custom)):
            assert old.params['compute']==new.params['compute'] and old.params['memory']==new.params['memory']
            assert set(new.params['operator_rules'])=={'first','second'}
            for j,n in enumerate(('first','second')):
                selected=(j+i)%2
                assert new.params['compute'][n]==paths[selected]
                assert tuple(old.params['operator_rules'][n]['options']['tiles'][k] for k in ('m','k','n'))==((2,3,5),(3,5,3))[selected]
                assert tuple(new.params['operator_rules'][n]['options']['tiles'][k] for k in ('m','k','n'))==((8,8,8),(6,7,10))[selected]
        for bad in (((0,1,1),(1,1,1)),((True,1,1),(1,1,1)),((1,1,1),)):
            with pytest.raises(ValueError):per_operator_mappings(spec,paths,'board/memory',matmul_tiles=bad)

def test_real_CLIP_search_charges_capacity_failures_and_saves_evaluated_winner(tmp_path):
    consumer=Path(__file__).with_name('clip_search_consumer.py');directory=tmp_path/'search'
    command([sys.executable,str(consumer),str(directory)],tmp_path,timeout=540)
    best=validate_saved(directory);assert best['best']['workloads'][0]['workload']['kind']=='clip_similarity'
    fresh_readonly(directory/'BEST.json',tmp_path)

def test_real_CLIP_offline_Agent_one_execution_and_readonly(clip_api,tmp_path):
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.agent_env.driver import run_agent
    config=core().clip_agent_config(layers=2,text_layers=2,dtype='fp16',layout='views');config['trace_path']=str(tmp_path/'agent-tools.jsonl')
    env=ExplorationEnv(config);result=run_agent(env=env,policy='scripted')
    assert result['evaluations_used']==env.used==1
    row=result['best']['workloads'][0];assert row['workload']['kind']=='clip_similarity'
    c=row['compiled_result']['request']['compilation'];assert row['mapping']==c['source']['mapping']
    assert set(c['program']['meta']['outputs'])=={'logits_per_image','logits_per_text'}
    p=tmp_path/'AGENT-RESULT.json';p.write_text(json.dumps(result,indent=2)+'\n');fresh_readonly(p,tmp_path)

def test_exported_CLIP_example_generates_configs_and_cli_compiles_prepares_evaluates(clip_api,tmp_path):
    export=Path(os.environ['VAL60_SOURCE_EXPORT']);example=tmp_path/'example';shutil.copytree(export/'examples/clip_compilation',example)
    source=tmp_path/'source.json';search=tmp_path/'search.json';agent=tmp_path/'agent.json'
    command([sys.executable,str(example/'scene.py'),'--dtype','bf16','--layout','views','--output',str(source),
        '--search-config',str(search),'--agent-config',str(agent)],tmp_path,timeout=540)
    document=json.loads(source.read_text());assert document['workload']['kind']=='clip_similarity'
    assert json.loads(search.read_text())['max_evaluations']==12 and json.loads(agent.read_text())['max_evaluations']==1
    cli=str(Path(sys.executable).with_name('archexplorer'));compiled=tmp_path/'compiled.json'
    command([cli,'compile',str(source),'--output',str(compiled)],tmp_path,timeout=540)
    prepared=command([cli,'prepare',str(compiled),'--options',str(example/'provider-options.json'),
        '--output',str(tmp_path/'new-request')],tmp_path,timeout=540)
    request=tmp_path/'request.json';request.write_text(prepared.stdout);result=tmp_path/'RESULT.json'
    command([cli,'evaluate',str(request),'--output',str(result)],tmp_path,timeout=540)
    saved=json.loads(result.read_text());c=saved['request']['compilation']
    assert set(c['program']['meta']['outputs'])=={'logits_per_image','logits_per_text'} and saved['evaluation']['latency_ns']>0
    fresh_readonly(result,tmp_path,'compiled')

def test_exported_example_rejects_document_collisions_and_invalid_tokens_without_partial_writes(tmp_path):
    export=Path(os.environ['VAL60_SOURCE_EXPORT']);script=export/'examples/clip_compilation/scene.py'
    same=tmp_path/'collision.json'
    for arguments,outputs in [(['--output',str(same),'--search-config',str(same)],[same])]:
        result=subprocess.run([sys.executable,str(script),*arguments],cwd=tmp_path,capture_output=True,text=True,timeout=120)
        assert result.returncode and all(not p.exists() for p in outputs)
    bad=deepcopy(frozen_inputs()['cases']['regular-2-fp32']['workload']['params']['inputs']);bad['token_ids'][0][0]=1.5
    roots=tmp_path/'bad.json';roots.write_text(json.dumps(bad));source=tmp_path/'bad-source.json';search=tmp_path/'bad-search.json'
    result=subprocess.run([sys.executable,str(script),'--inputs',str(roots),'--output',str(source),'--search-config',str(search)],
        cwd=tmp_path,capture_output=True,text=True,timeout=120)
    (tmp_path/'INVALID-CLI.json').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    assert result.returncode and not source.exists() and not search.exists()

def test_CLIP_explicit_rerun_reuses_completed_search_in_a_new_directory(tmp_path):
    from arch_explorer.exploration import rerun
    path=Path(os.environ['VAL60_SEARCH_RESULT']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL60_SEARCH_RESULT_SHA256']
    original=json.loads(raw);before=files(path.parent);directory=tmp_path/'new-rerun'
    result=rerun(original,output_dir=directory);assert result['evaluations_used']==1
    row=result['best']['workloads'][0];assert row['workload']['kind']=='clip_similarity'
    new=row['compiled_result']['request'];old=original['best']['workloads'][0]['compiled_result']['request']
    assert new!=old and directory.is_dir() and files(directory)
    assert str(directory) in json.dumps(new) and str(directory) not in json.dumps(old)
    assert files(path.parent)==before
