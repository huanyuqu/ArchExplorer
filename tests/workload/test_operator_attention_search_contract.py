"""Actual public search, budgets, installed CLI and independent readonly replay."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_operator_mapping_attention_contract import api,no_backend,files
from test_operator_mapping_extension_contract import reseal


def child_environment():
    env = {k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    return env


def command(argv,cwd,*,timeout=120):
    result = subprocess.run(argv,cwd=cwd,env=child_environment(),capture_output=True,text=True,timeout=timeout)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    return result


@pytest.mark.parametrize('mode',['regular','unknown'])
def test_real_search_recompiles_mapping_and_parameter_candidates_then_fresh_replay(api,monkeypatch,tmp_path,mode):
    import arch_explorer
    from arch_explorer.exploration import replay,rerun
    assert 'site-packages' in arch_explorer.__file__
    consumer = Path(__file__).with_name('operator_attention_search_consumer.py')
    run = tmp_path/mode
    command([sys.executable,str(consumer),str(run),mode],tmp_path)
    result = json.loads((run/'BEST.json').read_text())
    observed = json.loads((run/'OBSERVATIONS.json').read_text())
    assert observed['search_is_public'] and observed['evaluations_used']<=6
    before = files(run)
    code = r'''
from pathlib import Path
import json,sys
import pytest
import numpy
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.simulator.events import engine
from arch_explorer.simulator import api as simulator
from arch_explorer.exploration import replay
payload = json.loads(Path(sys.argv[1]).read_text())
def forbidden(*args,**kwargs):raise AssertionError('readonly replay created execution state')
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(ExplorationEnv,'__init__',forbidden)
    patch.setattr(simulator,'prepare_compiled',forbidden)
    patch.setattr(engine,'simulate',forbidden)
    patch.setattr(engine.ConsistentScheduler,'run',forbidden)
    assert replay(payload)==payload
print('new-process-readonly-source-replay')
'''
    completed = command([sys.executable,'-c',code,str(run/'BEST.json')],tmp_path)
    assert completed.stdout.strip()=='new-process-readonly-source-replay'
    assert files(run)==before
    with no_backend(monkeypatch):
        assert replay(result)==result
        for mutation in ('objective','mapping','evaluation'):
            bad = deepcopy(result)
            if mutation=='objective':bad['best']['score']['objective']+=1
            elif mutation=='mapping':bad['best']['workloads'][0]['mapping']['params']['rules']['matmul']['options']['tiles']['m']+=1
            else:bad['best']['workloads'][0]['evaluation']['latency_ns']+=1
            with pytest.raises((ValueError,TypeError,KeyError)):
                replay(reseal(bad))
    assert files(run)==before
    # Explicit rerun is a new CPU execution with a fresh bounded budget and paths.
    renewed = rerun(result,output_dir=tmp_path/'explicit-rerun')
    assert renewed['schema']==result['schema'] and renewed!=result
    assert renewed['evaluations_used']==len(result['best']['workloads'])
    assert [r['evaluation'] for r in renewed['best']['workloads']]==[r['evaluation'] for r in result['best']['workloads']]
    assert files(run)==before


def test_agent_free_preparation_failed_attempt_budget_and_readonly_replay(api,monkeypatch,tmp_path):
    from arch_explorer.archs.attention import attention_scene,attention_search_config
    from arch_explorer.agent_env import ExplorationEnv
    w,_ = api
    _,_,mapping = attention_scene()
    config = attention_search_config()
    config.update(max_evaluations=1,trace_path=str(tmp_path/'agent.jsonl'))
    env = ExplorationEnv(config)
    seed,latest = deepcopy(env.adf.to_dict()),deepcopy(env.latest)
    def ok(tool,**args):
        reply = env.call(tool,args)
        assert 'error' not in reply,reply
        return reply['result']
    with no_backend(monkeypatch):
        assert ok('operator_mappings')==w.operator_mappings()
        compilation = ok('compile_workload',mapping=mapping.to_dict(),workload_index=0)
        request = ok('prepare_compiled',compilation=compilation,output_dir=str(tmp_path/'request'),options={'trace':True})
    assert env.used==0
    result = ok('evaluate_request',payload=request)
    assert env.used==1 and result['schema']=='arch-explorer.compiled-result.v3'
    with no_backend(monkeypatch):
        assert ok('evaluate_request',payload=result)==result
    assert env.used==1 and env.adf.to_dict()==seed and env.latest==latest
    exhausted = env.call('evaluate_request',{'payload':request})
    assert exhausted['error']['code']=='evaluation_budget_exhausted' and env.used==1
    # A separate attempted invalid request consumes one unit; compile rejection does not.
    config['trace_path']=str(tmp_path/'invalid-agent.jsonl')
    failed = ExplorationEnv(config)
    compile_error = failed.call('compile_workload',{'mapping':mapping.to_dict(),'workload_index':999})
    assert 'error' in compile_error and failed.used==0
    bad = deepcopy(request);bad['binding']['sha256']='0'*64
    with no_backend(monkeypatch):
        invalid = failed.call('evaluate_request',{'payload':bad})
    assert 'error' in invalid and failed.used==1 and failed.latest is None


def test_installed_cli_catalog_and_compile_consume_unknown_mapping_without_new_name_list(api,tmp_path):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.archs.attention import attention_scene
    executable = str(Path(sys.executable).with_name('archexplorer'))
    catalog = json.loads(command([executable,'operator-mappings','--operator','val55.square'],tmp_path).stdout)
    assert any(row['id']=='val55.square.materialized' for row in catalog)
    adf,workload,mapping = attention_scene()
    graph = deepcopy(workload.params)
    shape = next(t['shape'] for t in graph['tensors'] if t['name']=='attention')
    graph['tensors'].append(dict(name='attention_squared',shape=shape,dtype='fp32',kind='activation',alias_of=None,strides=None))
    graph['operators'].append(dict(name='unknown_square',op='val55.square',inputs=['attention'],outputs=['attention_squared'],
        attrs={'shape':shape},label='VAL55 unknown CLI mapping',type='operator'))
    graph['meta']['outputs']=['attention_squared']
    spec = WorkloadSpec('graph',graph)
    source = tmp_path/'SPECS.json'
    source.write_text(json.dumps(dict(adf=adf.to_dict(),workload=spec.to_dict(),mapping=mapping.to_dict()),indent=2))
    compiled = tmp_path/'COMPILATION.json'
    command([executable,'compile',str(source),'--output',str(compiled)],tmp_path)
    result = json.loads(compiled.read_text())
    assert result['schema']=='arch-explorer.compilation.v3'
    assert 'val55.square' in result['source']['mapping']['params']['rules']
