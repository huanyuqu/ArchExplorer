"""Public, installed, bounded ONNX search with pass-through execution ledger."""
from copy import deepcopy
import json,sys
from pathlib import Path
from onnx_contract_cases import formal_spec

def consume(directory,key):
    import arch_explorer
    from arch_explorer.workload import api as w
    from arch_explorer.simulator import api as s
    from arch_explorer.archs.onnx_compilation import onnx_search_config
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.exploration import explore
    assert 'site-packages' in arch_explorer.__file__
    root=Path(directory);root.mkdir()
    spec,_=formal_spec(key);config=onnx_search_config(spec,candidate=0,layout='views')
    assert config['max_evaluations']==3 and config['mapping_samples']==1
    config['trace_path']=str(root/'tools.jsonl')
    calls=dict(compile=[],prepare=[],execute=[])
    original_compile,original_prepare,original_evaluate=w.compile_workload,s.prepare_compiled,s.evaluate_request
    def compile(adf,workload,mapping,**options):
        item=dict(adf=deepcopy(adf.to_dict()),workload=deepcopy(workload.to_dict()),mapping=deepcopy(mapping.to_dict()))
        calls['compile'].append(item);answer=original_compile(adf,workload,mapping,**options)
        item['binding']=deepcopy(answer['binding']);return answer
    def prepare(compilation,*args,**options):
        calls['prepare'].append(deepcopy(compilation['binding']))
        return original_prepare(compilation,*args,**options)
    def evaluate(value,*args,**options):
        if value.get('schema')=='arch-explorer.compiled-request.v3':calls['execute'].append(deepcopy(value['binding']))
        return original_evaluate(value,*args,**options)
    w.compile_workload=compile;s.prepare_compiled=prepare;s.evaluate_request=evaluate
    scores=[];env=ExplorationEnv(config);original=env.evaluate
    def observed(*args,**options):
        answer=original(*args,**options)
        if 'result' in answer and answer['result']['score']['objective'] is not None:scores.append(answer['result']['score']['objective'])
        return answer
    env.evaluate=observed
    (root/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    try:
        best=explore(env=env,policy='greedy')
        (root/'BEST.json').write_text(json.dumps(best,indent=2)+'\n')
    finally:
        (root/'OBSERVATIONS.json').write_text(json.dumps(dict(calls=calls,scores=scores,
            used=env.used,remaining=env.remaining),indent=2)+'\n')
    attempts=best['evaluation_attempts']
    assert best['evaluations_used']==env.used==len(attempts)<=config['max_evaluations']
    assert best['best']['score']['objective']==min(scores)
    assert len(calls['execute'])==sum(bool(r['execution_attempted']) for r in attempts)
    assert len(calls['prepare'])==sum(r['phase'] in ('prepare','evaluate','completed') for r in attempts)
    assert any(not row['valid'] for row in attempts),'1B capacity proposal must leave a recorded failure'
    for item in best['best']['workloads']:
        compiled=item['compiled_result']['request']['compilation']
        assert item['workload']['kind']=='onnx'
        assert item['mapping']==compiled['source']['mapping']
        assert item['compiled_result']['request']['binding'] in calls['execute']
        assert compiled['source']['workload']['params']['source']==spec.params['source']
    return best

if __name__=='__main__':consume(sys.argv[1],sys.argv[2])
