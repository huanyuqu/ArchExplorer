"""Observe installed CORE32 public search without altering its producer inputs."""
from copy import deepcopy
import json
from pathlib import Path
import sys


def consume(directory):
    import arch_explorer
    from arch_explorer.workload import api as w
    from arch_explorer.simulator import api as s
    from arch_explorer.archs.mla import mla_search_config
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.exploration import explore
    assert 'site-packages' in arch_explorer.__file__
    root=Path(directory);root.mkdir()
    calls=dict(compile=[],prepare=[],execute=[])
    compile_original,prepare_original,evaluate_original=w.compile_workload,s.prepare_compiled,s.evaluate_request
    def compile(adf,spec,mapping,**options):
        item=dict(adf=deepcopy(adf.to_dict()),workload=deepcopy(spec.to_dict()),mapping=deepcopy(mapping.to_dict()))
        calls['compile'].append(item)
        answer=compile_original(adf,spec,mapping,**options)
        item['binding']=deepcopy(answer['binding']);return answer
    def prepare(compilation,*args,**options):
        calls['prepare'].append(deepcopy(compilation['binding']))
        return prepare_original(compilation,*args,**options)
    def evaluate(value,*args,**options):
        if value.get('schema')=='arch-explorer.compiled-request.v3':
            calls['execute'].append(deepcopy(value['binding']))
        return evaluate_original(value,*args,**options)
    w.compile_workload=compile;s.prepare_compiled=prepare;s.evaluate_request=evaluate
    config=mla_search_config(candidate=0,layout='views')
    assert config['max_evaluations']==3 and config['mapping_samples']==1
    config['trace_path']=str(root/'tools.jsonl')
    (root/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    scores=[];env=ExplorationEnv(config);original=env.evaluate
    def observed(*args,**options):
        answer=original(*args,**options)
        if 'result' in answer and answer['result']['score']['objective'] is not None:
            scores.append(answer['result']['score']['objective'])
        return answer
    env.evaluate=observed
    try:
        best=explore(env=env,policy='sa')
        (root/'BEST.json').write_text(json.dumps(best,indent=2)+'\n')
    finally:
        # Snapshot the attempt list; later checker actions must not mutate it.
        (root/'OBSERVATIONS.json').write_text(json.dumps(dict(calls=calls,scores=scores,
            used=env.used,remaining=env.remaining),indent=2)+'\n')
    attempts=best['evaluation_attempts']
    assert best['evaluations_used']==env.used==len(attempts)
    assert env.used<=config['max_evaluations']
    assert best['best']['score']['objective']==min(scores)
    assert len(calls['execute'])==sum(bool(row['execution_attempted']) for row in attempts)
    assert len(calls['prepare'])==sum(row['phase'] in ('prepare','evaluate','completed') for row in attempts)
    assert any(not row['valid'] for row in attempts),'capacity-1 proposal must be represented'
    assert all(row['workload']['kind']=='mla_decode' for row in calls['compile'])
    for item in best['best']['workloads']:
        compiled=item['compiled_result']['request']['compilation']
        assert item['mapping']==compiled['source']['mapping']
        assert set(item['mapping']['params'])=={'compute','memory','storage','replicas','rules','operator_rules'}
        assert compiled['program']['meta']['outputs']==['Y']
        assert item['compiled_result']['request']['binding'] in calls['execute']
    return best


if __name__=='__main__':consume(sys.argv[1])
