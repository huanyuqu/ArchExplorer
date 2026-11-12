"""Pass-through observation of CORE29's existing public search, installed only."""
from copy import deepcopy
import json
from pathlib import Path
import sys


def consume(directory):
    import arch_explorer
    from arch_explorer.workload import api as w
    from arch_explorer.simulator import api as s
    assert 'site-packages' in arch_explorer.__file__
    root=Path(directory);root.mkdir();calls=dict(compile=[],prepare=[],execute=[])
    compile_original=w.compile_workload;prepare_original=s.prepare_compiled;evaluate_original=s.evaluate_request
    def compile(adf,spec,mapping,**kw):
        row=dict(adf=deepcopy(adf.to_dict()),workload=deepcopy(spec.to_dict()),mapping=deepcopy(mapping.to_dict()))
        calls['compile'].append(row);result=compile_original(adf,spec,mapping,**kw)
        row['binding']=deepcopy(result['binding']);return result
    def prepare(compilation,*args,**kw):
        calls['prepare'].append(deepcopy(compilation['binding']));return prepare_original(compilation,*args,**kw)
    def evaluate(payload,*args,**kw):
        if payload.get('schema')=='arch-explorer.compiled-request.v3':calls['execute'].append(deepcopy(payload['binding']))
        return evaluate_original(payload,*args,**kw)
    w.compile_workload=compile;s.prepare_compiled=prepare;s.evaluate_request=evaluate
    from arch_explorer.archs.multilayer import encoder_search_config
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.exploration import explore
    config=encoder_search_config(layers=2,dtype='bf16',layout='views',compare_layouts=True)
    config['trace_path']=str(root/'tools.jsonl')
    assert config['max_evaluations']==12 and config['mapping_samples']==4
    assert len(config['mapping_selectors']['compile_graph_v3'])==4
    (root/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    scores=[];env=ExplorationEnv(config);original=env.evaluate
    def observed(*args,**kwargs):
        envelope=original(*args,**kwargs)
        if 'result' in envelope:
            score=envelope['result']['score']['objective']
            if score is not None:scores.append(score)
        return envelope
    env.evaluate=observed
    try:
        best=explore(env=env,policy='sa')
        (root/'BEST.json').write_text(json.dumps(best,indent=2)+'\n')
    finally:
        (root/'OBSERVATIONS.json').write_text(json.dumps(dict(calls=calls,scores=scores,used=env.used),indent=2)+'\n')
    validate_saved(root)


def validate_saved(directory):
    root=Path(directory);best=json.loads((root/'BEST.json').read_text())
    observed=json.loads((root/'OBSERVATIONS.json').read_text());calls=observed['calls']
    attempts=best['evaluation_attempts']
    assert best['evaluations_used']==observed['used']==len(attempts)==12
    assert best['max_evaluations']==12 and best['cache_enabled'] is False
    assert [row['index'] for row in attempts]==list(range(12))
    assert best['best']['score']['objective']==min(observed['scores'])
    assert len(calls['compile'])==sum(r['phase']!='mapping_validation' for r in attempts)
    assert len(calls['prepare'])==sum(r['phase'] in ('prepare','evaluate','completed') for r in attempts)
    assert len(calls['execute'])==sum(bool(r['execution_attempted']) for r in attempts)
    failed=[r for r in attempts if not r['valid']]
    assert failed and all(r['phase']=='compile' and not r['execution_attempted'] for r in failed)
    assert all('storage allocation' in str(r.get('error','')).lower() and '> available 1 bytes' in
               str(r.get('error','')).lower() for r in failed)
    policies=set();placements=set();tiles=set()
    for call in calls['compile']:
        p=call['mapping']['params'];assert set(p)=={'compute','memory','rules','operator_rules'}
        ids={r['id'] for r in p['operator_rules'].values()}
        policies.add('views' if 'transpose.view' in ids else 'materialized')
        placements.add(json.dumps(p['compute'],sort_keys=True))
        tiles.add(json.dumps([r['options'] for r in p['operator_rules'].values() if r['id']=='matmul.tiled'],sort_keys=True))
    assert policies=={'materialized','views'} and len(placements)>=2 and len(tiles)>=2
    assert len({json.dumps(row['adf'],sort_keys=True) for row in calls['compile']})>=2
    for row in attempts:
        assert row['requested_mapping']
        if row['valid']:assert row['effective_mapping']
    for row in best['best']['workloads']:
        c=row['compiled_result']['request']['compilation']
        assert row['mapping']==c['source']['mapping']
        assert set(row['mapping']['params']['operator_rules'])==set(c['plan']['rules'])
        assert row['evaluation']==row['compiled_result']['evaluation']
    return best


if __name__=='__main__':consume(sys.argv[1])
