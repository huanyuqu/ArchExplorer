"""Observe real CLIP search through existing generic ledger and winner checks."""
from copy import deepcopy
import json,sys
from pathlib import Path
from view_storage_search_consumer import validate_saved

def consume(directory):
    import arch_explorer
    from arch_explorer.workload import api as w
    from arch_explorer.simulator import api as s
    from arch_explorer.archs.clip_compilation import clip_search_config
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.exploration import explore
    assert 'site-packages' in arch_explorer.__file__
    root=Path(directory);root.mkdir();calls=dict(compile=[],prepare=[],execute=[])
    compile_original=w.compile_workload;prepare_original=s.prepare_compiled;evaluate_original=s.evaluate_request
    def compile(adf,spec,mapping,**kw):
        row=dict(adf=deepcopy(adf.to_dict()),workload=deepcopy(spec.to_dict()),mapping=deepcopy(mapping.to_dict()))
        calls['compile'].append(row);answer=compile_original(adf,spec,mapping,**kw);row['binding']=deepcopy(answer['binding']);return answer
    def prepare(compilation,*a,**kw):
        calls['prepare'].append(deepcopy(compilation['binding']));return prepare_original(compilation,*a,**kw)
    def evaluate(value,*a,**kw):
        if value.get('schema')=='arch-explorer.compiled-request.v3':calls['execute'].append(deepcopy(value['binding']))
        return evaluate_original(value,*a,**kw)
    w.compile_workload=compile;s.prepare_compiled=prepare;s.evaluate_request=evaluate
    config=clip_search_config(layers=2,text_layers=2,dtype='bf16',layout='views',compare_layouts=True)
    assert config['max_evaluations']==12 and config['mapping_samples']==4
    assert len(config['mapping_selectors']['compile_graph_v3'])==4
    config['trace_path']=str(root/'tools.jsonl');(root/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    scores=[];env=ExplorationEnv(config);original=env.evaluate
    def observed(*a,**kw):
        answer=original(*a,**kw)
        if 'result' in answer and answer['result']['score']['objective'] is not None:scores.append(answer['result']['score']['objective'])
        return answer
    env.evaluate=observed
    try:
        best=explore(env=env,policy='sa');(root/'BEST.json').write_text(json.dumps(best,indent=2)+'\n')
    finally:(root/'OBSERVATIONS.json').write_text(json.dumps(dict(calls=calls,scores=scores,used=env.used),indent=2)+'\n')
    best=validate_saved(root)
    assert all(r['workload']['kind']=='clip_similarity' and r['workload']['params']['dtype']=='bf16' for r in calls['compile'])
    assert all(r['workload']['params']['image_batch']!=r['workload']['params']['text_batch'] for r in calls['compile'])
    for row in best['best']['workloads']:
        c=row['compiled_result']['request']['compilation']
        assert set(c['program']['meta']['outputs'])=={'logits_per_image','logits_per_text'}

if __name__=='__main__':consume(sys.argv[1])
