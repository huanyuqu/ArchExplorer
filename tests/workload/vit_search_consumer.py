"""Observe the existing real search; reuse its model-independent ledger audit."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from view_storage_search_consumer import validate_saved


def consume(directory):
    import arch_explorer
    from arch_explorer.workload import api as w
    from arch_explorer.simulator import api as s
    from arch_explorer.archs.vit_compilation import vit_search_config
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.exploration import explore
    assert 'site-packages' in arch_explorer.__file__
    root=Path(directory);root.mkdir()
    calls=dict(compile=[],prepare=[],execute=[])
    compile_original=w.compile_workload;prepare_original=s.prepare_compiled;evaluate_original=s.evaluate_request
    def compile(adf,spec,mapping,**kw):
        row=dict(adf=deepcopy(adf.to_dict()),workload=deepcopy(spec.to_dict()),mapping=deepcopy(mapping.to_dict()))
        calls['compile'].append(row)
        result=compile_original(adf,spec,mapping,**kw);row['binding']=deepcopy(result['binding']);return result
    def prepare(compilation,*args,**kw):
        calls['prepare'].append(deepcopy(compilation['binding']));return prepare_original(compilation,*args,**kw)
    def evaluate(value,*args,**kw):
        if value.get('schema')=='arch-explorer.compiled-request.v3':calls['execute'].append(deepcopy(value['binding']))
        return evaluate_original(value,*args,**kw)
    w.compile_workload=compile;s.prepare_compiled=prepare;s.evaluate_request=evaluate
    config=vit_search_config(layers=2,dtype='bf16',layout='views',compare_layouts=True)
    config['trace_path']=str(root/'tools.jsonl')
    assert config['max_evaluations']==12 and config['mapping_samples']==4
    assert len(config['mapping_selectors']['compile_graph_v3'])==4
    (root/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    scores=[];env=ExplorationEnv(config);original=env.evaluate
    def observed(*args,**kwargs):
        value=original(*args,**kwargs)
        if 'result' in value and value['result']['score']['objective'] is not None:
            scores.append(value['result']['score']['objective'])
        return value
    env.evaluate=observed
    try:
        best=explore(env=env,policy='sa')
        (root/'BEST.json').write_text(json.dumps(best,indent=2)+'\n')
    finally:
        (root/'OBSERVATIONS.json').write_text(json.dumps(dict(calls=calls,scores=scores,used=env.used),indent=2)+'\n')
    validate_saved(root)
    assert all(row['workload']['kind']=='vit_classifier' for row in calls['compile'])
    assert all(row['workload']['params']['dtype']=='bf16' for row in calls['compile'])


if __name__=='__main__':consume(sys.argv[1])
