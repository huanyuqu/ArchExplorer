"""Fresh-process VAL55 search observer using the real public searcher only."""
from copy import deepcopy
from pathlib import Path
import json
import math
import sys


def main(directory,mode,*,config_factory=None,mapping_template='compile_graph_v2'):
    out = Path(directory)
    out.mkdir()
    import arch_explorer
    import arch_explorer.workload.api as workload_api
    import arch_explorer.simulator.api as simulator_api
    assert 'site-packages' in arch_explorer.__file__,arch_explorer.__file__
    calls = dict(compile=[],prepare=[],evaluate=[])
    original_compile = workload_api.compile_workload
    original_prepare = simulator_api.prepare_compiled
    original_evaluate = simulator_api.evaluate_request

    def compile_record(adf,workload,mapping,**kwargs):
        entry = dict(adf=deepcopy(adf.to_dict()),workload=deepcopy(workload.to_dict()),mapping=deepcopy(mapping.to_dict()))
        calls['compile'].append(entry)
        result = original_compile(adf,workload,mapping,**kwargs)
        assert result['schema']=='arch-explorer.compilation.v3'
        entry['binding'] = deepcopy(result['binding'])
        return result

    def prepare_record(compilation,*args,**kwargs):
        calls['prepare'].append(deepcopy(compilation['binding']))
        return original_prepare(compilation,*args,**kwargs)

    def evaluate_record(payload,*args,**kwargs):
        if payload.get('schema')=='arch-explorer.compiled-request.v3':
            calls['evaluate'].append(deepcopy(payload['binding']))
        return original_evaluate(payload,*args,**kwargs)

    # Patch before importing Env/search, including any module-level API aliases.
    workload_api.compile_workload = compile_record
    simulator_api.prepare_compiled = prepare_record
    simulator_api.evaluate_request = evaluate_record
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.search import run_search
    from arch_explorer.exploration import explore
    assert explore is run_search
    if config_factory is None:
        from arch_explorer.adf import WorkloadSpec
        from arch_explorer.archs.attention import attention_workload,attention_search_config
        workload = attention_workload()
        if mode=='unknown':
            # An actual unknown mapping joins attention; no search policy branch changes.
            import val55_mapping_extension
            graph = deepcopy(workload.params)
            shape = next(t['shape'] for t in graph['tensors'] if t['name']=='attention')
            graph['tensors'].append(dict(name='attention_squared',shape=shape,dtype='fp32',
                kind='activation',alias_of=None,strides=None))
            graph['operators'].append(dict(name='unknown_square',op='val55.square',inputs=['attention'],
                outputs=['attention_squared'],attrs={'shape':shape},label='VAL55 installed unknown mapping',type='operator'))
            graph['meta']['outputs'] = ['attention_squared']
            workload = WorkloadSpec('graph',graph)
        config = attention_search_config(workload=workload)
    else:
        # A new public scene injects data/config only. The observer continues
        # to use the same real Env/searcher and exact accounting assertions.
        config = config_factory()
    config['trace_path'] = str(out/'tools.jsonl')
    assert config['random_seed']==0 and config['max_evaluations']==6
    assert len(config['mapping_selectors'][mapping_template])==2
    assert config['search']['max_steps']==3
    (out/'CONFIG.json').write_text(json.dumps(config,sort_keys=True,indent=2)+'\n')
    env = ExplorationEnv(config)
    evaluate_env = env.evaluate
    scores = []

    def observed(*args,**kwargs):
        envelope = evaluate_env(*args,**kwargs)
        if 'result' in envelope:
            value = envelope['result']['score']['objective']
            if value is not None:
                assert math.isfinite(value)
                scores.append(value)
        return envelope

    env.evaluate = observed
    result = run_search(env=env,policy='greedy')
    assert result['schema']=='archexplorer.compilation-search-result.v1'
    assert result['evaluations_used']==env.used<=6
    assert result['max_evaluations']==6 and result['cache_enabled'] is False
    attempts = result['evaluation_attempts']
    assert [row['index'] for row in attempts]==list(range(env.used))
    assert scores and result['best']['score']['objective']==min(scores)
    assert len(calls['compile'])==sum(r['phase']!='mapping_validation' for r in attempts)
    assert len(calls['prepare'])==sum(r['phase'] in ('prepare','evaluate','completed') for r in attempts)
    assert len(calls['evaluate'])==sum(bool(r['execution_attempted']) for r in attempts)
    assert any(not r['valid'] and r.get('error') for r in attempts)
    assert len({json.dumps(r['mapping'],sort_keys=True) for r in attempts})>=2
    assert len({json.dumps(r['adf'],sort_keys=True) for r in calls['compile']})>=2
    if mode=='unknown':
        assert calls['compile'] and all('val55.square' in {op['op'] for op in r['workload']['params']['operators']}
                                        for r in calls['compile'])
    for row in result['best']['workloads']:
        saved = row['compiled_result']
        assert saved['schema']=='arch-explorer.compiled-result.v3'
        assert saved['evaluation']==row['evaluation']
        assert saved['request']['compilation']['source']['adf']==result['best']['adf']
    (out/'BEST.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'OBSERVATIONS.json').write_text(json.dumps(dict(mode=mode,calls=calls,scores=scores,
        evaluations_used=env.used,attempts=attempts,search_is_public=True),sort_keys=True,indent=2)+'\n')


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
