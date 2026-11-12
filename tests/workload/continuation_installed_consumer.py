"""Consume an installed sole candidate and frozen CORE35 data, one stage per run.

No production substitutes and no source-tree imports. Run each stage inside
stage_capture (120s/8GiB) from outside any checkout. Source preparation does
not execute this module against an unsupported baseline.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from continuation_observations import bind_reference,compare_value,check_cast,check_ledger,digest,shaped,original_roots
from static_control_observations import classify,cast_observability,inspect_consumer_shapes


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(package,case_id,record,*,output,candidate_commit,candidate=0,layout='views',stage='numeric',budget=None,max_evaluations=3):
    import arch_explorer
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.archs.onnx_compilation import onnx_scene,onnx_search_config,onnx_agent_config
    from arch_explorer.workload import api as w
    from arch_explorer.simulator import api as s
    path=Path(arch_explorer.__file__).resolve()
    if 'site-packages' not in path.parts:raise ValueError('consume only the sole installed candidate')
    if len(candidate_commit)!=40 or any(c not in '0123456789abcdef' for c in candidate_commit):raise ValueError('explicit candidate commit required')
    if stage not in ('numeric','component','search','agent'):raise ValueError('unknown consumer stage')
    if type(max_evaluations) is not int or not 1<=max_evaluations<=3:raise ValueError('single selector stage admits at most three units')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    facts=dict(schema='arch-explorer.val64.installed-observation.v1',stage=stage,case=case_id,status='running',
        product=digest(path),candidate_commit=candidate_commit,candidate_binding='caller installation receipt must verify wheel/sdist source identity',
        calls=dict(import_onnx=0,compile=0,reference=0,prepare=0,evaluate=0),original={},mapped={},casts=[],
        call_counter_scope='direct calls only; search/Agent indirect execution is recorded by pass-through attempt receipts, not intercepted counters',
        numeric_acceptance='pending formal producer precision/physical contract review',
        physical_event_audit='pending actual observation adapter; reference code does not invent Events')
    dump=lambda:save(root/'FACTS.json',facts)
    dump()
    try:
        package=Path(package).resolve();data=json.loads((package/'INPUTS.json').read_text())
        if data.get('answers_included') is not False:raise ValueError('frozen input package must not contain answers')
        case=next(c for c in data['cases'] if c['id']==case_id)
        binding,types,oracle=bind_reference(case,package,record);facts['reference']=binding;dump()
        import onnx
        model=onnx.load(package/case['model'],load_external_data=False)
        facts['expected_original_root_bytes']=original_roots(model,case['input_shapes'])
        kwargs={} if budget is None else dict(resource_budget=deepcopy(budget))
        facts['calls']['import_onnx']+=1
        spec=w.import_onnx(package/case['model'],inputs=case['inputs'],input_shapes=case['input_shapes'],
            external_root=package/case['external_root'],**kwargs)
        description=w.describe_onnx(spec);facts['description']=description
        facts['original_coverage']=classify(description,types,oracle,model)
        save(root/'WORKLOAD.json',spec.to_dict());dump()
        if stage in ('search','agent'):
            from arch_explorer.agent_env import ExplorationEnv
            config=(onnx_search_config if stage=='search' else onnx_agent_config)(spec,candidate=candidate,layout=layout)
            config['max_evaluations']=min(max_evaluations,config['max_evaluations'])
            config['trace_path']=str(root/'TOOLS.jsonl');save(root/'CONFIG.json',config)
            env=ExplorationEnv(config)
            if stage=='search':
                from arch_explorer.exploration import explore
                result=explore(env=env,policy='greedy')
            else:
                from arch_explorer.agent_env.driver import run_agent
                result=run_agent(env=env,policy='scripted')
            save(root/'SAVED.json',result)
            facts['ledger']=check_ledger(result,limit=config['max_evaluations'])
            if result['evaluations_used']!=env.used:raise ValueError('environment used differs from saved receipt')
            facts['status']='observation-complete';dump();return facts
        adf,spec,mapping=onnx_scene(spec,candidate=candidate,layout=layout)
        save(root/'SCENE.json',dict(adf=adf.to_dict(),workload=spec.to_dict(),mapping=mapping.to_dict()))
        facts['calls']['compile']+=1;compiled=w.compile_workload(adf,spec,mapping)
        save(root/'COMPILED.json',compiled);dump()
        if facts['original_coverage']['static']:
            from arch_explorer.workload.graph import to_dict as graph_dict
            facts['calls']['graph_reconstruction']=1
            ordinary=graph_dict(w.workload_from_spec(spec));save(root/'ORDINARY-GRAPH.json',ordinary)
            facts['static_consumer_shape_attributes']=inspect_consumer_shapes(facts['original_coverage'],description,ordinary)
            dump()
        if stage=='component':
            facts['calls']['prepare']+=1
            request=s.prepare_compiled(compiled,output_dir=root/'request',options={'trace':True,'check_capacity':True})
            save(root/'REQUEST.json',request);dump()
            facts['calls']['evaluate']+=1;result=s.evaluate_request(request)
            save(root/'SAVED.json',result)
            facts['component_evaluation_returned']=True
            facts['component_numeric_output_not_claimed']=True
        else:
            facts['calls']['reference']+=1;original=w.execute_reference(spec)
            for observed in facts['original_coverage']['materialized']:
                name=observed['value']
                meta=description['values'][name];dtype=types[name];wanted=oracle[name]
                if meta['dtype']!=dtype:raise ValueError('original value declared dtype differs')
                actual=original['tensors'][meta['tensor']]
                facts['original'][name]=compare_value(actual,wanted,dtype=dtype,shape=meta['logical_shape'])
            facts['unobserved_casts']=[]
            for node in description['nodes']:
                if node['op_type']!='Cast':continue
                visibility=cast_observability(node,description['values'])
                if not visibility['observable']:
                    facts['unobserved_casts'].append(visibility);continue
                src=description['values'][node['inputs'][0]];dst=description['values'][node['outputs'][0]]
                left=shaped(original['tensors'][src['tensor']],src['logical_shape'],src['dtype'])
                facts['casts'].append(dict(node=node.get('name'),**check_cast(left,original['tensors'][dst['tensor']],
                    target_dtype=dst['dtype'],target_shape=dst['logical_shape'])))
            dump();facts['calls']['reference']+=1
            mapped=w.execute_reference(WorkloadSpec('graph',compiled['program']))
            names=[r['name'] for r in description['outputs']]
            if names!=case['output_names']:raise ValueError('public output order/names differ from frozen ModelProto')
            for item in description['outputs']:
                name=item['name'];facts['mapped'][name]=compare_value(mapped['outputs'][item['tensor']],oracle[name],
                    dtype=types[name],shape=description['values'][name]['logical_shape'])
            facts['all_public_outputs_observed']=True
            facts['all_original_values_cpu_observed']=not facts['original_coverage']['static']
        facts['status']='observation-complete';dump();return facts
    except BaseException as error:
        facts.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:dump()


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--case',required=True)
    p.add_argument('--reference-record',required=True);p.add_argument('--output',required=True)
    p.add_argument('--candidate-commit',required=True);p.add_argument('--candidate',type=int,choices=(0,1),default=0)
    p.add_argument('--layout',choices=('materialized','views'),default='views')
    p.add_argument('--stage',choices=('numeric','component','search','agent'),default='numeric')
    p.add_argument('--resource-budget');p.add_argument('--max-evaluations',type=int,default=3)
    a=p.parse_args();budget=None if a.resource_budget is None else json.loads(Path(a.resource_budget).read_text())
    result=run(a.package,a.case,a.reference_record,output=a.output,candidate_commit=a.candidate_commit,
        candidate=a.candidate,layout=a.layout,stage=a.stage,budget=budget,max_evaluations=a.max_evaluations)
    print(json.dumps(dict(status=result['status'],stage=result['stage'])))


if __name__=='__main__':main()
