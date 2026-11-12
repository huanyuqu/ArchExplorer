"""Installed typed CLI, budgeted search, Agent, explicit rerun and capacity controls."""
from copy import deepcopy
import hashlib,json,subprocess,sys
from pathlib import Path
from typed_onnx_contract_cases import imported,scene

def run(inputs,output):
    import arch_explorer
    from arch_explorer.adf import Param,find
    from arch_explorer.archs.onnx_compilation import onnx_scene,onnx_search_config,onnx_agent_config
    from arch_explorer.workload import api as w
    from arch_explorer.simulator import api as s
    from arch_explorer.exploration import explore,rerun
    from arch_explorer.agent_env import ExplorationEnv
    from arch_explorer.agent_env.driver import run_agent
    assert 'site-packages' in arch_explorer.__file__
    root=Path(output);root.mkdir(exist_ok=False);inputs=Path(inputs)
    facts=dict(status='running',product=arch_explorer.__file__,cli=[],capacity=[])
    save=lambda:(root/'FACTS.json').write_text(json.dumps(facts,indent=2)+'\n')
    save()
    try:
        spec,_,_=imported(inputs,'mlp-fp16-external')
        data=json.loads((inputs/'INPUTS.json').read_text());row=next(r for r in data['cases'] if r['id']=='mlp-fp16-external')
        for name,value in [('INPUTS',row['inputs']),('SHAPES',row['input_shapes']),('BUDGET',data['resource_budget'])]:
            (root/(name+'.json')).write_text(json.dumps(value)+'\n')
        cli=str(Path(sys.executable).with_name('archexplorer'))
        def command(label,args):
            argv=[cli,*args];p=subprocess.run(argv,cwd=root,capture_output=True,text=True,timeout=90)
            (root/(label+'-stdout.log')).write_text(p.stdout);(root/(label+'-stderr.log')).write_text(p.stderr)
            facts['cli'].append(dict(label=label,argv=argv,returncode=p.returncode));save()
            assert p.returncode==0,p.stderr
            return p.stdout
        command('import',['import-onnx',str(inputs/row['model']),'--inputs',str(root/'INPUTS.json'),
            '--input-shapes',str(root/'SHAPES.json'),'--external-root',str(inputs/row['external_root']),
            '--resource-budget',str(root/'BUDGET.json'),'--output',str(root/'WORKLOAD.json')])
        assert json.loads((root/'WORKLOAD.json').read_text())==spec.to_dict()
        adf,_,mapping=onnx_scene(spec,candidate=1,layout='views')
        (root/'SCENE.json').write_text(json.dumps(dict(adf=adf.to_dict(),workload=spec.to_dict(),mapping=mapping.to_dict()))+'\n')
        command('compile',['compile',str(root/'SCENE.json'),'--output',str(root/'COMPILED.json')])
        text=command('prepare',['prepare',str(root/'COMPILED.json'),'--output',str(root/'cli-request')])
        (root/'REQUEST.json').write_text(text)
        command('evaluate',['evaluate',str(root/'REQUEST.json'),'--output',str(root/'RESULT.json')])
        assert json.loads((root/'RESULT.json').read_text())['evaluation']['latency_ns']>0
        config=onnx_search_config(spec,candidate=0,layout='views');config['trace_path']=str(root/'search-tools.jsonl')
        env=ExplorationEnv(config);best=explore(env=env,policy='greedy')
        (root/'BEST.json').write_text(json.dumps(best,indent=2)+'\n')
        attempts=best['evaluation_attempts'];assert len(attempts)==best['evaluations_used']==env.used<=3
        assert any(not r['valid'] for r in attempts)
        good=[r['score']['objective'] for r in attempts if r['valid'] and r.get('score',{}).get('objective') is not None]
        if good:assert best['best']['score']['objective']==min(good)
        for item in best['best']['workloads']:
            assert item['mapping']==item['compiled_result']['request']['compilation']['source']['mapping']
            assert item['workload']['kind']=='onnx'
        facts['search']=dict(used=env.used,attempts=attempts,winner_bound=True);save()
        raw=(root/'BEST.json').read_bytes()
        old_files={str(p.relative_to(root)):(p.stat().st_size,p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
        renewed=rerun(best,output_dir=root/'explicit-rerun')
        assert renewed['evaluations_used']==len(best['best']['workloads'])==1
        assert (root/'BEST.json').read_bytes()==raw
        for name,stat in old_files.items():assert ((root/name).stat().st_size,(root/name).stat().st_mtime_ns)==stat
        assert list((root/'explicit-rerun').rglob('*'))
        (root/'RERUN.json').write_text(json.dumps(renewed,indent=2)+'\n')
        facts['rerun']=dict(used=renewed['evaluations_used'],new_directory=True,original_files_unchanged=True);save()
        config=onnx_agent_config(spec,candidate=1,layout='views');config['trace_path']=str(root/'agent-tools.jsonl')
        env=ExplorationEnv(config);agent=run_agent(env=env,policy='scripted')
        assert agent['evaluations_used']==env.used==1
        item=agent['best']['workloads'][0];assert item['mapping']==item['compiled_result']['request']['compilation']['source']['mapping']
        (root/'AGENT.json').write_text(json.dumps(agent,indent=2)+'\n');facts['agent']=dict(used=env.used,winner_bound=True);save()
        baseline=json.loads((root.parent/'main-mlp-fp16-external-m0-views-v3/FACTS.json').read_text())
        peak=int(baseline['storage']['peaks']['board/root_sram'])
        for capacity in (peak,peak-1):
            adf,spec,mapping,_,_=scene(inputs,'mlp-fp16-external',0,'views')
            find(adf,'board/root_sram').params['capacity']=Param(capacity,'B','independent installed capacity boundary')
            record=dict(bytes=capacity,expected='pass' if capacity==peak else 'CapacityError');facts['capacity'].append(record);save()
            try:
                compiled=w.compile_workload(adf,spec,mapping)
                request=s.prepare_compiled(compiled,output_dir=root/('capacity-'+str(capacity)),options={'trace':True,'check_capacity':True})
                answer=s.evaluate_request(request)
            except Exception as e:
                record.update(type=type(e).__name__,diagnostic=str(e));save()
                assert capacity==peak-1 and type(e).__name__=='CapacityError'
            else:
                record.update(type='returned',peak=answer['evaluation']['details']['stats']['memory_peak']);save()
                assert capacity==peak
        facts['status']='succeeded'
    except BaseException as e:facts.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:save()

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
