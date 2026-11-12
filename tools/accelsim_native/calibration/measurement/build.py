"""Compile fixed finite cubin and CPU oracle only; never invoke a CUDA host.

Each new output preserves the complete command and failure ledger. Sources are
snapshotted before subprocesses. No GPU probing, runtime load, or native build.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys

from design import BUDGET,WORK_ITEM,FUNCTION_PREFIX,budget,cases,values,sha,write_json
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from static_check import assess

HERE=Path(__file__).resolve().parent
CANDIDATE_SOURCES=('build.py','design.py','candidates.cu','oracle.cpp')


def build(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir();(out/'sources').mkdir();(out/'inputs').mkdir()
    # Host/tracer sources change in their own build. They are not inputs to
    # these candidates and must not enter the frozen candidate registry.
    for name in CANDIDATE_SOURCES:
        shutil.copy2(HERE/name,out/'sources'/name)
    cuobjdump=args.cuobjdump
    state={'schema':'arch-explorer.measurement-build.v2','work_item':WORK_ITEM,'status':'incomplete',
           'gpu_queries':0,'gpu_host_invocations':0,'gpu_launches':0,
           'sources':{p.name:sha(p) for p in (out/'sources').iterdir()},'commands':[],
           'tools':{str(p.resolve()):sha(p) for p in (args.nvcc,args.nvdisasm,args.cxx) if p is not None},
           'independent_cuobjdump':'requested separately; absent unless an explicit fixed tool is supplied',
           'python':{'executable':str(Path(sys.executable).resolve()),'version':sys.version},
           'source_contract':'candidate inputs only; compiler dependencies and assessment modules retained separately; no host/tracer identity claim'}
    def command(label,argv):
        log=out/'logs'/(label+'.log')
        row={'label':label,'argv':list(map(str,argv)),'timeout_s':120,'log':str(log)}
        state['commands'].append(row)
        def limits():
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
        try:
            with log.open('wb') as stream:
                done=subprocess.run(row['argv'],cwd=out,stdout=stream,stderr=subprocess.STDOUT,
                    timeout=120,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','LC_ALL':'C'},preexec_fn=limits)
            row['exit_code']=done.returncode
            if done.returncode:raise ValueError('offline command failed: '+label)
        finally:
            row['log_sha256']=sha(log);write_json(out/'BUILD.json',state)
        return log.read_text()
    def retain_dependencies(depfile):
        # Compiler-produced Make dependency grammar for these space-free
        # absolute paths; shlex preserves escaped spaces if a tool emits one.
        text=depfile.read_text().replace('\\\n',' ')
        paths=shlex.split(text.split(':',1)[1])
        retained=out/'compiler-inputs';retained.mkdir(exist_ok=True)
        rows=[]
        for name in dict.fromkeys(paths):
            p=Path(name);p=p if p.is_absolute() else out/p
            p=p.resolve();raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
            target=retained/(h+'-'+p.name)
            if not target.exists():target.write_bytes(raw)
            rows.append({'path':str(p),'bytes':len(raw),'sha256':h,'retained':str(target)})
        return {'path':str(depfile),'sha256':sha(depfile),'inputs':rows}
    def retain_assessment_modules():
        target=out/'assessment-modules';target.mkdir(exist_ok=False);rows=[]
        for name,module in sorted(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if not path or not (name.startswith('arch_explorer.') or name=='static_check'):continue
            p=Path(path).resolve()
            if p.suffix!='.py':continue
            dest=target/(name.replace('.','/')+'.py');dest.parent.mkdir(parents=True,exist_ok=True)
            raw=p.read_bytes();dest.write_bytes(raw)
            rows.append({'module':name,'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'retained':str(dest)})
        return rows
    try:
        # These readers are not compiled into a CUDA host, but their bytes are
        # part of the complete offline assessment evidence.
        shutil.copy2(HERE.parent/'static_check.py',out/'static_check.py')
        shutil.copy2(HERE.parent/'SOURCES.json',out/'UPSTREAM-SOURCES.json')
        state['assessment_sources']={p.name:sha(p) for p in
                                    (out/'static_check.py',out/'UPSTREAM-SOURCES.json')}
        expected=json.loads((HERE.parent/'SOURCES.json').read_text())
        upstream=out/'upstream';upstream.mkdir()
        state['upstream']={'commit':expected['commit'],'files':[]}
        for i,row in enumerate(expected['files']):
            p=args.upstream_dir/(str(i)+'-'+Path(row['path']).name)
            if sha(p)!=row['sha256']:raise ValueError('fixed public source mismatch')
            shutil.copy2(p,upstream/p.name)
            state['upstream']['files'].append(dict(row,retained=str(upstream/p.name)))
        version=command('nvcc-version',[args.nvcc,'--version'])
        if 'V12.8.93' not in version:raise ValueError('unreviewed CUDA compiler version')
        command('nvdisasm-version',[args.nvdisasm,'--version'])
        if cuobjdump:
            state['tools'][str(cuobjdump.resolve())]=sha(cuobjdump)
            command('cuobjdump-version',[cuobjdump,'--version'])
        cuda=args.nvcc.resolve().parent.parent
        state['compiler_components']={str(p):sha(p) for p in
            (cuda/'bin/ptxas',cuda/'bin/fatbinary',cuda/'bin/cudafe++',
             cuda/'nvvm/bin/cicc',cuda/'include/cuda.h') if p.is_file()}
        command('oracle-compile',[args.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-MD','-MF',out/'oracle.d',
                                  out/'sources/oracle.cpp','-o',out/'cpu-oracle'])
        state['oracle_dependencies']=retain_dependencies(out/'oracle.d')
        ptxlog=command('cubin',[args.nvcc,'-std=c++17','-arch=sm_120a','-O3','--cubin','-Xptxas=-v',
                              '-MD','-MF',out/'candidates.d',
                              out/'sources/candidates.cu','-o',out/'kernels.cubin'])
        state['cubin_dependencies']=retain_dependencies(out/'candidates.d')
        assembly=command('nvdisasm',[args.nvdisasm,'-hex',out/'kernels.cubin'])
        if cuobjdump:
            command('cuobjdump-elf',[cuobjdump,'--dump-elf',out/'kernels.cubin'])
            command('cuobjdump-resources',[cuobjdump,'--dump-resource-usage',out/'kernels.cubin'])
            command('cuobjdump-sass',[cuobjdump,'--dump-sass',out/'kernels.cubin'])
            state['independent_cuobjdump']='actual ELF/resources/SASS output retained'
        facts=assess((out/'kernels.cubin').read_bytes(),assembly)
        write_json(out/'STATIC.json',facts)
        candidate=cases()
        if set(facts)!={c['function'] for c in candidate}:raise ValueError('missing/extra compiled function')
        for c in candidate:
            f=facts[c['function']];r=f['resources']
            if f['static_native_subset']!='eligible':raise ValueError('full static native subset refused: '+c['id'])
            prefix=[i for i in f['instructions'] if i['reachability']=='in complete candidate flow']
            if not 0<len(prefix)<=4096:raise ValueError('per-warp dynamic record budget')
            if r['stack_frame_bytes'] or r['local_section_bytes']:raise ValueError('local/spill candidate refused')
            c.update(text_sha256=f['text_sha256'],text_bytes=f['text_bytes'],resources=r,
                     full_flow_instructions=len(prefix),opcode_histogram=f['flow_opcode_histogram'])
            for name,data in zip(('input','auxiliary','expected'),values(c['family'],c['steps'])):
                file=out/'inputs'/(c['id']+'.'+name+'.bin');file.write_bytes(data)
                c[name]={'name':file.relative_to(out).as_posix(),'sha256':sha(file),'bytes':len(data)}
            check=out/(c['id']+'.oracle.bin')
            command('oracle-'+c['id'],[out/'cpu-oracle',c['family'],c['steps'],
                    out/c['input']['name'],out/c['auxiliary']['name'],check])
            if check.read_bytes()!=(out/c['expected']['name']).read_bytes():raise ValueError('independent CPU oracle differs')
        actual_budget=budget(len(candidate))
        registry={'schema':'arch-explorer.measurement-registry.v2','work_item':WORK_ITEM,
                  'function_prefix':FUNCTION_PREFIX,'grid':[1,1,1],'block':[32,1,1],
                  'dynamic_shared':0,'cuda_stream':0,'budget':actual_budget,'cases':candidate,
                  'cubin':{'name':'kernels.cubin','sha256':sha(out/'kernels.cubin'),'bytes':(out/'kernels.cubin').stat().st_size},
                  'static':{'name':'STATIC.json','sha256':sha(out/'STATIC.json')},
                  'sources':state['sources'],'upstream':expected['commit'],
                  'physical_parameters':None,'hardware_accuracy':'not validated','gpu_observations':0}
        write_json(out/'REGISTRY.json',registry)
        from arch_explorer.components.accelsim.microbundle import load_registry
        accepted=load_registry(out/'REGISTRY.json')
        flow=sum(c['full_flow_instructions'] for c in candidate)
        static=sum(len(f['instructions']) for f in facts.values())
        bounds={'schema':'arch-explorer.measurement-capture-budget.v2','work_item':WORK_ITEM,
                'cases':len(candidate),'dynamic_records':flow,'static_instructions':static,
                'managed_record_channel_bytes':64+288*BUDGET['warp_records_per_case'],
                'per_case_dynamic_record_max':max(c['full_flow_instructions'] for c in candidate),
                'records_bytes':64*len(candidate)+288*flow,
                'static_bytes':40*static+len(candidate)*(64*2048+4096),
                'other_capture_bytes':2**20,
                'application_peak_bytes':max(c['input']['bytes']+c['auxiliary']['bytes']+
                                             2*c['expected']['bytes'] for c in candidate),
                'actual_execution_budget':actual_budget,
                'parent_state_plan_budget':{'cases':len(candidate),'measurement_hosts':len(candidate)+1,
                    'structure_launches':len(candidate),'diagnostic_launches':len(candidate),
                    'preparation_launches':2*len(candidate),'formal_launches':3*len(candidate),
                    'uninstrumented_launches':6*len(candidate),'application_launches':7*len(candidate),
                    'per_case_roles':{'diagnostic':1,'preparation':2,'formal':3},
                    'host_timeout_s':120,'total_window_s':1800,'close_reserve_s':60,
                    'warmup_launches':0,'helper_launches':0},'gpu_observations':0}
        bounds['all_cases_application_sum_bound_bytes']=sum(c['input']['bytes']+c['auxiliary']['bytes']+
                                                           2*c['expected']['bytes'] for c in candidate)
        bounds['aggregate_capture_max_bytes']=bounds['records_bytes']+bounds['static_bytes']+bounds['other_capture_bytes']
        if bounds['aggregate_capture_max_bytes']>actual_budget['capture_bytes'] or bounds['application_peak_bytes']>actual_budget['app_bytes']:
            raise ValueError('complete source/data budget exceeds the frozen ceiling')
        write_json(out/'CAPTURE-BUDGET.json',bounds)
        if any(sha(HERE/name)!=state['sources'][name] for name in CANDIDATE_SOURCES):
            raise ValueError('candidate build inputs changed during the offline build')
        state['assessment_modules']=retain_assessment_modules()
        state.update(status='built',registry_sha256=sha(out/'REGISTRY.json'),
                     registry_readback={'status':'accepted','cases':len(accepted.cases)},
                     host_tracer_catalog_generated=False,candidates=len(candidate),
                     ptxas_log_sha256=hashlib.sha256(ptxlog.encode()).hexdigest())
    except Exception as error:
        state['error']=type(error).__name__+': '+str(error);raise
    finally:write_json(out/'BUILD.json',state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('nvcc','nvdisasm','cxx','upstream-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cuobjdump',type=Path)
    x=build(p.parse_args());print(json.dumps({'status':x['status'],'candidates':x['candidates'],'gpu_queries':0,'gpu_launches':0}))
