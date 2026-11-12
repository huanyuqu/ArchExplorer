# SPDX-License-Identifier: Apache-2.0
"""Offline host compilation against the one generated registration contract."""
import argparse,json,subprocess
from pathlib import Path
from arch_explorer.components.accelsim.microbundle import load_registry
from arch_explorer.components.accelsim.microexperiment import ref
from arch_explorer.components.accelsim.microcatalog import (BUILD,POLICY,REPO_POLICY,source_input,
    verify_generation,cpu_environment,embedded_catalog,catalog_bytes)
from design import sha
from arch_explorer.components.accelsim.measurement_state import stateful,embedded
HERE=Path(__file__).resolve().parent
PREFIX='tools/accelsim_native/calibration/measurement/'
INPUTS=('timing_host.cpp','capture_host.cpp','host_io.hpp','host_common.hpp','catalog_types.hpp','host_build.py')

def build(args):
    root=args.bundle.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    src=out/'sources';src.mkdir();(out/'logs').mkdir()
    bundle=load_registry(root/'REGISTRY.json');generation=ref(args.catalog.resolve()/'CATALOG.json')
    g=verify_generation(bundle,generation);sources=[]
    for name in INPUTS+(('state_cuda.hpp',) if stateful(bundle) else ()):
        target=src/name;target.write_bytes((HERE/name).read_bytes());sources.append(source_input(target,repo=PREFIX+name))
    for name in ('catalog.hpp','capture_catalog.hpp')+(('measurement_state.hpp','state_executor.hpp') if stateful(bundle) else ()):
        target=src/name;target.write_bytes(Path(g['files'][name]['path']).read_bytes());sources.append(source_input(target,generated=name))
    target=src/POLICY.name;target.write_bytes(POLICY.read_bytes());sources.append(source_input(target,repo=REPO_POLICY))
    state={'schema':BUILD,'kind':'hosts','status':'incomplete','registry':ref(bundle.path),'generation':generation,
           'sources':sources,'tools':[ref(args.cxx)],'commands':[],'binaries':{},'hardware_executed':False,
           'gpu_queries':0,'gpu_host_invocations':0,'gpu_launches':0}
    def save(): (out/'HOST-BUILD.json').write_text(json.dumps(state,indent=2)+'\n')
    def command(label,argv):
        log=out/'logs'/(label+'.log');row={'label':label,'argv':list(map(str,argv))};state['commands'].append(row)
        with log.open('xb') as f:r=subprocess.run(row['argv'],cwd=out,env=cpu_environment(),stdout=f,stderr=subprocess.STDOUT,timeout=120)
        row.update(exit_code=r.returncode,log=ref(log));save()
        if r.returncode:raise ValueError('offline host compilation/inspection failed: '+label)
        return log.read_text()
    try:
        cuda=args.cuda_root.resolve();stub=next((p for p in (cuda/'lib/stubs',cuda/'lib64/stubs') if (p/'libcuda.so').is_file()),None)
        if stub is None:raise ValueError('fixed CUDA driver link stub missing')
        state['cuda']={'header':ref(cuda/'include/cuda.h'),'link_stub':ref(stub/'libcuda.so')}
        state['tools'] += [state['cuda']['header'],state['cuda']['link_stub']]
        common=[args.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(cuda/'include'),'-I'+str(src)]
        for role in ('timing','capture'):
            name=role+'_host';argv=common+[src/(name+'.cpp'),'-L'+str(stub),'-lcuda','-lcrypto']
            if role=='capture':argv+=['-ldl']
            command(name+'-compile',argv+['-o',out/name])
            dynamic=command(name+'-dynamic',['/usr/bin/readelf','-d',out/name]);symbols=command(name+'-undefined',['/usr/bin/nm','-u',out/name])
            if any(v.lower() in dynamic.lower() for v in ('nvbit','tracer','libcudart','libcupti')):raise ValueError('host links an instrumentation library')
            if 'cuLaunchKernel' not in symbols:raise ValueError('missing registered launch interface')
            if role=='timing' and ('cuEventElapsedTime' not in symbols or 'dlsym' in symbols):raise ValueError('timing isolation differs')
            if role=='capture' and ('cuEvent' in symbols or 'dlsym' not in symbols):raise ValueError('capture/event isolation differs')
            embedded_catalog(out/name,catalog_bytes(bundle))
            if stateful(bundle):embedded(out/name,bundle)
            state['binaries'][role]=ref(out/name)
        state['status']='built'
    except BaseException as e:state.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:save()
    return state

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('bundle','catalog','cuda-root','cxx','output'):p.add_argument('--'+n,type=Path,required=True)
    s=build(p.parse_args());print(json.dumps({'status':s['status'],'host_executions':0,'GPU_operations':0}))
