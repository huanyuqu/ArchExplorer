# SPDX-License-Identifier: Apache-2.0
"""Offline NVBit variant bound to the one generated registration directory.

No dlopen, host execution, CUDA call or device query is performed.
"""
import argparse,json,subprocess
from pathlib import Path
from arch_explorer.components.accelsim.microexperiment import ref
from arch_explorer.components.accelsim.microbundle import load_registry
from arch_explorer.components.accelsim.microcatalog import (BUILD,POLICY,REPO_POLICY,source_input,
    verify_generation,cpu_environment,embedded_catalog,catalog_bytes)
HERE=Path(__file__).resolve().parent
PREFIX='tools/accelsim_native/capture/'
ASSET='9c383591bc62be02dd0bcba51f98320525817de67d30d997ec3d618adcd3d618'

def build(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();src=out/'sources';src.mkdir()
    bundle=load_registry(args.registry);generation=ref(args.catalog.resolve()/'CATALOG.json');g=verify_generation(bundle,generation)
    sources=[]
    for name in ('microbenchmark/inject_funcs.cu','microbenchmark/tracer.cu','microbenchmark/build.py','reflection.h','records.h','json_util.h'):
        origin=HERE.parent/name;target=src/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(origin.read_bytes())
        sources.append(source_input(target,repo=PREFIX+name))
    gen=src/'generated';gen.mkdir()
    for name in ('capture_catalog.hpp',):
        target=gen/name;target.write_bytes(Path(g['files'][name]['path']).read_bytes());sources.append(source_input(target,generated=name))
    target=gen/POLICY.name;target.write_bytes(POLICY.read_bytes());sources.append(source_input(target,repo=REPO_POLICY))
    cuda=args.cuda.resolve();nvbit=args.nvbit.resolve()
    if ref(args.asset)['sha256']!=ASSET:raise ValueError('fixed NVBit asset changed')
    if '#define NVBIT_VERSION "1.8.1"' not in (nvbit/'core/nvbit.h').read_text() or 'URZ_BLACKWELL_PLUS = 255' not in (nvbit/'core/instr_types.h').read_text():raise ValueError('fixed Blackwell NVBit API required')
    nvcc=cuda/'bin/nvcc';state={'schema':BUILD,'kind':'tracer','status':'incomplete','registry':ref(bundle.path),
      'generation':generation,'sources':sources,'tools':[ref(nvcc),ref(args.asset)]+[ref(nvbit/'core'/n) for n in ('nvbit.h','nvbit_tool.h','instr_types.h','libnvbit.a')],
      'commands':[],'binaries':{},'hardware_executed':False}
    def save(): (out/'BUILD.json').write_text(json.dumps(state,indent=2)+'\n')
    common=[str(nvcc),'-std=c++17','-arch=sm_120a','-I'+str(nvbit/'core'),'-I'+str(gen),'-O3','-D_FORCE_INLINES']
    inject=src/'microbenchmark/inject_funcs.cu';tracer=src/'microbenchmark/tracer.cu'
    commands=[('inject',common+['-Xptxas','-astoolspatch','--keep-device-functions','-Xcompiler','-Wall,-fPIC','-c',str(inject),'-o',str(out/'inject.o')]),
      ('tracer',common+['-Xptxas','-cloning=no','-Xcompiler','-Wall,-fPIC','-dc',str(tracer),'-o',str(out/'tracer.o')]),
      ('link',common+['-shared','-Xcompiler','-fPIC',str(out/'tracer.o'),str(out/'inject.o'),str(nvbit/'core/libnvbit.a'),'-L'+str(cuda/'lib/stubs'),'-L'+str(cuda/'lib'),'-lcuda','-lcudart_static','-lcrypto','-ldl','-lrt','-Xcompiler','-pthread','-o',str(out/'capture.so')]),
      ('exports',['/usr/bin/nm','-D','--defined-only',str(out/'capture.so')])]
    env=cpu_environment();env['PATH']=str(cuda/'bin')+':/usr/bin:/bin'
    try:
        for label,argv in commands:
            log=out/'logs'/(label+'.log');row={'argv':argv};state['commands'].append(row)
            with log.open('xb') as f:r=subprocess.run(argv,cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
            row.update(exit_code=r.returncode,log=ref(log));save()
            if r.returncode:raise ValueError('offline tracer compilation failed: '+label)
        symbols=(out/'logs/exports.log').read_text()
        if not all(n in symbols for n in ('sim30_capture_info','sim30_capture_begin','sim30_capture_finish','ae_capture_catalog_v1')):raise ValueError('new catalog/transport exports missing')
        embedded_catalog(out/'capture.so',catalog_bytes(bundle));state['binaries']['tracer']=ref(out/'capture.so');state['status']='built'
    except BaseException as e:state.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:save()
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cuda','nvbit','asset','registry','catalog','output'):p.add_argument('--'+n,type=Path,required=True)
    s=build(p.parse_args());print(json.dumps({'status':s['status'],'tool':s['binaries'],'GPU_operations':0}))
if __name__=='__main__':main()
