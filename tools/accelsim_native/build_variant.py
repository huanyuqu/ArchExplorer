"""Build an isolated, patched CPU engine from fixed local upstream checkouts.

No download or GPU call. CUDA here supplies build headers/libraries only.
No upstream working tree is edited; all source writes are in --root (new).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASES={'accelsim':'1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea',
       'gpgpusim':'c24e8858eee552b3bc2919b147f14e2a88571775'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('accel-source','gpgpu-source','cuda','tools','root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.jobs<=16:raise ValueError('jobs must be 1..16')
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    cuda,tools=a.cuda.resolve(),a.tools.resolve()
    receipt={'upstream':BASES,'patches':{},'source_trees':{},'commands':[],'GPU_calls':0}
    with (root/'build.log').open('wb') as log:
        def run(argv,cwd,env=None):
            receipt['commands'].append({'argv':list(map(str,argv)),'cwd':str(cwd)})
            subprocess.run(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for key,source in (('accelsim',a.accel_source),('gpgpusim',a.gpgpu_source)):
            dest=root/key
            run(['git','clone','--shared','--no-checkout',str(source.resolve()),str(dest)],root)
            run(['git','checkout','--detach',BASES[key]],dest)
            patch=Path(__file__).parent/'patches'/(key+'-sm120-typed-v1.patch')
            receipt['patches'][key]=hashlib.sha256(patch.read_bytes()).hexdigest()
            run(['git','apply','--index','--binary',str(patch.resolve())],dest)
            receipt['source_trees'][key]=subprocess.check_output(['git','write-tree'],cwd=dest,text=True).strip()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CXX='g++ -Wl,--no-as-needed',
            PATH=f'{tools}/usr/bin:{cuda}/bin:/usr/bin:/bin',
            BISON_PKGDATADIR=f'{tools}/usr/share/bison',M4=f'{tools}/usr/bin/m4',
            CPLUS_INCLUDE_PATH=f'{tools}/usr/include:{root}/accelsim/util/tracer_nvbit/tracer_tool',
            LIBRARY_PATH=f'{tools}/usr/lib/x86_64-linux-gnu',LD_LIBRARY_PATH=f'{tools}/usr/lib/x86_64-linux-gnu',
            CUDA_INSTALL_PATH=str(cuda),GPGPUSIM_ROOT=f'{root}/gpgpusim',
            ACCELSIM_ROOT=f'{root}/accelsim/gpu-simulator',ACCELSIM_CONFIG='release',
            ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN='1',SIM28_BUILD_JOBS=str(a.jobs))
        receipt['environment']={k:v for k,v in env.items() if k not in os.environ or os.environ[k]!=v}
        try:
            run(['bash','-c','source "$GPGPUSIM_ROOT/setup_environment" release && make -j"$SIM28_BUILD_JOBS"'],root/'accelsim/gpu-simulator',env)
            receipt['status']='built'
            artifacts=[root/'accelsim/gpu-simulator/bin/release/accel-sim.out',*sorted((root/'gpgpusim/lib').glob('**/release/libcudart.so'))]
            receipt['artifacts']=[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in artifacts]
        finally:
            (root/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(root/'BUILD.json')

if __name__=='__main__':main()
