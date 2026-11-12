"""Read-only source/binary closure and original-cycle isolation evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from extract import block


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--built-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    engine=json.loads(a.engine.read_text()); prepared=json.loads((a.prepared/'PREPARE.json').read_text())
    assert engine['build']['source_trees']==prepared['build']['source_trees']
    for item in engine['build']['inputs'].values(): assert sha(item['path'])==item['sha256']
    for item in [engine['binary'],*engine['libraries']]: assert sha(item['path'])==item['sha256']
    checked={}
    for name in ['accelsim','gpgpusim']:
        root=a.built_root/name
        assert subprocess.check_output(['git','diff','--name-only'],cwd=root)==b''
        tree=subprocess.check_output(['git','write-tree'],cwd=root,text=True).strip()
        assert tree==engine['build']['source_trees'][name]
        checked[name]=tree
    extraction=json.loads((a.prepared/'EXTRACTION.json').read_text())
    gpu=(a.built_root/'gpgpusim/src/gpgpu-sim/gpu-sim.cc').read_text()
    _,_,body=block(gpu,'void gpgpu_sim::cycle()')
    assert body.count('  ae_accelsim::original_cycle();')==1
    body=body.replace('\n  ae_accelsim::original_cycle();','',1)
    cycle_sha=hashlib.sha256(body.encode()).hexdigest()
    assert cycle_sha==extraction['original_cycle_body_sha256']
    adapter=Path(engine['build']['inputs']['adapter.cc']['path']).read_text()
    for marker in ['bool can_accept(const Endpoint& node,const Transaction& tx) const',
                   'bool accept(const Endpoint& node,const Transaction& tx)']:
        _,_,passive=block(adapter,marker)
        for forbidden in ['core_cycle(', 'icnt_cycle(', 'cache_cycle(', 'dram_cycle(', 'icnt_transfer(', 'runtime->tick(']:
            assert forbidden not in passive,(marker,forbidden)
    _,_,step=block(adapter,'void step(gpgpu_sim* gpu)')
    assert 'runtime->tick()' in step and 'gpu->cycle()' not in step
    result={'schema':'arch-explorer.accelsim-component-source-audit.v1','status':'passed',
            'source_trees':checked,'original_cycle_body_sha256':cycle_sha,
            'oracle_cycle_body_preserved':True,'component_step_uses_generic_runtime':True,
            'accept_and_can_accept_have_no_clock_advances':True,
            'reproduced_from_current_pinned_recipe':True,
            'engine':{'path':str(a.engine.resolve()),'sha256':sha(a.engine)},
            'runtime':engine['build']['inputs']['runtime.hpp'],
            'inputs':engine['build']['inputs'],'binary':engine['binary'],'libraries':engine['libraries']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(a.output)


if __name__=='__main__': main()
