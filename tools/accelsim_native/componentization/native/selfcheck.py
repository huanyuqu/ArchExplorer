"""Bounded CPU mechanism checks; never runs a device or full model campaign.

Each invocation creates a new evidence directory. Native children run serially
in new process groups with 600s/8GiB hard bounds. Config and source originals,
commands, process exits, stdout/stderr and raw metrics are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time


def digest(path):
    return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()


def topology(clusters, partitions, subs, source=False):
    out=['arch-explorer.accelsim-topology.v1']
    def node(name,kind,index): out.append(f'node\t{name}\t{kind}\t{index}')
    def edge(a,ap,b,bp): out.append(f'connect\t{a}\t{ap}\t{b}\t{bp}')
    if source: node('source','source',0)
    else:
        node('icnt','icnt',0)
        for i in range(clusters):
            node(f'sm{i}','sm',i)
            edge(f'sm{i}','request','icnt',f'request{i}')
            edge('icnt',f'reply{i}',f'sm{i}','reply')
    for p in range(partitions): node(f'dram{p}','dram',p)
    for j in range(partitions*subs):
        node(f'l2{j}','l2',j)
        edge(f'l2{j}','dram_request',f'dram{j//subs}',f'request{j%subs}')
        edge(f'dram{j//subs}',f'reply{j%subs}',f'l2{j}','dram_reply')
        edge('source' if source else 'icnt',f'request{j}' if source else f'request_mem{j}',f'l2{j}','request')
        edge(f'l2{j}','reply','source' if source else 'icnt',f'reply{j}' if source else f'reply_mem{j}')
    return '\n'.join(out)+'\n'


def trace(blocks, pressure=False):
    count=18 if pressure else 4
    out=[f'''-kernel name = sim38_mechanism_{'pressure' if pressure else 'boundary'}
-kernel id = 1
-grid dim = ({blocks},1,1)
-block dim = (32,1,1)
-shmem = 0
-nregs = 8
-binary version = 90
-cuda stream id = 0
-enable lineinfo = 0
-nvbit version = synthetic-no-nvbit
-accelsim tracer version = 3
-shmem base_addr = 0x0
-local mem base_addr = 0x0
# synthetic mechanism input; no GPU capture
''']
    for block in range(blocks):
        out += ['#BEGIN_TB',f'thread block = {block},0,0','warp = 0',f'insts = {count}']
        if pressure:
            for i in range(8):
                addr=0x100000+block*0x10000+i*128
                out.append(f'{i*32:04x} ffffffff 1 R1 LDG.E 0 4 1 {addr:x} 4 0')
                out.append(f'{i*32+16:04x} ffffffff 0 STG.E 1 R1 4 1 {addr+0x1000000:x} 4 0')
            out += ['0100 ffffffff 1 R2 IADD3 1 R1 0 0','0110 ffffffff 0 EXIT 0 0 0']
        else:
            out += ['0000 ffffffff 1 R1 LDG.E 0 4 1 100000 4 0',
                    '0010 ffffffff 1 R2 IADD3 1 R1 0 0',
                    '0020 ffffffff 0 STG.E 1 R2 4 1 200000 4 0',
                    '0030 ffffffff 0 EXIT 0 0 0']
        out.append('#END_TB')
    return '\n'.join(out)+'\n'


def limits():
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(590,595))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def run(engine,stage,mode,source=False,expected_ok=True,expected_metrics=False):
    metrics=stage/'metrics.json'
    argv=[engine['binary']['path'],'-ae_mode',mode,'-ae_metrics',str(metrics)]
    if mode=='components': argv+=['-ae_topology',str(stage/'topology.tsv')]
    if source: argv+=['-ae_source',str(stage/'source.tsv')]
    argv+=['-config','gpgpusim.config','-config','limits.config','-trace','./kernelslist.g']
    env={'PATH':'/usr/bin:/bin','LC_ALL':'C','CUDA_VISIBLE_DEVICES':'',
         'LD_LIBRARY_PATH':':'.join(str(Path(lib['path']).parent) for lib in engine['libraries'])}
    process={'argv':argv,'cwd':str(stage),'environment':env,'started_realtime_ns':time.time_ns()}
    with (stage/'stdout').open('wb') as stdout,(stage/'stderr').open('wb') as stderr:
        child=subprocess.Popen(argv,cwd=stage,env=env,stdout=stdout,stderr=stderr,
                               start_new_session=True,preexec_fn=limits)
        process['pid']=child.pid
        process['start_tick']=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        try: process['exit_code']=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process['timeout']=True
            os.killpg(child.pid,signal.SIGKILL)
            process['exit_code']=child.wait()
        process['wait_completed']=True
    process['ended_realtime_ns']=time.time_ns()
    process['stdout_sha256']=digest(stage/'stdout'); process['stderr_sha256']=digest(stage/'stderr')
    (stage/'PROCESS.json').write_text(json.dumps(process,indent=2)+'\n')
    if not expected_ok:
        assert process['exit_code']!=0,stage
        if expected_metrics:
            assert metrics.exists(),stage
            data=json.loads(metrics.read_text())
            assert data['foreground_final']['total_cycles']>=1 and data['drain_ticks']==0
            return {'rejected_native_limit':True,'process':process,'metrics':data}
        assert not metrics.exists(),stage
        assert 'launching kernel name:' not in (stage/'stdout').read_text()
        return {'rejected_before_metrics':True,'process':process}
    assert process['exit_code']==0,(stage,(stage/'stderr').read_text()[-3000:])
    data=json.loads(metrics.read_text())
    assert data['final']['total_cycles']==data['foreground_final']['total_cycles']+data['drain_core_cycles']
    for key in ['total_instructions','total_cta']:
        assert data['final'][key]==data['foreground_final'][key]
    assert data['final']['system_drained'] is True and data['final']['pending']==0 and data['final']['native_busy']==0,stage
    assert data['lifecycle']['native_requests']==data['lifecycle']['native_destroyed']==data['lifecycle']['resource_ready']==data['lifecycle']['consumer_ready']==data['lifecycle']['physical_complete'],stage
    if mode=='components':
        assert data['original_cycle_calls']==0 and data['component_ticks']>0,stage
        assert data['routing']['accepted']>0 and data['routing']['completed']==data['lifecycle']['native_requests'],stage
    else: assert data['component_ticks']==0 and data['original_cycle_calls']>0 and data['routing']=={},stage
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--config-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['boundary','multikernel','pressure','eviction','source','negative'],required=True)
    a=p.parse_args(); root=a.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    engine=json.loads(a.engine.read_text())
    assert digest(engine['binary']['path'])==engine['binary']['sha256']
    for lib in engine['libraries']: assert digest(lib['path'])==lib['sha256']
    raw={}
    for row in (a.config_dir/'gpgpusim.config').read_text().splitlines():
        row=row.split('#',1)[0].strip()
        if row:
            key,value=row.split(None,1); assert key not in raw; raw[key]=value
    raw.update({'-gpgpu_n_chiplet_partition':'1','-gpgpu_lrc_enabled':'0'})
    if a.case in {'source','pressure','eviction'}:
        raw.update({'-gpgpu_simple_dram_model':'0','-gpgpu_clock_domains':'1000:1370:730:2110'})
    if a.case in {'pressure','eviction'}:
        raw.update({'-network_mode':'2','-icnt_in_buffer_limit':'2','-icnt_out_buffer_limit':'2',
                    '-gpgpu_dram_partition_queues':'4:4:4:4'})
    if a.case=='eviction':
        # Preserve native policies and line/sector size; reduce only sets/ways
        # to force dirty L2 eviction into the actual native DRAM write path.
        cache=raw['-gpgpu_cache:dl2'].split(',')
        geometry=cache[0].split(':'); assert len(geometry)==4
        geometry[1]='4'; geometry[3]='1'; cache[0]=':'.join(geometry)
        raw['-gpgpu_cache:dl2']=','.join(cache)
    clusters=int(raw['-gpgpu_n_clusters']); partitions=int(raw['-gpgpu_n_mem']); subs=int(raw['-gpgpu_n_sub_partition_per_mchannel'])
    base=root/'inputs'; base.mkdir()
    (base/'gpgpusim.config').write_text(''.join(f'{k} {v}\n' for k,v in sorted(raw.items())))
    (base/'limits.config').write_text('-gpgpu_max_cycle 10000000\n-gpgpu_max_insn 10000000\n')
    for f in a.config_dir.glob('*.icnt'): shutil.copy2(f,base/f.name)
    source=a.case=='source'
    (base/'topology.tsv').write_text(topology(clusters,partitions,subs,source))
    if source:
        (base/'kernelslist.g').write_text('')
        # Independent traffic: long misses followed by reuse plus stores.
        inputs=[('read',0x100000+i*0x1000) for i in range(256)]
        inputs += [('read',0x100000+i*0x1000) for i in range(64)]
        inputs += [('write',0x200000+i*0x1000) for i in range(64)]
        (base/'source.tsv').write_text('arch-explorer.accelsim-memory-source.v1\n'+''.join(f'{op}\t{addr:#x}\t32\n' for op,addr in inputs))
    else:
        (base/'kernelslist.g').write_text('kernel-1.traceg\n')
        (base/'kernel-1.traceg').write_text(trace(128 if a.case in {'pressure','eviction'} else 1,a.case in {'pressure','eviction'}))
        if a.case=='multikernel':
            (base/'kernel-2.traceg').write_text(trace(1).replace('-kernel id = 1','-kernel id = 2').replace('sim38_mechanism_boundary','sim38_mechanism_boundary_second'))
            (base/'kernelslist.g').write_text('kernel-1.traceg\nkernel-2.traceg\n')
    result={'case':a.case,'engine':str(a.engine.resolve()),'engine_sha256':digest(a.engine),'inputs':{f.name:digest(f) for f in base.iterdir()}}
    try:
        if a.case=='negative':
            mutations={'missing-edge':lambda s:'\n'.join(s.splitlines()[:-1])+'\n',
                       'duplicate-edge':lambda s:s+s.splitlines()[-1]+'\n',
                       'bad-port':lambda s:s.replace('\trequest_mem0\t','\tunknown_mem0\t'),
                       'wrong-index':lambda s:s.replace('node\tsm0\tsm\t0','node\tsm0\tsm\t1')}
            for label,mutate in mutations.items():
                stage=root/label; shutil.copytree(base,stage)
                f=stage/'topology.tsv'; f.write_text(mutate(f.read_text()))
                result[label]=run(engine,stage,'components',expected_ok=False)
            for label,key,value in [('chiplet','-gpgpu_n_chiplet_partition','2'),('lrc','-gpgpu_lrc_enabled','1')]:
                stage=root/label; shutil.copytree(base,stage)
                f=stage/'gpgpusim.config'; f.write_text(f.read_text().replace(f'{key} {raw[key]}\n',f'{key} {value}\n'))
                result[label]=run(engine,stage,'components',expected_ok=False)
            for mode in ['oracle','components']:
                stage=root/('limit-'+mode); shutil.copytree(base,stage)
                (stage/'limits.config').write_text('-gpgpu_max_cycle 1\n')
                result['limit-'+mode]=run(engine,stage,mode,expected_ok=False,expected_metrics=True)
        elif source:
            stage=root/'components'; shutil.copytree(base,stage)
            data=run(engine,stage,'components',True); result['components']=data
            assert data['source']['accepted']==data['source']['completed']==len(inputs)
            assert not any(key.startswith('sm') for key in data['phase_calls'])
            assert data['final']['instructions']==0 and data['final']['total_cta']==0
            assert sum(d['commands'] for d in data['final']['dram'])>0
            assert sum(d['activate'] for d in data['final']['dram'])>0
            assert sum(d['read'] for d in data['final']['dram'])>0
            assert sum(d['misses'] for d in data['final']['l2'])>0
            assert sum(d['hits'] for d in data['final']['l2'])>0
        else:
            for mode in ['oracle','components']:
                stage=root/mode; shutil.copytree(base,stage)
                result[mode]=run(engine,stage,mode)
            for key in ['final','foreground_final','drain_ticks','drain_core_cycles','kernels','ports','lifecycle']:
                assert result['oracle'][key]==result['components'][key],(a.case,key)
            assert result['components']['component_ticks']==result['oracle']['original_cycle_calls']
            if a.case=='multikernel':
                assert [k['uid'] for k in result['components']['kernels']]==[1,2]
                assert result['components']['final']['total_instructions']==256
                assert result['components']['final']['total_cta']==2
            if a.case in {'pressure','eviction'}:
                assert sum(d['commands'] for d in result['components']['final']['dram'])>0
                assert sum(result['components']['final'][k] for k in ['stall_core_to_icnt','stall_icnt_to_l2','stall_l2_to_icnt','stall_icnt_to_core'])>0
            if a.case=='eviction':
                assert sum(d['write']+d['writeback'] for d in result['components']['final']['dram'])>0
                assert result['components']['drain_ticks']>0 and result['components']['drain_core_cycles']>0
        result['status']='passed'
    except BaseException as error:
        result['status']='failed'; result['failure']=repr(error)
        raise
    finally:
        (root/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(root/'RESULTS.json')


if __name__=='__main__': main()
