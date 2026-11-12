# SPDX-License-Identifier: Apache-2.0
"""Exercise CPU arithmetic, real compiled dependency chains, and host I/O.

Only the no-CUDA io_check program and rejected CPU-oracle inputs are executed.
The timing and capture hosts are inspected as ELF files, never started.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

from design import WORK_ITEM,FUNCTION_PREFIX,cases,FAMILIES,HELDOUT,budget,sha,write_json,values
from resolution import event_ms,slope_interval
from arch_explorer.components.accelsim.microcatalog import BUILD
from arch_explorer.components.accelsim.microexperiment import ref

HERE=Path(__file__).resolve().parent


def check(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'sources').mkdir();(out/'logs').mkdir()
    for name in ('io_check.cpp','host_io.hpp','catalog_types.hpp','check_cpu.py','resolution.py'):
        shutil.copy2(HERE/name,out/'sources'/name)
    for name in ('catalog.hpp','capture_catalog.hpp','catalog_policy.hpp'):
        shutil.copy2(args.build/'catalog'/name,out/'sources'/name)
    state={'schema':'arch-explorer.measurement-cpu-check.v2','work_item':WORK_ITEM,'status':'incomplete',
           'gpu_queries':0,'gpu_host_invocations':0,'gpu_launches':0,'checks':[],'commands':[],
           'sources':{p.name:sha(p) for p in (out/'sources').iterdir()}}
    def test(name,condition):
        if not condition:raise ValueError('CPU check failed: '+name)
        state['checks'].append(name)
    def command(name,argv,expected=0):
        log=out/'logs'/(name+'.log')
        row={'label':name,'argv':list(map(str,argv)),'expected_return':expected,'log':str(log)}
        state['commands'].append(row)
        with log.open('wb') as stream:
            done=subprocess.run(row['argv'],stdout=stream,stderr=subprocess.STDOUT,timeout=120,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','LC_ALL':'C'})
        row.update(exit_code=done.returncode,log_sha256=sha(log))
        test(name+' return',done.returncode==expected)
    try:
        registry=json.loads((args.build/'REGISTRY.json').read_text())
        static=json.loads((args.build/'STATIC.json').read_text())
        test('new work item and function prefix',registry['work_item']==WORK_ITEM and registry['function_prefix']==FUNCTION_PREFIX)
        test('catalog-derived execution budget',registry['budget']==budget(len(registry['cases'])))
        test('fixed identities and partitions',
             [(c['id'],c['partition']) for c in registry['cases']]==[(c['id'],c['partition']) for c in cases()])
        for family,lengths in FAMILIES.items():
            test(family+' heldout is interior',min(lengths)<HELDOUT[family]<max(lengths))
        for case in registry['cases']:
            code=static[case['function']];n=case['steps'];family=case['family']
            original=next(c for c in cases() if c['id']==case['id'])
            test(case['id']+' explicit oracle identity',case['oracle']==original['oracle'])
            test(case['id']+' independent analytic oracle bytes',
                 tuple((args.build/case[k]['name']).read_bytes() for k in ('input','auxiliary','expected'))==values(family,n))
            flow=[row for row in code['instructions'] if row['reachability']=='in complete candidate flow']
            test(case['id']+' full prefix fits record budget',len(flow)==case['full_flow_instructions']<=4096)
            test(case['id']+' full instruction bytes',all(len(row['binary_hex'])==32 for row in code['instructions']))
            imads=[row['sass'].replace('.reuse','').split(' ',1)[1].split(', ') for row in flow if row['sass'].startswith('IMAD ')]
            if family=='quad':
                # Track definitions through final-register reuse. The compiler
                # may move a last result into a register whose old value is dead.
                definitions={};hits=Counter();serial=True
                for dst,src,mul,add in imads:
                    origin,depth=definitions.get(src,(src,0))
                    serial=serial and depth==hits[origin]
                    hits[origin]+=1;definitions[dst]=(origin,depth+1)
                test(case['id']+' four real independent chains',len(imads)==4*n and serial
                     and len(hits)==4 and set(hits.values())=={n})
            if family=='dep':
                test(case['id']+' real serial chain',len(imads)==n
                     and all(imads[i][1]==imads[i-1][0] for i in range(1,len(imads))))
            if family=='chase':
                test(case['id']+' dependent ordinary loads',code['flow_opcode_histogram'].get('LDG.E')==n+1)
            test(case['id']+' no spills',not code['resources']['local_section_bytes'] and not code['resources']['stack_frame_bytes'])
        host=json.loads((args.host_build/'HOST-BUILD.json').read_text())
        test('hosts built without execution',host['schema']==BUILD and host['kind']=='hosts'
             and host['status']=='built' and host['hardware_executed'] is False
             and host['gpu_host_invocations']==0)
        test('host and cubin share fixed registry',host['registry']==ref(args.build/'REGISTRY.json'))
        test('capturing host has no event symbols','cuEvent' not in (args.host_build/'logs/capture_host-undefined.log').read_text())
        test('timing host has no dlopen/dlsym entry',not any(x in (args.host_build/'logs/timing_host-undefined.log').read_text() for x in ('dlsym','dlopen')))
        for name in ('capture_host','timing_host'):
            raw=(args.host_build/name).read_bytes()
            test(name+' embeds actual registry and new work identity',
                 sha(args.build/'REGISTRY.json').encode() in raw and WORK_ITEM.encode() in raw
                 and b'arch-explorer.measurement-host-invocation.v2' in raw)
        delta=max(FAMILIES['dep'])-min(FAMILIES['dep'])
        interval=slope_interval(10000,20000,delta,500)
        test('double-end elapsed error retained',math.isclose(interval['half_width_ns_per_round'],1000/delta))
        test('500ns assumed bound needs 10us gap for 10pct',interval['relative_quantization_le_10pct'])
        test('three identical samples cannot shrink worst-case bound',slope_interval(sum([10000]*3)/3,sum([20000]*3)/3,delta,500)==interval)
        test('flat timing preserves negative interval',slope_interval(10000,10000,delta,500)['lower_ns_per_round']<0)
        refused=0
        for argv in ((1,2,0,500),(1,2,1,-1),(float('nan'),2,1,500)):
            try:slope_interval(*argv)
            except ValueError:refused+=1
        test('invalid timing algebra rejected',refused==3)
        command('io-compile',[args.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                              '-I'+str(out/'sources'),out/'sources/io_check.cpp','-lcrypto','-o',out/'cpu-io-check'])
        command('io-run',[out/'cpu-io-check',out/'io-output'])
        catalog=json.loads((out/'io-output/catalog.json').read_text())
        test('compiled C++ catalog work item and registry',catalog['work_item']==WORK_ITEM
             and catalog['registry_sha256']==sha(args.build/'REGISTRY.json')
             and catalog['samples_per_case']==registry['budget']['samples_per_case'])
        expected=[]
        for c in registry['cases']:
            expected.append({k:c[k] for k in ('id','function','family','steps','partition','text_sha256')}|
                            {'registers':c['resources']['registers_per_thread'],
                             'shared':c['resources']['static_shared_bytes'],'local':c['resources']['local_section_bytes'],
                             'files':[c[k] for k in ('input','auxiliary','expected')]})
        test('compiled C++ complete catalog matches raw registry',catalog['cases']==expected)
        row=json.loads((out/'io-output/roundtrip.json').read_text());decoded=event_ms(row['bits'])
        test('float32 original bits/ms/ns close exactly',row['ms']==decoded and row['ns']==decoded*1e6)
        test('JSON strings round trip',row['quoted']=='a\n\t\\"b')
        # Bad actual input, never fed to a CUDA host; exercise independent oracle refusal.
        chase=next(c for c in registry['cases'] if c['family']=='chase')
        bad=out/'bad-index.bin';bad.write_bytes(struct.pack('<32I',*([chase['oracle']['modulus']]*32)))
        command('oracle-reject-index',[args.build/'cpu-oracle','chase',str(chase['steps']),
                        args.build/chase['input']['name'],bad,out/'unwritten.bin'],expected=2)
        test('rejected oracle produced no output',not (out/'unwritten.bin').exists())
        command('oracle-reject-step',[args.build/'cpu-oracle','dep','0',
                        args.build/chase['input']['name'],bad,out/'unwritten-step.bin'],expected=2)
        state.update(status='passed',interval_example=interval,hardware_accuracy='blocked: no observations')
    except Exception as error:
        state['error']=type(error).__name__+': '+str(error);raise
    finally:write_json(out/'CPU-CHECK.json',state)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','host-build','cxx','output'):p.add_argument('--'+name,type=Path,required=True)
    s=check(p.parse_args());print(json.dumps({'status':s['status'],'gpu_host_invocations':0,'gpu_queries':0}))
