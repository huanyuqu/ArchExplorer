"""CPU-only compilation/ELF/disassembly checks. Never starts host or loads CUDA."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile

from verify_capture import text_sections,read_records

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    args=parser.parse_args(); build=args.build.resolve()
    receipt=json.loads((build/'BUILD.json').read_text())
    sections=text_sections(build/'kernels.cubin')
    assert set(sections)=={'sim28_u16','sim28_u32_u64','sim28_u128'}
    assembly=Path(receipt['disassembly']).read_text()
    assert '.target\tsm_120a' in assembly
    name=None; pending=None; decoded={n:[] for n in sections}
    for line in assembly.splitlines():
        start=re.search(r'\.section\s+\.text\.([^,\s]+)',line)
        if start: name=start[1]; continue
        if name not in decoded: continue
        first=re.search(r'^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*/\* 0x([0-9a-f]{16}) \*/',line)
        if first:
            pending=(int(first[1],16),first[2],int(first[3],16)); continue
        second=re.search(r'^\s*/\* 0x([0-9a-f]{16}) \*/',line)
        if second and pending:
            pc,sass,low=pending; high=int(second[1],16)
            raw=struct.pack('<QQ',low,high)
            assert raw==sections[name][pc:pc+16],(name,pc)
            sass=re.split(r'\s[&?]',sass,1)[0].rstrip('; ').strip()
            decoded[name].append({'pc':pc,'sass':sass,'binary_hex':raw.hex()})
            pending=None
    assert all(len(decoded[n])*16==len(body) for n,body in sections.items())
    expected={'sim28_u16':{'LDG.E.U16','STG.E.U16'},'sim28_u32_u64':{'LDG.E','STG.E','LDG.E.64','STG.E.64'},
        'sim28_u128':{'LDG.E.128','STG.E.128'}}
    facts={}
    for name,rows in decoded.items():
        opcodes=Counter(re.sub(r'^@!?[UP]+\d+\s+','',r['sass']).split()[0] for r in rows)
        assert expected[name]<=set(opcodes)
        assert any('desc[UR4]' in r['sass'] for r in rows)
        assert any(r['sass'].startswith('@P0 EXIT') for r in rows)
        facts[name]={'text_sha256':hashlib.sha256(sections[name]).hexdigest(),'text_bytes':len(sections[name]),
            'instructions':rows,'opcode_histogram':dict(opcodes),'descriptor_UR4_present':True,'tail_P0_EXIT_present':True}
    # Pure C++ oracle binary: no CUDA includes, libraries or device entry.
    exe=build/'cpu-oracle-test'
    with (build/'CPU-ORACLE-COMPILE.log').open('wb') as log:
        subprocess.run(['/usr/bin/g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(HERE/'oracle_test.cpp'),'-o',str(exe)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    run=subprocess.run([str(exe)],capture_output=True,text=True,check=True)
    (build/'CPU-ORACLE.log').write_text(run.stdout+run.stderr)
    cpu=json.loads(run.stdout)
    # Hand-packed synthetic ABI specimen is never called captured or measured.
    header=struct.pack('<8s14I',b'SIM28NV1',1,64,288,1,1,4096,0,1,4,1,1,128,1,1)
    record=struct.pack('<8I32Q',0,0,0,0,0,0xffffffff,0,0,*([0]*32))
    variants={'valid-synthetic':header+record,'wrong-magic':b'BADMAGIC'+header[8:]+record,
        'short-record':header+record[:-1],'trailing-data':header+record+b'x',
        'overflow':header[:32]+struct.pack('<I',1)+header[36:]+record,
        'exec-outside-active':header+struct.pack('<8I32Q',0,0,0,0,0,1,2,0,*([0]*32))}
    abi=[]
    with tempfile.TemporaryDirectory(prefix='sim28-synthetic-abi-') as tmp:
        for label,raw in variants.items():
            p=Path(tmp)/(label+'.bin');p.write_bytes(raw)
            try: result=read_records(p); outcome='passed'
            except ValueError: outcome='blocked'
            assert outcome==('passed' if label=='valid-synthetic' else 'blocked')
            abi.append({'synthetic_case':label,'outcome':outcome})
    result={'schema':'arch-explorer.sim28-capture-offline-check.v1','status':'passed','functions':facts,
        'cpu_oracle':cpu,'synthetic_abi_tests':abi,'nvbit_core_version':'1.8.1','GPU_queries':0,'GPU_executions':0,
        'host_or_tool_loaded':False,'scope':'offline compilation and source/cubin binary checks only; capture/compatibility not executed'}
    (build/'OFFLINE-CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','functions':{n:{k:v for k,v in f.items() if k!='instructions'} for n,f in facts.items()},
        'synthetic_abi_tests':len(abi),'cpu_oracle':cpu,'GPU_queries':0,'GPU_executions':0}))


if __name__=='__main__': main()
