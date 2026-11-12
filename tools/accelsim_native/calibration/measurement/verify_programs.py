# SPDX-License-Identifier: Apache-2.0
"""Interpret complete finite SASS on CPU arrays; never create a device trace."""
import argparse,ast,hashlib,json,re,struct,sys,types
from collections import Counter
from fractions import Fraction
from pathlib import Path
from design import (WORK_ITEM,FUNCTION_PREFIX,FAMILIES,HELDOUT,MUL,ADD,INPUT_SEED,
                    INPUT_STRIDE,INDEX_MUL,INDEX_ADD,INDEX_MODULUS,START_MUL,START_ADD,
                    affine,sha,write_json)


def reference(path):
    p=Path(path).resolve()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}


def offline_backend():
    """Use the exact pure file hash helper without loading the tracez adapter.

    capture_input/microbundle only import file_digest from trace_stream. The
    compressed-reader module otherwise initializes ctypes on import. Selecting
    its unchanged function AST permits a strict no-loading CPU audit. Validation
    rules and ELF readers are imported intact from the actual package.
    """
    import importlib.util
    spec=importlib.util.find_spec('arch_explorer.components.accelsim.trace_stream')
    path=Path(spec.origin);raw=path.read_bytes();parsed=ast.parse(raw)
    node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='file_digest')
    selected=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(selected)
    module=types.ModuleType(spec.name);module.__file__=str(path)
    module.__dict__.update(hashlib=hashlib,Path=Path)
    exec(compile(selected,str(path),'exec'),module.__dict__);sys.modules[spec.name]=module
    denied=[]
    def audit(event,args):
        if event in ('subprocess.Popen','ctypes.dlopen','fcntl.flock','os.system','os.posix_spawn','os.posix_spawnp'):
            denied.append(event);raise RuntimeError('offline read-only audit forbids execution/loading/locks')
        if event=='open' and str(args[0]).startswith(('/dev/nvidia','/proc/driver/nvidia')):
            denied.append(event);raise RuntimeError('offline audit forbids device files')
    sys.addaudithook(audit)
    return denied,{'module':spec.name,'source':reference(path),'selected_function':'file_digest',
                   'selected_AST_sha256':hashlib.sha256(ast.dump(selected).encode()).hexdigest(),
                   'compressed_reader_not_imported':True}


def words(raw):return struct.unpack('<'+'I'*(len(raw)//4),raw)
def packed(values):return struct.pack('<'+'I'*len(values),*values)


def verify(bundle_path,output):
    denied,helper=offline_backend()
    from arch_explorer.components.accelsim.microbundle import load_registry
    from arch_explorer.components.accelsim.capture_input import elf_texts
    bundle=load_registry(Path(bundle_path)/'REGISTRY.json');root=bundle.path.parent
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    registry=bundle.document;static=bundle.static;texts=elf_texts(bundle.file(registry['cubin']).read_bytes())
    expected_cases=[(family,n,'heldout' if n==HELDOUT[family] else 'fit') for family,ns in FAMILIES.items() for n in ns]
    assert [(c['family'],c['steps'],c['partition']) for c in bundle.cases.values()]==expected_cases
    assert registry['work_item']==WORK_ITEM and registry['function_prefix']==FUNCTION_PREFIX
    rows=[]
    for cid,c in bundle.cases.items():
        caseout=out/cid;caseout.mkdir();family,n=c['family'],c['steps']
        main,aux,expected=bundle.inputs(c);iterated=(root/(cid+'.oracle.bin')).read_bytes()
        if family=='dep':
            assert words(main)==tuple((INPUT_SEED+INPUT_STRIDE*i)&0xffffffff for i in range(32))
            assert words(aux)==(MUL,ADD)
        else:
            assert family=='chase' and words(main)==tuple((INDEX_MUL*i+INDEX_ADD)%INDEX_MODULUS for i in range(INDEX_MODULUS))
            assert words(aux)==tuple((START_MUL*i+START_ADD)%INDEX_MODULUS for i in range(32))
            visited=[];x=0
            while x not in visited:visited.append(x);x=words(main)[x]
            assert len(visited)==INDEX_MODULUS and x==0 and expected!=aux
        assert expected==iterated
        fact=static[c['function']];allcode=fact['instructions'];body=texts[c['function']]
        assert body==b''.join(bytes.fromhex(r['binary_hex']) for r in allcode)
        assert [r['pc'] for r in allcode]==list(range(0,len(body),16))
        flow=[r for r in allcode if r['reachability']=='in complete candidate flow']
        assert flow==allcode[:len(flow)] and flow[-1]['sass']=='EXIT'
        assert len(flow)==(n+11 if family=='dep' else 2*n+11)<=4096
        assert fact['static_native_subset']=='eligible' and not fact['blockers']
        assert all(re.fullmatch(r'NOP(?:\..*)?|BRA\s+`?\(\.L_[A-Za-z0-9_]+\)',r['sass']) for r in allcode[len(flow):])
        resources=c['resources']
        assert resources['registers_per_thread']==(9 if family=='dep' else 18)
        assert resources['static_shared_bytes']==resources['local_section_bytes']==resources['stack_frame_bytes']==0
        assert resources['compiler_reserved_shared_bytes']==64

        bases={'input':0x410000010,'auxiliary':0x510000020,'output':0x610000040}
        arrays={'input':list(words(main)),'auxiliary':list(words(aux)),'output':[0xa5a5a5a5]*32}
        bank={0x37c:0,0x358:0x1020304050607080,0x380:bases['input'],
              0x388:bases['output'] if family=='dep' else bases['auxiliary'],
              0x390:(ADD<<32)|MUL if family=='dep' else bases['output']}
        registers=[{} for _ in range(32)];memory=[];counts=Counter();last_load=[None]*32
        def clean(x):return x.removesuffix('.reuse')
        def get(rr,x):
            x=clean(x)
            if x in ('RZ','URZ'):return 0
            if re.fullmatch(r'(R|UR)\d+',x):
                if x not in rr:raise ValueError('register read before definition: '+x)
                return rr[x]
            return int(x,0)&0xffffffff
        def put(rr,x,value):rr[clean(x)]=value&0xffffffff
        def pair(x):
            m=re.fullmatch(r'(UR|R)(\d+)(?:\.64)?',clean(x))
            if not m:raise ValueError('unhandled 64-bit register pair: '+x)
            return m[1]+m[2],m[1]+str(int(m[2])+1)
        def get64(rr,x):
            lo,hi=pair(x);return get(rr,lo)|(get(rr,hi)<<32)
        def put64(rr,x,value):
            lo,hi=pair(x);put(rr,lo,value);put(rr,hi,value>>32)
        def memory_at(address):
            if address%4:raise ValueError('unaligned u32 address')
            for role,base in bases.items():
                offset=address-base
                if 0<=offset<4*len(arrays[role]):return role,offset//4
            raise ValueError('address outside synthetic program arrays')
        for instruction in flow:
            pc=instruction['pc'];sass=instruction['sass'].replace('.reuse','')
            if sass=='EXIT':break
            op,operands=sass.split(' ',1);args=operands.split(', ')
            for lane,rr in enumerate(registers):
                if op in ('LDC','LDC.64','LDCU.64'):
                    m=re.fullmatch(r'c\[0x0\]\[(0x[0-9a-f]+)\]',args[1]);assert m
                    value=bank[int(m[1],16)]
                    if op=='LDC':put(rr,args[0],value)
                    else:put64(rr,args[0],value)
                elif op=='S2R':
                    assert args[1]=='SR_TID.X';put(rr,args[0],lane)
                elif op=='IMAD':
                    dst,src,mul,add=args;put(rr,dst,get(rr,src)*get(rr,mul)+get(rr,add))
                elif op=='IMAD.WIDE.U32':
                    dst,src,mul,add=args;put64(rr,dst,(get(rr,src)*get(rr,mul)+get64(rr,add))&0xffffffffffffffff)
                elif op in ('LDG.E','STG.E'):
                    operand=args[1] if op=='LDG.E' else args[0]
                    m=re.fullmatch(r'desc\[(UR\d+)\]\[(R\d+\.64)\]',operand);assert m
                    assert get(rr,m[1])==(bank[0x358]&0xffffffff)
                    role,index=memory_at(get64(rr,m[2]))
                    memory.append((pc,lane,{'input':0,'auxiliary':1,'output':2}[role],index*4))
                    counts[op,role]+=1
                    if op=='LDG.E':
                        assert role!='output'
                        previous=last_load[lane]
                        if family=='chase':
                            if previous is None:assert role=='auxiliary' and index==lane
                            else:assert role=='input' and index==previous
                        else:assert role=='input' and index==lane
                        value=arrays[role][index];put(rr,args[0],value);last_load[lane]=value
                    else:
                        assert role=='output' and index==lane;arrays[role][index]=get(rr,args[1])
                else:raise ValueError('reachable SASS not independently interpreted: '+sass)
        actual=packed(arrays['output']);assert actual==expected==iterated
        (caseout/'sass-oracle.bin').write_bytes(actual)
        (caseout/'synthetic-array-offsets.bin').write_bytes(b''.join(struct.pack('<4I',*r) for r in memory))
        assert counts['STG.E','output']==32
        if family=='dep':
            assert counts['LDG.E','input']==32 and sum(counts.values())==64
            imads=[r['sass'].replace('.reuse','').split(' ',1)[1].split(', ') for r in flow if r['sass'].startswith('IMAD ')]
            assert len(imads)==n and all(imads[i][1]==imads[i-1][0] for i in range(1,len(imads)))
            dependency={'serial_IMAD_count':len(imads),'each_source_is_previous_destination':True}
        else:
            assert counts['LDG.E','auxiliary']==32 and counts['LDG.E','input']==32*n and sum(counts.values())==32*(n+2)
            a,b=affine(n,INDEX_MUL,INDEX_ADD,INDEX_MODULUS)
            dependency={'dependent_table_loads_per_lane':n,'each_index_is_previous_loaded_value':True,
                        'permutation_cycle':len(visited),'length_residue':n%INDEX_MODULUS,
                        'endpoint_affine':{'mul':a,'add':b,'modulus':INDEX_MODULUS},'output_equals_starts':False}
        row={'case':cid,'family':family,'steps':n,'partition':c['partition'],'function':c['function'],
             'kind':'CPU interpretation of complete actual SASS with synthetic arrays',
             'full_flow':len(flow),'all_static_instructions':len(allcode),'text_bytes':len(body),'text_sha256':hashlib.sha256(body).hexdigest(),
             'every_16B_instruction_bound_to_ELF':True,'only_dead_suffix_not_interpreted':True,'resources':resources,
             'oracles':{'python_affine':reference(bundle.file(c['expected'])),
                        'CPP_iteration':reference(root/(cid+'.oracle.bin')),'SASS_register_interpretation':reference(caseout/'sass-oracle.bin'),'equal':True},
             'address_oracle':{'bases_kind':'synthetic only, no CUDA pointer or capture','bases':bases,
                              'format':'LE u32(pc,lane,buffer_id,byte_offset); 0=input 1=auxiliary 2=output',
                              'offsets':reference(caseout/'synthetic-array-offsets.bin'),'aligned_and_in_bounds':True,
                              'counts':[{'opcode':op,'buffer':role,'count':v} for (op,role),v in counts.items()]},
             'dependency':dependency,'physical_parameters':None,'hardware_accuracy':'not validated'}
        write_json(caseout/'RESULT.json',row);rows.append(row)
    app=[c['input']['bytes']+c['auxiliary']['bytes']+2*c['expected']['bytes'] for c in bundle.cases.values()]
    bounds=json.loads((root/'CAPTURE-BUDGET.json').read_text())
    assert max(app)==bounds['application_peak_bytes']==1408 and sum(app)==bounds['all_cases_application_sum_bound_bytes']==7200
    assert bounds['managed_record_channel_bytes']==1179712 and bounds['aggregate_capture_max_bytes']<=16*2**20
    assert bounds['dynamic_records']==sum(r['full_flow'] for r in rows)
    assert bounds['static_instructions']==sum(r['all_static_instructions'] for r in rows)
    result={'schema':'arch-explorer.measurement-offline-program-check.v1','status':'passed','kind':'offline-CPU',
            'registry':reference(bundle.path),'cubin':reference(bundle.file(registry['cubin'])),'rows':rows,'bounds':bounds,
            'conditional_resolution':{'assumed_each_elapsed_abs_error_ns':1000,
              'formal_mean_shrinks_worst_case_bound':False,'endpoint_gap_for_10pct_ns':20000,
              'families':{f:{'fit_step_gap':max(n for n in ns if n!=HELDOUT[f])-min(n for n in ns if n!=HELDOUT[f]),
                'round_slope_error_half_width_ns':str(Fraction(2000,max(n for n in ns if n!=HELDOUT[f])-min(n for n in ns if n!=HELDOUT[f])))} for f,ns in FAMILIES.items()},
              'observed_hardware_times':None,'physical_parameters':None},
            'pure_helper_selection':helper,'forbidden_calls':denied,
            'measurement_hosts':0,'GPU_queries_cuInit_kernel_event_locks_leases_reservations':0,
            'ctypes_or_NVBit_dlopen':0,'native_builds_runs':0,'physical_parameters':None,'hardware_accuracy':'not validated'}
    write_json(out/'RESULTS.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();r=verify(args.bundle,args.output)
    print(json.dumps({'status':r['status'],'cases':len(r['rows']),'dynamic':r['bounds']['dynamic_records'],
                      'all_static':r['bounds']['static_instructions'],'forbidden_calls':r['forbidden_calls']}))
