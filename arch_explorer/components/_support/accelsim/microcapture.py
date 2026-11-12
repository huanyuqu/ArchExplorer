# Concrete Accel-Sim implementation; owned by the component provider.
"""Registered SIM30 originals -> the shared complete typed native renderer.

The static dictionary is lossless storage, not opcode rewriting. Dynamic global
addresses must come from the recorded MREF channel and match a separate input
oracle. This module never queries a device, executes a host, or invents a trace.
"""
from collections import Counter,defaultdict
import json,re,struct
from pathlib import Path
from arch_explorer.components.accelsim.trace import exact,digest,same_json
from arch_explorer.components.accelsim.trace_stream import file_digest
from arch_explorer.components.accelsim.microbundle import load_registry,json_file,words,BUDGET
from arch_explorer.components.accelsim.capture_input import _read_records,operands,render

SCHEMA='arch-explorer.accelsim-capture-binding.v2'
ROLES={'records','static','static_table','cubin','bundle','allocation','sample','output','preflight','invocation','supervisor','started','launch','command','environment','maps_before','maps_after','closed','capture'}
STATIC_FIELDS={'schema','nvbit','kernel','grid','block','buffer_bytes','functions','encoding','table_record_bytes','static_count','descriptions'}
DESCRIPTION_FIELDS={'sass','opcode','opcode_short','size','memory_space','is_load','is_store','is_extended','is_tma_mem','is_tma_tensor','has_pred','pred_num','pred_uniform','pred_neg','mref_count','operands'}
REGEX=r'\b(?:UR[0-9]+|URZ|UP[0-9]+|UPT|R[0-9]+|RZ|P[0-9]+|PT)\b'

def references(binding):
    exact(binding,{'schema','case','files'},'registered microbenchmark capture')
    if binding['schema']!=SCHEMA or type(binding['case']) is not str:raise ValueError('microbenchmark capture version/case')
    exact(binding['files'],ROLES,'complete microbenchmark original roles')
    files={}
    for role,row in binding['files'].items():
        exact(row,{'path','sha256'},'original capture reference')
        p=Path(row['path'])
        if not p.is_absolute() or not p.is_file() or p.stat().st_size>32*2**20 or file_digest(p)!=row['sha256']:
            raise ValueError('original capture missing/changed: '+role)
        files[role]=p.resolve()
    return files

def static_instructions(meta,table,case,compiled):
    exact(meta,STATIC_FIELDS,'complete dictionary static metadata')
    if (meta['schema']!='arch-explorer.sim30-nvbit-static.v1' or meta['nvbit']!='1.8.1' or meta['kernel']!=case['function']
        or meta['grid']!=case['grid'] or meta['block']!=case['block'] or not 0<meta['buffer_bytes']<2*2**20
        or meta['encoding']!='lossless-description-dictionary-v1' or meta['table_record_bytes']!=40):
        raise ValueError('original static version/function/geometry/encoding differs')
    desc=meta['descriptions'];n=meta['static_count'];functions=meta['functions']
    if not isinstance(desc,list) or not 0<len(desc)<=64 or any(len(json.dumps(d,separators=(',',':')).encode())>2048 for d in desc):raise ValueError('static dictionary bound')
    if type(n) is not int or not 0<n<=4352 or len(table)!=40*n or n!=len(compiled['instructions']):raise ValueError('complete static table bound/count')
    if len(functions)!=1 or set(functions[0])!={'id','name'} or functions[0]['name']!=case['function']:raise ValueError('unregistered related function')
    result={};used=set()
    for j in range(n):
        iid,fid,pc,idx,di,reserved,machine=struct.unpack_from('<6I16s',table,j*40)
        if iid in result or fid!=functions[0]['id'] or pc!=16*j or idx!=j or di>=len(desc) or reserved:
            raise ValueError('duplicate/reordered/unknown static table field')
        d=desc[di];exact(d,DESCRIPTION_FIELDS,'full original instruction descriptor');used.add(di)
        decoded=compiled['instructions'][j]
        if decoded['pc']!=pc or bytes.fromhex(decoded['binary_hex'])!=machine:raise ValueError('actual original instruction bytes differ from independent ELF disassembly')
        sass=decoded['sass'];pred=re.match(r'^@(!?)(UP|P)(\d+)\s+',sass)
        bare=sass[pred.end():] if pred else sass;op=bare.split()[0]
        if d['opcode']!=op or d['opcode_short']!=op.split('.')[0] or re.findall(REGEX,d['sass'])!=re.findall(REGEX,sass):raise ValueError('raw opcode/register metadata differs from independent disassembly')
        if d['has_pred']!=bool(pred) or (pred and (d['pred_num']!=int(pred[3]) or d['pred_uniform']!=(pred[2]=='UP') or d['pred_neg']!=(pred[1]=='!'))):raise ValueError('raw predicate metadata differs')
        for operand in d['operands']:
            common={'type','type_name','nbytes','str','is_neg','is_not','is_abs'}
            kind=operand['type_name'];extra={'REG':'reg','UREG':'reg','PRED':'pred','UPRED':'pred','CBANK':'cbank','MREF':'mref','MEM_DESC':'mem_desc','IMM_UINT64':'imm_uint64','IMM_DOUBLE':'imm_double_bits','GENERIC':'generic','TMA_PARAM_HANDLE':'tma_param_handle'}.get(kind)
            if extra is None:raise ValueError('unsupported raw operand kind')
            exact(operand,common|{extra},'raw original operand')
            if kind in ('REG','UREG','PRED','UPRED'):
                tokens=re.findall(REGEX,operand['str'])
                token=(('UR' if kind=='UREG' else 'UP' if kind=='UPRED' else 'R' if kind=='REG' else 'P')+
                       str(operand[extra]['num']))
                aliases={'R255':'RZ','UR255':'URZ','P7':'PT','UP7':'UPT'}
                if not tokens or tokens[0] not in (token,aliases.get(token)):raise ValueError('raw typed operand number conflicts with original printed operand')
        result[iid]=dict(d,id=iid,function_id=fid,offset=pc,idx=idx,binary_hex=machine.hex())
    if used!=set(range(len(desc))):raise ValueError('unused static descriptor is unrelated metadata')
    return result

def validate_addresses(bundle,case,instructions,streams,allocation):
    """Per-lane address/count oracle from concrete frozen inputs, not observed values."""
    main,aux,expected=bundle.inputs(case);next_values=words(main);starts=words(aux)
    buffers={b['name']:b for b in allocation['buffers']}
    names={'input','output','auxiliary'} if case['family']=='chase' else {'input','output'}
    if len(buffers)!=len(allocation['buffers']) or set(buffers)!=names:raise ValueError('actual live buffer set differs')
    sizes={'input':len(main),'output':len(expected),'auxiliary':len(aux)};ranges=[]
    for name,b in buffers.items():
        exact(b,{'name','address','bytes','element_bytes','output'},'actual live allocation')
        if type(b['address']) is not int or b['address']<=0 or b['address']+b['bytes']>2**64 or b['bytes']!=sizes[name] or b['element_bytes']!=4 or b['output']!=(name=='output'):
            raise ValueError('actual allocation value/extent/type differs')
        ranges.append((b['address'],b['address']+b['bytes']))
    if any(a[1]>b[0] for a,b in zip(sorted(ranges),sorted(ranges)[1:])) or sum(b['bytes'] for b in buffers.values())>BUDGET['app_bytes']:raise ValueError('overlapping/oversized actual allocations')
    if case['family']=='chase':
        if allocation['mul'] or allocation['add']:raise ValueError('chase has no scalar affine arguments')
    elif (allocation['mul'],allocation['add'])!=words(aux):raise ValueError('actual scalar arguments differ from frozen input')
    covered=defaultdict(Counter);indices=list(starts) if case['family']=='chase' else None;chases=[0]*32
    stream=streams[(0,0,0,0)]
    for row in stream:
        ins=instructions.get(row['id'])
        if ins is None:raise ValueError('dynamic record lacks original metadata')
        base=ins['opcode_short']
        if base not in ('LDG','STG'):continue
        _,_,size=operands(ins)
        for lane,address in enumerate(row['addresses']):
            if not row['mask']>>lane&1:continue
            matches=[name for name,b in buffers.items() if b['address']<=address and address+size<=b['address']+b['bytes']]
            if len(matches)!=1:raise ValueError('MREF outside actual allocation')
            name=matches[0];offset=address-buffers[name]['address']
            if (base=='STG')!=(name=='output'):raise ValueError('unexpected load/store allocation semantics')
            if case['family']=='chase' and name=='input':
                if size!=4 or offset!=4*indices[lane]:raise ValueError('actual dependent MREF does not follow independent pointer-chase oracle')
                indices[lane]=next_values[indices[lane]];chases[lane]+=1
            else:
                width=16 if case['family']=='quad' else 4
                start=lane*width
                if offset<start or offset+size>start+width or any(covered[name,lane][k] for k in range(offset,offset+size)):
                    raise ValueError('actual lane MREF/count differs from independent input/output oracle')
                covered[name,lane].update(range(offset,offset+size))
    for name in names:
        if case['family']=='chase' and name=='input':
            if chases!=[case['steps']]*32:raise ValueError('missing/extra pointer-chase dynamic load')
        else:
            width=16 if case['family']=='quad' else 4
            for lane in range(32):
                if set(covered[name,lane])!=set(range(lane*width,(lane+1)*width)):raise ValueError('incomplete independent lane memory coverage')
    return ranges

def derive(binding):
    files=references(binding);bundle=load_registry(files['bundle']);case=bundle.cases.get(binding['case'])
    if case is None or files['cubin']!=bundle.file(bundle.document['cubin']):raise ValueError('unknown/mismatched compiled bundle case')
    from arch_explorer.components.accelsim.microrecords import validate_case_attempt
    validated=validate_case_attempt(bundle,case,files,mode='structure')
    summary,_=json_file(files['capture'])
    exact(summary,{'schema','status','attempted','count','overflow','application_launches','total_application_launches','hardware_timing_consumed'},'complete original capture outcome')
    if summary['schema']!='arch-explorer.sim30-nvbit-capture.v1' or summary['status']!='captured' or summary['overflow']!=0 or summary['count']!=case['full_flow_instructions'] or summary['attempted']!=summary['count'] or summary['application_launches']!=1 or summary['total_application_launches']!=list(bundle.cases).index(case['id'])+1 or summary['hardware_timing_consumed'] is not False:raise ValueError('capture failure/overflow/launch/count differs')
    meta,_=json_file(files['static'],256*1024)
    instructions=static_instructions(meta,files['static_table'].read_bytes(),case,bundle.static[case['function']])
    grid,block,streams=_read_records(files['records'].read_bytes(),b'SIM30NV1',case['grid'],case['block'])
    if len(streams[(0,0,0,0)])!=case['full_flow_instructions']:raise ValueError('complete dynamic prefix count mismatch')
    ranges=validate_addresses(bundle,case,instructions,streams,validated['allocation'])
    return render(case['function'],validated['resource'],instructions,grid,block,streams,ranges,'SIM30NV1')
