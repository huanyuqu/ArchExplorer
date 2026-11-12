# Concrete Accel-Sim implementation; owned by the component provider.
"""Lossless SIM28NV1 observations -> typed native v7, with original ELF binding.

This is structural translation, not value simulation or a provenance signature.
Global memory addresses always come from the captured NVBit mref channel.
"""
from collections import defaultdict
import hashlib,json,math,re,struct
from pathlib import Path
from arch_explorer.components.accelsim.trace import exact,load_json,digest
from arch_explorer.components.accelsim.trace_stream import file_digest
from arch_explorer.components.accelsim.typed_trace import opcode,width,register

SCHEMA='arch-explorer.accelsim-capture-binding.v1'
ROLES={'records','static','cubin','registration','launch','preflight'}

def references(binding):
    if binding.get('schema') == 'arch-explorer.accelsim-capture-binding.v2':
        from arch_explorer.components.accelsim.microcapture import references as micro_references
        return micro_references(binding)
    exact(binding,{'schema','case','files'},'capture binding')
    if binding['schema']!=SCHEMA or type(binding['case']) is not str or not binding['case']:
        raise ValueError('versioned capture binding/case required')
    exact(binding['files'],ROLES,'capture binding files')
    result={}
    for name,row in binding['files'].items():
        exact(row,{'path','sha256'},'capture reference')
        p=Path(row['path'])
        if not p.is_absolute() or not p.is_file() or p.stat().st_size>16*2**20 or file_digest(p)!=row['sha256']:
            raise ValueError('original capture reference missing/changed/outside bound: '+name)
        result[name]=p.resolve()
    return result

def elf_texts(raw):
    if len(raw)<64 or raw[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<H',raw,18)[0]!=190:
        raise ValueError('original CUDA ELF64 little endian required')
    if (struct.unpack_from('<I',raw,48)[0]>>8)&255!=120:raise ValueError('original cubin must actually be SM120')
    pos=struct.unpack_from('<Q',raw,40)[0]
    size,count,names=struct.unpack_from('<HHH',raw,58)
    if size!=64 or not 0<names<count<=4096 or pos+size*count>len(raw):raise ValueError('ELF section bounds')
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,pos+i*size) for i in range(count)]
    string=sections[names];table=raw[string[4]:string[4]+string[5]]
    result={}
    for row in sections:
        n=row[0]
        if n>=len(table):raise ValueError('ELF name bound')
        end=table.find(b'\0',n)
        if end<0:raise ValueError('ELF unterminated name')
        name=table[n:end].decode('ascii')
        if name.startswith('.text.'):
            if name[6:] in result or row[4]+row[5]>len(raw):raise ValueError('ELF text bounds/duplicates')
            result[name[6:]]=raw[row[4]:row[4]+row[5]]
    return result

def _read_records(raw, expected_magic, expected_grid, expected_block):
    if len(raw)<64:raise ValueError('truncated SIM28NV1 header')
    h=struct.unpack_from('<8s14I',raw)
    magic,version,header,record,count,attempted,capacity,overflow,launches,*dims=h
    if (magic,version,header,record,capacity,overflow,launches)!=(expected_magic,1,64,288,4096,0,1):
        raise ValueError('unsupported/overflow/incomplete original capture header')
    if count!=attempted or not 0<count<=capacity or len(raw)!=64+count*288:
        raise ValueError('truncated/excess/partial capture records')
    grid,block=dims[:3],dims[3:]
    if grid!=expected_grid or block!=expected_block:raise ValueError('capture launch outside registered bounded ABI')
    streams=defaultdict(list)
    for offset in range(64,len(raw),288):
        iid,x,y,z,warp,active,executed,sequence,*addresses=struct.unpack_from('<8I32Q',raw,offset)
        if x>=grid[0] or y or z or warp>=math.prod(block)//32 or not active or executed&~active:
            raise ValueError('capture CTA/warp/active mask outside actual scope')
        if any(a for lane,a in enumerate(addresses) if not(executed>>lane&1)):
            raise ValueError('unexecuted lane must not invent an address')
        streams[(x,y,z,warp)].append(dict(id=iid,active=active,mask=executed,sequence=sequence,addresses=addresses))
    if set(streams)!={(cta,0,0,warp) for cta in range(grid[0]) for warp in range(math.prod(block)//32)}:
        raise ValueError('missing full CTA/warp capture stream')
    for stream in streams.values():
        stream.sort(key=lambda r:r['sequence'])
        if [r['sequence'] for r in stream]!=list(range(len(stream))):
            raise ValueError('duplicate/gap in actual per-warp dynamic sequence')
    return grid,block,streams

def read_records(raw):
    return _read_records(raw, b'SIM28NV1', [4,1,1], [128,1,1])

def _regs(kind,number,parts):
    limit=255 if kind in ('R','UR') else 7
    if type(number) is not int or not 0<=number<=limit:raise ValueError('unknown typed operand index')
    if number==limit:return [kind+str(number)]  # constants never expand across spaces
    if number+parts>limit:raise ValueError('vector crosses zero/true register boundary')
    return [kind+str(number+i) for i in range(parts)]

def _operand_reads(operand,parts=None):
    t=operand['type_name'];n=operand['nbytes']
    if type(n) is not int or n<0 or n>16:raise ValueError('unsupported operand width')
    if t in ('REG','UREG'):
        kind='R' if t=='REG' else 'UR'
        return _regs(kind,operand['reg']['num'],max(1,n//4,parts or 1))
    if t in ('PRED','UPRED'):
        return _regs('P' if t=='PRED' else 'UP',operand['pred']['num'],1)
    if t=='MREF':
        m=operand['mref'];result=[]
        if m['has_ra']:
            if m['ra_mod'] not in range(7):raise ValueError('unknown MREF R modifier')
            result+=_regs('R',m['ra_num'],2 if m['ra_mod']==6 else 1)
        if m['has_ur']:
            if m['ur_mod'] not in range(7):raise ValueError('unknown MREF UR modifier')
            result+=_regs('UR',m['ur_num'],2 if m['ur_mod']==6 else 1)
        if m['has_desc']:result+=_regs('UR',m['desc_ureg_num'],2)
        return result
    if t=='MEM_DESC':return _regs('UR',operand['mem_desc']['ureg_num'],2)
    if t=='CBANK':
        if operand['cbank']['has_reg_offset']:raise ValueError('indexed constant bank not supported')
        return []
    if t=='IMM_UINT64':return []
    if t=='GENERIC' and re.fullmatch(r'SR_(TID|CTAID)\.[XYZ]',operand['generic']):return []
    raise ValueError('unsupported observed operand kind: '+t)

def operands(ins):
    base=opcode(ins['opcode'])[0];ops=ins['operands']
    if ins['is_tma_mem'] or ins['is_tma_tensor']:
        raise ValueError('specialized instruction outside ordinary subset')
    if ins['is_extended'] and base not in ('LDG','STG','LDC','ULDC','LDCU'):
        raise ValueError('unknown extended instruction semantics')
    if base in ('STG','BRA','UBRA','EXIT','NOP'):nd=0
    elif base in ('ISETP','UISETP','PLOP3','UPLOP3'):nd=2
    elif base in ('IADD3','UIADD3') and len(ops)>1 and ops[1]['type_name'] in ('PRED','UPRED'):nd=3
    else:nd=1
    if len(ops)<nd:raise ValueError('missing observed output operands')
    size=width(ins['opcode']) if base in ('LDG','STG','LDC','ULDC','LDCU') else 0
    if size and ins['size']!=size:raise ValueError('NVBit/static opcode memory widths disagree')
    dst=[];src=[]
    for index,operand in enumerate(ops):
        parts=None
        if index<nd:
            if operand['type_name'] not in ('REG','UREG','PRED','UPRED'):raise ValueError('unknown destination operand')
            if base in ('LDG','LDC','ULDC','LDCU'):parts=max(1,size//4)
            elif base in ('IMAD','UIMAD') and 'WIDE' in ins['opcode'].split('.'):parts=2
            elif base=='IADD' and '64' in ins['opcode'].split('.'):parts=2
            dst+=_operand_reads(operand,parts)
        else:
            if base=='STG' and operand['type_name'] in ('REG','UREG'):parts=max(1,size//4)
            elif base=='IADD' and '64' in ins['opcode'].split('.') and operand['type_name'] in ('REG','UREG'):parts=2
            elif (base in ('IMAD','UIMAD') and 'WIDE' in ins['opcode'].split('.')
                    and index==len(ops)-1 and operand['type_name'] in ('REG','UREG')):parts=2
            src+=_operand_reads(operand,parts)
    dst=list(dict.fromkeys(dst));src=list(dict.fromkeys(src))
    if len(dst)>8 or len(src)>24:raise ValueError('typed native operand bound')
    if base in ('LDG','STG'):
        if ins['mref_count']!=1 or ins['memory_space']!=3:raise ValueError('ordinary global capture requires one actual NVBit global MREF')
        if bool(ins['is_load'])!=(base=='LDG') or bool(ins['is_store'])!=(base=='STG'):
            raise ValueError('observed load/store classification disagrees')
    return dst,src,size

def derive(binding):
    """Reconstruct exactly one complete actual trace, never call CUDA or native."""
    if binding.get('schema') == 'arch-explorer.accelsim-capture-binding.v2':
        from arch_explorer.components.accelsim.microcapture import derive as micro_derive
        return micro_derive(binding)
    files=references(binding)
    meta,_=load_json(files['static']);pre,_=load_json(files['preflight'])
    reg,_=load_json(files['registration']);launch,_=load_json(files['launch'])
    if reg.get('schema')!='arch-explorer.sim28-registration.v1':raise ValueError('registered original compilation required')
    entries=[k for k in reg['kernels'] if k['function']==binding['case']]
    if len(entries)!=1:raise ValueError('unique registered kernel required')
    entry=entries[0]
    if entry['cubin_sha256']!=binding['files']['cubin']['sha256'] or entry['grid']!=[4,1,1] or entry['block']!=[128,1,1]:
        raise ValueError('original cubin/registered launch differs')
    texts=elf_texts(files['cubin'].read_bytes())
    if digest(texts[binding['case']])!=entry['text_sha256']:raise ValueError('registered executable .text differs')
    if (meta['schema']!='arch-explorer.sim28-nvbit-static.v1' or meta['nvbit']!='1.8.1'
            or meta['kernel']!=binding['case'] or meta['grid']!=[4,1,1] or meta['block']!=[128,1,1]
            or not 0<meta['buffer_bytes']<2*2**20):
        raise ValueError('raw tracer/geometry/buffer identity differs')
    if (pre.get('status')!='passed' or pre['cc']!=[12,0] or pre.get('all_function_handles_live') is not True
            or pre.get('application_launches')!=0 or pre['grid']!=[4,1,1] or pre['block']!=[128,1,1]
            or {f['function'] for f in pre['functions']}!={k['function'] for k in reg['kernels']}
            or len(pre['functions'])!=len(reg['kernels']) or not all(f['legal'] for f in pre['functions'])):
        raise ValueError('live full-function resource lease did not pass')
    functions=[f for f in pre['functions'] if f['function']==binding['case']]
    if len(functions)!=1 or functions[0]['binary_version']!=120:raise ValueError('actual live function differs')
    resource=functions[0]
    if (launch.get('schema')!='arch-explorer.sim28-launch.v1' or launch.get('function')!=binding['case']
            or launch.get('case')!=entry['key'] or launch.get('grid')!=[4,1,1] or launch.get('block')!=[128,1,1]
            or launch.get('n')!=501 or launch.get('allocated_elements')!=512
            or launch.get('application_launch_budget')!=1 or launch.get('hardware_timing_requested') is not False
            or launch.get('buffer_bytes')!=meta['buffer_bytes']):
        raise ValueError('actual launch differs from registered complete scope')
    buffers=launch['buffers'];ranges=[]
    if not buffers or len({b['name'] for b in buffers})!=len(buffers):raise ValueError('unique live buffer names required')
    for b in buffers:
        if (type(b['address']) is not int or b['address']<=0 or type(b['element_bytes']) is not int
                or b['element_bytes'] not in (2,4,8,16) or b['bytes']!=512*b['element_bytes']
                or b['address']+b['bytes']>2**64):raise ValueError('actual allocation bound/width differs')
        ranges.append((b['address'],b['address']+b['bytes']))
    ordered=sorted(ranges)
    if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):raise ValueError('overlapping live allocations')
    if sum(b['bytes'] for b in buffers)+meta['buffer_bytes']>16*2**20:raise ValueError('complete data budget exceeded')
    fmap={f['id']:f['name'] for f in meta['functions']}
    if len(fmap)!=1 or set(fmap.values())!={binding['case']}:raise ValueError('device call/multifunction scope not supported')
    instructions={}
    for ins in meta['instructions']:
        pc=ins['offset'];name=fmap[ins['function_id']]
        if type(pc) is not int or pc<0 or pc%16 or pc+16>len(texts[name]) or bytes.fromhex(ins['binary_hex'])!=texts[name][pc:pc+16]:
            raise ValueError('NVBit static PC/binary differs from original executable ELF')
        if ins['id'] in instructions:raise ValueError('duplicate static instruction identity')
        instructions[ins['id']]=ins
    grid,block,streams=read_records(files['records'].read_bytes())
    return render(binding['case'], resource, instructions, grid, block, streams, ranges, 'SIM28NV1')

def render(name, resource, instructions, grid, block, streams, ranges, label):
    """Shared lossless straight-line renderer for both registered capture ABIs."""
    header=f'''-kernel name = {name}
-kernel id = 1
-grid dim = ({grid[0]},1,1)
-block dim = ({block[0]},1,1)
-shmem = {resource['static_shared']}
-nregs = {resource['registers']}
-binary version = 120
-cuda stream id = 0
-enable lineinfo = 0
-nvbit version = 1.8.1
-accelsim tracer version = 7
-shmem base_addr = 0x0
-local mem base_addr = 0x0
# {label} complete original observations; typed operands; no ISA relabel
'''
    out=[header.rstrip()]
    for cta in range(grid[0]):
        out+=['#BEGIN_TB',f'thread block = {cta},0,0']
        for warp in range(math.prod(block)//32):
            stream=streams[(cta,0,0,warp)]
            out += [f'warp = {warp}',f'insts = {len(stream)}']
            alive=0xffffffff;next_pc=0
            for r in stream:
                if r['id'] not in instructions or r['active']!=alive:raise ValueError('dynamic instruction/lane outside complete straight-line stream')
                ins=instructions[r['id']];dst,src,size=operands(ins)
                if ins['offset']!=next_pc or ins['opcode_short'] in ('BRA','UBRA'):
                    raise ValueError('missing/reordered instruction or unsupported taken control flow')
                next_pc+=16
                if ins['opcode_short']!=ins['opcode'].split('.')[0]:raise ValueError('raw opcode identity mismatch')
                if not ins['has_pred'] and r['mask']!=r['active']:raise ValueError('unpredicated execution mask disagrees')
                base=ins['opcode_short']
                addresses=[]
                if base in ('LDG','STG'):
                    addresses=[a for lane,a in enumerate(r['addresses']) if r['mask']>>lane&1]
                    if any(not any(lo<=a and a+size<=hi for lo,hi in ranges) for a in addresses):
                        raise ValueError('captured effective address outside actual live allocations')
                else:
                    if any(r['addresses']):raise ValueError('non-MREF instruction invents a global address')
                    if size:
                        # Fixed constant-bank *offsets*, not guessed global runtime
                        # addresses; indexed/dynamic CBANK is explicitly refused.
                        cb=[o['cbank'] for o in ins['operands'] if o['type_name']=='CBANK']
                        if len(cb)!=1 or cb[0]['has_reg_offset'] or not cb[0]['has_imm_offset'] or cb[0]['id']!=0 or not 0<=cb[0]['imm_offset']<65536:
                            raise ValueError('unsupported constant-bank address semantics')
                        addresses=[cb[0]['imm_offset']]*r['mask'].bit_count()
                fields=[f"{ins['offset']:04x}",f"{r['mask']:08x}",str(len(dst)),*dst,ins['opcode'],str(len(src)),*src,str(size)]
                if size:fields+=['0',*[f'{a:x}' for a in addresses]]
                guard=('UP' if ins['pred_uniform'] else 'P')+str(ins['pred_num']) if ins['has_pred'] else 'PT'
                out+=[' '.join(fields+['0','Guard',guard,'Active',f"{r['active']:08x}"])]
                if base=='EXIT':alive&=~r['mask']
            if alive:raise ValueError('capture does not complete all lanes through actual EXIT')
        out+=['#END_TB']
    return ('\n'.join(out)+'\n').encode()

