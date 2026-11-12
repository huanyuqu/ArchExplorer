# Concrete Accel-Sim implementation; owned by the component provider.
"""Bounded ordinary SM90 traceg/tracez reader, matching pinned trace_parser.

A trace is an observed dynamic instruction stream, not a static unique-PC IR.
No instructions, masks, warps or CTAs are removed. Native execution keeps its
original bytes. Python only validates input facts and completeness.
"""
from collections import Counter
import ctypes as C
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
from arch_explorer.components.accelsim.isa import HOPPER_OPCODES
from arch_explorer.components.accelsim.native_isa import require_adapter_isa

LIMITS=dict(file_bytes=64*2**20,text_bytes=256*2**20,chunk_bytes=8*2**20,line_bytes=65536,ctas=4096,records=2_000_000)
MODIFIERS=set('E U8 S8 U16 S16 U32 S32 U64 S64 8 16 32 64 128 L R HI LO LUT AND OR XOR EQ NE LT LE GT GE GEU NAN NUM F32 F64 RN RZ RM RP FTZ SAT WIDE HI X CC IADD MOV U SHF SYNC RED POPC BFLY IDX UP DOWN STRONG GPU SYS CTA PRIVATE CONSTANT NODEP ZD F16 F16x2 BF16 BF16x2 NTZ TRUNC CEIL FLOOR EX2 RCP RSQ SQRT SIN COS LG2 COPYSIGN ABS NEG CARRY RELAXED'.split())
SPECIAL={'GMMA','UTMALDG','UTMASTG','UTMAPF','UTMAREDG','UBLKCP','UBLKPF','UBLKRED','LDGSTS','LDGDEPBAR','DEPBAR','REPLAY_BEGIN','REPLAY_END'}
# These require tensor/async/barrier-value metadata or changing launch resources;
# ordinary operand parsing must not silently erase those fields.
SPECIAL |= {'REPLAY_START','WARPGROUP','UCGABAR_ARV','UCGABAR_WAIT','USETMAXREG','USETSHMSZ'}
SPECIAL_CATEGORIES={'TMA_OP','SPECIALIZED_UNIT_3_OP','SPECIALIZED_UNIT_5_OP','SYNCS_OP','ARRIVES_OP','STAS_OP','FENCE_OP'}


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


class TraceArtifact:
    def __init__(self,path,sha256):self.path=Path(path);self.sha256=sha256
    def stage(self,directory):
        name='kernel-1'+self.path.suffix;out=Path(directory)/name
        with self.path.open('rb') as src,out.open('xb') as dst:shutil.copyfileobj(src,dst,2**20)
        if file_digest(out)!=self.sha256:raise ValueError('trace changed between validation and native staging')
        return name


def _read(f,n):
    if n<0 or n>LIMITS['chunk_bytes']:raise ValueError('tracez read size exceeds bounded chunk contract')
    b=f.read(n)
    if len(b)!=n:raise ValueError('tracez truncated data/index')
    return b


@lru_cache(maxsize=1)
def _zstd():
    try:z=C.CDLL('libzstd.so.1')
    except OSError as e:raise ImportError('tracez validation requires native libzstd.so.1 (also an Accel-Sim dependency)') from e
    for name,args,result in [('ZSTD_getFrameContentSize',[C.c_void_p,C.c_size_t],C.c_ulonglong),('ZSTD_findFrameCompressedSize',[C.c_void_p,C.c_size_t],C.c_size_t),('ZSTD_isError',[C.c_size_t],C.c_uint),('ZSTD_createDCtx',[],C.c_void_p),('ZSTD_freeDCtx',[C.c_void_p],C.c_size_t),('ZSTD_decompress_usingDict',[C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],C.c_size_t)]:
        fn=getattr(z,name);fn.argtypes=args;fn.restype=result
    return z


def compressed_lines(path):
    """Read fixed tracez_format.h footer and one zstd subchunk at a time."""
    with Path(path).open('rb') as f:
        header=bytearray()
        while len(header)<=LIMITS['line_bytes']:
            c=_read(f,1)
            if c==b'\0':break
            header+=c
        else:raise ValueError('tracez header exceeds limit')
        ds,sub=struct.unpack('<II',_read(f,8))
        if ds>LIMITS['chunk_bytes'] or not 1<=sub<=65536:raise ValueError('tracez compression header unsupported')
        dictionary=_read(f,ds);start=f.tell();f.seek(-8,2);end=f.tell();index=struct.unpack('<Q',_read(f,8))[0]
        if not start<=index<end:raise ValueError('tracez footer pointer outside payload')
        f.seek(index);count=struct.unpack('<I',_read(f,4))[0]
        if not 1<=count<=LIMITS['ctas']:raise ValueError('tracez CTA index exceeds limit')
        tbs=[];total=0
        for _ in range(count):
            tb=struct.unpack('<11I',_read(f,44));warps=[]
            if not 1<=tb[-1]<=32:raise ValueError('tracez warp count invalid')
            for _ in range(tb[-1]):
                n,sc,offset=struct.unpack('<IIQ',_read(f,16))
                if not 1<=sc<=4096 or not 1<=n<=LIMITS['records']:raise ValueError('tracez warp index invalid')
                ns=struct.unpack('<'+str(sc)+'I',_read(f,sc*4))
                if sum(ns)!=n or any(not 1<=x<=sub for x in ns):raise ValueError('tracez subchunk instruction index mismatch')
                warps.append((n,sc,offset,ns));total+=n
            tbs.append((tb,warps))
        if f.tell()!=end or total>LIMITS['records']:raise ValueError('tracez index trailing bytes/record budget')
        spans=[]
        for tb,warps in tbs:
            for n,sc,off,ns in warps:
                if not start<=off<index:raise ValueError('tracez warp pointer invalid')
                f.seek(off)
                if struct.unpack('<II',_read(f,8))!=(n,sc):raise ValueError('tracez footer/body warp identity mismatch')
                for _ in ns:
                    size=struct.unpack('<I',_read(f,4))[0]
                    if not 0<size<=LIMITS['chunk_bytes'] or f.tell()+size>index:raise ValueError('tracez compressed range invalid')
                    f.seek(size,1)
                spans.append((off,f.tell()))
        cursor=start
        for lo,hi in sorted(spans):
            if lo!=cursor:raise ValueError('tracez overlapping or unreferenced execution payload')
            cursor=hi
        if cursor!=index:raise ValueError('tracez unreferenced trailing payload')
        yield from header.decode('ascii').splitlines()
        z=_zstd();ctx=z.ZSTD_createDCtx();db=C.create_string_buffer(dictionary)
        if not ctx:raise ValueError('zstd context allocation failed')
        try:
            for tb,warps in tbs:
                yield '#BEGIN_TB';yield 'thread block = '+','.join(map(str,tb[:3]))
                yield 'cluster id = '+','.join(map(str,tb[3:6]));yield 'cluster cta = '+','.join(map(str,tb[6:9]));yield 'cluster rank = '+str(tb[9])
                for w,(n,sc,off,ns) in enumerate(warps):
                    yield 'warp = '+str(w);yield 'insts = '+str(n);f.seek(off+8)
                    for expected in ns:
                        size=struct.unpack('<I',_read(f,4))[0];payload=_read(f,size);src=C.create_string_buffer(payload)
                        length=z.ZSTD_getFrameContentSize(src,size)
                        if not 0<length<=LIMITS['chunk_bytes'] or z.ZSTD_findFrameCompressedSize(src,size)!=size:raise ValueError('tracez frame size/content bound invalid')
                        out=C.create_string_buffer(length)
                        actual=z.ZSTD_decompress_usingDict(ctx,out,length,src,size,db,len(dictionary))
                        if z.ZSTD_isError(actual) or actual!=length:raise ValueError('tracez zstd decode failed')
                        lines=out.raw.decode('ascii').splitlines()
                        if len(lines)!=expected:raise ValueError('tracez actual subchunk record count differs')
                        yield from lines
                yield '#END_TB'
        finally:z.ZSTD_freeDCtx(ctx)


def text_lines(path):
    with Path(path).open('r',encoding='ascii') as f:
        while True:
            line=f.readline(LIMITS['line_bytes']+1)
            if not line:return
            if len(line)>LIMITS['line_bytes']:raise ValueError('trace line exceeds bound')
            yield line


def _reg(token,nregs):
    m=re.fullmatch(r'(R|UR|P|UP)([0-9]+)',token)
    if not m:raise ValueError('invalid native trace register '+token)
    value=int(m[2]);limit={'R':256,'UR':64,'P':8,'UP':8}[m[1]]
    if value>=limit or (m[1]=='R' and value>=nregs and value!=255):raise ValueError('trace register outside declared resources '+token)


def instruction(line,header):
    t=line.split()
    if header['lineinfo']:
        if not t or not t.pop(0).isdigit():raise ValueError('trace lineinfo invalid')
    try:
        pc=int(t.pop(0),16);mask=int(t.pop(0),16);nd=int(t.pop(0))
        if not 0<=pc<=0xffffffff or pc%16 or not 0<=mask<=0xffffffff or not 0<=nd<=1:raise ValueError('PC/mask/destination count invalid')
        for _ in range(nd):_reg(t.pop(0),header['nregs'])
        op=t.pop(0);parts=op.split('.');base=parts[0]
        if base in SPECIAL or HOPPER_OPCODES.get(base) in SPECIAL_CATEGORIES:raise ValueError('adapter: unsupported specialized opcode/operand format '+op)
        if base not in HOPPER_OPCODES:raise ValueError('native_isa: undefined Hopper opcode '+base)
        if any(x not in MODIFIERS for x in parts[1:]):raise ValueError('adapter: unsupported specialized opcode/operand format '+op)
        ns=int(t.pop(0))
        if not 0<=ns<=4:raise ValueError('native source count outside MAX_SRC=4')
        for _ in range(ns):_reg(t.pop(0),header['nregs'])
        width=int(t.pop(0));addresses=[];mode=None
        if width not in (0,1,2,4,8,16):raise ValueError('memory width unsupported')
        needs_addresses=base in {'LDG','STG','LDL','STL','LDS','STS','STAS','ATOMG','ATOM','RED','ATOMS','LDSM','LD','ST','LDGSTS'}
        if needs_addresses and not width:raise ValueError('missing native memory address fields')
        if width:
            mode=int(t.pop(0));active=mask.bit_count()
            if mode==0:addresses=[int(t.pop(0),16) for _ in range(active)]
            elif mode==1:
                address=int(t.pop(0),16);stride=int(t.pop(0))
                if not 0<=address<2**64 or not -2**31<=stride<2**31:raise ValueError('native base/stride integer range exceeded')
                addresses=[address+i*stride for i in range(active)]
            elif mode==2:
                address=int(t.pop(0),16)
                if not 0<=address<2**64:raise ValueError('native base integer range exceeded')
                addresses=[address] if active else []
                for _ in range(max(0,active-1)):
                    delta=int(t.pop(0))
                    if not -2**63<=delta<2**63:raise ValueError('native delta integer range exceeded')
                    address+=delta;addresses.append(address)
            else:raise ValueError('memory address encoding unsupported')
            if any(not 0<=a<2**64 for a in addresses):raise ValueError('memory address outside uint64')
        if header['version']==3:
            if len(t)!=1:int('invalid')
            int(t[0],16 if width and mode==0 else 10)
        else:
            pos=next((i for i,v in enumerate(t) if v in ('Val','NoVal')),None)
            if pos not in (1,2):raise ValueError('v6 immediate/value fields invalid')
            for value in t[:pos]:int(value,16 if width and mode==0 else 10)
            flag=t[pos];values=t[pos+1:]
            if flag=='Val':
                for _ in range(nd+ns):
                    n=int(values.pop(0))
                    if n not in (1,32):raise ValueError('v6 register value count invalid')
                    for _ in range(n):int(values.pop(0),16)
            if values:raise ValueError('v6 trailing or unexpected value fields')
        return pc,mask,base,op,width
    except (IndexError,StopIteration) as e:raise ValueError('truncated native instruction fields') from e


def facts(path, *, dialect='sm90'):
    if dialect not in ('sm90', 'sm120-typed-v1'):
        raise ValueError('unsupported native trace dialect')
    path=Path(path)
    if path.suffix not in ('.traceg','.tracez') or not 0<path.stat().st_size<=LIMITS['file_bytes']:raise ValueError('trace format/file byte bound exceeded')
    original=file_digest(path);lines=compressed_lines(path) if path.suffix=='.tracez' else text_lines(path)
    headers={};header=None;header_comment=False;tb=None;warp=None;left=0;alive=0;seen={};cta_facts=[];warp_facts=[];ops=Counter();masks=Counter();records=0;threads=0;repeats=0;pcset=set();bodyhash=None;decoded=0;memory=0
    def finish_warp():
        if warp is not None:
            if left or alive:raise ValueError('truncated warp stream or EXIT does not retire all block lanes')
            warp_facts.append(dict(id=warp,records=local_records,thread_instructions=local_threads,flow_sha256=bodyhash.hexdigest()))
    for raw in lines:
        decoded+=len(raw)+1
        if len(raw)>LIMITS['line_bytes'] or decoded>LIMITS['text_bytes']:raise ValueError('decoded trace resource bound exceeded')
        line=raw.strip()
        if not line:continue
        if header is None and line!='#BEGIN_TB':
            if line.startswith('#'):
                if header_comment:raise ValueError('adapter: native text header consumes exactly one comment before CTA body')
                header_comment=True
                continue
            if header_comment:raise ValueError('header fields after native header comment terminator')
            m=re.fullmatch(r'-([^=]+)\s*=\s*(.*)',line)
            if not m:raise ValueError('invalid trace header line')
            key=m[1].strip()
            if key in headers:raise ValueError('duplicate trace header '+key)
            headers[key]=m[2];continue
        if header is None:
            if path.suffix=='.traceg' and not header_comment:raise ValueError('adapter: native text reader requires a header comment before #BEGIN_TB')
            required={'kernel name','kernel id','grid dim','block dim','shmem','nregs','binary version','cuda stream id','enable lineinfo','nvbit version','accelsim tracer version','shmem base_addr','local mem base_addr'}
            if set(headers)!=required:raise ValueError('unsupported/missing native trace header fields')
            if dialect == 'sm90':
                require_adapter_isa('sm'+str(int(headers['binary version'])))
            elif int(headers['binary version']) != 120:
                raise ValueError('typed native trace requires actual SM120, never relabeled SM90')
            def dim(k):
                m=re.fullmatch(r'\(([0-9]+),([0-9]+),([0-9]+)\)',headers[k+' dim'])
                if not m:raise ValueError('invalid trace geometry')
                a=list(map(int,m.groups()))
                if min(a)<1:raise ValueError('zero trace dimension')
                return a
            grid,block=dim('grid'),dim('block');nt=math.prod(block);ct=math.prod(grid)
            header=dict(name=headers['kernel name'],grid=grid,block=block,stream=int(headers['cuda stream id']),kernel_id=int(headers['kernel id']),nregs=int(headers['nregs']),shmem=int(headers['shmem']),version=int(headers['accelsim tracer version']),lineinfo=int(headers['enable lineinfo']))
            versions = (3,6) if dialect == 'sm90' else (7,)
            if not 1<=nt<=1024 or not 1<=ct<=LIMITS['ctas'] or header['stream']!=0 or header['version'] not in versions or header['lineinfo'] not in (0,1) or not 1<=header['nregs']<=255 or header['shmem']<0:raise ValueError('trace launch/resources/version outside validated input domain')
            if not header['name'] or len(header['name'])>4096 or any(ord(c)<32 or ord(c)>126 for c in header['name']):raise ValueError('invalid native kernel name')
            for key in ('shmem base_addr','local mem base_addr'):
                if not 0<=int(headers[key],16)<2**64:raise ValueError('invalid base address')
        if line=='#BEGIN_TB':
            if tb is not None:raise ValueError('nested/incomplete CTA')
            tb='pending';warp=None;warp_facts=[];seen_warps=set();cluster={};continue
        if line=='#END_TB':
            finish_warp()
            if not isinstance(tb,tuple) or seen_warps!=set(range((nt+31)//32)):raise ValueError('missing/extra warp or CTA coordinates')
            if cluster and (set(cluster)!={'cluster id','cluster cta','cluster rank'} or cluster['cluster cta']!=[0,0,0] or cluster['cluster rank']!=[0] or cluster['cluster id'] not in ([0,0,0],list(tb))):raise ValueError('adapter: multi-CTA cluster execution unsupported')
            cta_facts.append(dict(id=list(tb),warps=sorted(warp_facts,key=lambda w:w['id'])));tb=None;warp=None;continue
        if line.startswith('thread block = '):
            if tb!='pending':raise ValueError('duplicate/outside CTA coordinates')
            tb=tuple(map(int,line.split('=',1)[1].split(',')))
            if len(tb)!=3 or tb in seen or any(not 0<=v<grid[i] for i,v in enumerate(tb)):raise ValueError('duplicate/out-of-range CTA')
            seen[tb]=True;continue
        if line.startswith(('cluster id =','cluster cta =','cluster rank =')):
            key,value=line.split('=',1);key=key.strip();vals=list(map(int,value.split(',')))
            if tb is None or warp is not None or key in cluster or any(v<0 for v in vals):raise ValueError('invalid cluster metadata')
            cluster[key]=vals
            continue
        if line.startswith('warp = '):
            if not isinstance(tb,tuple):raise ValueError('warp outside CTA')
            finish_warp();warp=int(line.split('=')[1])
            if warp in seen_warps or not 0<=warp<(nt+31)//32:raise ValueError('duplicate/out-of-range warp')
            seen_warps.add(warp);alive=(1<<min(32,nt-warp*32))-1;left=-1;pcset=set();bodyhash=hashlib.sha256();local_records=0;local_threads=0;continue
        if line.startswith('insts = '):
            if warp is None or left!=-1:raise ValueError('duplicate/outside warp count')
            left=int(line.split('=')[1])
            if not 1<=left<=LIMITS['records']:raise ValueError('warp record count invalid')
            continue
        if line.startswith('#'):raise ValueError('unexpected body marker')
        if warp is None or left<1:raise ValueError('unexpected/truncated dynamic record')
        if dialect == 'sm90':
            pc,mask,base,op,width=instruction(line,header)
        else:
            from arch_explorer.components.accelsim.typed_trace import instruction as typed_instruction
            pc,mask,base,op,width=typed_instruction(line,dict(header,alive_mask=alive))
        if mask&~alive:raise ValueError('record activates nonexistent or already EXITed lanes')
        if base=='EXIT':alive&=~mask
        repeats+=pc in pcset;pcset.add(pc);left-=1;records+=1;threads+=mask.bit_count();local_records+=1;local_threads+=mask.bit_count();memory+=width>0
        ops[op]+=1;masks[str(mask)]+=1;bodyhash.update((' '.join(line.split())+'\n').encode('ascii'))
        if records>LIMITS['records']:raise ValueError('trace instruction resource bound exceeded')
    if header is None or tb is not None or len(seen)!=ct:raise ValueError('truncated or missing CTA set')
    if file_digest(path)!=original:raise ValueError('trace changed during validation')
    summary=dict(name=header['name'],grid=grid,block=block,stream=header['stream'],ctas=ct,thread_instructions=threads,warp_streams=sum(len(c['warps']) for c in cta_facts),warp_instructions=records,nregs=header['nregs'],shmem=header['shmem'],kernel_id=header['kernel_id'],tracer_version=header['version'],isa='sm90',opcodes=dict(sorted(ops.items())),masks=dict(sorted(masks.items())),repeated_dynamic_pcs=repeats,memory_records=memory,
        flow_sha256=hashlib.sha256(json.dumps(cta_facts,sort_keys=True,separators=(',',':')).encode()).hexdigest(),uniform_predicate_scoreboard='native trace_driven.cc converts non-R dependencies to zero; software execution does not certify their hardware timing')
    if dialect == 'sm120-typed-v1':
        summary.update(isa='sm120', uniform_predicate_scoreboard='typed R/UR/P/UP native dependencies; synthetic banks; hardware timing unvalidated')
    return summary,TraceArtifact(path,original)
