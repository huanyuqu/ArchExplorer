# Concrete Accel-Sim implementation; owned by the component provider.
"""SM120 typed trace dialect. Logical semantics, never a hardware profile."""
import re

VARIANT='sm120-typed-v1'
SCHEMA='arch-explorer.accelsim-trace.v3'
BASES=frozenset({'LDC','ULDC','LDCU','S2UR','LDG','STG','MOV','MOV32I','S2R','CS2R','IMAD','IADD','IADD3','LEA','LOP3','SHF',
    'ISETP','PLOP3','UMOV','UIADD3','UIMAD','UISETP','ULEA','ULOP3','UPLOP3','R2UR','BRA','UBRA','EXIT','NOP'})
SUFFIXES=frozenset({'E','U8','S8','U16','S16','U32','S32','32','64','128','CONSTANT','BYPASS',
    'MOV','WIDE','LO','HI','X','AND','OR','XOR','NOT','LUT','R','L','SHL','SHR','ADD','EQ','NE','LT','LE','GT','GE','U64','S64'})
WIDTHS={'U8':1,'S8':1,'U16':2,'S16':2,'32':4,'U32':4,'S32':4,'64':8,'128':16}
MEMORY=frozenset({'LDC','ULDC','LDCU','LDG','STG'})

def opcode(op):
    tokens=op.split('.')
    if tokens[0] not in BASES or any(t not in SUFFIXES for t in tokens[1:]):
        raise ValueError('SM120 unsupported opcode/modifier/ordering: '+op)
    return tokens

def register(token,nregs=255):
    names={'RZ':('R',255),'URZ':('UR',255),'PT':('P',7),'UPT':('UP',7)}
    if token in names:return names[token]
    match=re.fullmatch(r'(UR|UP|R|P)([0-9]+)',token)
    if not match:raise ValueError('SM120 typed register required: '+token)
    kind,num=match[1],int(match[2])
    if num >= (256 if kind in ('R','UR') else 8) or (kind=='R' and num!=255 and num>=nregs):
        raise ValueError('SM120 register outside type/resource bounds')
    return kind,num

def width(op):
    tokens=opcode(op)
    choices={WIDTHS[t] for t in tokens[1:] if t in WIDTHS}
    if len(choices)>1:raise ValueError('ambiguous memory width')
    return next(iter(choices),4)

def instruction(line,header):
    """Parse exact v7 fields. Every memory address is explicit, including mask=0."""
    fields=line.split()
    try:
        if header['lineinfo']:int(fields.pop(0))
        pc=int(fields.pop(0),16);mask=int(fields.pop(0),16)
        if pc<0 or pc%16 or not 0<=mask<2**32:raise ValueError('SM120 PC/mask bound')
        nd=int(fields.pop(0))
        if not 0<=nd<=8:raise ValueError('SM120 output count bound')
        dst=[fields.pop(0) for _ in range(nd)]
        op=fields.pop(0);tokens=opcode(op);base=tokens[0]
        ns=int(fields.pop(0))
        if not 0<=ns<=24:raise ValueError('SM120 input count bound')
        src=[fields.pop(0) for _ in range(ns)]
        registers=[register(r,header['nregs']) for r in dst+src]
        size=int(fields.pop(0))
        if (base in MEMORY)!=(size>0):raise ValueError('SM120 complete memory fields required')
        if size:
            if size not in (1,2,4,8,16) or size!=width(op):raise ValueError('SM120 opcode/captured memory width mismatch')
            if fields.pop(0)!='0':raise ValueError('SM120 requires explicit addresses')
            addresses=[int(fields.pop(0),16) for _ in range(mask.bit_count())]
            if any(a<0 or a+size>2**64 for a in addresses):raise ValueError('SM120 effective address overflow')
            if base in ('LDG','LDC','ULDC','LDCU'):
                kind=registers[0][0] if base in ('ULDC','LDCU') and nd else 'R'
                real=[num for typ,num in registers[:nd] if typ==kind and num!=255]
                if real and (len(real)!=max(1,size//4) or real!=list(range(real[0],real[0]+len(real)))):
                    raise ValueError('SM120 load vector destination closure required')
        if len(fields)!=5 or fields[:2]!=['0','Guard'] or fields[3]!='Active':
            raise ValueError('SM120 exact immediate and explicit predicate control required')
        guard_kind,guard_num=register(fields[2],header['nregs'])
        before=int(fields[4],16)
        if not 0<before<2**32 or mask&~before or before&~header.get('alive_mask',0xffffffff):
            raise ValueError('SM120 observed pre-predicate mask differs from live lanes')
        if guard_kind not in ('P','UP') or ns+(guard_num!=7)>24:
            raise ValueError('SM120 predicate control type/capacity')
        return pc,mask,base,op,size
    except (IndexError,TypeError) as error:
        raise ValueError('SM120 truncated instruction') from error


def load_v3(workload,mapping,path,manifest,raw):
    from arch_explorer.components.accelsim.trace import exact,identity,same_json,digest
    from arch_explorer.components.accelsim.trace_v2 import _params
    from arch_explorer.components.accelsim.trace import _mapping_spec
    from arch_explorer.components.accelsim.trace_stream import facts,file_digest
    from arch_explorer.components.accelsim.profile import ACCELSIM_SHA,GPGPUSIM_SHA
    from arch_explorer.components.accelsim.variant import PROFILE
    _params(workload.params,'workload.params',True);_mapping_spec(mapping.to_dict(),'mapping')
    exact(manifest,{'schema','format','isa','tracer_version','source','engine','workload','mapping','commands','kernels','scope','capture_binding'},'trace v3')
    if manifest['schema']!=SCHEMA or manifest['format']!='traceg' or manifest['isa']!='sm120' or type(manifest['tracer_version']) is not int or manifest['tracer_version']!=7:
        raise ValueError('typed SM120 requires native traceg v7, no ISA relabel')
    if manifest['engine']!={'accelsim':ACCELSIM_SHA,'gpgpusim':GPGPUSIM_SHA,'profile':PROFILE,'variant':VARIANT}:
        raise ValueError('SM120 trace variant/profile identity differs')
    if manifest['scope']!='single-kernel-device-cycles' or not same_json(manifest['workload'],identity(workload)) or not same_json(manifest['mapping'],mapping.to_dict()):
        raise ValueError('SM120 workload/mapping/complete scope identity differs')
    source=manifest['source'];exact(source,{'kind','gpu','toolchain','description'},'SM120 source')
    if source['kind'] not in ('synthetic','hardware') or any(type(source[k]) is not str or not source[k] for k in source):
        raise ValueError('SM120 explicit source kind and provenance required')
    if (source['kind']=='synthetic')!=(source['gpu']=='none'):
        raise ValueError('synthetic SM120 mechanism must never claim a hardware source')
    if type(manifest['kernels']) is not list or len(manifest['kernels'])!=1:
        raise ValueError('SM120 supports one complete kernel only')
    k=manifest['kernels'][0]
    exact(k,{'file','sha256','original_sha256','name','grid','block','stream','thread_instructions','ctas','facts'},'SM120 kernel')
    if type(k['file']) is not str or not re.fullmatch(r'kernel-[A-Za-z0-9_-]+\.traceg',k['file']):
        raise ValueError('SM120 safe kernel basename required')
    actual,artifact=facts(path.parent/k['file'],dialect=VARIANT)
    if artifact.sha256!=k['sha256'] or k['sha256']!=k['original_sha256'] or not same_json(actual,k['facts']):
        raise ValueError('SM120 actual trace bytes/facts differ from manifest')
    if any(not same_json(k[key],actual[key]) for key in ('name','grid','block','stream','thread_instructions','ctas')):
        raise ValueError('SM120 complete kernel scope differs')
    for spec in (workload.params,mapping.params):
        for key in ('grid','block'):
            if spec[key]!=actual[key]:raise ValueError('SM120 actual launch differs')
    if workload.params['kernel']!=actual['name']:raise ValueError('SM120 kernel name differs')
    exact(manifest['commands'],{'file','sha256'},'SM120 commands')
    command=path.parent/'kernelslist.g'
    if manifest['commands']['file']!='kernelslist.g' or command.stat().st_size>4096 or command.read_bytes()!=(k['file']+'\n').encode() or file_digest(command)!=manifest['commands']['sha256']:
        raise ValueError('SM120 complete command/input binding differs')
    if source['kind']=='hardware':
        from arch_explorer.components.accelsim.capture_input import derive
        reconstructed=derive(manifest['capture_binding'])
        if digest(reconstructed)!=artifact.sha256:
            raise ValueError('SM120 original observed dynamic records do not reconstruct consumed trace')
    elif manifest['capture_binding'] is not None:
        raise ValueError('synthetic trace has no hardware capture binding')
    return manifest,artifact,digest(raw),actual


def create_request(trace_path,directory,*,source,trace_id,compiler,engine,adf=None,capture_binding=None):
    """Bind an original v7 trace to complete public input; no native/GPU execution."""
    import json,shutil
    from pathlib import Path
    from arch_explorer.adf import WorkloadSpec,MappingSpec
    from arch_explorer.components.accelsim.scenarios import sm120_mechanism_adf
    from arch_explorer.components.accelsim.trace import identity,digest
    from arch_explorer.components.accelsim.trace_stream import facts,file_digest
    from arch_explorer.components.accelsim.profile import ACCELSIM_SHA,GPGPUSIM_SHA
    from arch_explorer.components.accelsim.variant import PROFILE
    trace_path,directory=Path(trace_path).resolve(),Path(directory).resolve()
    actual,artifact=facts(trace_path,dialect=VARIANT)
    if not re.fullmatch(r'kernel-[A-Za-z0-9_-]+\.traceg',trace_path.name):raise ValueError('SM120 safe kernel basename required')
    directory.mkdir(exist_ok=False,parents=True)
    with trace_path.open('rb') as src,(directory/trace_path.name).open('xb') as out:shutil.copyfileobj(src,out)
    if file_digest(directory/trace_path.name)!=artifact.sha256:raise ValueError('trace changed while binding')
    workload=WorkloadSpec('operator',dict(op='trace_kernel',kernel=actual['name'],grid=actual['grid'],block=actual['block']))
    mapping=MappingSpec('accelsim_trace',dict(trace_id=trace_id,compiler=compiler,grid=actual['grid'],block=actual['block']))
    commands=(trace_path.name+'\n').encode();(directory/'kernelslist.g').write_bytes(commands)
    manifest=dict(schema=SCHEMA,format='traceg',isa='sm120',tracer_version=7,source=source,
        engine=dict(accelsim=ACCELSIM_SHA,gpgpusim=GPGPUSIM_SHA,profile=PROFILE,variant=VARIANT),
        workload=identity(workload),mapping=mapping.to_dict(),commands=dict(file='kernelslist.g',sha256=digest(commands)),
        kernels=[dict(file=trace_path.name,sha256=artifact.sha256,original_sha256=artifact.sha256,
            **{key:actual[key] for key in ('name','grid','block','stream','thread_instructions','ctas')},facts=actual)],
        scope='single-kernel-device-cycles',capture_binding=capture_binding)
    raw=(json.dumps(manifest,indent=2)+'\n').encode();mf=directory/'trace-manifest.json';mf.write_bytes(raw)
    workload.params['accelsim']=dict(manifest=str(mf),sha256=digest(raw))
    request=dict(schema='arch-explorer.accelsim-request.v1',adf=(adf or sm120_mechanism_adf()).to_dict(),
        workload=workload.to_dict(),mapping=mapping.to_dict(),options=dict(engine=str(Path(engine).resolve()),output_dir=str(directory/'runs')))
    (directory/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    # Validate the produced request using exactly the public consumption path.
    from arch_explorer.components.accelsim.trace import load_trace
    load_trace(workload,mapping)
    return request


