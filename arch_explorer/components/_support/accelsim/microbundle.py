# Concrete Accel-Sim implementation; owned by the component provider.
"""Finite offline microbenchmark registries and independent unsigned oracles.

Reads bytes only. A registry authorizes no device execution; execution additionally
requires its exact hash compiled into a host and a fresh coordinated lease.
"""
from dataclasses import dataclass
import json,math,re,struct
from collections import Counter
from pathlib import Path
from arch_explorer.components.accelsim.trace import exact,load_json,digest,same_json
from arch_explorer.components.accelsim.trace_stream import file_digest
from arch_explorer.components.accelsim.capture_input import elf_texts
from arch_explorer.components.accelsim.microcontract import REGISTRY_V2,CAPS,validate_budget,validate_header

SCHEMA='arch-explorer.sim30-measurement-registry.v1'
BUDGET=CAPS  # Historical public name; new actual budgets live in each registry.
FIELDS={'schema','grid','block','dynamic_shared','cuda_stream','budget','cases','cubin','static','sources','upstream','physical_parameters','hardware_accuracy','gpu_observations'}
CASE_FIELDS={'id','family','steps','partition','function','grid','block','text_sha256','text_bytes','resources','full_flow_instructions','opcode_histogram','input','auxiliary','expected'}
RESOURCE_FIELDS={'static_shared_bytes','local_section_bytes','compiler_reserved_shared_bytes','registers_per_thread','stack_frame_bytes'}

def integer(x,lo,hi,at):
    if type(x) is not int or not lo<=x<=hi:raise ValueError('bounded integer required: '+at)
    return x

def json_file(path,maximum=32*2**20):
    p=Path(path)
    if not p.is_file() or p.stat().st_size>maximum:raise ValueError('bounded original JSON required: '+str(p))
    x,raw=load_json(p);json.dumps(x,allow_nan=False)
    return x,raw

def reference(row,root):
    if set(row) not in ({'name','sha256'},{'name','sha256','bytes'}):raise ValueError('exact file reference required')
    name=Path(row['name'])
    if name.is_absolute() or '..' in name.parts or not name.parts:raise ValueError('bundle-relative original path required')
    p=(Path(root)/name).resolve()
    if not p.is_relative_to(Path(root).resolve()) or not p.is_file() or p.stat().st_size>32*2**20:
        raise ValueError('bundle artifact outside finite bound')
    if file_digest(p)!=row['sha256'] or ('bytes' in row and p.stat().st_size!=row['bytes']):
        raise ValueError('bundle artifact bytes changed: '+str(name))
    return p

def words(raw):
    if len(raw)%4:raise ValueError('u32 original byte length')
    return struct.unpack('<'+'I'*(len(raw)//4),raw)

def oracle(case,main,aux):
    """Closed-form affine composition independent of CUDA/C++ iterative code."""
    n=case['steps'];family=case['family'];inputs=words(main);extra=words(aux)
    rule=case.get('oracle')
    if rule is not None:
        exact(rule,{'kind','mul','add','modulus'},'registered numeric oracle')
        modulus=integer(rule['modulus'],2,2**32,'oracle modulus')
        mul=integer(rule['mul'],1,2**32-1,'oracle multiplier');add=integer(rule['add'],0,2**32-1,'oracle addend')
        if modulus&(modulus-1) or mul%2!=1:raise ValueError('finite power-of-two affine permutation required')
        if family=='chase':
            if rule['kind']!='affine-index-u32' or modulus>2**18 or len(inputs)!=modulus or len(extra)!=32 or any(x>=modulus for x in extra) or inputs!=tuple((mul*i+add)%modulus for i in range(modulus)):raise ValueError('registered complete index permutation differs')
            values=extra
        elif family in ('dep','quad'):
            if rule['kind']!='affine-u32' or modulus!=2**32 or len(inputs)!=(128 if family=='quad' else 32) or extra!=(mul,add):raise ValueError('registered integer recurrence differs')
            values=inputs
        else:raise ValueError('unregistered microbenchmark mechanism')
        mod=modulus
    elif family=='chase':
        if len(inputs)!=4096 or len(extra)!=32 or any(x>=4096 for x in inputs+extra):raise ValueError('chase finite input geometry')
        # Actual frozen permutation is checked, not inferred from observations.
        if tuple((29*i+37)%4096 for i in range(4096))!=inputs:raise ValueError('unsupported index permutation')
        mul,add,mod=29,37,4096;values=extra
    elif family in ('dep','quad'):
        if len(inputs)!=(128 if family=='quad' else 32) or extra!=(1664525,1013904223):raise ValueError('integer recurrence inputs/constants differ')
        mul,add,mod=extra[0],extra[1],2**32;values=inputs
    else:raise ValueError('unregistered microbenchmark family')
    a,b=1,0
    while n:
        if n&1:a,b=mul*a%mod,(mul*b+add)%mod
        mul,add=mul*mul%mod,(mul*add+add)%mod;n//=2
    return struct.pack('<'+'I'*len(values),*((a*v+b)%mod for v in values))

@dataclass
class Bundle:
    path: Path
    document: dict
    sha256: str
    static: dict
    texts: dict
    cases: dict
    bounds: dict
    program_root: Path = None
    def file(self,row):return reference(row,self.program_root or self.path.parent)
    def inputs(self,case):return tuple(self.file(case[n]).read_bytes() for n in ('input','auxiliary','expected'))


def load_registry(path):
    path=Path(path).resolve();data,raw=json_file(path)
    from arch_explorer.components.accelsim.measurement_state import REGISTRY,load_registry as state_registry
    if data.get('schema')==REGISTRY:return state_registry(path,data,raw)
    current=data.get('schema')==REGISTRY_V2
    exact(data,FIELDS|({'work_item','function_prefix'} if current else set()),'finite measurement registry')
    if data['schema'] not in (SCHEMA,REGISTRY_V2) or data['grid']!=[1,1,1] or data['block']!=[32,1,1] or data['dynamic_shared']!=0 or data['cuda_stream']!=0:
        raise ValueError('unsupported complete single-warp registry version/geometry')
    if current:validate_header(data)
    else:validate_budget(data['budget'],12,legacy=True)
    limits=data['budget']
    if data['physical_parameters'] is not None or data['hardware_accuracy']!='not validated' or data['gpu_observations']!=0:
        raise ValueError('offline registry budget or physical claim changed')
    if type(data['sources']) is not dict or not data['sources'] or any(not re.fullmatch('[0-9a-f]{64}',v) for v in data['sources'].values()):raise ValueError('offline source identity required')
    if data['upstream']!='4dd36a8d23ba90a02fab0f1a8f8e6d697617fdc7':raise ValueError('unreviewed design source')
    cubin=reference(data['cubin'],path.parent);texts=elf_texts(cubin.read_bytes())
    static,_=json_file(reference(data['static'],path.parent));cases={};groups={};flow=all_static=0
    if type(data['cases']) is not list or len(data['cases'])!=limits['cases']:raise ValueError('exact finite case denominator required')
    for c in data['cases']:
        exact(c,CASE_FIELDS|({'oracle'} if current else set()),'registered case')
        n=integer(c['steps'],1,4096,'steps');family=c['family']
        prefix=data['function_prefix'] if current else 'sim30'
        if family not in ('dep','quad','chase') or c['id']!=f'{family}-{n}' or c['function']!=f'{prefix}_{family}_{n}' or c['id'] in cases:
            raise ValueError('unknown/duplicate complete function')
        if c['grid']!=data['grid'] or c['block']!=data['block'] or c['partition'] not in ('fit','heldout'):raise ValueError('case geometry/partition differs')
        body=texts.get(c['function']);facts=static.get(c['function'])
        if body is None or facts is None or digest(body)!=c['text_sha256'] or len(body)!=c['text_bytes'] or not same_json(c['resources'],facts['resources']):
            raise ValueError('compiled complete function/text/resources binding differs')
        exact(c['resources'],RESOURCE_FIELDS,'compiled resource facts')
        for k,v in c['resources'].items():integer(v,0,2**20,k)
        if not 1<=c['resources']['registers_per_thread']<=255 or c['resources']['static_shared_bytes'] or c['resources']['stack_frame_bytes'] or c['resources']['local_section_bytes']:
            raise ValueError('ordinary no-local/no-shared finite kernel required')
        if facts['text_sha256']!=c['text_sha256'] or facts['static_native_subset']!='eligible':raise ValueError('whole static subset was not eligible')
        instructions=facts['instructions'];prefix=[i for i in instructions if i['reachability']=='in complete candidate flow']
        if len(prefix)!=integer(c['full_flow_instructions'],1,limits['warp_records_per_case'],'flow count') or not same_json(c['opcode_histogram'],facts['flow_opcode_histogram']):raise ValueError('compiled full flow count/opcodes changed')
        if len(instructions)*16!=len(body) or any(i['pc']!=j*16 or bytes.fromhex(i['binary_hex'])!=body[j*16:j*16+16] for j,i in enumerate(instructions)):
            raise ValueError('complete independent disassembly does not bind every original instruction')
        histogram=Counter(i['sass'].split()[0] for i in prefix)
        if dict(histogram)!=c['opcode_histogram'] or histogram['EXIT']!=1 or prefix[-1]['sass']!='EXIT' or any(i['sass'].startswith('@') or any(x in i['sass'].split()[0] for x in ('BRA','BAR','CLOCK')) for i in prefix):raise ValueError('unsupported complete static execution prefix')
        if (family in ('dep','quad') and histogram['IMAD']!=n*(4 if family=='quad' else 1)) or (family=='chase' and histogram['LDG.E']!=n+1):raise ValueError('semantic family/step count differs from actual SASS')
        main,aux,expected=(reference(c[k],path.parent).read_bytes() for k in ('input','auxiliary','expected'))
        if oracle(c,main,aux)!=expected:raise ValueError('independent input/output oracle differs')
        if len(main)+len(aux)+2*len(expected)>limits['app_bytes']:raise ValueError('static app buffer budget exceeded')
        cases[c['id']]=c;groups.setdefault(family,[]).append(c);flow+=len(prefix);all_static+=len(instructions)
    if set(texts)!=set(static) or set(texts)!={c['function'] for c in cases.values()}:raise ValueError('extra/missing compiled functions')
    for family,group in groups.items():
        fit=sorted(c['steps'] for c in group if c['partition']=='fit');held=[c['steps'] for c in group if c['partition']=='heldout']
        if len(fit)!=3 or len(held)!=1 or not fit[0]<held[0]<fit[-1] or len({c['steps'] for c in group})!=4:
            raise ValueError('three fit and one interior timing-heldout per family required')
    count=len(cases)
    bounds={'dynamic_records':flow,'static_instructions':all_static,'records_bytes':64*count+288*flow,
            'static_bytes':40*all_static+count*(64*2048+4096),'other_capture_bytes':2**20}
    bounds['aggregate_capture_max_bytes']=bounds['records_bytes']+bounds['static_bytes']+bounds['other_capture_bytes']
    if bounds['aggregate_capture_max_bytes']>limits['capture_bytes']:raise ValueError('static total capture bound exceeded; do not start')
    return Bundle(path,data,digest(raw),static,texts,cases,bounds)
