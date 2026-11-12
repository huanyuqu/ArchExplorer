# Concrete Accel-Sim implementation; owned by the component provider.
"""Frozen registration generation, offline ELF binding and a pure CPU gate.

Registry owns the program set. The embedded bytes are also the production
tracer's runtime directory. State/membership decisions live only in the shared
C++ policy; no production library is loaded by this module.
"""
import argparse,hashlib,json,os,struct,subprocess,tempfile
from pathlib import Path
from arch_explorer.components.accelsim.microbundle import load_registry,json_file
from arch_explorer.components.accelsim.microcontract import modern,work_item,budget
from arch_explorer.components.accelsim.microexperiment import ref,write_new
from arch_explorer.components.accelsim.microrecords import original
from arch_explorer.components.accelsim.measurement_state import stateful

MAGIC=b'ARCHCATALOG0001\0'
SECTION='.ae_catalog'
GEN='arch-explorer.measurement-catalog-generation.v1'
BIND='arch-explorer.measurement-catalog-binding.v1'
BUILD='arch-explorer.measurement-boundary-build.v1'
POLICY=Path(__file__).with_name('catalog_policy.hpp')
HARNESS=Path(__file__).with_name('catalog_harness.cpp')
REPO_POLICY='arch_explorer/components/accelsim/catalog_policy.hpp'
REPO_HARNESS='arch_explorer/components/accelsim/catalog_harness.cpp'
MEASUREMENT='tools/accelsim_native/calibration/measurement/'
CAPTURE='tools/accelsim_native/capture/'
BUILD_INPUTS={
    'policy':{REPO_POLICY,REPO_HARNESS},
    'hosts':{REPO_POLICY,'catalog.hpp','capture_catalog.hpp'}|{MEASUREMENT+n for n in
             ('timing_host.cpp','capture_host.cpp','host_io.hpp','host_common.hpp','catalog_types.hpp','host_build.py')},
    'tracer':{REPO_POLICY,'capture_catalog.hpp'}|{CAPTURE+n for n in
              ('microbenchmark/inject_funcs.cu','microbenchmark/tracer.cu','microbenchmark/build.py','reflection.h','records.h','json_util.h')},
}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def catalog_bytes(bundle):
    if not modern(bundle):raise ValueError('historical registry is read-only; explicit new registration required')
    names=[c['function'] for c in bundle.cases.values()]
    # load_registry is the sole validator of registered program identities.
    strings=[work_item(bundle)]+names
    encoded=[s.encode('ascii') for s in strings]
    identifiers=[bundle.sha256,bundle.document['cubin']['sha256'],digest(POLICY.read_bytes())]
    tail=b''.join(struct.pack('<H',len(x))+x for x in encoded)
    total=16+20+192+len(tail)+32
    body=MAGIC+struct.pack('<5I',1,total,len(names),budget(bundle)['capture_launches'],budget(bundle)['warp_records_per_case'])
    body+=b''.join(s.encode('ascii') for s in identifiers)+tail
    return body+hashlib.sha256(body).digest()


def capture_header(blob):
    data=','.join(str(b) for b in blob)
    return ('// Generated only from the frozen registry; runtime parses these same bytes.\n'
      '#pragma once\n#include "catalog_policy.hpp"\nnamespace ae_catalog_generated {\n'
      'static const unsigned char kBlob[] __attribute__((section(".ae_catalog"),used,aligned(1))) = {'+data+'};\n'
      'inline ae_catalog::Catalog contract(){return ae_catalog::Catalog::parse(kBlob,sizeof(kBlob));}\n'
      '}\n').encode()


def host_header(bundle):
    q=json.dumps
    def file(x):return '{'+','.join((q(x['name']),q(x['sha256']),str(x['bytes'])))+'}'
    doc=bundle.document
    lines=['// Generated from the same frozen registration as the capture tool.',
      '#pragma once','#include "catalog_types.hpp"','#include "capture_catalog.hpp"','namespace sim30 {',
      'inline constexpr const char* kRegistrySha='+q(bundle.sha256)+';',
      'inline constexpr const char* kWorkItem='+q(work_item(bundle))+';',
      'inline constexpr unsigned kSamplesPerCase='+str(budget(bundle)['samples_per_case'])+';',
      'inline const File kCubin='+file(doc['cubin'])+';',
      'inline constexpr std::array<Case,'+str(len(bundle.cases))+'> kCases={{']
    for c in bundle.cases.values():
        r=c['resources'];lines.append('{'+','.join((q(c['id']),q(c['family']),str(c['steps']),q(c['partition']),q(c['function']),
          q(c['text_sha256']),str(r['registers_per_thread']),str(r['static_shared_bytes']),str(r['local_section_bytes']),
          file(c['input']),file(c['auxiliary']),file(c['expected'])))+'},')
    lines+=['}};','} // namespace sim30']
    if stateful(bundle):lines+=['#define AE_MEASUREMENT_STATE_V1 1','#include "measurement_state.hpp"']
    return ('\n'.join(lines+[''])).encode()


def generated_values(bundle):
    values={'catalog.bin':catalog_bytes(bundle),'capture_catalog.hpp':capture_header(catalog_bytes(bundle)),
            'catalog.hpp':host_header(bundle),'catalog_policy.hpp':POLICY.read_bytes()}
    if stateful(bundle):
        from arch_explorer.components.accelsim.measurement_state import generated
        values.update(generated(bundle))
    return values


def generate(registry,output):
    bundle=load_registry(registry);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    values=generated_values(bundle)
    for name,raw in values.items():
        with (out/name).open('xb') as f:f.write(raw)
    result={'schema':GEN,'registry':ref(bundle.path),'policy_source':ref(POLICY),
      'catalog_sha256':digest(values['catalog.bin']),'files':{n:ref(out/n) for n in values},
      'source_owner':'installed microcatalog generator; registry is the only program-set authority',
      'resource_authority':False}
    return write_new(out/'CATALOG.json',result)


def verify_generation(bundle,generation):
    g=json_file(original(generation))[0]
    if set(g)!={'schema','registry','policy_source','catalog_sha256','files','source_owner','resource_authority'} or g['schema']!=GEN:
        raise ValueError('unknown/incomplete catalog generation contract')
    if g['registry']!=ref(bundle.path) or g['resource_authority'] is not False:raise ValueError('catalog registry/authority identity differs')
    expected=generated_values(bundle)
    if set(g['files'])!=set(expected) or g['catalog_sha256']!=digest(expected['catalog.bin']):raise ValueError('catalog generation identity differs')
    if original(g['policy_source']).read_bytes()!=POLICY.read_bytes():raise ValueError('different production CPU policy source')
    for name,raw in expected.items():
        if original(g['files'][name]).read_bytes()!=raw:raise ValueError('generated '+name+' differs from complete registry/policy')
    return g


def elf_sections(path):
    """Bounded ELF64 little-endian sections; no loader or external utility."""
    path=Path(path)
    if not path.is_file() or path.stat().st_size>128*2**20:raise ValueError('bounded actual ELF required')
    raw=path.read_bytes()
    if len(raw)<64 or raw[:7]!=b'\x7fELF\x02\x01\x01' or struct.unpack_from('<H',raw,18)[0]!=62:
        raise ValueError('x86-64 little-endian ELF required')
    off=struct.unpack_from('<Q',raw,40)[0];size,count,index=struct.unpack_from('<HHH',raw,58)
    if size!=64 or not 0<count<65535 or index>=count or off+64*count>len(raw):raise ValueError('bounded complete ELF section table required')
    sections=[struct.unpack_from('<IIQQQQIIQQ',raw,off+64*i) for i in range(count)]
    strings=sections[index]
    if strings[1]!=3 or strings[4]+strings[5]>len(raw):raise ValueError('ELF section-name table differs')
    names=raw[strings[4]:strings[4]+strings[5]];result={}
    def string(data,n):
        if n>=len(data):raise ValueError('ELF string offset differs')
        end=data.find(b'\0',n)
        if end<0:raise ValueError('unterminated ELF string')
        return data[n:end].decode('ascii')
    for index,section in enumerate(sections):
        name=string(names,section[0])
        if not name:continue
        if name in result:raise ValueError('duplicate ELF section '+name)
        if section[1]!=8 and section[4]+section[5]>len(raw):raise ValueError('ELF section exceeds actual bytes')
        result[name]={'type':section[1],'flags':section[2],'bytes':b'' if section[1]==8 else raw[section[4]:section[4]+section[5]],
                      'link':section[6],'index':index,'entry_size':section[9]}
    needed=[]
    for s in result.values():
        if s['type']!=6:continue
        if s['link']>=count:raise ValueError('ELF dynamic string-table link exceeds section count')
        target=sections[s['link']]
        if target[1]!=3 or target[4]+target[5]>len(raw):raise ValueError('ELF dynamic string table differs')
        table=raw[target[4]:target[4]+target[5]]
        if s['entry_size']!=16 or len(s['bytes'])%16:raise ValueError('ELF dynamic entry alignment differs')
        for i in range(0,len(s['bytes']),16):
            tag,value=struct.unpack_from('<QQ',s['bytes'],i)
            if tag==1:needed.append(string(table,value))
    return result,needed


def embedded_catalog(path,expected):
    sections,needed=elf_sections(path)
    if SECTION not in sections:raise ValueError('actual ELF lacks '+SECTION+' registration v1; legacy/absent directory is not ready')
    s=sections[SECTION]
    if s['type']!=1 or not s['flags']&2 or s['flags']&1:raise ValueError('catalog must be one allocated read-only ELF PROGBITS section')
    if s['bytes']!=expected:raise ValueError('actual ELF catalog differs from complete ordered registry/policy, even if its checksum was recomputed')
    return {'elf':ref(path),'catalog_sha256':digest(s['bytes']),'bytes':len(s['bytes']),'needed':needed}


def source_input(path,*,repo=None,generated=None):
    if (repo is None)==(generated is None):raise ValueError('one compile input owner required')
    return {'file':ref(path),'repo':repo,'generated':generated}


def verify_build(manifest,kind,binary_name,binary,bundle,generation,source=None):
    b=json_file(original(manifest))[0]
    if b.get('schema')!=BUILD or b.get('kind')!=kind or b.get('status')!='built' or b.get('hardware_executed') is not False:
        raise ValueError('completed offline '+kind+' build contract required')
    if b['binaries'].get(binary_name)!=binary:raise ValueError('actual binary substituted across build boundary')
    if kind!='policy' and (b['registry']!=ref(bundle.path) or b['generation']!=generation):raise ValueError('build registry/generation differs')
    g=None if kind=='policy' else verify_generation(bundle,generation)
    archive=None
    if source is not None:
        import tarfile
        origin=json_file(original(source))[0]
        archive=tarfile.open(original({'path':origin['archive'],'sha256':origin['archive_sha256']}))
    seen=set()
    try:
        for row in b['sources']:
            if set(row)!={'file','repo','generated'} or (row['repo'] is None)==(row['generated'] is None):raise ValueError('unknown compile-source owner')
            raw=original(row['file']).read_bytes();owner=row['repo'] or row['generated']
            if owner in seen:raise ValueError('duplicate compiled source')
            seen.add(owner)
            if row['generated'] is not None:
                if g is None or row['generated'] not in g['files'] or raw!=original(g['files'][row['generated']]).read_bytes():raise ValueError('compiled generated input differs')
            elif archive is not None:
                try:member=archive.extractfile(row['repo'])
                except KeyError as exc:raise ValueError('compiled source missing from installed archive: '+owner) from exc
                if member is None or member.read()!=raw:raise ValueError('compiled source differs from installed source archive: '+owner)
        wanted=BUILD_INPUTS[kind]
        if stateful(bundle) and kind=='hosts':
            wanted=wanted|{'measurement_state.hpp','state_executor.hpp',MEASUREMENT+'state_cuda.hpp'}
        if seen!=wanted:raise ValueError('incomplete/extra '+kind+' compile inputs: '+repr(sorted(seen^wanted)))
        if not b['tools']:raise ValueError('missing offline compiler/tool identities')
        for tool in b['tools']:original(tool)
        if not b['commands'] or any(r['exit_code']!=0 for r in b['commands']):raise ValueError('incomplete offline compilation')
        commands=[r['argv'] for r in b['commands']]
        for row in b['commands']:original(row['log'])
        if not any('-o' in args and args[args.index('-o')+1]==binary['path'] for args in commands):
            raise ValueError('actual binary has no successful compile/link command')
        for row in b['sources']:
            if row['repo'] and row['repo'].endswith(('.cpp','.cu')) and not any(row['file']['path'] in args for args in commands):
                raise ValueError('declared translation unit absent from compile commands')
    finally:
        if archive is not None:archive.close()
    return b


def build_policy(output,cxx):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);src=out/'sources';src.mkdir()
    for p in (POLICY,HARNESS):(src/p.name).write_bytes(p.read_bytes())
    executable=out/'catalog-policy';log=out/'compile.log'
    args=[str(Path(cxx).resolve()),'-std=c++17','-O2','-Wall','-Wextra','-Werror',
       '-DAE_CATALOG_POLICY_SHA256="'+digest(POLICY.read_bytes())+'"',
       '-DAE_CATALOG_HARNESS_SHA256="'+digest(HARNESS.read_bytes())+'"',str(src/HARNESS.name),'-lcrypto','-o',str(executable)]
    with log.open('xb') as f:
        r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=120,env=cpu_environment())
    b={'schema':BUILD,'kind':'policy','status':'built' if r.returncode==0 else 'failed','registry':None,'generation':None,
       'sources':[source_input(src/POLICY.name,repo=REPO_POLICY),source_input(src/HARNESS.name,repo=REPO_HARNESS)],
       'tools':[ref(cxx)],'commands':[{'argv':args,'exit_code':r.returncode,'log':ref(log)}],
       'binaries':{'policy':ref(executable)} if r.returncode==0 else {},'hardware_executed':False}
    receipt=write_new(out/'BUILD.json',b)
    if r.returncode:raise ValueError('pure CPU policy build failed; retained '+str(receipt))
    return {'executable':ref(executable),'build':receipt}


def cpu_environment():
    return {'PATH':'/usr/bin:/bin','LC_ALL':'C','CUDA_VISIBLE_DEVICES':''}


def verify_policy(policy,bundle,generation,source=None):
    executable=original(policy['executable']);sections,needed=elf_sections(executable)
    allowed={'libcrypto.so.3','libstdc++.so.6','libgcc_s.so.1','libm.so.6','libc.so.6'}
    if not set(needed)<=allowed:raise ValueError('policy harness has an unsupported/device dependency')
    for name,expected in (('.ae_policy',digest(POLICY.read_bytes())),('.ae_harness',digest(HARNESS.read_bytes()))):
        if name not in sections or sections[name]['bytes']!=expected.encode()+b'\0':raise ValueError('actual CPU harness source identity differs')
        if sections[name]['type']!=1 or not sections[name]['flags']&2 or sections[name]['flags']&1:
            raise ValueError('CPU harness identity must be allocated read-only PROGBITS')
    verify_build(policy['build'],'policy','policy',policy['executable'],bundle,generation,source)
    return executable


def verify_binding(bundle,binding,*,source=None,run_policy=False):
    b=json_file(original(binding))[0]
    if set(b)!={'schema','registry','generation','tracer','hosts','policy'} or b['schema']!=BIND or b['registry']!=ref(bundle.path) or set(b['hosts'])!={'capture','timing'}:
        raise ValueError('complete versioned registration boundary binding required')
    expected=catalog_bytes(bundle);g=verify_generation(bundle,b['generation']);artifacts={}
    for role,item in [('tracer',b['tracer'])]+list(b['hosts'].items()):
        if set(item)!={'binary','build'}:raise ValueError('actual binary and compile closure required')
        artifacts[role]=embedded_catalog(original(item['binary']),expected)
        if stateful(bundle) and role in ('capture','timing'):
            from arch_explorer.components.accelsim.measurement_state import embedded
            artifacts[role]['measurement_state']=embedded(original(item['binary']),bundle)
        verify_build(item['build'],'tracer' if role=='tracer' else 'hosts',role,item['binary'],bundle,b['generation'],source)
    exe=verify_policy(b['policy'],bundle,b['generation'],source)
    result={'schema':'arch-explorer.measurement-catalog-readiness.v1','status':'passed','registry':ref(bundle.path),
       'catalog_sha256':digest(expected),'binding':binding,'artifacts':artifacts,'policy':b['policy'],
       'shared_policy_executed':False,'resource_authority':False,'hardware_accuracy':'not validated','physical_parameters':None}
    if run_policy:
        with tempfile.TemporaryDirectory(prefix='ae-catalog-cpu-') as tmp:
            a=Path(tmp)/'expected.bin';a.write_bytes(expected);actual=Path(tmp)/'actual.bin'
            # Exact ELF section, not a sidecar, is consumed by the production rule.
            actual.write_bytes(elf_sections(original(b['tracer']['binary']))[0][SECTION]['bytes'])
            p=subprocess.run([str(exe),'check',str(a),str(actual)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             timeout=15,env=cpu_environment())
        result['policy_result']={'exit_code':p.returncode,'stdout':p.stdout.decode(),'stderr':p.stderr.decode()}
        if p.returncode:raise ValueError('shared production CPU catalog compatibility refused')
        result['shared_policy_executed']=True
    return result


def policy_sequence(bundle,binding,commands,*,source=None):
    verify_binding(bundle,binding,source=source)
    b=json_file(original(binding))[0];exe=verify_policy(b['policy'],bundle,b['generation'],source)
    with tempfile.TemporaryDirectory(prefix='ae-catalog-cpu-') as tmp:
        path=Path(tmp)/'catalog.bin';path.write_bytes(elf_sections(original(b['tracer']['binary']))[0][SECTION]['bytes'])
        p=subprocess.run([str(exe),'sequence',str(path)],input=('\n'.join(commands)+'\n').encode(),stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,timeout=15,env=cpu_environment())
    result={'kind':'pure-CPU-production-policy','commands':commands,'exit_code':p.returncode,
            'stdout':p.stdout.decode(),'stderr':p.stderr.decode(),'resource_authority':False}
    if p.returncode:raise ValueError('shared production begin/state policy refused: '+p.stdout.decode()+p.stderr.decode())
    return result


def design_binding(bundle,design):
    if design.get('schema') not in ('arch-explorer.measurement-design.v3','arch-explorer.measurement-state-device-design.v1') or 'catalog' not in design:
        raise ValueError('historical design is read-only; new runs require actual ELF registration binding')
    b=json_file(original(design['catalog']))[0]
    if b.get('schema')!=BIND or b.get('registry')!=design['registry']:
        raise ValueError('frozen design and catalog registry differ')
    if b['tracer']['binary']!=design['tracer'] or {k:v['binary'] for k,v in b['hosts'].items()}!=design['hosts']:
        raise ValueError('frozen design substituted a host/tracer outside its compiled catalog binding')
    return b


def require_design_catalog(design,*,source=None,run_policy=False):
    bundle=load_registry(original(design['registry']));design_binding(bundle,design)
    return verify_binding(bundle,design['catalog'],source=source,run_policy=run_policy)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    g=s.add_parser('generate');g.add_argument('--registry',required=True);g.add_argument('--output',required=True)
    h=s.add_parser('build-policy');h.add_argument('--output',required=True);h.add_argument('--cxx',default='/usr/bin/g++')
    r=s.add_parser('ready')
    for name in ('registry','binding','source','output'):r.add_argument('--'+name,required=True)
    a=p.parse_args(argv)
    if a.command=='generate':result=generate(a.registry,a.output)
    elif a.command=='build-policy':result=build_policy(a.output,a.cxx)
    else:
        from arch_explorer.components.accelsim.microcampaign import check_source_installation
        source=ref(a.source);check_source_installation(source)
        result=verify_binding(load_registry(a.registry),ref(a.binding),source=source,run_policy=True)
        write_new(a.output,result)
    print(json.dumps(result))


if __name__=='__main__':main()
