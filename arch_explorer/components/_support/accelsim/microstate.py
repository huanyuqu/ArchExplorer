# Concrete Accel-Sim implementation; owned by the component provider.
"""Versioned measurement-state closure and CPU Driver orchestration.

The real host and this CPU boundary use the same compiled executor. This module
never acquires resources or starts a production measurement host. Hardware
collection of this new method requires a separately reviewed work package.
"""
import argparse,json,subprocess,time,uuid
from collections import Counter
from fractions import Fraction
from pathlib import Path
from arch_explorer.components._support.paths import package_root
from arch_explorer.components.accelsim.trace import exact,same_json,digest
from arch_explorer.components.accelsim.microbundle import load_registry,json_file
from arch_explorer.components.accelsim.microrecords import original
from arch_explorer.components.accelsim.microexperiment import ref,write_new,seal,QUALITY
from arch_explorer.components.accelsim.measurement_state import (REGISTRY,PLAN,stateful,validate_plan,sha_plan,slots,
                                plan_bytes,embedded,EXECUTOR,CPU)
from arch_explorer.components.accelsim.state_lifecycle import validate_lifecycle
from arch_explorer.components.accelsim.state_observation import read_observation,quality

DESIGN='arch-explorer.measurement-state-design.v1'
OUTCOME='arch-explorer.measurement-state-outcomes.v1'
HANDOFF='arch-explorer.measurement-state-handoff.v1'
REPLAY='arch-explorer.measurement-state-replay.v1'
PREFIX='arch_explorer/components/accelsim/'
CPU_FAULTS=('preparation','quality','timeout','cleanup','launch','allocate-input','allocate-output',
            'upload-input','event-stop','allocate-output-handle','event-stop-handle',
            'cleanup-sync','free-output','allocate-auxiliary','upload-auxiliary','free-auxiliary')

def make_registry(program_registry,case_ids,preparation,work_item):
    from arch_explorer.components.accelsim.microcontract import CAPS
    programs=load_registry(original(program_registry));n=len(case_ids)
    if set(preparation)!=set(case_ids):raise ValueError('preparation must be frozen for every selected complete program')
    from arch_explorer.components.accelsim.measurement_state import LIFECYCLE
    p={'schema':PLAN,'lifecycle':LIFECYCLE,'cases':[{'case':cid,'diagnostic':1,'preparation':preparation[cid],'formal':3} for cid in case_ids]}
    validate_plan(p,{cid:programs.cases[cid] for cid in case_ids})
    b=dict(CAPS);b.update(cases=n,host_processes=n+1,capture_launches=n,timing_launches=3*n,
        diagnostic_launches=n,preparation_launches=sum(preparation.values()),
        uninstrumented_launches=4*n+sum(preparation.values()),application_launches=5*n+sum(preparation.values()))
    return {'schema':REGISTRY,'work_item':work_item,'program_registry':program_registry,'case_ids':case_ids,
            'measurement_state':p,'budget':b,'physical_parameters':None,'hardware_accuracy':'not validated','gpu_observations':0}

def build_cpu(registry,generation,output,cxx='/usr/bin/g++',measurement_sources=None):
    from arch_explorer.components.accelsim.microcatalog import verify_generation,cpu_environment
    bundle=load_registry(registry);g=verify_generation(bundle,generation)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);src=out/'sources';src.mkdir()
    sources={}
    for name in ('catalog.hpp','capture_catalog.hpp','catalog_policy.hpp','measurement_state.hpp','state_executor.hpp'):
        raw=original(g['files'][name]).read_bytes();(src/name).write_bytes(raw);sources[name]=ref(src/name)
    for path in (CPU,EXECUTOR):
        target=src/path.name
        if not target.exists():target.write_bytes(path.read_bytes())
        elif target.read_bytes()!=path.read_bytes():raise ValueError('installed executor differs from generation')
        sources[path.name]=ref(target)
    root=Path(measurement_sources) if measurement_sources else package_root().parent/'tools/accelsim_native/calibration/measurement'
    for name in ('catalog_types.hpp','host_io.hpp'):
        path=root/name
        (src/name).write_bytes(path.read_bytes());sources[name]=ref(src/name)
    exe=out/'state-cpu';log=out/'compile.log'
    args=[str(Path(cxx).resolve()),'-std=c++17','-O2','-Wall','-Wextra','-Werror',
          '-DAE_STATE_EXECUTOR_SHA256="'+digest(EXECUTOR.read_bytes())+'"',
          '-DAE_STATE_CPU_SHA256="'+digest(CPU.read_bytes())+'"','-I'+str(src),str(src/CPU.name),'-lcrypto','-o',str(exe)]
    with log.open('xb') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=120,env=cpu_environment())
    receipt={'schema':'arch-explorer.measurement-state-build.v1','status':'built' if r.returncode==0 else 'failed',
             'generation':generation,'registry':ref(bundle.path),'sources':sources,'cxx':ref(cxx),'argv':args,
             'exit_code':r.returncode,'log':ref(log),'executable':ref(exe) if r.returncode==0 else None,
             'hardware_executed':False}
    result=write_new(out/'BUILD.json',receipt)
    if r.returncode:raise ValueError('CPU Driver build failed: '+str(log))
    embedded(exe,bundle)
    return {'executable':ref(exe),'build':result}

def check_cpu_binding(bundle,binding,generation,source=None):
    from arch_explorer.components.accelsim.microcatalog import verify_generation,elf_sections,embedded_catalog,catalog_bytes
    exact(binding,{'executable','build'},'CPU boundary binding')
    b=json_file(original(binding['build']))[0];exe=original(binding['executable']);g=verify_generation(bundle,generation)
    if b['schema']!='arch-explorer.measurement-state-build.v1' or b['status']!='built' or b['hardware_executed'] is not False or b['exit_code']!=0 or b['executable']!=binding['executable'] or b['registry']!=ref(bundle.path) or b['generation']!=generation:raise ValueError('CPU boundary compile identity differs')
    expected={'catalog.hpp','capture_catalog.hpp','catalog_policy.hpp','measurement_state.hpp','state_executor.hpp','state_cpu.cpp','catalog_types.hpp','host_io.hpp'}
    if set(b['sources'])!=expected:raise ValueError('CPU Driver compile closure is incomplete')
    import tarfile
    archive=None
    if source:
        origin=json_file(original(source))[0];archive=tarfile.open(original({'path':origin['archive'],'sha256':origin['archive_sha256']}))
    try:
        for name,row in b['sources'].items():
            raw=original(row).read_bytes()
            if name in g['files']:
                if raw!=original(g['files'][name]).read_bytes():raise ValueError('compiled CPU generation changed')
            elif archive:
                repo=PREFIX+name if name=='state_cpu.cpp' else 'tools/accelsim_native/calibration/measurement/'+name
                member=archive.extractfile(repo)
                if member is None or member.read()!=raw:raise ValueError('CPU compile source differs from archive')
        if archive and archive.extractfile(PREFIX+'state_executor.hpp').read()!=EXECUTOR.read_bytes():raise ValueError('installed executor not in source archive')
    finally:
        if archive:archive.close()
    original(b['log']);original(b['cxx'])
    if '-o' not in b['argv'] or b['argv'][b['argv'].index('-o')+1]!=str(exe) or b['sources']['state_cpu.cpp']['path'] not in b['argv']:raise ValueError('CPU executable lacks its actual compile command')
    sections,needed=elf_sections(exe)
    embedded_catalog(exe,catalog_bytes(bundle))
    if not set(needed)<={'libcrypto.so.3','libstdc++.so.6','libgcc_s.so.1','libm.so.6','libc.so.6'}:raise ValueError('CPU Driver links a device/unknown dependency')
    for name,path in (('.ae_state_exec',EXECUTOR),('.ae_state_cpu',CPU)):
        s=sections.get(name)
        if s is None or s['type']!=1 or not s['flags']&2 or s['flags']&1 or s['bytes']!=digest(path.read_bytes()).encode()+b'\0':raise ValueError('actual CPU source identity section differs')
    return embedded(exe,bundle)

def make_design(bundle,source,catalog,cpu,method,native):
    """No observations here; all conditions and budgets frozen before execution."""
    if not stateful(bundle):raise ValueError('explicit state registry required')
    for row in (source,catalog,method,native):original(row)
    b=json_file(original(catalog))[0]
    d={'schema':DESIGN,'kind':'synthetic','no_resource_authority':True,'run_id':uuid.uuid4().hex,
       'registry':ref(bundle.path),'source':source,'catalog':catalog,'cpu':cpu,'method':method,'native':native,
       'plan_sha256':sha_plan(bundle),'budget':bundle.document['budget'],'quality':QUALITY,
       'state_evidence':{'clock':None,'cache':None},'native_state':'cold-initialization-assumption',
       'created_realtime_ns':time.time_ns(),'physical_parameters':None,'hardware_accuracy':'not validated'}
    d['design_sha256']=seal(d);return d

def validate_design(design_file,*,installation=False):
    d=json_file(design_file)[0];fields={'schema','kind','no_resource_authority','run_id','registry','source','catalog','cpu',
        'method','native','plan_sha256','budget','quality','state_evidence','native_state','created_realtime_ns',
        'physical_parameters','hardware_accuracy','design_sha256'}
    exact(d,fields,'state design')
    raw=dict(d);claimed=raw.pop('design_sha256')
    bundle=load_registry(original(d['registry']))
    if (not stateful(bundle) or d['schema']!=DESIGN or d['kind']!='synthetic' or d['no_resource_authority'] is not True
        or seal(raw)!=claimed or d['plan_sha256']!=sha_plan(bundle) or not same_json(d['budget'],bundle.document['budget'])
        or not same_json(d['quality'],QUALITY) or d['physical_parameters'] is not None or d['hardware_accuracy']!='not validated'
        or d['native_state']!='cold-initialization-assumption'):raise ValueError('frozen state design/budget/quality/scope changed')
    for row in (d['source'],d['method'],d['native']):original(row)
    from arch_explorer.components.accelsim.microcatalog import verify_binding
    b=json_file(original(d['catalog']))[0]
    verify_binding(bundle,d['catalog'],source=d['source'],run_policy=installation)
    check_cpu_binding(bundle,d['cpu'],b['generation'],source=d['source'])
    if installation:
        from arch_explorer.components.accelsim.microcampaign import check_source_installation
        check_source_installation(d['source'])
    validate_state_evidence(d,d['created_realtime_ns'],2**63-1)
    return bundle,d

def validate_state_evidence(design,begin,end):
    exact(design['state_evidence'],{'clock','cache'},'independent state evidence')
    for role,row in design['state_evidence'].items():
        if row is None:continue
        e=json_file(original(row))[0]
        exact(e,{'schema','run_id','role','begin_realtime_ns','end_realtime_ns','value','independent_of_events'},'state evidence')
        if (e['schema']!='arch-explorer.measurement-independent-state.v1' or e['run_id']!=design['run_id']
            or e['role']!=role or e['independent_of_events'] is not True or e['value'] is None
            or type(e['begin_realtime_ns']) is not int or type(e['end_realtime_ns']) is not int
            or not begin<=e['begin_realtime_ns']<=e['end_realtime_ns']<=end):raise ValueError('state evidence wrong run/time/independence')

def validate_handoff(bundle,design,handoff,*,acquired_ns=None):
    exact(handoff,{'schema','kind','no_resource_authority','run_id','registry_sha256','plan_sha256','budget',
        'resource_owner','issued_realtime_ns','absolute_deadline_realtime_ns','close_reserve_s'},'role-bound handoff')
    if (handoff['schema']!=HANDOFF or handoff['kind']!='synthetic' or handoff['no_resource_authority'] is not True
        or handoff['run_id']!=design['run_id'] or handoff['registry_sha256']!=bundle.sha256
        or handoff['plan_sha256']!=sha_plan(bundle) or not same_json(handoff['budget'],design['budget'])
        or handoff['resource_owner']!='synthetic-no-device-owner' or handoff['close_reserve_s']!=60
        or type(handoff['issued_realtime_ns']) is not int or type(handoff['absolute_deadline_realtime_ns']) is not int
        or not design['created_realtime_ns']<handoff['issued_realtime_ns']<handoff['absolute_deadline_realtime_ns']
           <=handoff['issued_realtime_ns']+design['budget']['total_window_s']*10**9):raise ValueError('new role budget and fixed absolute handoff required')
    soft=handoff['absolute_deadline_realtime_ns']-60*10**9
    if acquired_ns is not None and not handoff['issued_realtime_ns']<=acquired_ns<soft:raise ValueError('absolute handoff expired; acquisition cannot extend it')
    return soft

def validate_native(bundle,reference):
    """Read completed public protocols; never build/run the native engine."""
    from arch_explorer.components.accelsim.protocol import consume_request,preflight_protocol
    x=json_file(original(reference))[0]
    exact(x,{'schema','scope','rows','parameters','hardware_state'},'native state boundary')
    if x['schema']!='arch-explorer.measurement-state-native.v1' or x['scope']!='existing-complete-program-software-only' or set(x['rows'])!=set(bundle.cases) or x['parameters']['physical_parameters'] is not None or x['hardware_state']!='cold-initialization-assumption':raise ValueError('native program/physical state differs')
    parameters=x['parameters']
    exact(parameters,{'physical_parameters','hardware_accuracy','scenario','adf','native_config','unknown'},'unchanged native assumptions')
    if parameters['hardware_accuracy']!='not validated':raise ValueError('native physical accuracy claim')
    adf=json_file(original(parameters['adf']))[0];config=json_file(original(parameters['native_config']))[0]
    for cid,row in x['rows'].items():
        exact(row,{'request','prediction','review','protocol'},'native closure')
        for r in row.values():original(r)
        request=json_file(original(row['request']))[0];p=consume_request(request)
        spec=json_file(original(row['review']))[0];proof=json_file(original(row['protocol']))[0]
        if (p['actual']['name']!=bundle.cases[cid]['function'] or spec['request']!=row['request']['path'] or not same_json(preflight_protocol(spec),proof) or proof['status']!='passed'
            or not same_json(request['adf'],adf) or not same_json(p['config'].artifacts,config)):raise ValueError('native complete program/protocol/physical parameter substitution')
        frozen={r['role']:r for r in spec['frozen']}
        if any(frozen['prediction'][k]!=row['prediction'][k] for k in ('path','sha256')):raise ValueError('native prediction outside frozen protocol')
    return x

def _freeze(design,bundle,stage,stamp,*,parent=None,fit=None):
    return {'schema':'arch-explorer.measurement-state-freeze.v1','kind':'synthetic',
        'run_id':design['run_id'],'design_sha256':design['design_sha256'],'registry_sha256':bundle.sha256,
        'plan_sha256':sha_plan(bundle),'stage':stage,'created_realtime_ns':stamp,'native':design['native'],
        'parameters_changed':False,'parent':parent,'fit_only':fit,'physical_parameters':None}

def _journal(root,design,events,stage,stamp,payload):
    row={'schema':'arch-explorer.measurement-state-journal.v1','run_id':design['run_id'],
         'design_sha256':design['design_sha256'],'index':len(events),'previous':events[-1] if events else None,
         'stage':stage,'synthetic_realtime_ns':stamp,'payload':payload}
    r=write_new(root/'journal'/f'{len(events):03d}.json',row);events.append(r);return r

def rehearse(design_file,output,*,fault=None):
    """Starts only the source/ELF-verified pure CPU Driver, never a GPU host."""
    bundle,d=validate_design(design_file,installation=True);validate_native(bundle,d['native'])
    if fault not in (None,)+CPU_FAULTS:raise ValueError('finite CPU fault required')
    if fault and fault.endswith('-auxiliary') and not any(c['family']=='chase' for c in bundle.cases.values()):
        raise ValueError('auxiliary fault requires a registered chase case')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    write_new(root/'SYNTHETIC.json',{'schema':'arch-explorer.measurement-state-fixture.v1','kind':'synthetic',
        'no_resource_authority':True,'real_world_resource_authority':False,'design':ref(design_file),'run_id':d['run_id']})
    issued=max(time.time_ns(),d['created_realtime_ns']+1)
    handoff={'schema':HANDOFF,'kind':'synthetic','no_resource_authority':True,'run_id':d['run_id'],
        'registry_sha256':bundle.sha256,'plan_sha256':sha_plan(bundle),'budget':d['budget'],
        'resource_owner':'synthetic-no-device-owner','issued_realtime_ns':issued,
        'absolute_deadline_realtime_ns':issued+d['budget']['total_window_s']*10**9,'close_reserve_s':60}
    hr=write_new(root/'SYNTHETIC-HANDOFF.json',handoff);soft=validate_handoff(bundle,d,handoff,acquired_ns=issued)
    stamp=issued+1000;events=[];processes=[];qualities={};error=None;cleanup=True
    _journal(root,d,events,'registered',stamp,{'design':ref(design_file),'handoff':hr})
    initial=write_new(root/'freezes/initial.json',_freeze(d,bundle,'initial',stamp+1000));stamp+=2000
    _journal(root,d,events,'initial',stamp,{'freeze':initial})
    final=None;fit_ref=None
    for partition in ('fit','heldout'):
        if error:break
        if partition=='heldout':
            fit={'schema':'arch-explorer.measurement-state-fit-only.v1','run_id':d['run_id'],
                 'cases':[k for k,c in bundle.cases.items() if c['partition']=='fit'],'quality':qualities.copy(),
                 'status':'passed','physical_parameters':None,'parameters_changed':False,'scope':'formal-quality only; no physical fit'}
            fit_ref=write_new(root/'fit-only.json',fit);stamp+=1000
            _journal(root,d,events,'fit-only',stamp,{'fit':fit_ref})
            final=write_new(root/'freezes/final.json',_freeze(d,bundle,'final',stamp+1000,parent=initial,fit=fit_ref));stamp+=2000
            _journal(root,d,events,'final',stamp,{'freeze':final})
        freeze=initial if partition=='fit' else final
        for cid,c in bundle.cases.items():
            if c['partition']!=partition:continue
            index=len(processes);out=root/f'cpu-{index:02d}-{cid}'
            host_start=stamp+1000;host_end=min(soft,host_start+d['budget']['host_timeout_s']*10**9)
            args=[str(original(d['cpu']['executable'])),'--bundle',str(bundle.path.parent),'--case',cid,
                  '--output',str(out),'--run',d['run_id'],'--freeze',freeze['sha256'],
                  '--start-ns',str(host_start),'--deadline-ns',str(host_end)]
            selected_fault=fault if ((fault=='preparation' and any(s['role']=='preparation' for s in slots(bundle,cid)))
                or (fault and fault.endswith('-auxiliary') and c['family']=='chase')
                or (fault!='preparation' and not (fault and fault.endswith('-auxiliary')) and index==0)) else None
            if selected_fault:args+=['--fault',selected_fault]
            stdout=root/f'cpu-{index:02d}.stdout';stderr=root/f'cpu-{index:02d}.stderr'
            from arch_explorer.components.accelsim.microcatalog import cpu_environment
            with stdout.open('xb') as a,stderr.open('xb') as b:
                p=subprocess.run(args,stdout=a,stderr=b,timeout=d['budget']['host_timeout_s'],env=cpu_environment())
            ledger=ref(out/'LEDGER.json');value=json_file(original(ledger))[0]
            # Timestamp domain is the explicit injected CPU clock, not elapsed hardware time.
            stamp=max(stamp+1000,value['ended_ns'])+1000
            receipt={'schema':'arch-explorer.measurement-state-cpu-process.v1','kind':'synthetic','run_id':d['run_id'],
                'case':cid,'partition':partition,'freeze':freeze,'argv':args,'exit_code':p.returncode,
                'stdout':ref(stdout),'stderr':ref(stderr),'ledger':ledger,'index':index,'wait_completed':True}
            pr=write_new(root/f'process-{index:02d}.json',receipt);processes.append(pr)
            formal=[s['elapsed_ns'] for s in value['slots'] if s['role']=='formal' and s['status']=='returned']
            q=quality(formal) if len(formal)==3 else None
            if q is not None:qualities[cid]=q
            derived=validate_lifecycle(bundle,cid,value)
            cleanup=cleanup and derived['cleanup_complete']
            _journal(root,d,events,'observation',stamp,{'process':pr,'quality':q})
            if p.returncode or value['status']!='returned' or q is None or q['status']!='passed':
                error='fixed C++ execution/quality refused; no additional observations';break
    stamp+=1000
    _journal(root,d,events,'failed' if error else 'closed',stamp,{'error':error})
    close=write_new(root/'CLOSE.json',{'schema':'arch-explorer.measurement-state-close.v1','kind':'synthetic',
        'no_resource_authority':True,'run_id':d['run_id'],'all_CPU_children_reaped':True,'all_injected_resources_reclaimed':cleanup,
        'real_GPU_hosts':0,'real_locks':0,'real_leases':0,'ended_ns':stamp+1000,
        'absolute_deadline_realtime_ns':handoff['absolute_deadline_realtime_ns'],'common_GPU_lock_touched':False})
    outcome={'schema':OUTCOME,'kind':'synthetic','no_resource_authority':True,'run_id':d['run_id'],'design':ref(design_file),
        'status':'failed' if error else 'closed','error':error,'handoff':hr,'initial':initial,'fit_only':fit_ref,'final':final,
        'processes':processes,'journal':events,'close':close,'denominator':None,'physical_parameters':None,'hardware_accuracy':'not validated'}
    outcome['denominator']=denominator(bundle,processes)
    write_new(root/'OUTCOMES.json',outcome)
    result=replay_run(design_file,root);write_new(root/'REPLAY.json',result)
    return result

def denominator(bundle,processes):
    by_case={}
    for p in processes:
        process=json_file(original(p))[0];cid=process['case']
        if cid in by_case:raise ValueError('case was retried or replaced')
        by_case[cid]=json_file(original(process['ledger']))[0]['slots']
    rows=[]
    for cid in bundle.cases:
        actual=by_case.get(cid)
        if actual is not None and len(actual)!=len(slots(bundle,cid)):raise ValueError('missing or extra execution slots')
        for i,planned in enumerate(slots(bundle,cid)):
            rows.append({'case':cid,**planned,'status':'not_started' if actual is None else actual[i]['status'],
                         'attempted':False if actual is None else actual[i]['attempted'],
                         'launch_attempted':False if actual is None else actual[i]['launch_attempted']})
    return rows

def _host(bundle,design,process,freeze,soft):
    cid=process['case'];reference=process['ledger']
    result=read_observation(bundle,cid,reference,design['run_id'],freeze['sha256'])
    ledger=result['ledger'];q=result['quality'];should_pass=result['passed'];derived=result['resources']
    if ledger['kind']!='synthetic':raise ValueError('CPU boundary cannot consume a hardware observation')
    hostroot=original(reference).parent
    if set(hostroot.iterdir())!=result['files']|{hostroot/'COMMAND.json',hostroot/'SYNTHETIC.json'}:raise ValueError('unlisted/missing C++ raw execution original')
    command=json_file(hostroot/'COMMAND.json')[0]
    if command.get('argv')!=process['argv'] or command.get('kind')!='synthetic' or command.get('device_operations')!=0:raise ValueError('actual C++ command was replaced')
    if (process['exit_code']==0)!=should_pass:raise ValueError('CPU process exit hides execution/quality/cleanup failure')
    start=int(process['argv'][process['argv'].index('--start-ns')+1]);end=int(process['argv'][process['argv'].index('--deadline-ns')+1])
    if (end!=min(soft,start+design['budget']['host_timeout_s']*10**9) or ledger['start_ns']!=start or ledger['deadline_ns']!=end
        or start<=json_file(original(freeze))[0]['created_realtime_ns'] or start>=soft):raise ValueError('freeze/deadline did not precede ALL observation roles')
    if process['argv'][0]!=design['cpu']['executable']['path'] or process['argv'][process['argv'].index('--case')+1]!=cid:raise ValueError('actual CPU command differs from frozen source/case')
    if process['argv'][process['argv'].index('--run')+1]!=design['run_id'] or process['argv'][process['argv'].index('--freeze')+1]!=freeze['sha256']:raise ValueError('CPU invocation identity substituted')
    return ledger,q,should_pass,derived

def replay_run(design_file,window,*,reader_source=None):
    """The versioned branch of the public microclosure replay entry point."""
    bundle,d=validate_design(design_file);validate_native(bundle,d['native'])
    if reader_source:
        from arch_explorer.components.accelsim.microcampaign import check_source_installation
        check_source_installation(reader_source)
    root=Path(window).resolve();out=json_file(root/'OUTCOMES.json')[0]
    fields={'schema','kind','no_resource_authority','run_id','design','status','error','handoff','initial','fit_only','final',
        'processes','journal','close','denominator','physical_parameters','hardware_accuracy'}
    exact(out,fields,'state outcome')
    if (out['schema']!=OUTCOME or out['kind']!='synthetic' or out['no_resource_authority'] is not True
        or out['run_id']!=d['run_id'] or out['design']!=ref(design_file) or out['status'] not in ('closed','failed')
        or out['physical_parameters'] is not None or out['hardware_accuracy']!='not validated'):raise ValueError('state outcome identity/physical scope changed')
    marker=json_file(root/'SYNTHETIC.json')[0]
    if marker.get('kind')!='synthetic' or marker.get('no_resource_authority') is not True or marker.get('real_world_resource_authority') is not False or marker.get('design')!=ref(design_file) or marker.get('run_id')!=d['run_id']:raise ValueError('synthetic state boundary lost')
    handoff=json_file(original(out['handoff']))[0];soft=validate_handoff(bundle,d,handoff)
    initial=json_file(original(out['initial']))[0]
    if initial!=_freeze(d,bundle,'initial',initial['created_realtime_ns']) or not handoff['issued_realtime_ns']<initial['created_realtime_ns']<soft:raise ValueError('all initial predictions must freeze before any uninstrumented role')
    fitids=[k for k,c in bundle.cases.items() if c['partition']=='fit'];heldids=[k for k,c in bundle.cases.items() if c['partition']=='heldout']
    order=fitids+heldids;qualities={};passed=True;cleanup=True;last=initial['created_realtime_ns'];seen=[];host_records=[]
    for index,pr in enumerate(out['processes']):
        p=json_file(original(pr))[0]
        exact(p,{'schema','kind','run_id','case','partition','freeze','argv','exit_code','stdout','stderr','ledger','index','wait_completed'},'CPU process receipt')
        if (not passed or index>=len(order) or p['case']!=order[index] or p['run_id']!=d['run_id']
            or p['schema']!='arch-explorer.measurement-state-cpu-process.v1' or p['kind']!='synthetic' or p['index']!=index
            or p['wait_completed'] is not True or p['partition']!=bundle.cases[p['case']]['partition']):raise ValueError('reordered/retried/after-failure/foreign process')
        freeze=out['initial'] if p['partition']=='fit' else out['final']
        if freeze is None or p['freeze']!=freeze:raise ValueError('heldout first-diagnostic/preparation lacks final freeze')
        original(p['stdout']);original(p['stderr'])
        ledger,q,passed,derived=_host(bundle,d,p,freeze,soft)
        if ledger['ended_ns']<last:raise ValueError('injected observation time precedes previous stage')
        last=ledger['ended_ns'];cleanup=cleanup and derived['cleanup_complete'];seen.append(p['case'])
        if q:qualities[p['case']]=q
        host_records.append((pr,p,ledger,q,derived))
    if out['final'] is not None:
        if not set(fitids)<=set(seen) or any(qualities.get(k,{}).get('status')!='passed' for k in fitids):raise ValueError('final freeze before all formal fit-only quality')
        fit=json_file(original(out['fit_only']))[0]
        want={'schema':'arch-explorer.measurement-state-fit-only.v1','run_id':d['run_id'],'cases':fitids,
              'quality':{k:qualities[k] for k in fitids},'status':'passed','physical_parameters':None,
              'parameters_changed':False,'scope':'formal-quality only; no physical fit'}
        if fit!=want:raise ValueError('fit consumed heldout observations or changed parameters')
        final=json_file(original(out['final']))[0]
        if final!=_freeze(d,bundle,'final',final['created_realtime_ns'],parent=out['initial'],fit=out['fit_only']):raise ValueError('final freeze retuned native or role strategy')
        last_fit=max(r[2]['ended_ns'] for r in host_records if r[1]['partition']=='fit')
        if not last_fit<final['created_realtime_ns']<soft:raise ValueError('final freeze did not follow fit-only observations')
    elif out['fit_only'] is not None or any(c in heldids for c in seen):raise ValueError('heldout or fit result without final freeze')
    expected_terminal='closed' if passed and len(seen)==len(order) else 'failed'
    if out['status']!=expected_terminal or (out['error'] is None)!=(expected_terminal=='closed'):raise ValueError('outcome terminal hides C++/quality failure or missing roles')
    expected_denominator=denominator(bundle,out['processes'])
    if out['denominator']!=expected_denominator:raise ValueError('formal/all-execution denominator altered')
    expected_ledgers={original(p['ledger']).resolve() for _,p,_,_,_ in host_records}
    if set(root.glob('cpu-*/LEDGER.json'))!=expected_ledgers:raise ValueError('hidden/unlisted host transcript')
    close=json_file(original(out['close']))[0]
    if (close.get('schema')!='arch-explorer.measurement-state-close.v1' or close.get('kind')!='synthetic'
        or close.get('no_resource_authority') is not True or close.get('run_id')!=d['run_id']
        or close.get('all_CPU_children_reaped') is not True or close.get('all_injected_resources_reclaimed')!=cleanup
        or close.get('real_GPU_hosts')!=0 or close.get('real_locks')!=0 or close.get('real_leases')!=0
        or close.get('common_GPU_lock_touched') is not False or close.get('absolute_deadline_realtime_ns')!=handoff['absolute_deadline_realtime_ns']
        or not last<=close['ended_ns']<=handoff['absolute_deadline_realtime_ns']):raise ValueError('resource cleanup/deadline falsely closed')
    previous=None;stages=[];payloads=[];stamp=d['created_realtime_ns']
    for i,r in enumerate(out['journal']):
        e=json_file(original(r))[0]
        if e.get('run_id')!=d['run_id'] or e.get('design_sha256')!=d['design_sha256'] or e.get('index')!=i or e.get('previous')!=previous or not stamp<e['synthetic_realtime_ns']<=close['ended_ns']:raise ValueError('journal chain/run/time changed')
        stamp=e['synthetic_realtime_ns'];previous=r;stages.append(e['stage']);payloads.append(e['payload'])
    expected=['registered','initial']
    if out['final'] is not None:expected+=['observation']*len(fitids)+['fit-only','final']+['observation']*len([c for c in seen if c in heldids])
    else:expected+=['observation']*len(seen)
    expected+=[expected_terminal]
    if stages!=expected or payloads[0]!={'design':ref(design_file),'handoff':out['handoff']} or payloads[1]!={'freeze':out['initial']}:raise ValueError('journal stage closure differs')
    observed=[x for stage,x in zip(stages,payloads) if stage=='observation']
    if observed!=[{'process':pr,'quality':q} for pr,_,_,q,_ in host_records] or payloads[-1]!={'error':out['error']}:raise ValueError('journal differs from C++ observation originals')
    if out['final'] is not None:
        if payloads[stages.index('fit-only')]!={'fit':out['fit_only']} or payloads[stages.index('final')]!={'freeze':out['final']}:raise ValueError('journal freeze/fit references changed')
    total_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and p!=root/'REPLAY.json')
    if total_bytes>d['budget']['host_raw_bytes']:raise ValueError('complete retained original budget exceeded')
    validate_state_evidence(d,d['created_realtime_ns'],close['ended_ns'])
    counts={r:dict(Counter(x['status'] for x in expected_denominator if x['role']==r)) for r in ('diagnostic','preparation','formal')}
    return {'schema':REPLAY,'replay_status':'replayed','kind':'synthetic','collection_outcome':out['status'],
        'run_id':d['run_id'],'plan_sha256':sha_plan(bundle),'budget':d['budget'],'denominator':expected_denominator,
        'role_counts':counts,'formal_quality':qualities,'stages':stages,'cleanup_complete':cleanup,
        'CPU_children_reaped':True,'resource_lifecycles':{p['case']:life for _,p,_,_,life in host_records},
        'hardware_execution':False,'real_GPU_hosts':0,'real_locks':0,'real_leases':0,
        'structure_evidence':'existing complete programs selected by reference; no new structural launch',
        'physical_parameters':None,'hardware_accuracy':'not validated','clock_mhz':None,'cache_state':None,
        'comparison':{'state_compatibility':'refused','native_state':d['native_state'],
                      'observation_state':'diagnostic/preparation-conditioned; cache/clock unknown',
                      'native_mape':None,'native_worst':None,'reason':'CPU observations and native cold-state assumptions are not equivalent physical-state evidence'},
        'source_epochs':{'collection':d['source'],'reader':reader_source},'total_original_bytes':total_bytes}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    build=s.add_parser('build-cpu')
    for n in ('registry','generation','output','measurement-sources'):build.add_argument('--'+n,required=True)
    build.add_argument('--cxx',default='/usr/bin/g++')
    ready=s.add_parser('ready');ready.add_argument('--design',required=True);ready.add_argument('--output',required=True)
    demo=s.add_parser('rehearse');demo.add_argument('--design',required=True);demo.add_argument('--output',required=True);demo.add_argument('--fault',choices=CPU_FAULTS)
    a=p.parse_args(argv)
    if a.command=='build-cpu':r=build_cpu(a.registry,ref(a.generation),a.output,a.cxx,a.measurement_sources)
    elif a.command=='ready':
        b,d=validate_design(a.design,installation=True);r={'status':'passed','registry':ref(b.path),'plan_sha256':sha_plan(b),
            'budget':d['budget'],'hardware_operations':0,'resource_authority':False};write_new(a.output,r)
    else:r=rehearse(a.design,a.output,fault=a.fault)
    print(json.dumps(r));return 0

if __name__=='__main__':raise SystemExit(main())
