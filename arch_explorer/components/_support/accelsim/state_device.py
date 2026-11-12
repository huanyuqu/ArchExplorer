# Concrete Accel-Sim implementation; owned by the component provider.
"""Finite state campaign; explicit resource/device capabilities, shared C++ roles.

Import is CPU-only. Actual capabilities require idle admission or a historical
coordinated entry. Synthetic capabilities exercise this identical stage machine and never
select production hosts. Observations are validated by state_observation.
"""
import argparse,json,os,sys,time
from pathlib import Path
from arch_explorer.components.accelsim.trace import exact,same_json
from arch_explorer.components.accelsim.microbundle import load_registry,json_file
from arch_explorer.components.accelsim.microrecords import original
from arch_explorer.components.accelsim.microexperiment import ref,write_new,seal,make_design as base_design,prediction_freeze,validate_prediction_freeze
from arch_explorer.components.accelsim.measurement_state import stateful,sha_plan,slots
from arch_explorer.components.accelsim.microcampaign import Collector,check_installation,check_source_installation
from arch_explorer.components.accelsim.state_observation import read_observation

DESIGN='arch-explorer.measurement-state-device-design.v1'
OUTCOME='arch-explorer.measurement-state-device-outcomes.v1'
PROTOCOL='arch-explorer.measurement-state-device-protocol.v1'
PROCESS='arch-explorer.measurement-state-device-process.v1'

def validate_extension(bundle,d):
    from arch_explorer.components.accelsim.state_resource import snapshot_policy
    p=d['state_protocol'];exact(p,{'schema','cpu','snapshot_policy','native_state'},'device state protocol')
    if (not stateful(bundle) or len(bundle.cases)>8 or p['schema']!=PROTOCOL or d['plan_sha256']!=sha_plan(bundle)
        or p['snapshot_policy']!=snapshot_policy() or p['native_state']!='cold-initialization-assumption'
        or any([s['role'] for s in slots(bundle,cid)]!=['diagnostic','preparation','preparation','formal','formal','formal'] for cid in bundle.cases)):
        raise ValueError('new full device-state role protocol required')
    from arch_explorer.components.accelsim.microstate import check_cpu_binding
    binding=json_file(original(d['catalog']))[0]
    check_cpu_binding(bundle,p['cpu'],binding['generation'],source=d['source'])

def make_design(bundle,source,engine,hosts,tracer,history,method,catalog,cpu):
    from arch_explorer.components.accelsim.state_resource import snapshot_policy
    d=base_design(bundle,source,engine,hosts,tracer,history,method,catalog=catalog)
    d.pop('design_sha256');d['schema']=DESIGN
    d['plan_sha256']=sha_plan(bundle)
    d['state_protocol']={'schema':PROTOCOL,'cpu':cpu,
        'snapshot_policy':snapshot_policy(),'native_state':'cold-initialization-assumption'}
    d['design_sha256']=seal(d);validate_extension(bundle,d);return d

def validate_design(path,*,installation=False):
    from arch_explorer.components.accelsim.microexperiment import validate_design as validate_base
    d=json_file(path)[0];b=load_registry(original(d['registry']))
    if d.get('schema')!=DESIGN:raise ValueError('explicit device-state design required')
    validate_base(b,d)
    if installation:
        check_source_installation(d['source'])
        from arch_explorer.components.accelsim.microcatalog import require_design_catalog
        require_design_catalog(d,source=d['source'],run_policy=True)
    return b,d

def denominator(bundle,processes,bindings):
    bycase={p['case']:p for p in processes if p['case'] is not None};rows=[]
    capture=next((p for p in processes if p['phase']=='structure'),None)
    attempts={};reported={}
    if capture:
        raw=json_file(original(capture['raw']))[0];root=Path(raw['output'])
        if (root/'failure.json').exists():attempts={x['case']:x['attempted'] for x in json_file(root/'failure.json')[0]['cases']}
        for cid in bundle.cases:
            sample=root/cid/'sample-0.json'
            if sample.exists() and json_file(sample)[0].get('status')=='returned':reported[cid]=True
    for cid in bundle.cases:
        attempted=cid in bindings or cid in reported or bool(attempts.get(cid))
        rows.append({'case':cid,'role':'structure','sequence':0,'attempted':attempted,'launch_attempted':attempted,
                     'status':'returned' if cid in bindings or cid in reported else 'failed' if attempted else 'not_started'})
        p=bycase.get(cid);ledger=json_file(original(p['ledger']))[0] if p and p['ledger'] else None
        for s in slots(bundle,cid):
            actual=ledger['slots'][s['sequence']] if ledger else None
            rows.append({**s,'case':cid,'attempted':False if actual is None else actual['attempted'],
                         'launch_attempted':False if actual is None else actual['launch_attempted'],
                         'status':'not_started' if actual is None else actual['status']})
    return rows

class Journal:
    def __init__(self,campaign):self.c=campaign
    def structures(self,bindings):self.c.event('structure',{'bindings':bindings})
    def initial_predictions(self,freeze):self.c.event('initial',{'freeze':freeze})

class DeviceTransport(Collector):
    """Reuse the established process/lock/capture/native mechanisms, new roles."""
    def __init__(self,c):
        self.c=c
        self.design_file=c.design_file;self.design=c.design;self.bundle=c.bundle;self.budget=c.design['budget']
        self.origin=check_installation(self.design);self.handoff_ref=c.handoff_ref;self.handoff=c.handoff
        self.root=c.root;self.processes=[];self.capture_bytes=0;self.lease_ref=None
        self.lock_fd=None;self.opened=False;self.deadline=None;self.started=None;self.journal=Journal(c)

    def acquire(self):
        from arch_explorer.components.accelsim.state_idle import REQUEST, acquire
        if self.handoff.get('schema')!=REQUEST:
            return super().acquire()
        admission_ref,admission=acquire(self,self.handoff_ref)
        self.handoff_ref=self.c.handoff_ref=admission_ref
        self.handoff=self.c.handoff=admission
        self.issue_lease()

class ActualCapabilities:
    kind='hardware'
    def __init__(self,c):
        if (c.root/'SYNTHETIC.json').exists():raise ValueError('fixture root cannot open actual capabilities')
        self.c=c;self.t=DeviceTransport(c);self.queries=set()
    def acquire(self):self.t.acquire();return self.t.lease_ref
    def capture(self):
        try:return self.t.structures()
        finally:
            if self.t.processes:
                pr=self.t.processes[-1];raw=json_file(original(pr))[0]
                frame={'schema':PROCESS,'kind':self.kind,'run_id':self.c.design['run_id'],'index':len(self.c.frames),
                    'case':None,'phase':'structure','raw':pr,'ledger':None,'before':None,'after':None,
                    'started_realtime_ns':raw['started_realtime_ns'],'ended_realtime_ns':raw.get('ended_realtime_ns',time.time_ns()),
                    'reaped':raw['status'] in ('returned','failed','timeout'),'execution_status':raw['status'],'error':None}
                write_new(self.c.root/f'frame-{len(self.c.frames):02d}.json',frame);self.c.frames.append(frame)
    def predict(self,output,bindings):return self.t.predict(output,bindings)
    def now(self):return time.time_ns()
    def snapshot(self,cid,position,uuid_hex):
        from arch_explorer.components.accelsim.state_resource import snapshot_policy,validate_snapshot
        import subprocess
        if cid not in self.c.bundle.cases or position not in ('before','after'):raise ValueError('unregistered state query case/position; no device operation')
        key=(cid,position)
        if key in self.queries or len(self.queries)>=2*len(self.c.bundle.cases):raise ValueError('duplicate/extra state query; no device operation')
        self.queries.add(key)
        if self.t.remaining()<=6:raise TimeoutError('insufficient bounded state-query margin; no extra device work')
        policy=snapshot_policy();start=time.time_ns();mono=time.monotonic_ns()
        args=policy['query'];timeout=min(policy['timeout_s'],self.t.remaining())
        stdout=self.c.root/'state'/f'{cid}-{position}.stdout';stderr=stdout.with_suffix('.stderr')
        try:
            p=subprocess.run(args,capture_output=True,timeout=timeout,check=False,env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
            code=p.returncode;out=p.stdout;err=p.stderr
        except subprocess.TimeoutExpired as e:code=-9;out=e.stdout or b'';err=e.stderr or b''
        stdout.parent.mkdir(exist_ok=True);stdout.write_bytes(out);stderr.write_bytes(err)
        row={'schema':'arch-explorer.measurement-state-snapshot.v1','kind':self.kind,'no_resource_authority':False,
            'run_id':self.c.design['run_id'],'case':cid,'position':position,'uuid_hex':uuid_hex,'argv':args,
            'status':'unknown','exit_code':code,'started_realtime_ns':start,'ended_realtime_ns':time.time_ns(),
            'started_monotonic_ns':mono,'ended_monotonic_ns':time.monotonic_ns(),'stdout':ref(stdout),'stderr':ref(stderr),
            'values':{'sm_mhz':None,'memory_mhz':None,'pstate':None},'reason':'state unavailable; no retry'}
        try:
            fields=out.decode().strip().split(',')
            if code==0 and len(fields)==4 and fields[1].strip().isdigit() and fields[2].strip().isdigit():
                row.update(status='returned',reason=None,values={'sm_mhz':int(fields[1]),'memory_mhz':int(fields[2]),'pstate':fields[3].strip()})
        except (ValueError,UnicodeError):pass
        result=write_new(self.c.root/'state'/f'{cid}-{position}.json',row)
        view={**self.c.design,'absolute_deadline_realtime_ns':self.c.handoff['absolute_deadline_realtime_ns'],'close_reserve_s':60}
        validate_snapshot(row,view,cid,position,uuid_hex,synthetic=False)
        return result
    def observe(self,cid,freeze):
        uuid=json_file(self.c.capture_output/'preflight.json')[0]['uuid_hex']
        before=self.snapshot(cid,'before',uuid);error=None;attempt_at=len(self.t.processes)
        try:output,pr=self.t.host(self.c.bundle.cases[cid]['partition'],freeze,cid)
        except BaseException as e:
            error=type(e).__name__+': '+str(e)
            if len(self.t.processes)==attempt_at:raise
            pr=self.t.processes[-1];output=Path(json_file(original(pr))[0]['output'])
        raw=json_file(original(pr))[0];after=None
        try:
            if self.t.remaining()>6:after=self.snapshot(cid,'after',uuid)
            else:error=error or 'after state query not_started: fixed cleanup/query margin'
        except TimeoutError:error=error or 'after state query not_started: work cutoff'
        frame={'schema':PROCESS,'kind':self.kind,'run_id':self.c.design['run_id'],'index':len(self.c.frames),
            'case':cid,'phase':self.c.bundle.cases[cid]['partition'],'raw':pr,
            'ledger':ref(output/'LEDGER.json') if (output/'LEDGER.json').exists() else None,
            'before':before,'after':after,'started_realtime_ns':raw['started_realtime_ns'],
            'ended_realtime_ns':raw['ended_realtime_ns'],'reaped':raw['status'] in ('returned','failed','timeout'),
            'execution_status':raw['status'],'error':error}
        return frame
    def release(self):
        from arch_explorer.components.accelsim.microguard import processes,descendants,kill_owned,reap
        owned=descendants(processes(),os.getpid());kill_owned(owned)
        until=min(time.monotonic()+10,time.monotonic()+max(0,(self.c.handoff['absolute_deadline_realtime_ns']-time.time_ns())/1e9-1))
        while True:
            reap();table=processes();remaining={pid:tick for pid,tick in owned.items() if pid in table and table[pid][1]==tick}
            if not remaining or time.monotonic()>=until:break
            kill_owned(remaining);time.sleep(.02)
        if remaining:raise RuntimeError('own descendant cleanup incomplete; no successful release')
        self.t.release()
        reaps=[ref(p) for p in sorted(self.c.root.glob('reap-*.json'))]
        return {'lease':self.t.lease_ref,'raw_processes':self.t.processes,'reaps':reaps,
            'all_owned_hosts_reaped':len(reaps)==len(self.t.processes),
            'common_lock_released':self.t.lock_fd is None,'real_locks':int(self.t.opened or getattr(self.t,'resource_lock_acquired',False)),'real_leases':int(self.t.opened),
            'ended_realtime_ns':time.time_ns(),'owned_process_cleanup':{'observed':owned,'remaining':remaining}}

class Campaign:
    def __init__(self,design_file,handoff_file,output,*,capabilities=ActualCapabilities):
        self.design_file=Path(design_file).resolve();self.bundle,self.design=validate_design(self.design_file,installation=True)
        self.kind=capabilities.kind;self.handoff_ref=ref(handoff_file);self.handoff=json_file(original(self.handoff_ref))[0]
        from arch_explorer.components.accelsim.state_resource import validate_handoff
        from arch_explorer.components.accelsim.state_idle import REQUEST, validate_request
        if self.handoff.get('schema')==REQUEST:
            if self.kind!='hardware':raise ValueError('idle requests cannot authorize CPU fixture boundaries')
            validate_request(self.bundle,self.design,self.handoff)
        else:validate_handoff(self.bundle,self.design,self.handoff,synthetic=self.kind=='synthetic')
        if self.kind=='hardware':check_installation(self.design)
        self.root=Path(output).resolve()
        if self.kind=='synthetic':
            marker=json_file(self.root/'SYNTHETIC.json')[0]
            if marker.get('kind')!='synthetic' or marker.get('no_resource_authority') is not True or marker.get('run_id')!=self.design['run_id']:raise ValueError('marked synthetic boundary required')
        else:self.root.mkdir(parents=True,exist_ok=False)
        self.frames=[];self.journal=[];self.bindings={};self.initial=self.final=self.fit=self.lease=None
        self.parameters=None;self.observations=[];self.capture_output=None
        self.io=capabilities(self)
    def event(self,stage,payload):
        e={'schema':'arch-explorer.measurement-state-device-event.v1','kind':self.kind,'run_id':self.design['run_id'],
            'design':ref(self.design_file),'index':len(self.journal),'previous':self.journal[-1] if self.journal else None,
            'stage':stage,'realtime_ns':self.io.now(),'payload':payload}
        r=write_new(self.root/'journal'/f'{len(self.journal):03d}.json',e);self.journal.append(r)
    def execute(self):
        error=None;predictions=None
        try:
            self.lease=self.io.acquire();self.event('acquired',{'handoff':self.handoff_ref,'lease':self.lease})
            self.capture_output,self.bindings=self.io.capture()
            self.initial,predictions,self.parameters=self.io.predict(self.capture_output,self.bindings)
            for partition in ('fit','heldout'):
                if partition=='heldout':
                    qualities={json_file(original(r))[0]['case']:json_file(original(r))[0]['quality'] for r in self.observations}
                    self.fit=write_new(self.root/'fit-only.json',{'schema':'arch-explorer.measurement-state-fit-only.v2',
                        'run_id':self.design['run_id'],'cases':self.design['order']['fit'],'quality':qualities,
                        'parameters_changed':False,'physical_parameters':None,'state_compatibility':False,'status':'passed'})
                    self.event('fit-only',{'fit':self.fit})
                    self.final=prediction_freeze(self.bundle,self.design,predictions,self.parameters,self.root/'freezes',
                        stage='final',fit_result=json_file(original(self.fit))[0],parent=self.initial)
                    self.event('final',{'freeze':self.final})
                freeze=self.initial if partition=='fit' else self.final
                for cid in self.design['order'][partition]:
                    f=self.io.observe(cid,freeze);fr=write_new(self.root/f'frame-{len(self.frames):02d}.json',f);self.frames.append(f)
                    result=None
                    if f['ledger']:
                        result=read_observation(self.bundle,cid,f['ledger'],self.design['run_id'],freeze['sha256'])
                    row={'case':cid,'partition':partition,'frame':fr,'freeze':freeze,
                         'quality':None if result is None else result['quality'],
                         'resources':None if result is None else result['resources'],
                         'passed':bool(result and result['passed'] and f['execution_status']=='returned')}
                    rr=write_new(self.root/'observations'/(cid+'.json'),row);self.observations.append(rr)
                    self.event('observation',{'record':rr})
                    if not row['passed']:raise ValueError('first execution/quality/resource failure; no next case or retry')
        except BaseException as e:error=type(e).__name__+': '+str(e)
        finally:
            try:release=self.io.release()
            except BaseException as e:
                error=error or ('resource release incomplete: '+str(e))
                release={'lease':self.lease,'raw_processes':[],'reaps':[],'all_owned_hosts_reaped':False,'common_lock_released':False,'real_locks':None,'real_leases':None}
            self.event('failed' if error else 'closed',{'error':error})
            close=write_new(self.root/'RESOURCE-CLOSE.json',{'schema':'arch-explorer.measurement-state-device-close.v1',
                'kind':self.kind,'run_id':self.design['run_id'],'handoff':self.handoff_ref,**release,'ended_realtime_ns':self.io.now()})
            out={'schema':OUTCOME,'kind':self.kind,'no_resource_authority':self.kind=='synthetic','run_id':self.design['run_id'],
                'design':ref(self.design_file),'handoff':self.handoff_ref,'lease':self.lease,'status':'failed' if error else 'closed',
                'error':error,'bindings':self.bindings,'initial':self.initial,'fit_only':self.fit,'final':self.final,
                'frames':[ref(self.root/f'frame-{i:02d}.json') for i in range(len(self.frames))],
                'observations':self.observations,'journal':self.journal,'resource_close':close,
                'denominator':denominator(self.bundle,self.frames,self.bindings),'parameters':self.parameters,
                'physical_parameters':None,'hardware_accuracy':'not validated','retry_allowed':False}
            write_new(self.root/'OUTCOMES.json',out)
        return error
