# Concrete Accel-Sim implementation; owned by the component provider.
"""Finite SIM30 collection controller. Import/check/replay are CPU only.

Only the explicit ``run`` command starts hosts. It requires a newly coordinated
handoff, the exact installed software/design, the common flock and an empty
external-compute check. A failed slot is never replaced or retried.
"""
from arch_explorer.components._support.paths import package_root
import argparse
from fractions import Fraction
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from arch_explorer.components.accelsim.microbundle import load_registry, json_file, BUDGET
from arch_explorer.components.accelsim.microrecords import (original, case_files, validate_case_attempt,
                           attempt_reference, load_attempt, ENV_KEYS)
from arch_explorer.components.accelsim.microexperiment import (ref, write_new, validate_design, prediction_freeze,
                               fit_intervals, sample_quality, Journal)
from arch_explorer.components.accelsim.trace import exact, same_json
from arch_explorer.components.accelsim.microcontract import budget,modern,work_item,identity,schema,validate_handoff


def check_installation(design):
    origin=check_source_installation(design['source'])
    if origin.get('kind')=='synthetic':raise ValueError('synthetic boundary source cannot open a hardware campaign')
    from arch_explorer.components.accelsim.microcatalog import require_design_catalog
    require_design_catalog(design,source=design['source'],run_policy=True)
    return origin


def check_source_installation(source):
    """Reader identity and the collection's frozen source are separate eras."""
    origin, _ = json_file(original(source))
    archive = original({'path':origin['archive'],'sha256':origin['archive_sha256']})
    if str(Path(sys.executable).absolute()) != origin['installed_python']:
        raise ValueError('controller must use the frozen installed interpreter')
    package = package_root()
    for name, sha in origin['installed_modules'].items():
        original({'path':str(package/name),'sha256':sha})
    for name, sha in origin.get('installed_resources',{}).items():
        original({'path':str(package/name),'sha256':sha})
    if not origin['clean'] or not archive.is_file(): raise ValueError('clean source archive missing')
    return origin


def total_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def raw_budget(root):
    root=Path(root)
    capture=sum(f.stat().st_size for name in ('records.bin','static.bin','static.json','capture.json') for f in root.glob('host-*/*/'+name))
    raw=sum(total_bytes(d) for d in root.glob('host-*') if d.is_dir())+sum(f.stat().st_size for pat in ('process-*.stdout','process-*.stderr') for f in root.glob(pat))-capture
    return {'capture_bytes':capture,'host_raw_bytes':raw}

def require_raw_budget(root,limits=None):
    limits=BUDGET if limits is None else limits
    sizes=raw_budget(root)
    if sizes['capture_bytes']>limits['capture_bytes'] or sizes['host_raw_bytes']>limits['host_raw_bytes']:raise ValueError('aggregate original file budget exceeded')
    return sizes


def check_handoff(design, handoff):
    data, _ = json_file(original(handoff))
    if not isinstance(design,dict) or 'registry' not in design:raise ValueError('frozen registered design required before handoff')
    bundle=load_registry(original(design['registry']));validate_handoff(bundle,design,data)
    if data['issued_realtime_ns'] > time.time_ns():
        raise ValueError('handoff predates new design or is in the future')
    from arch_explorer.components.accelsim.state_resource import handoff_evidence
    evidence=handoff_evidence(data)
    if not evidence: raise ValueError('actual resource admission originals missing')
    for row in evidence: original(row)
    if not Path(data['lock_file']).is_absolute(): raise ValueError('absolute common lock required')
    return data


def native_assumption_scenario(preflight):
    """The preregistered scenario, reusable for CPU consumption after closure."""
    from arch_explorer.adf import Param
    from arch_explorer.components.accelsim.scenarios import sm120_mechanism_adf, device_scenario
    facts,_=json_file(preflight)
    assumptions=sm120_mechanism_adf()
    shared=assumptions.child('cluster').child('l1_shared')
    shared.params['gpgpu_adaptive_cache_config']=Param(0,'1','synthetic fixed shared allocation; actual adaptive allocation policy remains unknown')
    assumptions.networks[0].params['booksim_k']=Param(facts['limits']['sm_count']+80,'count','synthetic SIM30 network node count; physical topology and timing unknown')
    return device_scenario(assumptions,preflight)


class Collector:
    def __init__(self, design_file, handoff_file, output):
        self.design_file = Path(design_file).resolve(); self.design, _ = json_file(self.design_file)
        self.bundle = load_registry(original(self.design['registry']));self.budget=budget(self.bundle)
        from arch_explorer.components.accelsim.measurement_state import stateful
        if stateful(self.bundle):raise ValueError('state registry requires the explicit state controller; legacy hardware controller cannot authorize roles')
        validate_design(self.bundle,self.design); self.origin = check_installation(self.design)
        self.handoff_ref = ref(handoff_file); self.handoff = check_handoff(self.design,self.handoff_ref)
        self.root = Path(output).resolve(); self.root.mkdir(parents=True,exist_ok=False)
        self.processes = []; self.capture_bytes = 0; self.lease_ref = None
        self.lock_fd = None; self.opened = False; self.deadline = None; self.started = None
        self.journal = Journal(self.root/'journal',self.bundle,self.design)

    def remaining(self):
        value = self.deadline-time.monotonic()
        if value <= 0: raise TimeoutError('whole-window deadline expired; no retry')
        return value

    def acquire(self):
        # No CUDA/library/host initialization. Device management is reached only
        # after a new actual resource handoff and common-lock acquisition.
        remaining=self.budget['total_window_s']
        if modern(self.bundle):
            remaining=min(remaining,(self.handoff['absolute_deadline_realtime_ns']-time.time_ns())/1e9-self.handoff['close_reserve_s'])
            if remaining<=0:raise TimeoutError('absolute reservation expired before lock/host; no retry')
        self.lock_fd = os.open(self.handoff['lock_file'],os.O_RDWR)
        try: fcntl.flock(self.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            os.close(self.lock_fd); self.lock_fd = None; raise
        self.started = time.monotonic(); self.deadline = self.started+remaining
        if modern(self.bundle):
            self.deadline=min(self.deadline,time.monotonic()+(self.handoff['absolute_deadline_realtime_ns']-time.time_ns())/1e9-self.handoff['close_reserve_s'])
            self.remaining()
        args = ['/usr/bin/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']
        completed = subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=min(10,self.remaining()),check=False)
        (self.root/'external-compute.stdout').write_bytes(completed.stdout)
        (self.root/'external-compute.stderr').write_bytes(completed.stderr)
        write_new(self.root/'external-compute.json',{'argv':args,'exit_code':completed.returncode,'stdout':ref(self.root/'external-compute.stdout'),'stderr':ref(self.root/'external-compute.stderr'),'checked_realtime_ns':time.time_ns(),'before_any_host':True})
        if completed.returncode or completed.stdout.strip(): raise ValueError('external compute nonempty/unknown; no host started')
        self.issue_lease()

    def issue_lease(self):
        """Bind a locally admitted or coordinated window to the same host lease."""
        now = time.time_ns()
        deadline=now+int(self.remaining()*1e9)
        if modern(self.bundle):deadline=min(deadline,self.handoff['absolute_deadline_realtime_ns']-self.handoff['close_reserve_s']*10**9)
        lease = {'schema':schema(self.bundle,'lease'),'run_id':self.design['run_id'],'work_item':work_item(self.bundle),'design_sha256':self.design['design_sha256'],'registry_sha256':self.bundle.sha256,'new_window':True,'handoff':self.handoff_ref,'issued_realtime_ns':now,'deadline_realtime_ns':deadline,'device_ordinal':0,'lock_file':self.handoff['lock_file'],'budget':self.budget}
        from arch_explorer.components.accelsim.measurement_state import stateful,sha_plan
        if stateful(self.bundle):lease['plan_sha256']=sha_plan(self.bundle)
        self.lease_ref = write_new(self.root/'lease.json',lease); self.opened = True

    def host(self, phase, freeze, cid=None):
        self.remaining()
        index = len(self.processes)
        order = ['structure']+self.design['order']['fit']+self.design['order']['heldout']
        if index >= self.budget['host_processes'] or order[index] != ('structure' if phase=='structure' else cid):
            raise ValueError('duplicate/extra/out-of-order host slot')
        freeze_ref=None if phase=='structure' else freeze
        freeze_sha=freeze if phase=='structure' else freeze['sha256']
        output = self.root/(f'host-{index:02d}-'+(cid or 'structure'))
        host = self.design['hosts']['capture' if phase=='structure' else 'timing']
        host_start=time.time_ns()
        argv = [str(original(host)),'--bundle',str(self.bundle.path.parent),'--output',str(output),'--lease',self.lease_ref['path'],'--lease-sha',self.lease_ref['sha256'],'--freeze',freeze_sha,'--lock-file',self.handoff['lock_file'],'--lock-fd',str(self.lock_fd)]
        if phase != 'structure': argv += ['--case',cid]
        from arch_explorer.components.accelsim.measurement_state import stateful,slots
        if stateful(self.bundle):argv+=['--run',self.design['run_id'],'--start-ns',str(host_start),'--deadline-ns',str(min(json_file(original(self.lease_ref))[0]['deadline_realtime_ns'],host_start+self.budget['host_timeout_s']*10**9))]
        env = {'PATH':'/usr/bin:/bin','LC_ALL':'C','CUDA_VISIBLE_DEVICES':'0','CUDA_MODULE_LOADING':'EAGER'}
        if phase == 'structure':
            nvdisasm=original(self.origin['runtime_tools']['nvdisasm'])
            env.update(LD_PRELOAD=str(original(self.design['tracer'])),NVDISASM=str(nvdisasm),PATH=str(nvdisasm.parent)+':/usr/bin:/bin')
        recorded_env = {k:env.get(k) for k in ENV_KEYS}
        st = os.fstat(self.lock_fd)
        account = {'process_index':index,'case_ids':list(self.bundle.cases) if phase=='structure' else [cid],'launch_quota':len(self.bundle.cases) if phase=='structure' else len(slots(self.bundle,cid)) if stateful(self.bundle) else self.budget['samples_per_case']}
        stdout = self.root/(f'process-{index:02d}.stdout'); stderr = self.root/(f'process-{index:02d}.stderr')
        receipt = {'schema':schema(self.bundle,'process-receipt'),**identity(self.bundle),'kind':'hardware','phase':phase,'registry_sha256':self.bundle.sha256,'design_sha256':self.design['design_sha256'],'design':ref(self.design_file),'freeze_sha256':freeze_sha,'prediction_freeze':freeze_ref,'lease':self.lease_ref,'command':argv,'environment':recorded_env,'launch_environment':env,'pid':None,'supervisor_pid':os.getpid(),'exit_code':None,'status':'starting','started_realtime_ns':host_start,'started_monotonic_ns':time.monotonic_ns(),'host':host,'output':str(output),'stdout':None,'stderr':None,'lock':{'path':self.handoff['lock_file'],'device':st.st_dev,'inode':st.st_ino,'fd':self.lock_fd,'exclusive':True},'budget_account':account}
        attempt = write_new(self.root/(f'process-{index:02d}.starting.json'),receipt)
        self.processes.append(attempt)  # Attempt is consumed before Popen, even on failure.
        killed = None
        with stdout.open('xb') as out, stderr.open('xb') as err:
            p = subprocess.Popen(argv,stdout=out,stderr=err,env=env,pass_fds=(self.lock_fd,),start_new_session=True)
            receipt['pid'] = p.pid
            try:
                while p.poll() is None:
                    if time.monotonic_ns()-receipt['started_monotonic_ns'] > self.budget['host_timeout_s']*10**9 or time.monotonic() >= self.deadline:
                        killed = 'timeout'
                    try: require_raw_budget(self.root,self.budget)
                    except ValueError: killed = 'aggregate original file budget exceeded'
                    if phase == 'structure' and output.exists():
                        capture_size = sum(f.stat().st_size for name in ('records.bin','static.bin','static.json','capture.json') for f in output.glob('*/'+name))
                        if capture_size > self.budget['capture_bytes']: killed = 'capture original budget exceeded'
                    if killed:
                        os.killpg(p.pid,signal.SIGKILL); break
                    time.sleep(0.1)
            finally:
                if p.poll() is None: os.killpg(p.pid,signal.SIGKILL)
                code=p.wait()
            code = p.wait()  # Reap before releasing the common lock.
        receipt.update(exit_code=code,status='returned' if code==0 and killed is None else ('timeout' if killed=='timeout' else 'failed'),ended_realtime_ns=time.time_ns(),ended_monotonic_ns=time.monotonic_ns(),stdout=ref(stdout),stderr=ref(stderr))
        final = write_new(self.root/(f'process-{index:02d}.json'),receipt); self.processes[-1] = final
        self.record_reap(final)
        if code or killed: raise ValueError('host failed; originals retained, no retry: '+str(final['path']))
        require_raw_budget(self.root,self.budget)
        return output,final

    def structures(self):
        from arch_explorer.components.accelsim.microcapture import SCHEMA, ROLES, derive
        output, receipt = self.host('structure',self.design['design_sha256'])
        bindings = {}
        for cid, c in self.bundle.cases.items():
            paths = case_files(output,cid,0,receipt['path'])
            paths.update(records=output/cid/'records.bin',static=output/cid/'static.json',static_table=output/cid/'static.bin',capture=output/cid/'capture.json',bundle=self.bundle.path,cubin=self.bundle.file(self.bundle.document['cubin']))
            if set(paths)!=ROLES: raise ValueError('closed original capture roles differ')
            binding = {'schema':SCHEMA,'case':cid,'files':{k:ref(p) for k,p in paths.items()}}
            raw = derive(binding)  # Independent static bytes, every address, output and host identity.
            trace = self.root/'derived'/(cid+'.traceg'); trace.parent.mkdir(exist_ok=True)
            with trace.open('xb') as f:f.write(raw)
            bindings[cid] = write_new(self.root/'bindings'/(cid+'.json'),binding)
        self.journal.structures(bindings)
        return output, bindings

    def predict(self, capture_output, bindings):
        from arch_explorer.components.accelsim.typed_trace import create_request
        from arch_explorer.components.accelsim.microcapture import derive
        from arch_explorer.components.accelsim.backend import evaluate_request
        from arch_explorer.components.accelsim.protocol import prepare_review,preflight_protocol
        from arch_explorer.components.accelsim.config import compile_adf
        pre = capture_output/'preflight.json'
        facts, _ = json_file(pre)
        adf = native_assumption_scenario(pre)
        config = compile_adf(adf)
        adf_ref = write_new(self.root/'assumed-adf.json',adf.to_dict())
        cfg_ref = write_new(self.root/'assumed-config.json',config.artifacts)
        parameters = {'physical_parameters':None,'hardware_accuracy':'not validated','scenario':'new observed SM/resources with unchanged synthetic clock/cache/interconnect/timing assumptions','adf':adf_ref,'native_config':cfg_ref,'unknown':['physical clocks','latency and issue','cache and queue decomposition','interconnect','instruction cache']}
        predictions = {}
        for cid in self.bundle.cases:
            self.remaining()
            binding, _ = json_file(original(bindings[cid]))
            trace = self.root/'native-traces'/cid/'kernel-1.traceg'; trace.parent.mkdir(parents=True,exist_ok=False)
            with trace.open('xb') as f:f.write(derive(binding))
            request = create_request(trace,self.root/'requests'/cid,source={'kind':'hardware','gpu':facts['gpu'],'toolchain':'NVBit1.8.1 SIM30NV1 transport with unchanged SM120 original bytes','description':'Complete newly collected structure; hardware timing has not yet been observed'},trace_id=work_item(self.bundle).lower()+'-'+cid,compiler='frozen offline sm_120a cubin',engine=original(self.design['engine']),adf=adf,capture_binding=binding)
            request['options'].update(timeout_s=min(120,max(1,int(self.remaining()))),max_cycles=1_000_000,memory_limit_mb=4096)
            if self.design.get('state_protocol'):request['options']['output_dir']=str(self.root/'native-runs'/cid)
            request_file = self.root/'native-requests'/(cid+'.json'); write_new(request_file,request)
            cpu_start=time.time_ns();cpu_mono=time.monotonic_ns()
            try:result = evaluate_request(request)
            finally:
                if self.design.get('state_protocol'):
                    candidates=list((self.root/'native-runs'/cid).rglob('run.json'))
                    if len(candidates)>1:raise ValueError('extra native CPU process for one candidate')
                    run_file=candidates[0] if candidates else self.root/'native-runs'/cid/'run.json'
                    run=json_file(run_file)[0] if run_file.exists() else {}
                    write_new(self.root/'native-cpu'/(cid+'.json'),{'schema':'arch-explorer.measurement-state-native-cpu.v1','run_id':self.design['run_id'],'case':cid,'kind':'cpu','request':ref(request_file),'engine':self.design['engine'],'started_realtime_ns':cpu_start,'ended_realtime_ns':time.time_ns(),'started_monotonic_ns':cpu_mono,'ended_monotonic_ns':time.monotonic_ns(),'run':ref(run_file) if run_file.exists() else None,'pid':run.get('pid'),'exit_code':run.get('exit_code'),'status':run.get('status','not_started')})
            prediction = write_new(self.root/'predictions'/(cid+'.json'),result)
            review_dir = self.root/'reviews'/cid
            spec = prepare_review(request_file,original(prediction),original(self.design['source']),self.origin['archive'],original(self.design['history']),review_dir)
            proof = preflight_protocol(spec)
            proof_ref = write_new(review_dir/'result.json',proof)
            if proof['status'] != 'passed': raise ValueError('installed native protocol rejected: '+str(proof_ref))
            predictions[cid] = {'request':ref(request_file),'prediction':prediction,'review':ref(review_dir/'protocol.json'),'protocol':proof_ref}
        freeze = prediction_freeze(self.bundle,self.design,predictions,parameters,self.root/'freezes',stage='initial')
        self.journal.initial_predictions(freeze)
        return freeze, predictions, parameters

    def timings(self, phase, freeze):
        target = {}
        for cid in self.design['order'][phase]:
            output, receipt = self.host(phase,freeze,cid)
            rows = [attempt_reference(self.bundle,cid,output,i,receipt['path']) for i in range(3)]
            quality = self.journal.timing(cid,rows,freeze)
            target[cid] = rows
            write_new(self.root/'observations'/(cid+'.json'),{'case':cid,'phase':phase,'originals':rows,'quality':quality})
            if quality['status'] != 'passed': raise ValueError('preregistered sample quality refused; no additional observations')
        return target

    def orchestrate(self):
        """The same phase sequence for actual boundaries and CPU rehearsals."""
        capture, bindings = self.structures()
        initial, predictions, parameters = self.predict(capture,bindings)
        fit = self.timings('fit',initial)
        intervals = fit_intervals(self.bundle,self.design,fit)
        write_new(self.root/'fit-only.json',intervals)
        if intervals['status'] != 'passed': raise ValueError('fit interval consistency refused; heldout not opened')
        final = prediction_freeze(self.bundle,self.design,predictions,parameters,self.root/'freezes',stage='final',fit_result=intervals,parent=initial)
        self.journal.final_predictions(final)
        heldout = self.timings('heldout',final)
        write_new(self.root/'comparison.json',compare(self.bundle,predictions,heldout,intervals))
        self.journal.close()

    def record_reap(self, process):
        value=json_file(original(process))[0]
        write_new(self.root/(f'reap-{value["budget_account"]["process_index"]:02d}.json'),
                  {'schema':schema(self.bundle,'process-reap'),**identity(self.bundle),'run_id':self.design['run_id'],
                   'pid':value['pid'],'process':process,'exit_code':value['exit_code'],
                   'wait_completed':True,'ended_realtime_ns':time.time_ns()})

    def release(self):
        if self.lock_fd is not None:
            fcntl.flock(self.lock_fd,fcntl.LOCK_UN); os.close(self.lock_fd); self.lock_fd=None

    def replay_outcome(self):
        from arch_explorer.components.accelsim.microclosure import replay_run
        return replay_run(self.design_file,self.root)

    def execute(self):
        from arch_explorer.components.accelsim.microclosure import context,denominator,V2
        error = None
        try:
            self.acquire()
            self.orchestrate()
        except BaseException as e:
            error = type(e).__name__+': '+str(e)
            self.journal._denominator=denominator(self.bundle,self.processes)
            if self.journal.state not in ('failed','closed'): self.journal.fail(error,'controller')
        finally:
            self.release()
            reaps=[ref(p) for p in sorted(self.root.glob('reap-*.json'))]
            closed=write_new(self.root/'RESOURCE-CLOSE.json',{'schema':schema(self.bundle,'controller-close'),**identity(self.bundle),
                       'run_id':self.design['run_id'],'design_sha256':self.design['design_sha256'],
                       'collection_source':self.design['source'],'reaps':reaps,'remaining_owned_processes':[],
                       'common_lock_released':self.lock_fd is None,'all_owned_hosts_reaped':len(reaps)==len(self.processes),
                       'ended_realtime_ns':time.time_ns()})
            evidence=write_new(self.root/'EVIDENCE.json',context(self.design_file,self.root,original(closed)))
            write_new(self.root/'OUTCOMES.json',{'schema':schema(self.bundle,'outcomes') if modern(self.bundle) else V2,**identity(self.bundle),'run_id':self.design['run_id'],'status':'closed' if error is None else 'failed','error':error,'processes':self.processes,'denominator':denominator(self.bundle,self.processes),'common_lock_released':self.lock_fd is None,'all_owned_hosts_reaped':len(reaps)==len(self.processes),'phase_elapsed_s':None if self.started is None else time.monotonic()-self.started,'hardware_accuracy':'not validated','retry_allowed':False,'evidence':evidence})
        replay=self.replay_outcome()
        write_new(self.root/'REPLAY.json',replay)
        if 'native_timing_comparison' in replay:write_new(self.root/'native-timing-comparison.json',replay['native_timing_comparison'])
        return error


def compare(bundle, predictions, heldout, intervals):
    rows = []
    for cid, attempts in heldout.items():
        quality = sample_quality(bundle,attempts)
        values = [load_attempt(bundle,a)['elapsed_ns'] for a in attempts]
        result, _ = json_file(original(predictions[cid]['prediction']))
        native = result['evaluation']['latency_ns']; mean = sum(values)/len(values)
        lower,upper = (float(Fraction(x)) for x in intervals['heldout_service_intervals_ns'][cid])
        rows.append({'case':cid,'all_hardware_ns':values,'hardware_mean_ns':mean,'native_ns_under_assumed_clock':native,'absolute_percentage_error':abs(native-mean)/mean,'conditional_service_interval_ns':[lower,upper],'service_mean_inside_conditional_interval':lower<=mean<=upper,'quality':quality})
    return {'schema':'arch-explorer.sim30-comparison.v1','rows':rows,'native_mape':sum(r['absolute_percentage_error'] for r in rows)/len(rows),'native_worst':max(r['absolute_percentage_error'] for r in rows),'hardware_accuracy':'not validated','scope':'timing-heldout known structures; uncalibrated explicit assumed native profile'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command',required=True)
    check = sub.add_parser('check'); check.add_argument('registry')
    run = sub.add_parser('run'); run.add_argument('--design',required=True); run.add_argument('--handoff',required=True); run.add_argument('--output',required=True)
    args = p.parse_args(argv)
    if args.command == 'check':
        b = load_registry(args.registry); print(json.dumps({'status':'passed','registry_sha256':b.sha256,'bounds':b.bounds,'device_operations':0}))
    else:
        error = Collector(args.design,args.handoff,args.output).execute()
        print(json.dumps({'status':'failed' if error else 'closed','error':error,'output':args.output}))
        return 1 if error else 0
    return 0

if __name__ == '__main__': raise SystemExit(main())
