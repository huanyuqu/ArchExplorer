# Concrete Accel-Sim implementation; owned by the component provider.
"""Read-only reconstruction of explicit device-state campaign records."""
from collections import Counter
from pathlib import Path
from arch_explorer.components.accelsim.trace import exact,same_json
from arch_explorer.components.accelsim.microbundle import json_file
from arch_explorer.components.accelsim.microrecords import original,validate_process_context,validate_preflight,ENV_KEYS
from arch_explorer.components.accelsim.microexperiment import ref,validate_prediction_freeze
from arch_explorer.components.accelsim.state_device import validate_design,denominator,OUTCOME,PROCESS
from arch_explorer.components.accelsim.state_observation import read_observation
from arch_explorer.components.accelsim.state_resource import validate_handoff,validate_snapshot

def doc(r):return json_file(original(r))[0]
def require(x,message):
    if not x:raise ValueError('device state replay: '+message)

def _saved_process_identity(root,raw,owned=None):
    """Bind the host's saved /proc stat to its supervisor and process session."""
    path=root/'process-stat-before.txt'
    require(path.is_file() and path.stat().st_size<=65536,'missing/oversized saved process identity')
    value=path.read_text()
    try:
        pid,tail=value.split(' (',1)
        command,tail=tail.rsplit(') ',1)
        fields=tail.split()
        pid=int(pid);parent=int(fields[1]);group=int(fields[2]);session=int(fields[3]);tick=int(fields[19])
    except (ValueError,IndexError):
        raise ValueError('device state replay: malformed saved process identity') from None
    require(bool(command) and pid==raw['pid'] and parent==raw['supervisor_pid']
            and group==session==pid and tick>0,'saved process identity differs from supervised host/session')
    if owned is not None:
        require(owned.get(str(pid))==tick,'saved process start tick differs from outer supervisor')
    return tick

def _hardware_host(bundle,d,f,freeze,soft):
    raw=doc(f['raw']);lease=validate_process_context(bundle,raw);root=Path(raw['output'])
    synthetic=doc(d['source']).get('kind')=='synthetic'
    # Injected capture receipts preserve the strict host format. Authority is
    # established by the frozen source and outer synthetic marker, not a relabel.
    if not synthetic:require(raw['kind']=='hardware','actual host boundary kind differs from frozen source')
    require(f['started_realtime_ns']==raw['started_realtime_ns'] and f['ended_realtime_ns']==raw['ended_realtime_ns'],
            'process frame replaced actual clock boundaries')
    require(f['execution_status']==raw['status'] and f['reaped'] is True,'frame hid process failure/reaping')
    if raw['status']!='returned' and not (root/'preflight.json').exists():
        require(not (root/'LEDGER.json').exists(),'executor ledger lacks preflight identity')
        return None,None
    invocation=doc(ref(root/'invocation.json'));command=doc(ref(root/'command.json'))
    require(command['argv']==raw['command'],'actual host command differs')
    require(invocation['host_sha256']==raw['host']['sha256'] and invocation['pid']==raw['pid']
            and invocation['registry_sha256']==bundle.sha256 and invocation['lease_sha256']==raw['lease']['sha256']
            and invocation['freeze_sha256']==raw['freeze_sha256'],'actual host invocation identity differs')
    # Environment has an existing independent loader/maps validator via full
    # capture derivation. Timing additionally closes all recorded loader values.
    env=doc(ref(root/'environment.json'))['values']
    require(set(env)==ENV_KEYS,'missing actual loader environment')
    if f['phase']!='structure':
        require(env==raw['environment'],'timing loader variables differ')
        require(not any(env.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH','CUDA_INJECTION32_PATH','NVBIT_TOOL','NVBIT_TOOL_PATH','CUDA_PROFILE','CUDA_ENABLE_PROFILING')),'instrumented timing environment')
        for name in ('maps-before-cuInit.txt','maps-before-close.txt'):
            if (root/name).exists():
                data=(root/name).read_text().lower()
                require(not any(x in data for x in ('nvbit','tracer.so','libcupti','ncu')),'instrumentation in timing maps')
    pre=doc(ref(root/'preflight.json'));validate_preflight(bundle,pre)
    require(pre['pid']==raw['pid'],'preflight belongs to another process')
    if not synthetic:_saved_process_identity(root,raw)
    session=doc(ref(root/'STATE-SESSION-CLEANUP.json'))
    require(session['schema']=='arch-explorer.measurement-state-session-cleanup.v2'
            and session['run']==d['run_id'] and session['registry_sha256']==bundle.sha256,'session cleanup identity differs')
    require(session['context']==pre['context'] and session['module'] not in ('0','0x0','(nil)'),'session cleanup borrowed absent/foreign initialized handles')
    complete=(session['module_unload_return']==0 and session['context_destroy_return']==0
              and session['all_module_context_handles_destroyed'] is True)
    require(session['cleanup_complete']==complete,'session hides a cleanup API failure')
    for name in ('module','context'):
        if session[name] not in ('0','0x0','(nil)'):
            require(session[name+'_unload_attempted' if name=='module' else 'context_destroy_attempted'] is True,'missing acquired session-handle cleanup')
    if raw['status']=='returned':
        require(complete,'returned host hides failed session cleanup')
        closed=doc(ref(root/'closed.json'))
        quota=len(bundle.cases) if f['phase']=='structure' else 6
        require(closed['application_launches']==closed['planned_application_launches']==quota and closed['unstarted_application_launches']==0 and closed['all_module_context_handles_destroyed'] is True,'returned host omitted registered roles/context cleanup')
        if f['phase']!='structure':require(session['state_executor_entered'] is True and session['state_resources_released'] is True,'returned timing host skipped shared executor or resource cleanup')
    state=doc(ref(root/'STATE-INVOCATION.json'))
    require(state['run']==d['run_id'] and state['plan_sha256']==d['plan_sha256']
            and state['registry_sha256']==bundle.sha256 and state['freeze_sha256']==raw['freeze_sha256'],
            'actual state invocation plan/run/freeze differs')
    if f['phase']!='structure':
        require(state['case']==f['case'] and state['planned_application_launches']==6,'state host did not bind every role')
        require(state['start_ns']==raw['started_realtime_ns'] and state['deadline_ns']==min(soft,lease['deadline_realtime_ns'],state['start_ns']+120*10**9),'host deadline/phase not frozen before initialization')
    return pre,complete

def _cpu_host(d,f):
    raw=doc(f['raw']);exact(raw,{'schema','kind','no_resource_authority','run_id','case','argv','executable','pid','start_tick',
        'wait_completed','exit_code','stdout','stderr','ledger','clock_injected','started_realtime_ns','ended_realtime_ns',
        'observed_started_realtime_ns','observed_ended_realtime_ns'},'actual injected CPU process')
    require(raw['schema']=='arch-explorer.measurement-state-device-cpu-process.v1' and raw['kind']=='synthetic'
            and raw['no_resource_authority'] is True and raw['clock_injected'] is True and raw['wait_completed'] is True,
            'CPU boundary lost explicit injection/reaping identity')
    require(raw['run_id']==d['run_id'] and raw['case']==f['case'] and raw['ledger']==f['ledger']
            and raw['executable']==d['state_protocol']['cpu']['executable'],'CPU executable/run/case substituted')
    require(raw['started_realtime_ns']==f['started_realtime_ns'] and raw['ended_realtime_ns']==f['ended_realtime_ns'],'injected clock frame differs')
    original(raw['stdout']);original(raw['stderr'])
    root=original(f['ledger']).parent;cmd=doc(ref(root/'COMMAND.json'));marker=doc(ref(root/'SYNTHETIC.json'))
    require(cmd['argv']==raw['argv'] and marker['kind']=='synthetic','CPU original command/marker differs')
    ledger=doc(f['ledger'])
    expected=('timeout' if ledger['ended_ns']>ledger['deadline_ns'] else
              'returned' if raw['exit_code']==0 and ledger['status']=='returned' else 'failed')
    require(f['execution_status']==expected and f['reaped'] is True,'CPU frame hides actual executor/timeout terminal')
    return raw

def replay_run(design_file,window,*,reader_source=None,evidence=None):
    b,d=validate_design(design_file);root=Path(window).resolve();out=json_file(root/'OUTCOMES.json')[0]
    exact(out,{'schema','kind','no_resource_authority','run_id','design','handoff','lease','status','error','bindings','initial',
        'fit_only','final','frames','observations','journal','resource_close','denominator','parameters','physical_parameters',
        'hardware_accuracy','retry_allowed'},'device state outcome')
    synthetic=out['kind']=='synthetic';require(out['kind'] in ('hardware','synthetic'),'explicit capability kind required')
    require(out['schema']==OUTCOME and out['run_id']==d['run_id'] and out['design']==ref(design_file)
            and out['no_resource_authority'] is synthetic and out['physical_parameters'] is None
            and out['hardware_accuracy']=='not validated' and out['retry_allowed'] is False,'outcome scope changed')
    marker=root/'SYNTHETIC.json';require(marker.exists()==synthetic,'CPU fixture cannot be relabelled as hardware')
    if synthetic:
        m=json_file(marker)[0];require(m['kind']=='synthetic' and m['no_resource_authority'] is True
             and m['run_id']==d['run_id'] and m['design']==ref(design_file),'fixture boundary/run lost')
    else:
        require(doc(d['source']).get('kind')!='synthetic','synthetic source cannot become device authority')
    if reader_source:
        from arch_explorer.components.accelsim.microcampaign import check_source_installation
        check_source_installation(reader_source)
    handoff=doc(out['handoff'])
    from arch_explorer.components.accelsim.state_idle import REQUEST,ADMISSION,validate_request,validate_observation
    idle_query=None
    if handoff.get('schema')==REQUEST:
        require(not synthetic and out['status']=='failed' and out['lease'] is None
                and not any(out[k] for k in ('frames','bindings','initial','final','fit_only','observations')),
                'pending/failed idle request cannot authorize application work')
        validate_request(b,d,handoff)
        query_path=root/'IDLE-OBSERVATION.json'
        if query_path.exists():
            idle_query=validate_observation(handoff,doc(ref(query_path)))
            require(idle_query['status']!='idle','successful idle observation lost its admission/lease')
    else:
        validate_handoff(b,d,handoff,synthetic=synthetic)
        if handoff.get('schema')==ADMISSION:
            require(out['handoff']==ref(root/'IDLE-ADMISSION.json')
                    and handoff['observation']==ref(root/'IDLE-OBSERVATION.json'),'foreign idle window originals')
            idle_query=doc(handoff['observation'])
    soft=handoff['absolute_deadline_realtime_ns']-60*10**9
    view={**d,'absolute_deadline_realtime_ns':handoff['absolute_deadline_realtime_ns'],'close_reserve_s':60}
    frames=[doc(r) for r in out['frames']];order=['structure']+d['order']['fit']+d['order']['heldout']
    require(len(frames)<=len(order),'extra host budget')
    for i,f in enumerate(frames):
        exact(f,{'schema','kind','run_id','index','case','phase','raw','ledger','before','after','started_realtime_ns',
            'ended_realtime_ns','reaped','execution_status','error'},'device host frame')
        require(f['schema']==PROCESS and f['kind']==out['kind'] and f['run_id']==d['run_id'] and f['index']==i
                and order[i]==('structure' if f['case'] is None else f['case']),'reordered/foreign/duplicate host')
        require(handoff['issued_realtime_ns']<=f['started_realtime_ns']<=f['ended_realtime_ns']<handoff['absolute_deadline_realtime_ns']
                and f['started_realtime_ns']<soft and f['ended_realtime_ns']-f['started_realtime_ns']<=120*10**9+10**9,
                'host outside absolute/fixed budget')
    capture=frames[0] if frames else None
    if capture and idle_query:
        require(idle_query['status']=='idle' and idle_query['ended_realtime_ns']<capture['started_realtime_ns'],
                'application work before known-idle observation')
    if capture:require(capture['phase']=='structure' and capture['ledger'] is None,'capture is not a timing role')
    from arch_explorer.components.accelsim.microcapture import derive
    for cid,r in out['bindings'].items():
        require(cid in b.cases and capture is not None,'unknown/unattempted structure')
        binding=doc(r);require(binding['files']['supervisor']==capture['raw'],'structure borrowed foreign process')
        derive(binding)
    if out['initial'] is not None:
        require(set(out['bindings'])==set(b.cases),'predictions before complete structures')
        initial=validate_prediction_freeze(b,d,out['initial'],'initial')
        require(capture['ended_realtime_ns']<initial['created_realtime_ns']<soft,'initial native freeze precedes structures or misses cutoff')
        for cid,row in initial['native_predictions'].items():
            request=doc(row['request'])
            # Native requests must retain this run's exact complete raw binding.
            params=request['workload']['params']['accelsim']
            manifest=doc({'path':params['manifest'],'sha256':params['sha256']})
            if not synthetic:require(same_json(manifest['capture_binding'],doc(out['bindings'][cid])),'native request borrowed another capture')
    else:initial=None;require(len(frames)<=1,'observation before initial predictions')
    native_cpu=[]
    if not synthetic:
        for path in sorted((root/'native-cpu').glob('*.json')):
            row=json_file(path)[0];cid=row['case']
            require(cid in b.cases and row['run_id']==d['run_id'] and row['kind']=='cpu' and row['engine']==d['engine'], 'foreign native CPU process')
            require(handoff['issued_realtime_ns']<=row['started_realtime_ns']<=row['ended_realtime_ns']<handoff['absolute_deadline_realtime_ns']
                    and 0<=row['ended_monotonic_ns']-row['started_monotonic_ns']<=121*10**9,'native CPU work outside frozen total/host bound')
            if row['run'] is not None:
                run=doc(row['run']);require(row['pid']==run.get('pid') and row['exit_code']==run.get('exit_code') and row['status']==run['status'],'native PID/exit record differs')
            if initial:
                require(row['request']==initial['native_predictions'][cid]['request'] and row['status']=='complete' and row['exit_code']==0
                    and row['ended_realtime_ns']<initial['created_realtime_ns'],'all initial native runs must finish before first observation')
            native_cpu.append(ref(path))
        require(len(native_cpu)<=len(b.cases),'extra native process budget')
        if initial:require({doc(r)['case'] for r in native_cpu}==set(b.cases),'missing actual native CPU process denominator')
    observations=[doc(r) for r in out['observations']];require(len(observations)==max(0,len(frames)-1),'lost process observation')
    passed=True;resources={};qualities={};last=initial['created_realtime_ns'] if initial else d['created_realtime_ns'];snapshot_count=0
    pre_capture=None
    if capture:
        raw=doc(capture['raw']);prepath=Path(raw['output'])/'preflight.json'
        pre_capture=doc(ref(prepath)) if prepath.exists() else None
        if handoff.get('schema')==ADMISSION and pre_capture:
            require(pre_capture['uuid_hex']==handoff['gpu_uuid'][4:].replace('-',''),
                    'live CUDA device differs from idle-admitted UUID')
        _hardware_host(b,d,capture,None,soft)
    for f,r in zip(frames[1:],observations):
        require(passed,'normal execution after first failure')
        cid=f['case'];freeze=out['initial'] if b.cases[cid]['partition']=='fit' else out['final']
        require(freeze is not None and r['case']==cid and r['partition']==b.cases[cid]['partition']
                and r['frame']==out['frames'][f['index']] and r['freeze']==freeze,'role phase/freeze changed')
        if not synthetic:_hardware_host(b,d,f,freeze,soft)
        raw=_cpu_host(d,f) if synthetic else doc(f['raw'])
        result=read_observation(b,cid,f['ledger'],d['run_id'],freeze['sha256']) if f['ledger'] else None
        require(result is None or result['ledger']['kind']==out['kind'],'CPU transcript promoted to hardware')
        passed=bool(result and result['passed'] and f['execution_status']=='returned')
        require(r['quality']==(None if result is None else result['quality']) and r['resources']==(None if result is None else result['resources'])
                and r['passed']==passed,'observation hid C++ quality/resource failure')
        if result:
            ledger=result['ledger']
            hostroot=original(f['ledger']).parent
            if synthetic:
                require(set(hostroot.iterdir())==result['files']|{hostroot/'COMMAND.json',hostroot/'SYNTHETIC.json'},'unlisted CPU execution original')
            else:
                mandatory={'invocation.json','STATE-INVOCATION.json','command.json','environment.json','preflight.json','maps-before-cuInit.txt','process-stat-before.txt','STATE-SESSION-CLEANUP.json'}
                optional={'closed.json','failure.json','maps-before-close.txt'}
                require(set(hostroot.iterdir())-result['files'] <= {hostroot/n for n in mandatory|optional} and {hostroot/n for n in mandatory}<=set(hostroot.iterdir()),'unlisted/missing hardware state original')
                require(ledger['context']==doc(ref(hostroot/'preflight.json'))['context'],'ledger context differs from live preflight')
            require(ledger['start_ns']==f['started_realtime_ns'] and ledger['ended_ns']<=f['ended_realtime_ns'], 'actual C++ time outside process')
            require((raw['exit_code']==0)==result['passed'],'exit code hides executor/quality failure')
            args=raw['argv'] if synthetic else raw['command'];opts=dict(zip(args[1::2],args[2::2]))
            require(opts.get('--run')==d['run_id'] and opts.get('--case')==cid and opts.get('--freeze')==freeze['sha256']
                    and int(opts['--start-ns'])==ledger['start_ns'] and int(opts['--deadline-ns'])==ledger['deadline_ns'],'C++ invocation phase/run/deadline differs')
            resources[cid]=result['resources'];qualities[cid]=result['quality']
        require(last<f['started_realtime_ns'],'observation before previous freeze/stage')
        last=f['ended_realtime_ns'];scope={'run_id':d['run_id'],'case':cid,'started_realtime_ns':f['started_realtime_ns'],'ended_realtime_ns':f['ended_realtime_ns']}
        require(f['before'] is not None,'missing preregistered pre-diagnostic state observation')
        for position in ('before','after'):
            snap=f[position]
            if snap is None:
                require(not passed or f['ended_realtime_ns']>=soft-6*10**9 and f['error'] and 'margin' in f['error'],'successful host omitted available after snapshot');continue
            require(pre_capture is not None,'snapshot has no live UUID reference')
            value=doc(snap)
            require(value['started_realtime_ns']>doc(freeze)['created_realtime_ns'],'state evidence observed before relevant prediction freeze')
            validate_snapshot(value,view,cid,position,pre_capture['uuid_hex'],scope,synthetic=synthetic);snapshot_count+=1
    expected_snapshots={original(f[key]).resolve() for f in frames[1:] for key in ('before','after') if f[key] is not None}
    actual_snapshots=set((root/'state').glob('*.json'))
    require(expected_snapshots<=actual_snapshots and len(actual_snapshots)<=2*len(b.cases),'hidden/missing/excess state query originals')
    orphan_queries=actual_snapshots-expected_snapshots
    require(len(orphan_queries)<=1 and (not orphan_queries or out['status']=='failed' and len(frames)<len(order)),'state query outside failed next-host boundary')
    for path in orphan_queries:
        cid=order[len(frames)];row=json_file(path)[0]
        require(path.name==cid+'-before.json' and row['case']==cid and row['position']=='before','unregistered query position/case')
        freeze=out['initial'] if b.cases[cid]['partition']=='fit' else out['final']
        require(freeze is not None and row['started_realtime_ns']>doc(freeze)['created_realtime_ns'] and pre_capture is not None,'orphan query before its frozen phase')
        try:validate_snapshot(row,view,cid,'before',pre_capture['uuid_hex'],synthetic=synthetic)
        except ValueError as error:
            require(out['error']=='ValueError: '+str(error),'query refusal did not stop the original controller')
    fitids=d['order']['fit'];heldids=d['order']['heldout']
    if out['fit_only'] is not None:
        require(set(fitids)<=set(qualities) and all(qualities[c] and qualities[c]['status']=='passed' for c in fitids),'final freeze before all fit-only formal quality')
        fit=doc(out['fit_only']);require(fit=={'schema':'arch-explorer.measurement-state-fit-only.v2','run_id':d['run_id'],'cases':fitids,
            'quality':{c:qualities[c] for c in fitids},'parameters_changed':False,'physical_parameters':None,'state_compatibility':False,'status':'passed'},'fit consumed heldout or changed assumptions')
    if out['final'] is not None:
        require(out['fit_only'] is not None,'final freeze lacks fit-only record')
        final=validate_prediction_freeze(b,d,out['final'],'final',fit_result=doc(out['fit_only']),parent=out['initial'])
        lastfit=max(f['ended_realtime_ns'] for f in frames[1:] if f['case'] in fitids)
        require(lastfit<final['created_realtime_ns']<soft,'final freeze outside fit/cutoff')
        for f in frames[1:]:
            if f['case'] in heldids:require(final['created_realtime_ns']<f['started_realtime_ns'],'heldout diagnostic/preparation leaked before final')
    else:require(not any(f['case'] in heldids for f in frames[1:]),'heldout observations without final freeze')
    require(out['denominator']==denominator(b,frames,out['bindings']),'all-role denominator differs from actual C++ attempts')
    terminal='closed' if passed and len(frames)==len(order) else 'failed'
    require(out['status']==terminal and (out['error'] is None)==(terminal=='closed'),'terminal hides failed or missing roles')
    previous=None;stages=[];payload=[];stamp=d['created_realtime_ns']
    for i,r in enumerate(out['journal']):
        e=doc(r);require(e['index']==i and e['previous']==previous and e['run_id']==d['run_id'] and e['design']==ref(design_file)
              and e['kind']==out['kind'] and e['realtime_ns']>=stamp,'journal scope/order/time changed')
        previous=r;stamp=e['realtime_ns'];stages.append(e['stage']);payload.append(e['payload'])
    expected=[]
    if out['lease'] is not None:expected.append('acquired')
    if out['bindings']:expected.append('structure')
    if initial:expected.append('initial')
    if out['final'] is None:expected+=['observation']*len(observations)+(['fit-only'] if out['fit_only'] else [])
    else:expected+=['observation']*len(fitids)+['fit-only','final']+['observation']*(len(observations)-len(fitids))
    expected.append(terminal)
    expected_payload=[]
    if out['lease'] is not None:expected_payload.append({'handoff':out['handoff'],'lease':out['lease']})
    if out['bindings']:expected_payload.append({'bindings':out['bindings']})
    if initial:expected_payload.append({'freeze':out['initial']})
    obs_payload=[{'record':r} for r in out['observations']]
    if out['final'] is None:expected_payload+=obs_payload+([{'fit':out['fit_only']}] if out['fit_only'] else [])
    else:expected_payload+=obs_payload[:len(fitids)]+[{'fit':out['fit_only']},{'freeze':out['final']}]+obs_payload[len(fitids):]
    expected_payload.append({'error':out['error']})
    require(payload==expected_payload,'journal payload differs from actual phase evidence')
    require(stages==expected and payload[-1]=={'error':out['error']},'stage closure differs')
    if out['lease'] is not None:require(payload[0]=={'handoff':out['handoff'],'lease':out['lease']},'acquisition references changed')
    close=doc(out['resource_close']);require(close['run_id']==d['run_id'] and close['kind']==out['kind']
        and close['handoff']==out['handoff'] and stamp<=close['ended_realtime_ns']<handoff['absolute_deadline_realtime_ns'], 'resource close scope/time changed')
    require(close['lease']==out['lease'],'resource close replaced lease')
    require(close['all_owned_hosts_reaped'] is True,'incomplete own process reaping')
    require(close['common_lock_released'] is (None if synthetic else True),'actual versus simulated lock closure differs')
    require(close['raw_processes']==[f['raw'] for f in frames],'resource closure lost an attempted process')
    if synthetic:
        require(close['real_locks']==close['real_leases']==0,'CPU rehearsal claims actual authority')
        cpu=[doc(f['raw']) for f in frames[1:]]
        expected_reaps=[{'pid':x['pid'],'start_tick':x['start_tick'],'wait_completed':True,'exit_code':x['exit_code'],'raw':f['raw']} for f,x in zip(frames[1:],cpu)]
        require(close['cpu_reaps']==expected_reaps,'actual CPU child wait/identity records differ')
    else:
        require(not close['owned_process_cleanup']['remaining'],'registered/native descendant cleanup incomplete')
        require(len(close['reaps'])==len(frames),'missing per-host wait evidence')
        for f,r in zip(frames,close['reaps']):
            reap=doc(r);raw=doc(f['raw']);require(reap['process']==f['raw'] and reap['pid']==raw['pid'] and reap['wait_completed'] is True
                and reap['exit_code']==raw['exit_code'],'wait evidence contradicts actual process')
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and p.name not in ('REPLAY.json','FINAL-REPLAY.json'))
    require(total<=d['budget']['host_raw_bytes']+d['budget']['capture_bytes'],'complete retained original budget exceeded')
    external=None
    if evidence is not None:
        # A direct-use run may lose a nonblocking lock to another user, or a
        # different user may take it after our release. Preserve that global
        # observation; only our own processes and lock must be released.
        external=doc(evidence);require(external['run_id']==d['run_id'] and not external['remaining_owned_processes']
            and external['all_owned_processes_reaped'] is True and external['closed_before_absolute_deadline'] is True
            and external['absolute_deadline_realtime_ns']==handoff['absolute_deadline_realtime_ns']
            and (type(external['common_lock_rechecked_free']) is bool if handoff.get('schema') in (REQUEST,ADMISSION)
                 else external['common_lock_rechecked_free'] is True),'outer absolute supervisor did not establish resource closure')
        if not synthetic:
            for f in frames:
                raw=doc(f['raw']);hostroot=Path(raw['output'])
                if (hostroot/'preflight.json').exists():_saved_process_identity(hostroot,raw,external['owned_processes'])
    return {'schema':'arch-explorer.measurement-state-device-replay.v1','kind':'synthetic' if synthetic else 'hardware-record-replay',
        'replay_status':'replayed','collection_outcome':out['status'],'collection_error':out['error'],'run_id':d['run_id'],
        'denominator':out['denominator'],'role_counts':{r:dict(Counter(x['status'] for x in out['denominator'] if x['role']==r)) for r in ('structure','diagnostic','preparation','formal')},
        'native_cpu_processes':native_cpu,'verified_structure_count':len(out['bindings']),'formal_quality':qualities,'resource_lifecycles':resources,'snapshot_count':len(actual_snapshots),'unbound_failed_query_records':[ref(p) for p in sorted(orphan_queries)],'stages':stages,
        'resource_close':out['resource_close'],'outer_supervisor':evidence,'source_epochs':{'collection':d['source'],'reader':reader_source},
        'state_compatibility':False,'native_state':d['state_protocol']['native_state'],'observed_state':'fixed preparation; unknown cache/clock; snapshots instantaneous',
        'physical_mape':None,'physical_worst':None,'physical_parameters':None,'hardware_accuracy':'not validated','total_original_bytes':total}
