# Concrete Accel-Sim implementation; owned by the component provider.
"""Offline closure of a finite campaign, shared by its producer and reader.

An outcome is a claim, not an observation. This module reconciles its complete
denominator with the explicitly scoped originals and replays Journal's actual
transitions. Read success never changes the collection's terminal state.
Historical v1 outcomes require an explicit, hash-bound evidence context; v2
producers embed that same context. No current resource permission is inferred.
"""
from collections import Counter
from pathlib import Path
import re

from arch_explorer.components.accelsim.trace import exact, same_json
from arch_explorer.components.accelsim.microbundle import load_registry, json_file
from arch_explorer.components.accelsim.microrecords import (original, case_files, validate_process, validate_sample,
                           validate_case_attempt, load_attempt, attempt_reference)
from arch_explorer.components.accelsim.microexperiment import (ref, write_new, validate_design, Journal,
                              fit_intervals, sample_quality)
from arch_explorer.components.accelsim.microcontract import (budget,modern,identity,identity_fields,check_identity,schema as record_schema)

CONTEXT = 'arch-explorer.sim30-closure-context.v1'
V1 = 'arch-explorer.sim30-outcomes.v1'
V2 = 'arch-explorer.sim30-outcomes.v2'
MEASUREMENT_V2='arch-explorer.measurement-outcomes.v2'
OUTCOME_FIELDS = {'schema','run_id','status','error','processes','denominator',
                  'common_lock_released','all_owned_hosts_reaped','phase_elapsed_s',
                  'hardware_accuracy','retry_allowed'}
ROW_FIELDS = {'case','mode','sample','status','started','original'}


def document(reference):
    return json_file(original(reference))[0]


def scope_files(roots, subdir, pattern):
    """Only declared campaign roots, never an ancestor-wide recursive search."""
    return sorted({p.resolve() for root in roots for p in (Path(root)/subdir).glob(pattern)})


def context(design_file, root, resource_close, *, source_epochs=None, recovery=None):
    root = Path(root).resolve()
    design = json_file(design_file)[0]
    bundle=load_registry(original(design['registry']))
    return {'schema':record_schema(bundle,'closure-context'),**identity(bundle),'design':ref(design_file),'collection_source':design['source'],
            'raw_roots':[str(root)],
            'journal':[ref(p) for p in scope_files([root],'journal','*.json')],
            'observations':[ref(p) for p in scope_files([root],'observations','*.json')],
            'resource_close':ref(resource_close),
            'source_epochs':None if source_epochs is None else ref(source_epochs),
            'recovery':recovery or {}}


def denominator(bundle, processes):
    """Build producer rows from owned process directories, including failed slots."""
    slots = {}
    for reference in processes:
        process = document(reference)
        mode = 'structure' if process['phase']=='structure' else 'timing'
        output = Path(process['output'])
        for cid in process['budget_account']['case_ids']:
            for i in range(1 if mode=='structure' else 3):
                start = output/cid/f'sample-{i}.started.json'
                raw = output/cid/f'sample-{i}.json'
                if not start.exists() and not raw.exists():continue
                key = (cid,mode,i)
                if key in slots:raise ValueError('duplicate owned attempt; no retry')
                slots[key] = {'started':ref(start) if start.exists() else None,
                              'original':ref(raw) if raw.exists() else None,
                              'status':json_file(raw)[0]['status'] if raw.exists() else 'attempted'}
    return [dict(case=cid,mode=mode,sample=i,**slots.get((cid,mode,i),
                 {'status':'not_started','started':None,'original':None}))
            for cid in bundle.cases for mode,count in (('structure',1),('timing',3)) for i in range(count)]


def _resource(bundle,design, outcomes, processes, evidence):
    close = document(evidence)
    end = max([p['ended_realtime_ns'] for p in processes] or [design['created_realtime_ns']])
    if outcomes['common_lock_released'] is not True or outcomes['all_owned_hosts_reaped'] is not True:
        raise ValueError('collection did not close its owned resources')
    if not modern(bundle) and close.get('schema')=='arch-explorer.sim30-absolute-supervisor-close.v1':
        # Bounded v1 compatibility: actual supervisor's complete PID/start-tick
        # ledger, reaping, lock recheck and absolute deadline, not two booleans.
        exact(close,{'schema','exit_code','signal_soft_stop_sent','forced_cleanup','owned_processes',
                     'remaining_owned_processes','all_owned_processes_reaped','common_lock_rechecked_free',
                     'ended_realtime_ns','hard_deadline_realtime_ns','closed_before_absolute_deadline'},'absolute resource close')
        owned=close['owned_processes']
        if (type(owned) is not dict or any(type(v) is not int or v<=0 for v in owned.values())
            or any(str(p['pid']) not in owned for p in processes)
            or close['remaining_owned_processes']!={} or close['all_owned_processes_reaped'] is not True
            or close['common_lock_rechecked_free'] is not True
            or close['ended_realtime_ns']<end
            or close['closed_before_absolute_deadline'] is not (close['ended_realtime_ns']<close['hard_deadline_realtime_ns'])
            or (close['exit_code']==0)!=(outcomes['status']=='closed')):
            raise ValueError('outcome contradicts supervisor reaping/deadline/exit evidence')
    elif close.get('schema')==record_schema(bundle,'controller-close'):
        exact(close,identity_fields(bundle,{'schema','run_id','design_sha256','collection_source','reaps','remaining_owned_processes',
                     'common_lock_released','all_owned_hosts_reaped','ended_realtime_ns'}),'controller resource close')
        check_identity(bundle,close,'controller-close')
        if (close['run_id']!=design['run_id'] or close['design_sha256']!=design['design_sha256']
            or close['collection_source']!=design['source'] or close['remaining_owned_processes']!=[]
            or close['common_lock_released'] is not True or close['all_owned_hosts_reaped'] is not True
            or close['ended_realtime_ns']<end or len(close['reaps'])!=len(processes)):
            raise ValueError('controller resource close identity/remaining processes differs')
        for r,p in zip(close['reaps'],processes):
            reap=document(r)
            exact(reap,identity_fields(bundle,{'schema','run_id','pid','process','exit_code','wait_completed','ended_realtime_ns'}),'owned process reap')
            check_identity(bundle,reap,'process-reap')
            if (reap['run_id']!=design['run_id']
                or document(reap['process'])!=p or reap['pid']!=p['pid'] or reap['exit_code']!=p['exit_code']
                or reap['wait_completed'] is not True or not p['ended_realtime_ns']<=reap['ended_realtime_ns']<=close['ended_realtime_ns']):
                raise ValueError('close lacks matching actual process wait evidence')
    else:raise ValueError('unknown resource closure evidence version')
    return {'evidence':evidence,'consistent':True,'current_resource_authority':False}


def _epochs(design, ctx):
    source=document(design['source'])
    result={'collection_source':design['source'],'collection_commit':source.get('source_sha'),
            'reader_source':None,'postclosure_is_prediction':False}
    if ctx['collection_source']!=design['source']:raise ValueError('reader source substituted for collection source')
    if ctx['source_epochs'] is not None:
        epochs=document(ctx['source_epochs'])
        exact(epochs,{'before_GPU','after_closed_window','original_freezes_preserved'},'source epochs')
        before,after=epochs['before_GPU'],epochs['after_closed_window']
        if (before['source']!=design['source'] or before['commit']!=source.get('source_sha')
            or document(before['design'])!=design or epochs['original_freezes_preserved'] is not True
            or after['commit']!=document(after['source']).get('source_sha')):
            raise ValueError('collection/reader source era falsely rebound')
        result['postclosure_source']=after['source'];result['postclosure_commit']=after['commit']
    return result


def _processes(bundle, design, outcomes, roots):
    processes=[];paths=set();outputs=set();pids=set()
    for index,row in enumerate(outcomes['processes']):
        path=original(row);p=document(row)
        if path in paths or p['output'] in outputs or p['pid'] in pids:raise ValueError('duplicate process/output identity')
        paths.add(path);outputs.add(p['output']);pids.add(p['pid'])
        if (p['design_sha256']!=design['design_sha256'] or document(p['design'])!=design
            or p['registry_sha256']!=bundle.sha256 or p['budget_account']['process_index']!=index
            or Path(p['output']).resolve().parent not in roots):
            raise ValueError('process does not belong to this design/run/registered scope/order')
        if p['status']=='returned':validate_process(bundle,p)
        else:
            from arch_explorer.components.accelsim.microrecords import validate_process_context
            validate_process_context(bundle,p)
            if p['status'] not in ('failed','timeout') or type(p['exit_code']) is not int:
                raise ValueError('unknown/incomplete terminal process state')
        if processes and (p['started_realtime_ns']<processes[-1]['ended_realtime_ns'] or processes[-1]['status']!='returned'):
            raise ValueError('overlapping hosts or continued execution after failed process')
        processes.append(p)
    return processes


def _rows(bundle, design, outcomes, processes, roots):
    expected={(cid,m,i) for cid in bundle.cases for m,n in (('structure',1),('timing',3)) for i in range(n)}
    indexed={};verified={};actual={}
    if type(outcomes['denominator']) is not list:raise ValueError('complete outcome denominator must be a list')
    for row in outcomes['denominator']:
        exact(row,ROW_FIELDS,'outcome denominator row')
        if type(row['sample']) is not int:raise ValueError('sample index is not an integer')
        key=(row['case'],row['mode'],row['sample'])
        if key not in expected or key in indexed:raise ValueError('missing/duplicate/foreign denominator row')
        if row['status'] not in ('not_started','attempted','returned','failed'):raise ValueError('unknown attempt terminal state')
        indexed[key]=row
    if set(indexed)!=expected:raise ValueError('incomplete registered denominator')
    for process,pref in zip(processes,outcomes['processes']):
        output=Path(process['output']);mode='structure' if process['phase']=='structure' else 'timing'
        for cid in bundle.cases:
            d=output/cid
            for file in sorted(d.glob('sample-*.json')):
                match=re.fullmatch(r'sample-(\d+)\.(started\.)?json',file.name)
                if match is None:continue
                i=int(match[1]);key=(cid,mode,i)
                if key not in expected or cid not in process['budget_account']['case_ids']:
                    raise ValueError('extra/unregistered raw attempt')
                slot=actual.setdefault(key,{'process':process,'receipt':pref})
                if slot['receipt']!=pref:raise ValueError('same slot attempted by multiple processes')
                role='started' if match[2] else 'original'
                if role in slot:raise ValueError('duplicate actual slot record')
                slot[role]=ref(file)
    for key,row in indexed.items():
        found=actual.get(key,{})
        if row['started']!=found.get('started') or row['original']!=found.get('original'):
            raise ValueError('denominator is not closed to actual started/return originals')
        if row['status']=='not_started':
            if found:raise ValueError('not_started has actual observation/attempt')
            continue
        if not row['started']:raise ValueError('claimed return/attempt has no launch-start original')
        started=document(row['started']);p=found['process'];cid,mode,i=key
        if (started.get('schema')!='arch-explorer.sim30-started.v1' or started.get('case')!=cid
            or started.get('sample_index')!=i or started.get('pid')!=p['pid']
            or started.get('freeze_sha256')!=p['freeze_sha256'] or started.get('registry_sha256')!=bundle.sha256
            or started.get('timed') is not (mode=='timing')
            or not p['started_realtime_ns']<=started.get('host_realtime_ns',0)<=p['ended_realtime_ns']):
            raise ValueError('started original case/run/process/phase differs')
        if row['status']=='attempted':
            if row['original'] is not None:raise ValueError('attempted slot already has a return record')
            continue
        if row['original'] is None:raise ValueError('terminal attempt missing return original')
        raw=document(row['original'])
        if raw.get('status')!=row['status'] or raw.get('case')!=cid or raw.get('mode')!=mode or raw.get('sample_index')!=i or raw.get('pid')!=p['pid'] or raw.get('freeze_sha256')!=p['freeze_sha256']:
            raise ValueError('outcome status differs from original actual return identity')
        if row['status']=='returned':
            validate_sample(bundle,bundle.cases[cid],raw,mode)
            if p['status']=='returned':
                files=case_files(p['output'],cid,i,found['receipt']['path'])
                verified[key]=validate_case_attempt(bundle,bundle.cases[cid],files,mode)
    # An unlisted owned host/observation in either the supplied view or the
    # declared original root must not disappear just because no row refers to it.
    declared_outputs={Path(p['output']).resolve() for p in processes}
    for root in roots:
        for directory in root.glob('host-*'):
            if directory.is_dir() and directory.resolve() not in declared_outputs and list(directory.glob('*/sample-*.json')):
                raise ValueError('orphan owned host originals outside process denominator')
    return indexed,verified


def _observations(bundle, outcomes, ctx, roots, indexed, verified):
    paths=[original(r) for r in ctx['observations']]
    if len(set(paths))!=len(paths) or set(paths)!=set(scope_files(roots,'observations','*.json')):
        raise ValueError('orphan/missing observation in declared campaign scope')
    observations={}
    for reference in ctx['observations']:
        obs=document(reference);exact(obs,{'case','phase','originals','quality'},'phase observation')
        cid=obs['case']
        if cid not in bundle.cases or cid in observations or obs['phase']!=bundle.cases[cid]['partition']:
            raise ValueError('duplicate/foreign/heldout-contaminated phase observation')
        if len(obs['originals'])!=3:raise ValueError('three original observations required')
        seen=set()
        for row in obs['originals']:
            a=load_attempt(bundle,row);sample=a['sample'];key=(cid,'timing',sample['sample_index'])
            if row['case']!=cid or key in seen or key not in verified or indexed[key]['status']!='returned':
                raise ValueError('orphan observation or unreturned denominator')
            seen.add(key)
            if row['files']['sample']!=indexed[key]['original'] or row['files']['started']!=indexed[key]['started']:
                raise ValueError('observation is not the actual denominator original')
        if obs['quality']!=sample_quality(bundle,obs['originals']):raise ValueError('observation quality differs from raw samples')
        observations[cid]=obs
    for cid in bundle.cases:
        complete=all((cid,'timing',i) in verified for i in range(3))
        if complete and cid not in observations:raise ValueError('returned timing process has no closed phase observation')
    return observations


def _journal(bundle, design, ctx, roots, observations, rows, processes):
    refs=ctx['journal'];paths=[original(r) for r in refs]
    available=scope_files(roots,'journal','*.json')
    if len(set(paths))!=len(paths) or set(paths)!=set(available) or not refs:raise ValueError('missing/extra journal original')
    events=[];previous=None;stamp=design['created_realtime_ns']
    for index,row in enumerate(refs):
        e=document(row)
        exact(e,identity_fields(bundle,{'schema','run_id','index','previous','state','realtime_ns','payload'}),'journal event')
        check_identity(bundle,e,'journal-event')
        if (e['run_id']!=design['run_id']
            or type(e['index']) is not int or e['index']!=index or e['previous']!=previous
            or type(e['realtime_ns']) is not int or e['realtime_ns']<stamp):
            raise ValueError('journal chain/run/time changed')
        events.append((row,e));previous=row;stamp=e['realtime_ns']
    journal=Journal(roots[0]/'journal',bundle,design,_replay_events=events,_denominator=list(rows.values()))
    consumed=set()
    for row,e in events[1:]:
        state,payload=e['state'],e['payload']
        prior_stamp=events[e['index']-1][1]['realtime_ns']
        if state=='structures':
            journal.structures(payload)
            for r in payload.values():
                binding=document(r);p=document(binding['files']['supervisor'])
                if p not in processes or not prior_stamp<=p['started_realtime_ns']<=p['ended_realtime_ns']<=e['realtime_ns']:raise ValueError('structure journal lacks earlier actual process')
        elif state=='initial_predictions':
            journal.initial_predictions(payload)
            if not prior_stamp<=document(payload)['created_realtime_ns']<=e['realtime_ns']:raise ValueError('initial prediction freeze is not between structure and timing')
        elif state in ('fit','heldout'):
            cid=payload.get('case');obs=observations.get(cid)
            if obs is None or obs['phase']!=state:raise ValueError('journal timing has no matching actual observation')
            p=load_attempt(bundle,obs['originals'][0])['process']
            if not prior_stamp<=p['started_realtime_ns']<=p['ended_realtime_ns']<=e['realtime_ns']:raise ValueError('journal timing happened before its phase or after acceptance')
            journal.timing(cid,obs['originals'],journal.initial if state=='fit' else journal.final);consumed.add(cid)
        elif state=='final_predictions':
            journal.final_predictions(payload)
            if not prior_stamp<=document(payload)['created_realtime_ns']<=e['realtime_ns']:raise ValueError('final prediction freeze is not between fit and heldout')
        elif state=='failed':
            # A quality refusal emits its own failure transition from timing().
            if journal.state!='failed':journal.fail(payload.get('reason'),payload.get('at'))
        elif state=='closed':journal.close()
        else:raise ValueError('unknown/out-of-order journal phase')
    if len(journal.files)!=len(events) or journal.state not in ('failed','closed'):
        raise ValueError('journal has no genuine terminal transition')
    if consumed!=set(observations):raise ValueError('observation outside actual accepted phase prefix')
    if any(p['ended_realtime_ns']>events[-1][1]['realtime_ns'] for p in processes):raise ValueError('owned process continued after journal terminal')
    if journal.initial:
        initial=document(journal.initial)
        if initial['created_realtime_ns']>next(e['realtime_ns'] for _,e in events if e['state']=='initial_predictions'):
            raise ValueError('initial freeze written after its journal transition')
    if journal.final:
        final=document(journal.final)
        if final['created_realtime_ns']>next(e['realtime_ns'] for _,e in events if e['state']=='final_predictions'):
            raise ValueError('final freeze written after its journal transition')
    # A visible replacement/extra freeze must not be silently ignored on failure.
    declared={original(r) for r in (journal.initial,journal.final) if r is not None}
    if set(scope_files(roots,'freezes','*.json'))!=declared:raise ValueError('orphan/replaced prediction freeze')
    return journal,events


def _recoveries(bundle,ctx,processes):
    """Postclosure protocols are useful CPU evidence, never prospective freezes."""
    result={}
    for kind,row in ctx['recovery'].items():
        if kind not in ('structures','native'):raise ValueError('unknown postclosure recovery role')
        data=document(row);rows=data.get('rows')
        if not isinstance(rows,list) or len(rows)!=len(bundle.cases) or {r.get('case') for r in rows}!=set(bundle.cases):raise ValueError('incomplete/foreign postclosure case set')
        if data.get('hardware_reexecuted') is not False:raise ValueError('recovery must not claim a new device observation')
        if kind=='structures':
            from arch_explorer.components.accelsim.microcapture import derive
            if data.get('raw_capture_unchanged') is not True:raise ValueError('recovery modified old captures')
            for item in rows:
                binding=document(item['binding']);trace=original(item['trace'])
                if item['status']!='passed' or binding['case']!=item['case'] or document(binding['files']['supervisor']) not in processes or derive(binding)!=trace.read_bytes():raise ValueError('postclosure structure does not match actual old original')
        else:
            from arch_explorer.components.accelsim.protocol import preflight_protocol,consume_request
            if data.get('schema')!='arch-explorer.sim30-postclosure-native.v1' or data.get('software_status')!='passed' or data.get('native_rebuilt') is not False or data.get('physical_parameters') is not None or data.get('heldout_accuracy') is not None or data.get('hardware_accuracy')!='not validated':raise ValueError('recovery software/physical scope differs')
            for item in rows:
                current=consume_request(document(item['request']))
                spec=document(item['review']);proof=document(item['protocol']);prediction=document(item['prediction'])
                if (item['hardware_timing_ns'] is not None or current['actual']['name']!=bundle.cases[item['case']]['function']
                    or spec['request']!=item['request']['path'] or preflight_protocol(spec)!=proof
                    or proof.get('status')!='passed' or proof.get('hardware_executed') is not False):raise ValueError('postclosure native protocol differs')
                fixed={r['role']:r for r in spec['frozen']}
                if any(fixed[role]['path']!=item[key]['path'] or fixed[role]['sha256']!=item[key]['sha256'] for role,key in (('prediction','prediction'),('source-origin','source'))):raise ValueError('recovery source/prediction reference changed')
                if ctx['source_epochs'] is None or item['source']!=document(ctx['source_epochs'])['after_closed_window']['source']:raise ValueError('recovery reader era not explicitly separated')
                if item['native']!=prediction['evaluation']['details']['native']:raise ValueError('postclosure native result metadata differs')
        result[kind]={'evidence':row,'scope':'CPU recovery after failed window; never an initial prediction','entries':len(rows),'status':'verified'}
    return result


def replay_run(design_file, window, *, evidence=None, reader_source=None):
    """Read only. v1 evidence is explicit; v2 carries the same closed context."""
    if json_file(Path(window)/'OUTCOMES.json')[0].get('schema')=='arch-explorer.measurement-state-device-outcomes.v1':
        from arch_explorer.components.accelsim.state_device_reader import replay_run as device_replay
        return device_replay(design_file,window,reader_source=reader_source,evidence=evidence)
    if json_file(Path(window)/'OUTCOMES.json')[0].get('schema')=='arch-explorer.measurement-state-outcomes.v1':
        from arch_explorer.components.accelsim.microstate import replay_run as state_replay
        return state_replay(design_file,window,reader_source=reader_source)
    return _replay_run(design_file,window,evidence=evidence,reader_source=reader_source,synthetic=False)


def _replay_run(design_file, window, *, evidence=None, reader_source=None, synthetic=False):
    window=Path(window).resolve();outcomes=json_file(window/'OUTCOMES.json')[0]
    schema=outcomes.get('schema');extra=({'evidence'} if schema in (V2,MEASUREMENT_V2) else set())|({'work_item'} if schema==MEASUREMENT_V2 else set())
    exact(outcomes,OUTCOME_FIELDS|extra,'collection outcome')
    if schema not in (V1,V2,MEASUREMENT_V2) or outcomes['status'] not in ('closed','failed'):raise ValueError('unknown outcome version/terminal state')
    if (outcomes['phase_elapsed_s'] is not None and (type(outcomes['phase_elapsed_s']) not in (int,float) or not 0<=outcomes['phase_elapsed_s']<=1800)) or type(outcomes['processes']) is not list:raise ValueError('invalid phase duration/process ledger')
    if (outcomes['status']=='closed')!=(outcomes['error'] is None) or (outcomes['error'] is not None and (not isinstance(outcomes['error'],str) or not outcomes['error'])):
        raise ValueError('collection terminal state contradicts error')
    if outcomes['hardware_accuracy']!='not validated' or outcomes['retry_allowed'] is not False:raise ValueError('unsupported physical claim/retry')
    design=json_file(design_file)[0];bundle=load_registry(original(design['registry']));validate_design(bundle,design)
    if modern(bundle)!=(schema==MEASUREMENT_V2):raise ValueError('outcome identity version differs from frozen registry')
    if modern(bundle):check_identity(bundle,outcomes,'outcomes')
    if outcomes['run_id']!=design['run_id']:raise ValueError('foreign outcome run_id')
    ctx=document(outcomes['evidence']) if schema in (V2,MEASUREMENT_V2) else (document(evidence) if evidence is not None else None)
    if ctx is None:raise ValueError('historical v1 requires explicit original evidence context')
    exact(ctx,identity_fields(bundle,{'schema','design','collection_source','raw_roots','journal','observations','resource_close','source_epochs','recovery'}),'closed campaign context')
    check_identity(bundle,ctx,'closure-context')
    if document(ctx['design'])!=design:raise ValueError('context design differs')
    declared=[Path(r).resolve() for r in ctx['raw_roots']]
    if not declared or len(set(declared))!=len(declared) or any(not Path(r).is_absolute() or not Path(r).is_dir() for r in ctx['raw_roots']):raise ValueError('explicit existing raw campaign scope required')
    roots=list(dict.fromkeys(declared+[window]))
    markers=[r/'SYNTHETIC.json' for r in roots if (r/'SYNTHETIC.json').exists()]
    fixture_source=document(design['source']).get('kind')=='synthetic'
    if bool(markers)!=synthetic or fixture_source!=synthetic:raise ValueError('synthetic boundary fixture cannot be consumed as hardware collection')
    for marker in markers:
        m=json_file(marker)[0]
        if m.get('kind')!='synthetic' or m.get('no_resource_authority') is not True or m.get('run_id')!=design['run_id'] or document(m['design'])!=design:raise ValueError('synthetic marker lost its explicit run/design boundary')
    epochs=_epochs(design,ctx)
    if reader_source is not None:
        from arch_explorer.components.accelsim.microcampaign import check_source_installation
        check_source_installation(reader_source);epochs['reader_source']=reader_source
    processes=_processes(bundle,design,outcomes,roots)
    indexed,verified=_rows(bundle,design,outcomes,processes,roots)
    obs=_observations(bundle,outcomes,ctx,roots,indexed,verified)
    journal,events=_journal(bundle,design,ctx,roots,obs,indexed,processes)
    if journal.state!=outcomes['status']:raise ValueError('outcome terminal differs from actual journal')
    terminal=events[-1][1]['payload']
    if journal.state=='failed' and terminal.get('at')=='controller' and terminal.get('reason')!=outcomes['error']:
        raise ValueError('outcome error differs from the original controller failure')
    if outcomes['status']=='closed' and (any(r['status']!='returned' for r in indexed.values()) or len(processes)!=budget(bundle)['host_processes']):
        raise ValueError('closed requires every actual registered return')
    resource=_resource(bundle,design,outcomes,processes,ctx['resource_close'])
    if document(ctx['resource_close'])['ended_realtime_ns']<events[-1][1]['realtime_ns']:raise ValueError('resource close preceded journal terminal')
    fit=fit_intervals(bundle,design,journal.fit) if set(journal.fit)==set(design['order']['fit']) else None
    comparison=None
    if journal.state=='closed':
        from arch_explorer.components.accelsim.microcampaign import compare
        comparison=compare(bundle,document(journal.final)['native_predictions'],journal.heldout,fit)
        for name,value in (('fit-only.json',fit),('comparison.json',comparison)):
            files={p.resolve() for r in roots for p in [r/name] if p.exists()}
            if len(files)!=1 or json_file(next(iter(files)))[0]!=value:raise ValueError('phase result file missing/replaced')
    else:
        for root in roots:
            if (root/'fit-only.json').exists() and (fit is None or json_file(root/'fit-only.json')[0]!=fit):raise ValueError('orphan/replaced fit phase result')
            if (root/'comparison.json').exists():raise ValueError('failed prefix cannot claim a completed comparison')
    result={'schema':record_schema(bundle,'replay') if modern(bundle) else 'arch-explorer.sim30-replay.v2',**identity(bundle),'replay_status':'replayed','kind':'synthetic' if synthetic else 'hardware-record-replay',
            'collection_outcome':outcomes['status'],'collection_error':outcomes['error'],'run_id':design['run_id'],
            'denominator':outcomes['denominator'],'attempt_counts':{mode:dict(Counter(r['status'] for r in indexed.values() if r['mode']==mode)) for mode in ('structure','timing')},
            'verified_structure_count':sum(k[1]=='structure' for k in verified),
            'stages':{'terminal':journal.state,'states':[e['state'] for _,e in events],'fit_cases':list(journal.fit),'heldout_cases':list(journal.heldout),
                      'initial_predictions':journal.initial,'final_predictions':journal.final,'complete':journal.state=='closed'},
            'timing':{cid:{'phase':x['phase'],'quality':x['quality']} for cid,x in obs.items()},
            'fit_intervals':fit,'comparison':comparison,'source_epochs':epochs,'resource_close':resource,
            'hardware_accuracy':'not validated','physical_parameters':None,'postclosure_recovery':{}}
    result['postclosure_recovery']=_recoveries(bundle,ctx,processes)
    if modern(bundle):
        from arch_explorer.components.accelsim.microcomparison import compare_registered
        result['native_timing_comparison']=compare_registered(bundle,design,outcomes,journal,fit,kind=result['kind'])
    return result


def main(argv=None):
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('design');p.add_argument('window');p.add_argument('--evidence');p.add_argument('--reader-source');p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    result=replay_run(a.design,a.window,evidence=None if a.evidence is None else ref(a.evidence),reader_source=None if a.reader_source is None else ref(a.reader_source))
    write_new(a.output,result)


if __name__=='__main__':main()
