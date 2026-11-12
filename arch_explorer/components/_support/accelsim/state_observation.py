# Concrete Accel-Sim implementation; owned by the component provider.
"""Shared read-only semantics for actual CPU/CUDA C++ state observations."""
from fractions import Fraction
from pathlib import Path
from arch_explorer.components.accelsim.trace import digest
from arch_explorer.components.accelsim.microbundle import json_file
from arch_explorer.components.accelsim.microrecords import original
from arch_explorer.components.accelsim.measurement_state import sha_plan,plan_bytes,slots
from arch_explorer.components.accelsim.state_lifecycle import HOST,validate_lifecycle

def quality(values):
    if len(values)!=3:raise ValueError('three original formal samples required')
    v=[Fraction(x) for x in values];m=sum(v)/3
    if min(v)<=0:raise ValueError('positive original event intervals required')
    ratio=(max(v)-min(v))/m
    return {'status':'passed' if ratio<=Fraction('0.03') else 'refused','all_formal_ns':list(map(str,v)),
            'mean_ns':str(m),'range_over_mean':str(ratio),'threshold':'0.03',
            'assumed_event_interval_abs_error_ns':'1000','bound_is_conditional_not_guaranteed':True,
            'samples_shrink_event_error_bound':False}

def read_observation(bundle,cid,reference,run_id,freeze_sha256):
    """Verify the actual C++ transcript, not a Python-produced execution log."""
    import math,struct
    ledger=json_file(original(reference))[0]
    if (ledger.get('schema')!=HOST or ledger.get('kind') not in ('synthetic','hardware')
        or ledger.get('run')!=run_id or ledger.get('case')!=cid
        or ledger.get('registry_sha256')!=bundle.sha256 or ledger.get('plan_sha256')!=sha_plan(bundle)
        or ledger.get('freeze_sha256')!=freeze_sha256):raise ValueError('C++ host run/registry/plan/freeze differs; v2 resource evidence required')
    if (ledger.get('function')!=bundle.cases[cid]['function'] or ledger.get('partition')!=bundle.cases[cid]['partition']
        or ledger.get('directory_sha256')!=digest(plan_bytes(bundle))):raise ValueError('host program/directory differs')
    derived=validate_lifecycle(bundle,cid,ledger)
    def raw_scope(row):
        if any(row.get(k)!=ledger.get(k) for k in ('run','case','context','function','partition','registry_sha256','plan_sha256','directory_sha256','freeze_sha256')):
            raise ValueError('raw sample borrowed a foreign case/context/resource scope')
    if ledger.get('hardware_core_mhz') is not None or ledger.get('physical_parameters') is not None or ledger.get('hardware_accuracy')!='not validated' or ledger.get('retry_allowed') is not False or ledger.get('no_resource_authority') is not (ledger['kind']=='synthetic'):
        raise ValueError('Driver transcript attempted a physical/resource claim')
    for key,expected in zip(('input','auxiliary','expected'),bundle.inputs(bundle.cases[cid])):
        if original(ledger[key]).read_bytes()!=expected:raise ValueError('C++ actually consumed different input/oracle')
    if original(ledger['output_initial']).read_bytes()!=b'\xa5'*len(bundle.inputs(bundle.cases[cid])[2]):raise ValueError('output reset bytes changed')
    planned=slots(bundle,cid);actual=ledger['slots'];files=set()
    if len(actual)!=len(planned):raise ValueError('C++ omitted/added a launch role')
    failed=False;formal=[]
    for expected,row in zip(planned,actual):
        if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('execution role relabeled/reordered')
        status=row['status']
        if status not in ('returned','failed','not_started') or type(row['attempted']) is not bool or type(row['launch_attempted']) is not bool:raise ValueError('invalid C++ attempt state')
        if (status=='not_started')!= (not row['attempted']) or (row['launch_attempted'] and not row['attempted']):raise ValueError('lost attempt/launch denominator')
        if failed and status!='not_started':raise ValueError('failure followed by an extra/replacement execution')
        if status=='failed':failed=True
        if status!='not_started':
            raw=json_file(original(row['original']))[0];wanted=dict(row);wanted.pop('original')
            raw_scope(raw)
            if raw.get('slot')!=wanted or raw.get('run')!=run_id or raw.get('freeze_sha256')!=freeze_sha256 or raw.get('plan_sha256')!=sha_plan(bundle):raise ValueError('slot original differs from C++ terminal ledger')
        for role in ('started','event','actual_output','original'):
            r=row.get(role)
            if r is None:continue
            path=original(r).resolve()
            if path in files or not path.is_relative_to(original(reference).parent):raise ValueError('raw file reused across execution roles or outside host')
            files.add(path)
        if row['attempted']:
            start=json_file(original(row['started']))[0]
            raw_scope(start)
            if start.get('run')!=run_id or start.get('freeze_sha256')!=freeze_sha256 or any(start['slot'][k]!=v for k,v in expected.items()):raise ValueError('launch-start original identity differs')
        if row['event'] is not None:
            event=json_file(original(row['event']))[0]
            raw_scope(event)
            if (event['slot']['event_elapsed_ms_float32_bits']!=row['event_elapsed_ms_float32_bits']
                or any(event['slot'][k]!=v for k,v in expected.items())
                or any(event['slot'][k]!=row[k] for k in ('start_event','end_event'))):raise ValueError('event original role/pair/float32 differs')
        if status=='returned':
            bits=row['event_elapsed_ms_float32_bits'];ms=struct.unpack('<f',struct.pack('<I',bits))[0]
            if not math.isfinite(ms) or ms<=0 or row['event_elapsed_ms']!=ms or row['elapsed_ns']!=ms*1e6:raise ValueError('raw float32 event original differs')
            if row['event'] is None:raise ValueError('returned sample lacks actual elapsed original')
            if not row['returncodes'] or any(v!=0 for v in row['returncodes'].values()):raise ValueError('successful sample hides Driver error')
            raw=original(row['actual_output']).read_bytes()
            if raw!=bundle.inputs(bundle.cases[cid])[2] or row['correct'] is not True:raise ValueError('diagnostic/preparation/formal numerical oracle differs')
            if row['role']=='formal':formal.append(Fraction(ms)*10**6)
        elif row.get('actual_output') is not None:original(row['actual_output'])
    trace=ledger['trace'];launch=[t for t in trace if t['operation']=='launch']
    if [t['index'] for t in trace]!=list(range(len(trace))):raise ValueError('Driver transcript order/index changed')
    if len(launch)!=sum(r['launch_attempted'] for r in actual):raise ValueError('hidden/removed launch differs from all-role denominator')
    if [t['sequence'] for t in launch]!=[r['sequence'] for r in actual if r['launch_attempted']]:raise ValueError('launch order differs from role original')
    for row in actual:
        if row['attempted']:
            if (type(row['started_realtime_ns']) is not int or type(row['ended_realtime_ns']) is not int
                or not ledger['start_ns']<=row['started_realtime_ns']<=row['ended_realtime_ns']<=ledger['ended_ns']
                or row['started_realtime_ns']>=ledger['deadline_ns']):raise ValueError('role observation outside frozen host/absolute time')
        if row['actual_output'] is not None:
            correct=original(row['actual_output']).read_bytes()==bundle.inputs(bundle.cases[cid])[2]
            if row['correct']!=correct:raise ValueError('preparation/output correctness falsely reported')
    counts={'planned':len(actual),'attempted':sum(r['attempted'] for r in actual),'launch_attempted':sum(r['launch_attempted'] for r in actual),
            'returned':sum(r['status']=='returned' for r in actual),'failed':sum(r['status']=='failed' for r in actual),
            'not_started':sum(r['status']=='not_started' for r in actual),'formal_planned':3,
            'formal_returned':sum(r['role']=='formal' and r['status']=='returned' for r in actual)}
    if ledger['counts']!=counts:raise ValueError('C++ summary count differs from complete role originals')
    pairs={(r['start_event'],r['end_event']) for r in actual if r['start_event'] is not None}
    if len(pairs)>1 or (any(r['launch_attempted'] for r in actual) and len(pairs)!=1):raise ValueError('event-pair lifecycle not shared by all registered roles')
    hostroot=original(reference).parent
    state_files=files|{hostroot/n for n in ('LEDGER.json','input.bin','auxiliary.bin','expected.bin','output.initial.bin')}
    q=quality(formal) if len(formal)==3 else None
    if q is not None and (ledger['quality']['status']!=q['status'] or ledger['quality']['max_sample_range_over_mean']!='0.03'
        or ledger['quality']['assumed_event_interval_abs_error_ns']!='1000' or ledger['quality']['samples_shrink_assumed_bound'] is not False):raise ValueError('C++ formal quality changed frozen threshold/endpoint assumption')
    if q is not None and (not math.isclose(ledger['quality']['formal_mean_ns'],float(Fraction(q['mean_ns'])),rel_tol=1e-14)
        or not math.isclose(ledger['quality']['formal_range_over_mean'],float(Fraction(q['range_over_mean'])),rel_tol=1e-14)):raise ValueError('C++ formal statistics differ from raw float32 originals')
    should_pass=all(r['status']=='returned' for r in actual) and derived['cleanup_complete'] and q is not None and q['status']=='passed'
    if (ledger['status']=='returned')!=should_pass:raise ValueError('host/exit status hides execution/quality/cleanup failure')
    return {'ledger':ledger,'quality':q,'passed':should_pass,'resources':derived,'files':state_files}
