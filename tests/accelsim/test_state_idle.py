"""Real CPU lock/file boundaries with injected management bytes; no device use."""
import copy
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from arch_explorer.simulator.accelsim import state_idle as idle
from arch_explorer.simulator.accelsim.microexperiment import ref
from arch_explorer.simulator.accelsim.microcampaign import Collector
from arch_explorer.simulator.accelsim.state_resource import validate_handoff
from test_state_resource import fixture, GPU_UUID


def xml(*, uuid=GPU_UUID, pids=(), utilization='0 %', processes=True):
    rows=''.join('<process_info><pid>'+str(pid)+'</pid><type>G</type></process_info>' for pid in pids)
    process='<processes>'+rows+'</processes>' if processes else ''
    return ('<?xml version="1.0"?><nvidia_smi_log><gpu><uuid>'+uuid+'</uuid>'
            '<utilization><gpu_util>'+utilization+'</gpu_util></utilization>'
            '<fb_memory_usage><used>2 MiB</used></fb_memory_usage>'+process+'</gpu></nvidia_smi_log>').encode()


def setup(tmp_path):
    b,d,_=fixture(tmp_path)
    d['created_realtime_ns']=time.time_ns()-10**9
    lock=tmp_path/'common.lock';lock.touch()
    request=idle.make_request(b,d,lock,GPU_UUID)
    path=tmp_path/'REQUEST.json';path.write_text(json.dumps(request))
    root=tmp_path/'window';root.mkdir()
    transport=SimpleNamespace(bundle=b,design=d,budget=d['budget'],root=root,
        lock_fd=None,resource_lock_acquired=False,deadline=None,started=None)
    transport.remaining=lambda:transport.deadline-time.monotonic()
    return b,d,request,ref(path),transport


def release(t):
    if t.lock_fd is not None:
        fcntl.flock(t.lock_fd,fcntl.LOCK_UN);os.close(t.lock_fd);t.lock_fd=None


def runner_for(data, *, code=0, stderr=b''):
    def run(argv,**kwargs):
        assert argv==['/usr/bin/nvidia-smi','--id=0','-q','-x']
        assert 0<kwargs['timeout']<=10 and kwargs['check'] is False
        return SimpleNamespace(stdout=data,stderr=stderr,returncode=code)
    return run


def test_idle_admits_without_owner_and_holds_same_lock(tmp_path):
    b,d,request,r,t=setup(tmp_path)
    def query(argv,**kwargs):
        other=os.open(request['lock_file'],os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:os.close(other)
        return runner_for(xml())(argv,**kwargs)
    try:
        admission_ref,data=idle.acquire(t,r,runner=query)
        assert data['schema']==idle.ADMISSION and admission_ref==ref(t.root/'IDLE-ADMISSION.json')
        assert not ({'resource_owner','coordinator','coordination_evidence'}&set(data))
        assert validate_handoff(b,d,data)==data
        # The existing lease writer uses the admitted evidence and unchanged role budget.
        t.handoff_ref=admission_ref;t.handoff=data;t.opened=False
        Collector.issue_lease(t)
        lease=json.loads(Path(t.lease_ref['path']).read_text())
        assert lease['handoff']==admission_ref and lease['budget']==request['budget']
        assert lease['plan_sha256']==request['plan_sha256']
        assert lease['deadline_realtime_ns']<=request['absolute_deadline_realtime_ns']-60*10**9
        for key,value in [('run_id','foreign'),('plan_sha256','e'*64),('gpu_uuid','GPU-'+ '0'*8+'-'+'0'*4+'-'+'0'*4+'-'+'0'*4+'-'+'0'*12)]:
            bad=copy.deepcopy(data);bad[key]=value
            with pytest.raises(ValueError):validate_handoff(b,d,bad)
        with pytest.raises(ValueError,match='synthetic'):validate_handoff(b,d,data,synthetic=True)
    finally:release(t)


@pytest.mark.parametrize('raw,code,stderr',[(xml(pids=[123]),0,b''),(xml(utilization='20 %'),0,b''),
    (xml(uuid='GPU-00000000-0000-0000-0000-000000000000'),0,b''),(xml(processes=False),0,b''),
    (xml(utilization='N/A'),0,b''),(b'not XML',0,b''),(xml(),1,b'error'),(xml(),0,b'warning')])
def test_busy_unknown_never_admit_or_lease(tmp_path,raw,code,stderr):
    _,_,_,r,t=setup(tmp_path)
    try:
        with pytest.raises(ValueError,match='occupied or unknown'):
            idle.acquire(t,r,runner=runner_for(raw,code=code,stderr=stderr))
        assert not (t.root/'IDLE-ADMISSION.json').exists() and not (t.root/'lease.json').exists()
        q=json.loads((t.root/'IDLE-OBSERVATION.json').read_text())
        assert q['status'] in ('busy','unknown')
    finally:release(t)


def test_busy_common_lock_never_queries(tmp_path):
    _,_,request,r,t=setup(tmp_path)
    other=os.open(request['lock_file'],os.O_RDWR);fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    calls=[]
    try:
        with pytest.raises(BlockingIOError):idle.acquire(t,r,runner=lambda *a,**kw:calls.append(a))
        assert not calls and t.lock_fd is None and not (t.root/'IDLE-OBSERVATION.json').exists()
    finally:os.close(other)


def test_consumed_run_is_not_retried_after_release(tmp_path):
    _,_,_,r,t=setup(tmp_path);calls=[]
    try:idle.acquire(t,r,runner=runner_for(xml()))
    finally:release(t)
    t.root=tmp_path/'replacement-window';t.root.mkdir()
    try:
        with pytest.raises(FileExistsError):idle.acquire(t,r,runner=lambda *a,**kw:calls.append(a))
        assert not calls
    finally:release(t)


def test_timeout_preserves_unknown_and_partial_raw(tmp_path):
    _,_,_,r,t=setup(tmp_path)
    def timeout(*a,**kw):raise subprocess.TimeoutExpired(a[0],kw['timeout'],output=b'partial XML',stderr=b'timeout')
    try:
        with pytest.raises(ValueError,match='occupied or unknown'):idle.acquire(t,r,runner=timeout)
        q=json.loads((t.root/'IDLE-OBSERVATION.json').read_text())
        assert q['exit_code'] is None and q['status']=='unknown'
        assert (t.root/'idle.stdout').read_bytes()==b'partial XML'
    finally:release(t)


def test_expired_request_does_not_open_lock_or_query(tmp_path):
    b,d,data,r,t=setup(tmp_path);data['issued_realtime_ns']-=2000*10**9
    data['absolute_deadline_realtime_ns']-=2000*10**9;d['created_realtime_ns']=data['issued_realtime_ns']-1
    Path(r['path']).write_text(json.dumps(data));calls=[]
    with pytest.raises(ValueError,match='expired'):idle.acquire(t,ref(r['path']),runner=lambda *a,**kw:calls.append(a))
    assert t.lock_fd is None and not calls


def test_rehashed_idle_claim_cannot_hide_busy_bytes(tmp_path):
    b,d,_,r,t=setup(tmp_path)
    try:
        _,data=idle.acquire(t,r,runner=runner_for(xml()))
        query_path=Path(data['observation']['path']);q=json.loads(query_path.read_text())
        raw=Path(q['stdout']['path']);raw.write_bytes(xml(pids=[777]));q['stdout']=ref(raw)
        query_path.write_text(json.dumps(q));data['observation']=ref(query_path)
        with pytest.raises(ValueError,match='raw query'):validate_handoff(b,d,data)
    finally:release(t)


def test_new_request_is_only_a_plan_not_an_admission(tmp_path):
    b,d,data,_,_=setup(tmp_path)
    assert idle.validate_request(b,d,data)==data
    with pytest.raises(ValueError):validate_handoff(b,d,data)


@pytest.mark.parametrize('busy', [False, True])
def test_production_campaign_acquire_and_public_failed_prefix(tmp_path, monkeypatch, busy):
    """Inject only management bytes/installation for a temp CPU lock, no host."""
    from arch_explorer.simulator.accelsim import state_device as device
    from arch_explorer.simulator.accelsim import state_device_reader as reader
    b,d,request,request_ref,_=setup(tmp_path)
    source=tmp_path/'SOURCE.json';source.write_text(json.dumps({'kind':'hardware-format-unit-fixture'}))
    d['source']=ref(source)
    d['state_protocol']={'native_state':'cold-initialization-assumption'}
    d['order']={'structure':list(b.cases),'fit':[c for c,x in b.cases.items() if x['partition']=='fit'],
                'heldout':[c for c,x in b.cases.items() if x['partition']=='heldout']}
    design_file=tmp_path/'DESIGN.json';design_file.write_text(json.dumps(d))
    c=object.__new__(device.Campaign)
    c.design=d;c.bundle=b;c.design_file=design_file;c.root=tmp_path/'window'
    c.kind='hardware';c.handoff=request;c.handoff_ref=request_ref
    c.frames=[];c.journal=[];c.bindings={};c.initial=c.final=c.fit=c.lease=None
    c.parameters=None;c.observations=[];c.capture_output=None
    monkeypatch.setattr(device,'check_installation',lambda d:{'kind':'test-injected-installation'})
    monkeypatch.setattr(subprocess,'run',runner_for(xml(pids=[333] if busy else [])))
    t=device.DeviceTransport(c)
    def capture():raise ValueError('explicit CPU stop before any measurement host')
    def release_boundary():
        t.release()
        return {'lease':t.lease_ref,'raw_processes':[],'reaps':[],
                'all_owned_hosts_reaped':True,'common_lock_released':t.lock_fd is None,
                'real_locks':int(t.resource_lock_acquired),'real_leases':int(t.opened),
                'owned_process_cleanup':{'observed':{},'remaining':{}}}
    c.io=SimpleNamespace(t=t,now=time.time_ns,capture=capture,release=release_boundary)
    c.io.acquire=lambda:device.ActualCapabilities.acquire(c.io)
    error=c.execute()
    assert error and (('occupied or unknown' if busy else 'explicit CPU stop') in error)
    assert t.lock_fd is None and c.frames==[] and c.initial is None
    assert (c.lease is None)==busy
    assert c.handoff['schema']==(idle.REQUEST if busy else idle.ADMISSION)
    monkeypatch.setattr(reader,'validate_design',lambda p:(b,d))
    proof=reader.replay_run(design_file,c.root)
    assert proof['collection_outcome']=='failed'
    assert all(x['status']=='not_started' for x in proof['denominator'])
    assert proof['stages']==(['failed'] if busy else ['acquired','failed'])
    external=tmp_path/'CLOSE.json'
    external.write_text(json.dumps({'run_id':d['run_id'],'remaining_owned_processes':{},
        'all_owned_processes_reaped':True,'closed_before_absolute_deadline':True,
        'absolute_deadline_realtime_ns':request['absolute_deadline_realtime_ns'],
        'common_lock_rechecked_free':False,'owned_processes':{}}))
    closed=reader.replay_run(design_file,c.root,evidence=ref(external))
    assert closed['collection_outcome']=='failed' and closed['outer_supervisor']==ref(external)
