"""Inject query bytes into the actual boundary; never start a device process."""
import subprocess,time
from types import SimpleNamespace
import pytest
from arch_explorer.simulator.accelsim.state_device import ActualCapabilities

def boundary(tmp_path,monkeypatch,uuid='1'*32):
    now=time.time_ns();calls=[]
    def query(*args,**kwargs):
        calls.append((args,kwargs))
        return SimpleNamespace(returncode=0,stdout=('GPU-'+uuid+', 100, 200, P0\n').encode(),stderr=b'')
    monkeypatch.setattr(subprocess,'run',query)
    io=object.__new__(ActualCapabilities);io.queries=set()
    io.c=SimpleNamespace(root=tmp_path,bundle=SimpleNamespace(cases={'case':{}}),
        design={'run_id':'synthetic-query-test','created_realtime_ns':now-1000000},
        handoff={'absolute_deadline_realtime_ns':now+600*10**9})
    io.t=SimpleNamespace(remaining=lambda:100)
    return io,calls

def test_query_admission_rejects_duplicate_before_runner(tmp_path,monkeypatch):
    io,calls=boundary(tmp_path,monkeypatch)
    io.snapshot('case','before','1'*32)
    with pytest.raises(ValueError,match='duplicate/extra'):io.snapshot('case','before','1'*32)
    assert len(calls)==1

def test_unknown_case_never_reaches_runner(tmp_path,monkeypatch):
    io,calls=boundary(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='unregistered'):io.snapshot('other','before','1'*32)
    assert not calls

def test_wrong_device_is_refused_in_actual_boundary(tmp_path,monkeypatch):
    io,calls=boundary(tmp_path,monkeypatch,uuid='2'*32)
    with pytest.raises(ValueError,match='UUID'):io.snapshot('case','before','1'*32)
    assert len(calls)==1
    # The attempted key remains consumed; a refusal cannot cause a replacement.
    with pytest.raises(ValueError,match='duplicate/extra'):io.snapshot('case','before','1'*32)
    assert len(calls)==1
