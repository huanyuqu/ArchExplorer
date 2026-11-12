"""Finite identity, symbolic oracle and error limits without a device or engine."""
from copy import deepcopy
from types import SimpleNamespace
import struct
import pytest
from arch_explorer.simulator.accelsim.microcontract import CAPS,REGISTRY_V2,validate_budget,validate_handoff,schema,identity
from arch_explorer.simulator.accelsim.microbundle import oracle
from arch_explorer.simulator.accelsim.microcomparison import errors

def bundle(count=4):
    b=dict(CAPS,cases=count,host_processes=count+1,capture_launches=count,timing_launches=3*count)
    return SimpleNamespace(document={'schema':REGISTRY_V2,'work_item':'WP-SIM-31','budget':b})

def test_budget_follows_registered_case_count_and_no_hidden_launches():
    b=bundle();validate_budget(b.document['budget'],4)
    for key,value in [('cases',12),('host_processes',13),('timing_launches',36),('warmup_launches',1),('host_timeout_s',121)]:
        bad=deepcopy(b.document['budget']);bad[key]=value
        with pytest.raises(ValueError):validate_budget(bad,4)

def test_old_identity_cannot_authorize_new_work_item():
    b=bundle();design={'run_id':'new','created_realtime_ns':100}
    h={'schema':schema(b,'resource-handoff'),'work_item':'WP-SIM-31','run_id':'new',
       'resource_released':True,'external_compute_empty':True,'coordinator':'resident_2e1684fdfc1b',
       'resource_owner':'cpu-fixture-not-an-owner','coordination_evidence':[{'path':'/placeholder','sha256':'0'*64}],
       'issued_realtime_ns':200,'lock_file':'/unopened-CPU-fixture','absolute_deadline_realtime_ns':1000000000200,'close_reserve_s':60}
    validate_handoff(b,design,h)
    for fields in ({'work_item':'WP-SIM-30'},{'run_id':'old'},{'schema':'arch-explorer.sim30-resource-handoff.v1'},
                   {'close_reserve_s':0},{'absolute_deadline_realtime_ns':1801000000200}):
        bad=dict(h,**fields)
        with pytest.raises(ValueError):validate_handoff(b,design,bad)

def test_new_affine_index_oracle_is_independent_of_iterative_chain():
    table=[(61*i+113)%128 for i in range(128)];start=[(5*i+7)%128 for i in range(32)];actual=start[:]
    for _ in range(1664):actual=[table[i] for i in actual]
    case={'family':'chase','steps':1664,'oracle':{'kind':'affine-index-u32','mul':61,'add':113,'modulus':128}}
    pack=lambda a:struct.pack('<'+'I'*len(a),*a)
    assert oracle(case,pack(table),pack(start))==pack(actual)
    table[1]^=1
    with pytest.raises(ValueError):oracle(case,pack(table),pack(start))

def test_native_error_sign_and_conditional_uncertainty_are_separate():
    r=errors(15000,[12000]*3,1000)
    assert r['signed_error_ns']==3000 and r['absolute_percentage_error']==.25
    assert r['conditional_signed_error_ns']==[2000,4000] and not r['physical_parameters_identified']
    r=errors(9000,[12000]*3,1000)
    assert r['signed_error_ns']==-3000 and r['conditional_direction'].startswith('native underprediction')
    r=errors(10,[10]*3,1000)
    assert r['conditional_relative_error'] is None and not r['samples_reduce_assumed_bound']


def test_expired_absolute_window_never_spawns_worker(tmp_path,monkeypatch):
    from arch_explorer.simulator.accelsim import microguard
    def forbidden(*a,**k):raise AssertionError('no worker may start after absolute cutoff')
    monkeypatch.setattr(microguard.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError):
        microguard.supervise(['unused'],tmp_path/'expired',1,work='CPU',run_id='test')


def test_worker_spawn_failure_still_records_resource_close(tmp_path):
    import time
    from arch_explorer.simulator.accelsim.microguard import supervise
    root=tmp_path/'unstarted'
    close=supervise(['/nonexistent/cpu-worker'],root,time.time_ns()+10**10,
                    work='CPU-SELF-CHECK',run_id='test',reserve_s=2,kind='synthetic')
    assert close['error'].startswith('FileNotFoundError:')
    assert close['all_owned_processes_reaped'] and close['remaining_owned_processes']=={}
    assert close['common_lock_rechecked_free'] is None and (root/'CLOSE.json').is_file()


def test_declared_smaller_original_budget_is_enforced(tmp_path):
    from arch_explorer.simulator.accelsim.microcampaign import require_raw_budget
    raw=tmp_path/'host-00'/'synthetic-case'/'records.bin';raw.parent.mkdir(parents=True)
    raw.write_bytes(b'0123456789')
    with pytest.raises(ValueError):require_raw_budget(tmp_path,{'capture_bytes':4,'host_raw_bytes':100})
