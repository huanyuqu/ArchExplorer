"""Contract mutations that would authorize unregistered work or mix state."""
import copy
from types import SimpleNamespace
import pytest
from arch_explorer.simulator.accelsim import measurement_state as s
from arch_explorer.simulator.accelsim.microstate import validate_handoff,quality,validate_state_evidence
from arch_explorer.simulator.accelsim.microexperiment import ref,write_new

def bundle():
    p={'schema':s.PLAN,'lifecycle':s.LIFECYCLE,'cases':[
        {'case':'a','diagnostic':1,'preparation':0,'formal':3},
        {'case':'b','diagnostic':1,'preparation':2,'formal':3}]}
    return SimpleNamespace(document={'schema':s.REGISTRY,'measurement_state':p},
      cases={'a':{'id':'a','function':'demo_a','partition':'fit'},'b':{'id':'b','function':'demo_b','partition':'heldout'}},sha256='a'*64)

def test_fixed_plan_cannot_relabel_or_adapt():
    b=bundle();s.validate_plan(b.document['measurement_state'],b.cases)
    assert [x['role'] for x in s.slots(b,'b')]==['diagnostic','preparation','preparation','formal','formal','formal']
    for field,value in [('diagnostic',0),('preparation',9),('formal',4),('preparation',True)]:
        p=copy.deepcopy(b.document['measurement_state']);p['cases'][0][field]=value
        with pytest.raises(ValueError):s.validate_plan(p,b.cases)
    p=copy.deepcopy(b.document['measurement_state']);p['cases'].reverse()
    with pytest.raises(ValueError):s.validate_plan(p,b.cases)

def test_handoff_role_budget_and_fixed_deadline():
    b=bundle();d={'run_id':'test','created_realtime_ns':100,'budget':{'total_window_s':1800,'preparation_launches':2}}
    h={'schema':'arch-explorer.measurement-state-handoff.v1','kind':'synthetic','no_resource_authority':True,
       'run_id':'test','registry_sha256':b.sha256,'plan_sha256':s.sha_plan(b),'budget':d['budget'],
       'resource_owner':'synthetic-no-device-owner','issued_realtime_ns':200,
       'absolute_deadline_realtime_ns':200+1800*10**9,'close_reserve_s':60}
    assert validate_handoff(b,d,h,acquired_ns=300)==200+1740*10**9
    for edit in ({'schema':'arch-explorer.measurement-resource-handoff.v2'},
                 {'budget':{'total_window_s':1800,'preparation_launches':0}},
                 {'absolute_deadline_realtime_ns':300+1800*10**9},{'kind':'hardware'}):
        with pytest.raises(ValueError):validate_handoff(b,d,dict(h,**edit))
    with pytest.raises(ValueError):validate_handoff(b,d,h,acquired_ns=h['absolute_deadline_realtime_ns'])

def test_clock_cannot_be_borrowed_from_wrong_run_or_time(tmp_path):
    e={'schema':'arch-explorer.measurement-independent-state.v1','run_id':'other','role':'clock',
       'begin_realtime_ns':10,'end_realtime_ns':20,'value':1980,'independent_of_events':True}
    f=write_new(tmp_path/'state.json',e);d={'run_id':'own','state_evidence':{'clock':f,'cache':None}}
    with pytest.raises(ValueError):validate_state_evidence(d,0,30)
    d['run_id']='other'
    with pytest.raises(ValueError):validate_state_evidence(d,15,30)
    e['independent_of_events']=False;f=write_new(tmp_path/'event-derived.json',e);d['state_evidence']['clock']=f
    with pytest.raises(ValueError):validate_state_evidence(d,0,30)

def test_quality_never_drops_first_or_shrinks_bound():
    assert quality([32000,25184,19744])['status']=='refused'
    assert quality([20000,20100,20200])['status']=='passed'
    assert quality([20000,20100,20200])['assumed_event_interval_abs_error_ns']=='1000'
    with pytest.raises(ValueError):quality([20100,20200])
