"""Outcome consistency tests independent of CUDA, private traces or an engine."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest

from arch_explorer.simulator.accelsim.microclosure import denominator, _rows, replay_run, V1
from arch_explorer.simulator.accelsim.microexperiment import write_new


def test_registered_denominator_accounts_for_started_and_failed_raw_slots(tmp_path):
    bundle=SimpleNamespace(cases={'x':{},'y':{}})
    output=tmp_path/'host-00-structure';(output/'x').mkdir(parents=True)
    started=write_new(output/'x/sample-0.started.json',{'status':'about_to_attempt_launch'})
    returned=write_new(output/'x/sample-0.json',{'status':'failed'})
    process=write_new(tmp_path/'process.json',{'phase':'structure','output':str(output),'budget_account':{'case_ids':['x','y']}})
    rows=denominator(bundle,[process])
    assert len(rows)==8
    assert rows[0]=={'case':'x','mode':'structure','sample':0,'status':'failed','started':started,'original':returned}
    assert sum(r['status']=='not_started' for r in rows)==7
    with pytest.raises(ValueError,match='duplicate'):denominator(bundle,[process,process])


@pytest.mark.parametrize('mutation', ['missing','duplicate','foreign','returned-without-start','unknown','bool-index'])
def test_bidirectional_denominator_refuses_nonphysical_completion(tmp_path,mutation):
    b=SimpleNamespace(cases={'x':{}});rows=denominator(b,[])
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(deepcopy(rows[0]))
    elif mutation=='foreign':rows[0]['case']='alien'
    elif mutation=='returned-without-start':rows[1]['status']='returned'
    elif mutation=='unknown':rows[0]['status']='passed'
    else:rows[0]['sample']=False
    with pytest.raises(ValueError):_rows(b,{}, {'denominator':rows,'processes':[]},[],[tmp_path])


@pytest.mark.parametrize('status',['passed','success','timeout','not_started',None])
def test_reader_accepts_only_actual_producer_terminal_enumeration(tmp_path,status):
    row={'schema':V1,'run_id':'foreign','status':status,'error':None,'processes':[],
         'denominator':[],'common_lock_released':True,'all_owned_hosts_reaped':True,
         'phase_elapsed_s':0,'hardware_accuracy':'not validated','retry_allowed':False}
    write_new(tmp_path/'OUTCOMES.json',row)
    with pytest.raises(ValueError,match='terminal'):replay_run(tmp_path/'no-design.json',tmp_path)


def test_observation_files_outside_declared_scope_are_not_scanned(tmp_path):
    from arch_explorer.simulator.accelsim.microclosure import scope_files
    declared=tmp_path/'campaign';other=tmp_path/'unrelated'
    wanted=write_new(declared/'observations/one.json',{})
    write_new(other/'observations/noise.json',{})
    assert scope_files([declared],'observations','*.json')==[Path(wanted['path'])]
