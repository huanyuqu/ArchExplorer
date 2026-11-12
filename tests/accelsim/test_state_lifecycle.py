"""Resource-state counterexamples, independent of the C++ execution tests."""
import copy
from types import SimpleNamespace
import pytest
from arch_explorer.simulator.accelsim.state_lifecycle import validate_lifecycle,HOST
from arch_explorer.simulator.accelsim.trace import digest

def fixture():
    b=SimpleNamespace(cases={'case':{'family':'dep'}},document={'budget':{'app_bytes':1024}},
                      inputs=lambda c:(b'\x01\0\0\0',b'\x03\0\0\0\x01\0\0\0',b'\x04\0\0\0'))
    l={'schema':HOST,'run':'run','context':'ctx','case':'case','slots':[{'status':'not_started','attempted':False}],
       'trace':[],'ended_ns':20,'deadline_ns':100,'application_bytes':0,
       'cleanup':{'status':'closed','all_resources_released':True,'remaining_events':[],'remaining_allocations':[]}}
    return b,l

def call(l,op,name=None,handle=0,*,rc=0,size=0,sha=None,kind='allocation'):
    r=[] if name is None else [{'kind':kind,'name':name,'handle':handle,'bytes':size,'sha256':sha}]
    l['trace'].append({'index':len(l['trace']),'sequence':None,'operation':op,'return_code':rc,
                       'scope':{'run':l['run'],'case':'case','context':l['context']},'resources':r})

def test_no_object_early_failure_can_be_reclaimed():
    b,l=fixture();call(l,'allocate_input','input',rc=2,size=4)
    r=validate_lifecycle(b,'case',l)
    assert r['cleanup_complete'] and r['acquired']==[] and r['initialization_failure']['return_code']==2

def acquired_error():
    b,l=fixture();call(l,'allocate_input','input',handle=11,size=4)
    call(l,'allocate_output','output',handle=12,size=4,rc=2)
    call(l,'cleanup_context_synchronize');call(l,'free_output','output',handle=12);call(l,'free_input','input',handle=11)
    l['application_bytes']=8
    return b,l

def test_nonzero_handle_on_error_is_acquired_and_must_be_freed():
    b,l=acquired_error();r=validate_lifecycle(b,'case',l)
    assert r['cleanup_complete'] and len(r['acquired'])==2
    assert r['acquired'][1]['create_return_code']==2
    del l['trace'][2:]
    with pytest.raises(ValueError,match='missing cleanup'):validate_lifecycle(b,'case',l)

@pytest.mark.parametrize('change',['duplicate','foreign','wrong_handle','missing_init','return_code','missing_one'])
def test_semantic_mutations_of_partial_initialization(change):
    b,l=acquired_error()
    if change=='duplicate':l['trace'].insert(4,copy.deepcopy(l['trace'][3]))
    elif change=='foreign':l['trace'][3]['scope']['case']='other'
    elif change=='wrong_handle':l['trace'][3]['resources'][0]['handle']=11
    elif change=='missing_init':del l['trace'][0]
    elif change=='return_code':l['trace'][3]['return_code']=201
    elif change=='missing_one':del l['trace'][3]
    for i,t in enumerate(l['trace']):t['index']=i
    with pytest.raises(ValueError):validate_lifecycle(b,'case',l)

def test_cleanup_sync_failure_cannot_be_erased_by_later_success():
    b,l=acquired_error();l['trace'][2]['return_code']=201
    l['cleanup'].update(status='failed',all_resources_released=False)
    r=validate_lifecycle(b,'case',l)
    assert not r['cleanup_complete'] and not r['remaining_allocations']
    l['cleanup'].update(status='closed',all_resources_released=True)
    with pytest.raises(ValueError,match='cleanup contradicts'):validate_lifecycle(b,'case',l)

def test_destroy_error_is_derived_even_if_claimed_list_is_empty():
    b,l=fixture();call(l,'allocate_input','input',handle=11,size=4);call(l,'allocate_output','output',handle=12,size=4)
    call(l,'upload_input','input',handle=11,size=4,sha=digest(b'\x01\0\0\0'))
    call(l,'event_create_start','start',handle=21,kind='event');call(l,'event_create_stop','stop',handle=22,kind='event',rc=2)
    call(l,'cleanup_context_synchronize');call(l,'event_destroy_stop','stop',handle=22,kind='event',rc=201)
    call(l,'event_destroy_start','start',handle=21,kind='event');call(l,'free_output','output',handle=12);call(l,'free_input','input',handle=11)
    l['application_bytes']=8;l['cleanup'].update(status='failed',all_resources_released=False,remaining_events=[22])
    r=validate_lifecycle(b,'case',l);assert r['remaining_events']==[22] and not r['cleanup_complete']
    l['cleanup']['remaining_events']=[]
    with pytest.raises(ValueError,match='reported remaining'):validate_lifecycle(b,'case',l)
