from arch_explorer.adf import evaluate,MappingSpec
from arch_explorer.devices.pim import neupims
from test_neupims_instruction_program import small_workload
from arch_explorer.simulator.events.neupims_memory import TransactionMemory
from arch_explorer.components.neupims_instructions import Region


def test_public_shared_program_and_resource_conservation():
    e=evaluate(neupims(64),small_workload(),MappingSpec('pim_offload',
        {'npu_execution':'instructions','memory_execution':'transactions'}),cost_model=False,trace=True)
    d=e.details['neupims_transactions'];commands=e.details['command_trace']
    assert d['requested_burst_bytes'][0]==64*(d['commands']['RD']+d['forwarded_reads']+d['merged_reads']+d['pending_unique_reads'])
    assert d['requested_burst_bytes'][1]==64*(d['commands'].get('WR',0)+d['merged_writes']+d['pending_unique_writes'])
    assert d['maximum_ingress']<=32
    assert d['commands']['COMPS_READRES']==2*(1+4)
    assert len({(r['channel'],r['time']) for r in commands})==len(commands)
    assert all(r['resource']!='legacy-components' for r in e.breakdown)
    assert len(e.details['neupims_instructions']['operations'])==4
    assert {r['kind'] for r in e.details['transaction_trace']}=={'READ','WRITE','GWRITE','HEADER','COMPS_READRES'}


def test_persistent_address_and_queue_backpressure():
    m=TransactionMemory(4096,4,trace=True);r=Region('a',0,0,1,32768,32768,2);done=[]
    m.submit(r,False,0,lambda t:done.append(t))
    end=m.wait(lambda:len(done)==1,0)
    cursor=m.addresses.cursor
    m.submit(r,True,end+1,lambda t:done.append(t))
    m.wait(lambda:len(done)==2,end+1)
    assert m.addresses.cursor==2*cursor
    assert all(c.max_queue<=32 for c in m.channels)
    # The original low-occupancy write buffer has no idle flush. Advance
    # without inventing a drain and account for still-pending writes.
    m.step(m.last+500)
    assert len(done)==2
    assert sum(c.counts.get('WR',0)+len(c.pending_writes) for c in m.channels)*64==m.physical[1]
