from fractions import Fraction
import pytest
from arch_explorer.components.execution_profiles import AddressLayout, ControllerPolicy, InstructionPolicy
from arch_explorer.components.transaction_controller import Channel, Timing, Transaction
from arch_explorer.components.neupims_instructions import matmul_tiles
from arch_explorer.simulator.events.neupims_memory import TransactionMemory
from arch_explorer.simulator.events.instruction_execution import InstructionProgramModel, LinearAddressStream, matrix_program
from arch_explorer.simulator.events.controller_execution import ControllerExecutionModel
from arch_explorer.simulator.events.protocol import ExecutionRequest
from arch_explorer.simulator.events.engine import Task, TaskGraph, Resource, simulate


@pytest.mark.parametrize('layout',[AddressLayout(),AddressLayout(channels=2,ranks=1,groups=2,banks_per_group=2,columns=8,burst_bytes=32,rows=64)])
def test_address_bijection_and_second_topology(layout):
    for ch in range(layout.channels):
        address=layout.encode(ch,layout.ranks-1,layout.banks-1,layout.rows-1,layout.columns-1)
        assert layout.decode(address)==(ch,layout.ranks-1,layout.banks-1,layout.rows-1,layout.columns-1)
    c=Channel(ranks=layout.ranks,banks=layout.banks,layout=layout)
    tx=Transaction('READ',layout.encode(bank=layout.banks-1,row=2),0)
    assert c.submit(tx,0)
    for tick in range(80):c.step(tick)
    assert tx.returned is not None
    assert [r['command'] for r in c.trace]==['ACT','RD']
    assert len(c.group_rd[0])==layout.groups
    with pytest.raises(ValueError):layout.decode(layout.row_stride*layout.rows)
    with pytest.raises(ValueError):Channel(layout=layout,ranks=99)


def test_invalid_topology_policy_timing():
    with pytest.raises(ValueError):AddressLayout(channels=3)
    with pytest.raises(ValueError):ControllerPolicy(activation_window=0)
    with pytest.raises(ValueError):Timing(refi=0)


@pytest.mark.parametrize('burst_bytes',[32,64])
def test_pure_no_pim_matrix_and_independent_demand(burst_bytes):
    layout=AddressLayout(channels=2,ranks=1,groups=2,banks_per_group=2,columns=8,rows=1024,burst_bytes=burst_bytes)
    policy=ControllerPolicy(pim_enabled=False)
    instances=[]
    def factory():
        memory=TransactionMemory(layout=layout,policy=policy,
            addresses=LinearAddressStream({'a':0,'b':8192,'c':16384},layout))
        model=InstructionProgramModel(memory,trace=True);instances.append(model);return model
    graph=TaskGraph()
    graph.bind_execution('synthetic',factory,('memory.port',))
    payload=matrix_program(lambda:matmul_tiles(2,16,16,8,65536),8,8,
        policy=InstructionPolicy(spad_bytes=65536,accum_bytes=65536))
    graph.add(Task('matrix',execution=ExecutionRequest('m','synthetic',operation='program',payload=payload)))
    graph.add_resource(Resource('independent',1))
    graph.add(Task('parallel',demand={'independent':10}))
    first=simulate(graph,'conservative')
    second=simulate(graph,'conservative')
    assert first.latency_ns==second.latency_ns
    assert instances[0] is not instances[1] and instances[0].memory is not instances[1].memory
    assert next(r for r in first.records if r.name=='parallel').end_ns==pytest.approx(10,abs=1e-12)
    report=instances[0].memory.report()
    assert not any(k in report['commands'] for k in ('GWRITE','HEADER','COMPS_READRES'))
    assert report['logical_bytes']==[(2*16+16*16)*2,2*16*2]
    assert report['requested_burst_bytes'][0]==layout.burst_bytes*(report['commands']['RD']+report['forwarded_reads']+report['merged_reads']+report['pending_unique_reads'])


def test_controller_backpressure_no_mutation_and_early_write():
    c=Channel(queue_size=1)
    model=ControllerExecutionModel(c,Fraction(1,2))
    write=ExecutionRequest('write','memory',operation='WRITE',payload={'address':0})
    assert model.submit(write,0)
    state=(len(c.writes),c.rejected,model.serial,len(model.pending))
    assert not model.submit(ExecutionRequest('second','memory',operation='WRITE',payload={'address':64}),0)
    assert state==(len(c.writes),c.rejected,model.serial,len(model.pending))
    model.advance(0)
    events=model.advance(Fraction(1,2))
    assert any(e.kind=='consumer_ready' and e.request_id=='write' for e in events)
    assert not any(e.kind=='physical_complete' for e in events)
    # Read forwards from the same physical write buffer, not a second controller.
    assert model.submit(ExecutionRequest('read','memory',available_time=1,operation='READ',payload={'address':0}),1)
    model.advance(1);events=model.advance(Fraction(3,2))
    assert any(e.request_id=='read' and e.kind=='consumer_ready' for e in events)
    assert c.forwarded==1
