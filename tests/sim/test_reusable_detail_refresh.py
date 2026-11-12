from dataclasses import asdict
from arch_explorer.components.transaction_controller import Channel
from arch_explorer.simulator.events.neupims_memory import TransactionMemory


def test_idle_optimization_preserves_every_refresh_and_recovery_tick():
    memory=TransactionMemory(128,1,trace=True)
    reference=Channel()
    for tick in range(8200):
        memory.step(tick)
        reference.step(tick)
    for c in memory.channels:
        assert c.trace==reference.trace
        assert c.deadline==reference.deadline and c.bank_cursor==reference.bank_cursor
        assert c.busy==reference.busy
        assert [[asdict(b) for b in rank] for rank in c.state]==[[asdict(b) for b in rank] for rank in reference.state]


def test_queue_available_and_physical_recovery_are_not_consumer_completion():
    from arch_explorer.simulator.events.controller_execution import ControllerExecutionModel
    from arch_explorer.simulator.events.protocol import ExecutionRequest
    c=Channel(queue_size=1)
    model=ControllerExecutionModel(c)
    assert model.submit(ExecutionRequest('w','memory',operation='WRITE',payload={'address':0}),0)
    events=[]
    for tick in range(100):
        if model.next_event_time() is None:break
        events.extend(model.advance(tick))
    consumer=[e.time for e in events if e.kind=='consumer_ready']
    physical=[e.time for e in events if e.kind=='physical_complete']
    ready=[e.time for e in events if e.kind=='resource_ready']
    assert consumer==[1] and physical==ready
    assert physical[0]>consumer[0]
    # Source issue is ACT then WR; physical recovery = WR+WL+burst+tWR.
    wr=next(r['time'] for r in c.trace if r['command']=='WR')
    assert physical==[wr+c.t.wl+c.t.burst+c.t.wr]
