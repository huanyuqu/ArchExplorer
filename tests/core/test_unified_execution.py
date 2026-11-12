from fractions import Fraction
import pytest
from arch_explorer.simulator.events.protocol import ExecutionRequest as Request, ExecutionEvent as Event, DemandAdapter
from arch_explorer.simulator.events.engine import TaskGraph, Task, Resource, simulate


class Queue:
    def __init__(self): self.pending=None; self.serial=0
    def submit(self,request,now):
        if self.pending is not None: return False
        self.pending=(request.request_id,now+Fraction(1,3));return True
    def next_event_time(self): return self.pending[1] if self.pending else None
    def advance(self,now):
        ident,t=self.pending;assert t==now
        self.pending=None;self.serial+=3
        return [Event(now,0,self.serial,ident,'physical_complete'),
                Event(now,1,self.serial+1,ident,'consumer_ready'),
                Event(now,2,self.serial+2,ident,'queue_available')]


def graph():
    g=TaskGraph();g.bind_execution('q',Queue,resources=['physical'])
    for i in range(2): g.add(Task(str(i),execution=Request(str(i),'q')))
    return g


def test_backpressure_exact_mixed_fresh():
    g=graph();g.add_resource(Resource('a',1));g.add(DemandAdapter({'a':2}).task('independent'))
    for _ in range(2):
        t=simulate(g,'conservative')
        assert t.records[0].end_ns==Fraction(1,3)
        assert t.records[1].start_ns==Fraction(1,3)
        assert t.records[1].end_ns==Fraction(2,3)
        assert t.records[2].start_ns==0 and t.latency_ns==2
        assert len(t.stats['execution_events'])==6
    with pytest.raises(ValueError,match='conservative'):simulate(g)


def test_double_charge():
    g=graph();g.add_resource(Resource('physical',1));g.add(Task('bad',demand={'physical':1}))
    with pytest.raises(ValueError,match='both'):simulate(g,'conservative')


def test_deadlock_and_time_reversal():
    class Never(Queue):
        def submit(self,r,t):return False
    g=TaskGraph();g.bind_execution('q',Never);g.add(Task('x',execution=Request('x','q')))
    with pytest.raises(RuntimeError,match='deadlock'):simulate(g,'conservative')
    class Back(Queue):
        def next_event_time(self):return -1
    g.execution_models['q']=Back
    with pytest.raises(ValueError,match='time'):simulate(g,'conservative')


def test_future_request_and_no_hidden_callback():
    g=TaskGraph();g.bind_execution('q',Queue)
    g.add(Task('x',execution=Request('x','q',available_time=2)))
    t=simulate(g,'conservative');assert t.records[0].start_ns==2
    assert t.latency_ns==Fraction(7,3)
    with pytest.raises(ValueError):Task('x',execution=Request('x','q'),latency_ns=1)


def test_early_consumer_does_not_release_queue_or_block_independent():
    class Early(Queue):
        def submit(self,request,now):
            if self.pending:return False
            self.pending=[request.request_id,now+1,now+3];return True
        def next_event_time(self):return self.pending[1] if self.pending else None
        def advance(self,now):
            ident,t,end=self.pending
            if t<end:
                self.pending[1]=end
                return [Event(now,0,0,ident,'consumer_ready')]
            self.pending=None
            return [Event(now,0,1,ident,'physical_complete'),Event(now,1,2,ident,'queue_available')]
    g=graph();g.execution_models['q']=Early
    g.add_resource(Resource('a',1));g.add(Task('a',demand={'a':10}))
    t=simulate(g,'conservative')
    assert [(r.start_ns,r.end_ns) for r in t.records]==[(0,1),(3,4),(0,10)]
    assert [e['time'] for e in t.stats['execution_events']]==[1,3,3,4,6,6]


def test_no_consistent_call(monkeypatch):
    from arch_explorer.simulator.events.engine import ConsistentScheduler
    def fail(*args):raise AssertionError('old solver invoked')
    monkeypatch.setattr(ConsistentScheduler,'run',fail)
    g=TaskGraph();g.add_resource(Resource('a',1));g.add(Task('x',demand={'a':2}))
    assert simulate(g,'conservative').latency_ns==2
