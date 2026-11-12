import random
import pytest
from arch_explorer.simulator.events.engine import TaskGraph, Task, Resource, simulate


def compare(g):
    a,b=simulate(g),simulate(g,'conservative')
    assert a.latency_ns == pytest.approx(b.latency_ns, rel=1e-12, abs=1e-12)
    assert a.resource_usage==b.resource_usage
    assert a.memory_peak==b.memory_peak
    for x,y in zip(a.records,b.records):
        for k in ('start_ns','release_ns','end_ns'):
            assert getattr(x,k)==pytest.approx(getattr(y,k), rel=1e-12, abs=1e-12),(x,y,k)


@pytest.mark.parametrize('arbitration',['fifo','exclusive','priority','fair_share'])
def test_random_dag(arbitration):
    for seed in range(40):
        rng=random.Random(seed);g=TaskGraph(iterations=2)
        for r in ('a','b'):g.add_resource(Resource(r,2,arbitration))
        s=g.add(Task('weights',kind='storage',static=True,memory=('mem',10)))
        for i in range(12):
            deps=[j for j in range(len(g.tasks)) if rng.random()<.15]
            g.add(Task(str(i),demand={r:rng.randint(1,8) for r in ('a','b') if rng.random()<.65},
                       latency_ns=rng.randint(0,3),weight=rng.randint(1,3)), deps)
        compare(g)


def test_overlap_oracle():
    g=TaskGraph();g.add_resource(Resource('a',1));g.add_resource(Resource('b',1))
    a=g.add(Task('a',demand={'a':10},latency_ns=2));b=g.add(Task('b',demand={'b':4}))
    g.add(Task('join'),[a,b]);t=simulate(g,'conservative')
    assert [(r.start_ns,r.release_ns,r.end_ns) for r in t.records]==[(0,10,12),(0,4,4),(12,12,12)]
    compare(g)


def test_calendar_storage_distance():
    g=TaskGraph(3);g.add_resource(Resource('a',2))
    s=g.add(Task('s',kind='storage',memory=('x',32)))
    a=g.add(Task('a',demand={'a':8}),[s]);b=g.add(Task('calendar',end_time=lambda t:t+2),[a])
    g.add_edge(b,a,1);compare(g)
