"""Hardware-consistent scheduler: Fig. 6(a), Fig. 6(b) and exact-reference agreement."""
import random

import pytest

from arch_explorer.simulator.events.engine import Resource, Task, TaskGraph, simulate
from reference import global_event_reference, time_step_reference


def fig6a_graph():
    """Fig. 6(a): A, C, E share hardware 1; B, D share hardware 2; unit capacity, fair sharing."""
    graph = TaskGraph()
    graph.add_resource(Resource("hw1", 1.0, "fair_share", "1"))
    graph.add_resource(Resource("hw2", 1.0, "fair_share", "1"))
    d = graph.add(Task("D", demand={"hw2": 100}))
    a = graph.add(Task("A", demand={"hw1": 50}), [d])
    graph.add(Task("E", demand={"hw1": 200}), [d])
    b = graph.add(Task("B", demand={"hw2": 100}), [a])
    graph.add(Task("C", demand={"hw1": 50}), [b])
    return graph


def times(trace):
    return {(r.name, r.iteration): (r.start_ns, r.end_ns) for r in trace.records}


def assert_same_times(got, want):
    assert got.keys() == want.keys()
    for key in want:
        assert got[key] == pytest.approx(want[key], rel=1e-7, abs=1e-6), (key, got[key], want[key])


def test_fig6a_retained_schedulers_match_independent_contention_oracle():
    consistent = times(simulate(fig6a_graph(), "consistent"))
    assert consistent[("D", 0)] == pytest.approx((0, 100))
    assert consistent[("A", 0)] == pytest.approx((100, 200))
    assert consistent[("B", 0)] == pytest.approx((200, 300))
    # E keeps 50 units of work at t=300 and shares hardware 1 with C until both end at 400.
    assert consistent[("E", 0)] == pytest.approx((100, 400))
    assert consistent[("C", 0)] == pytest.approx((300, 400))
    conservative = times(simulate(fig6a_graph(), "conservative"))
    assert_same_times(consistent, conservative)
    reference = times(global_event_reference(fig6a_graph()))
    assert_same_times(consistent, reference)


def test_fig6b_truncation_at_first_completion():
    """v2, v3 share a zone; v3 is truncated at End(v2); v5 joins v3[2] when v4 completes."""
    graph = TaskGraph()
    graph.add_resource(Resource("pe", 1.0, "fair_share", "1"))
    graph.add_resource(Resource("other", 1.0, "fair_share", "1"))
    v1 = graph.add(Task("v1", demand={"other": 10}))
    graph.add(Task("v2", demand={"pe": 10}), [v1])
    graph.add(Task("v3", demand={"pe": 100}), [v1])
    v4 = graph.add(Task("v4", demand={"other": 30}), [v1])
    graph.add(Task("v5", demand={"pe": 10}), [v4])
    trace = simulate(graph, "consistent")
    result = times(trace)
    assert_same_times(result, times(global_event_reference(graph)))
    assert result[("v2", 0)] == pytest.approx((10, 30))
    assert result[("v5", 0)] == pytest.approx((40, 60))
    assert result[("v3", 0)] == pytest.approx((10, 130))   # 10 + 10 alone + 10 shared + 70 alone
    v3 = next(r for r in trace.records if r.name == "v3")
    assert v3.portions == 3                                 # v3[1], v3[2][1] (shared with v5), v3[2][2]


def test_speculation_rolls_back_and_still_matches_reference():
    rollbacks = 0
    for seed in range(200, 260):
        rng = random.Random(seed)
        graph = random_graph(rng, 14, 3, 2, ["fair_share"])
        trace = simulate(graph, "consistent")
        rollbacks += trace.stats.get("rollbacks", 0)
        reference = global_event_reference(random_graph(random.Random(seed), 14, 3, 2, ["fair_share"]))
        assert_same_times(times(trace), times(reference))
    assert rollbacks > 0


def random_graph(rng, n_tasks, n_resources, iterations, policies, latency=False, static=False):
    graph = TaskGraph(iterations)
    for r in range(n_resources):
        graph.add_resource(Resource(f"r{r}", rng.choice([0.5, 1.0, 2.0]), rng.choice(policies)))
    for t in range(n_tasks):
        used = rng.sample(range(n_resources), rng.choice([0, 1, 1, 1, 2]) if n_resources > 1 else 1)
        demand = {f"r{r}": float(rng.randint(1, 40)) for r in used}
        deps = rng.sample(range(t), min(t, rng.randint(0, 2)))
        is_static = static and t < 2 and not deps
        graph.add(Task(f"t{t}", kind="compute" if demand else "sync", demand=demand,
                       latency_ns=float(rng.randint(0, 5)) if latency else 0.0, static=is_static), deps)
    if iterations > 1 and n_tasks > 3:
        graph.add_edge(n_tasks - 1, 1 if not graph.tasks[1].static else 2, 1)
    return graph


@pytest.mark.parametrize("seed", range(60))
def test_random_graphs_match_global_event_reference(seed):
    rng = random.Random(seed)
    policies = [["fair_share"], ["fifo"], ["priority"], ["fair_share", "fifo", "exclusive", "priority"]][seed % 4]
    make = lambda: random_graph(random.Random(seed), rng.randint(3, 14), rng.randint(1, 4),
                                rng.choice([1, 1, 2, 3]), policies, latency=seed % 3 == 0, static=seed % 5 == 0)
    rng_state = rng.getstate()
    consistent = simulate(make(), "consistent")
    rng.setstate(rng_state)
    reference = global_event_reference(make())
    assert_same_times(times(consistent), times(reference))
    assert consistent.latency_ns == pytest.approx(reference.latency_ns, rel=1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_fair_sharing_matches_independent_time_step_reference(seed):
    rng = random.Random(1000 + seed)
    graph = TaskGraph(rng.choice([1, 2]))
    for r in range(2):
        graph.add_resource(Resource(f"r{r}", 1.0, "fair_share", "1"))
    for t in range(7):
        deps = rng.sample(range(t), min(t, rng.randint(0, 2)))
        graph.add(Task(f"t{t}", demand={f"r{rng.randint(0, 1)}": float(rng.randint(1, 20))}), deps)
    dt = 1 / 64
    start, end = time_step_reference(graph, dt)
    trace = simulate(graph, "consistent")
    for record in trace.records:
        key = (record.task, record.iteration)
        assert record.end_ns == pytest.approx(end[key], abs=8 * dt), record.name


def test_fifo_and_priority_arbitration():
    def graph(policy):
        g = TaskGraph()
        g.add_resource(Resource("u", 1.0, policy, "1"))
        g.add(Task("long", demand={"u": 30}))
        short = g.add(Task("short", demand={"u": 10}))
        g.add(Task("tail", demand={}, latency_ns=50), [short])
        return g
    fifo = times(simulate(graph("fifo")))
    assert fifo[("long", 0)] == pytest.approx((0, 30)) and fifo[("short", 0)][1] == pytest.approx(40)
    fair = times(simulate(graph("fair_share")))
    assert fair[("short", 0)][1] == pytest.approx(20) and fair[("long", 0)][1] == pytest.approx(40)
    # "short" has more downstream work (50 ns tail), so priority serves it first.
    prio = times(simulate(graph("priority")))
    assert prio[("short", 0)] == pytest.approx((0, 10)) and prio[("long", 0)][1] == pytest.approx(40)


def test_iterations_pipeline_and_storage_lifetime():
    g = TaskGraph(iterations=3)
    g.add_resource(Resource("load", 1.0, "fair_share", "B/ns"))
    g.add_resource(Resource("pe", 1.0, "fair_share", "1"))
    weights = g.add(Task("w", kind="storage", static=True, memory=("dram", 100)))
    buf = g.add(Task("act", kind="storage", memory=("sram", 10)))
    load = g.add(Task("load", kind="comm", demand={"load": 10}), [buf])
    g.add(Task("mm", demand={"pe": 10}), [load, weights])
    trace = simulate(g)
    t = times(trace)
    assert t[("load", 1)] == pytest.approx((10, 20))      # iteration 1 loads while iteration 0 computes
    assert t[("mm", 2)][1] == pytest.approx(40)
    assert trace.memory_peak["dram"] == 100
    assert trace.memory_peak["sram"] == 30                 # all three activation buffers alive at once
    assert trace.latency_ns == pytest.approx(global_event_reference(g).latency_ns)


def test_invalid_graphs_are_rejected():
    g = TaskGraph()
    with pytest.raises(KeyError):
        g.add(Task("x", demand={"missing": 1}))
    g.add_resource(Resource("r", 1.0))
    a = g.add(Task("a", demand={"r": 1}))
    b = g.add(Task("b", demand={"r": 1}), [a])
    g.add_edge(b, a)
    with pytest.raises(ValueError, match="cycle"):
        simulate(g)
    with pytest.raises(ValueError):
        Resource("bad", 0.0)
    with pytest.raises(ValueError):
        simulate(TaskGraph(), "nope")
