"""Independent phase contract through the exported component-author interface."""
from copy import deepcopy

from arch_explorer.components.api import ExecutionEvent, ExecutionRequest, Task, TaskGraph, simulate


class PhasedQueue:
    """One slot: consumer at +5, physical completion +7, reusable at +9 ns."""
    def __init__(self):
        self.pending = None
        self.serial = 0

    def submit(self, request, now):
        if self.pending is not None:
            return False
        self.pending = [request.request_id, now, 0]
        return True

    def next_event_time(self):
        if self.pending is None:
            return None
        return self.pending[1] + (5, 7, 9)[self.pending[2]]

    def advance(self, now):
        assert now == self.next_event_time()
        ident, _, index = self.pending
        kind = ("consumer_ready", "physical_complete", "resource_ready")[index]
        event = ExecutionEvent(now, index, self.serial, ident, kind)
        self.serial += 1
        if index == 2:
            self.pending = None
        else:
            self.pending[2] += 1
        return [event]


def test_backpressure_has_no_side_effect_and_three_distinct_readiness_times():
    queue = PhasedQueue()
    assert queue.submit(ExecutionRequest("first", "queue"), 0)
    before = deepcopy(queue.__dict__)
    assert not queue.submit(ExecutionRequest("rejected", "queue"), 1)
    assert queue.__dict__ == before
    events = []
    while queue.next_event_time() is not None:
        events.extend(queue.advance(queue.next_event_time()))
    assert [(e.time, e.kind) for e in events] == [
        (5, "consumer_ready"), (7, "physical_complete"), (9, "resource_ready")]


def test_consumer_can_run_before_physical_completion_but_queue_cannot_reuse():
    graph = TaskGraph()
    graph.bind_execution("queue", PhasedQueue)
    first = graph.add(Task("first", execution=ExecutionRequest("first", "queue")))
    graph.add(Task("second", execution=ExecutionRequest("second", "queue")))
    graph.add(Task("consumer", latency_ns=2), deps=[first])
    trace = simulate(graph, "conservative")
    assert [(r.name, r.start_ns, r.end_ns) for r in trace.records] == [
        ("first", 0, 5), ("second", 9, 14), ("consumer", 5, 7)]
    first_events = trace.stats["execution_events"][:3]
    assert [(e["time"], e["kind"]) for e in first_events] == [
        (5, "consumer_ready"), (7, "physical_complete"), (9, "resource_ready")]
