"""Check real scheduler allocation/free events against graph consumer lifetimes."""
from collections import Counter

from arch_explorer.simulator.events.engine import ConsistentScheduler


def storage_events(graph, trace):
    assert graph.iterations == 1
    records = {record.task: record for record in trace.records}
    engine = ConsistentScheduler(graph)
    observed = engine.run()
    assert observed.memory_peak == trace.memory_peak
    assert [(r.task, r.start_ns, r.end_ns) for r in observed.records] == [
        (r.task, r.start_ns, r.end_ns) for r in trace.records]
    expected, frees = [], {}
    for task in graph.tasks:
        if task.kind != 'storage' or task.memory is None:
            continue
        owner, size = task.memory
        expected.append((records[task.id].start_ns, 1, owner, size))
        if not task.static:
            consumers = graph.succs[task.id]
            assert all(distance == 0 for _, distance in consumers)
            free = max((records[dst].end_ns for dst, _ in consumers),
                       default=records[task.id].end_ns)
            frees[task.id] = free
            expected.append((free, 0, owner, -size))
    # release_ns on a storage record denotes activation, not freeing memory.
    assert Counter(engine.memory_events) == Counter(expected)
    return records, frees


def assert_held_through_casts(graph, records, frees, scratch, casts):
    assert graph.tasks[scratch].kind == 'storage' and casts
    consumers = {task for task, distance in graph.succs[scratch] if distance == 0}
    assert set(casts) <= consumers
    assert records[scratch].start_ns <= min(records[task].start_ns for task in casts)
    assert frees[scratch] >= max(records[task].end_ns for task in casts)
