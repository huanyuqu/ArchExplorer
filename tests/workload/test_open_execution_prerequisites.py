"""Existing DAG foundation for VAL50; not new implementation/API acceptance.

Durations are deliberately specified task latencies, not fitted device costs.
The checks use actual scheduler memory events, never storage release_ns as free.
"""
from collections import Counter

import pytest

from arch_explorer.simulator.events.engine import ConsistentScheduler, Task, TaskGraph


@pytest.mark.parametrize('serial', [False, True])
@pytest.mark.parametrize('extra_consumer_ns', [0, 4])
def test_decomposed_fork_join_temporaries_follow_real_consumers(serial, extra_consumer_ns):
    graph = TaskGraph()
    left_shift = graph.add(Task('left.shift', latency_ns=2))
    left_tmp = graph.add(Task('left.tmp', kind='storage', memory=('scratch', 32)), [left_shift])
    left_square = graph.add(Task('left.square', latency_ns=5 + extra_consumer_ns), [left_tmp])
    right_shift = graph.add(Task('right.shift', latency_ns=3), [left_square] if serial else [])
    right_tmp = graph.add(Task('right.tmp', kind='storage', memory=('scratch', 32)), [right_shift])
    right_square = graph.add(Task('right.square', latency_ns=2), [right_tmp])
    join = graph.add(Task('join', latency_ns=1), [left_square, right_square])
    engine = ConsistentScheduler(graph)
    trace = engine.run()
    records = {record.task: record for record in trace.records}

    left_end = 7 + extra_consumer_ns
    right_begin = left_end if serial else 0
    right_end = right_begin + 5
    assert records[left_shift].start_ns == 0
    assert records[right_shift].start_ns == right_begin
    assert records[left_square].end_ns == left_end
    assert records[right_square].end_ns == right_end
    assert records[join].start_ns == max(left_end, right_end)
    assert trace.latency_ns == max(left_end, right_end) + 1
    assert trace.memory_peak['scratch'] == (32 if serial else 64)
    assert Counter(engine.memory_events) == Counter([
        (2, 1, 'scratch', 32), (left_end, 0, 'scratch', -32),
        (right_begin + 3, 1, 'scratch', 32), (right_end, 0, 'scratch', -32),
    ])
    assert records[left_tmp].release_ns == 2 < left_end
    assert records[right_tmp].release_ns == right_begin + 3 < right_end


@pytest.mark.parametrize('late_reader_ns', [9, 13])
def test_temporary_free_waits_for_all_direct_readers(late_reader_ns):
    graph = TaskGraph()
    temporary = graph.add(Task('intermediate', kind='storage', memory=('scratch', 32)))
    fast = graph.add(Task('first.read', kind='comm', latency_ns=2), [temporary])
    late = graph.add(Task('second.read', kind='comm', latency_ns=late_reader_ns), [temporary])
    graph.add(Task('result', latency_ns=1), [fast, late])
    engine = ConsistentScheduler(graph)
    trace = engine.run()
    assert engine.memory_events == [(0.0, 1, 'scratch', 32),
                                    (float(late_reader_ns), 0, 'scratch', -32)]
    assert trace.latency_ns == late_reader_ns + 1


@pytest.mark.parametrize('late_reader_ns', [9, 13])
def test_late_reader_and_next_temporary_overlap_requires_64_bytes(late_reader_ns):
    graph = TaskGraph()
    first = graph.add(Task('first.tmp', kind='storage', memory=('scratch', 32)))
    fast = graph.add(Task('first.fast_read', kind='comm', latency_ns=2), [first])
    late = graph.add(Task('first.late_read', kind='comm', latency_ns=late_reader_ns), [first])
    second = graph.add(Task('next.tmp', kind='storage', memory=('scratch', 32)), [fast])
    use = graph.add(Task('next.use', latency_ns=4), [second])
    graph.add(Task('return', latency_ns=1), [fast, late, use])
    engine = ConsistentScheduler(graph)
    trace = engine.run()
    records = {record.task: record for record in trace.records}
    assert (records[second].start_ns, records[use].end_ns) == (2, 6)
    assert records[late].end_ns == late_reader_ns
    # The first 32B allocation is still consumed during the second's 2–6ns.
    assert trace.memory_peak['scratch'] == 64
    assert Counter(engine.memory_events) == Counter([
        (0, 1, 'scratch', 32), (late_reader_ns, 0, 'scratch', -32),
        (2, 1, 'scratch', 32), (6, 0, 'scratch', -32),
    ])


def test_free_and_allocate_at_same_time_reuses_32_bytes():
    graph = TaskGraph()
    first = graph.add(Task('first.tmp', kind='storage', memory=('scratch', 32)))
    read = graph.add(Task('read', kind='comm', latency_ns=2), [first])
    second = graph.add(Task('next.tmp', kind='storage', memory=('scratch', 32)), [read])
    use = graph.add(Task('next.use', latency_ns=4), [second])
    engine = ConsistentScheduler(graph)
    trace = engine.run()
    records = {record.task: record for record in trace.records}
    assert records[read].end_ns == records[second].start_ns == 2
    assert records[use].end_ns == 6
    assert Counter(engine.memory_events) == Counter([
        (0, 1, 'scratch', 32), (2, 0, 'scratch', -32),
        (2, 1, 'scratch', 32), (6, 0, 'scratch', -32),
    ])
    # Freeing a fully consumed buffer at t=2 permits allocating another at t=2.
    assert trace.memory_peak['scratch'] == 32


@pytest.mark.parametrize('mode', ['static', 'unused', 'iterations'])
def test_lifetime_fix_preserves_static_unused_and_per_iteration_storage(mode):
    graph = TaskGraph(iterations=2 if mode == 'iterations' else 1)
    storage = graph.add(Task('buffer', kind='storage', memory=('scratch', 32),
                             static=mode == 'static'))
    if mode != 'unused':
        graph.add(Task('fast', latency_ns=2), [storage])
        graph.add(Task('late', latency_ns=9), [storage])
    engine = ConsistentScheduler(graph)
    engine.run()
    if mode == 'static':
        assert engine.memory_events == [(0.0, 1, 'scratch', 32)]
    elif mode == 'unused':
        assert Counter(engine.memory_events) == Counter([
            (0, 1, 'scratch', 32), (0, 0, 'scratch', -32)])
    else:
        # Each iteration owns a distinct allocation and its own last reader.
        assert Counter(engine.memory_events) == Counter([
            (0, 1, 'scratch', 32), (0, 1, 'scratch', 32),
            (9, 0, 'scratch', -32), (18, 0, 'scratch', -32)])
