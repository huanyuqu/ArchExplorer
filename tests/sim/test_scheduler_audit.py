"""New commit-safety observations, checked against a non-speculative oracle."""
import itertools

import pytest

from arch_explorer.simulator.events.engine import ConsistentScheduler, Resource, Task, TaskGraph
from scheduler_oracle import Job, exact_reference


def graph_of(jobs, *, iterations=1, capacities=None, distance_edges=()):
    graph = TaskGraph(iterations)
    for r, c in (capacities or {'pe': 1, 'gate': 1}).items():
        graph.add_resource(Resource(r, c, 'fair_share'))
    ids = {}
    for j in jobs:
        ids[j.name] = graph.add(Task(j.name, kind='storage' if j.memory else 'compute',
                                    demand={j.resource: j.work} if j.work else {},
                                    latency_ns=j.delay, weight=j.weight,
                                    static=j.static, memory=j.memory or None))
    for j in jobs:
        for p in j.parents:
            graph.add_edge(ids[p], ids[j.name])
    for src, dst, distance in distance_edges:
        graph.add_edge(ids[src], ids[dst], distance)
    return graph


class AuditScheduler(ConsistentScheduler):
    def __init__(self, graph, reference):
        super().__init__(graph)
        self.reference = reference
        self.commit_observations = []
        self.rollback_observations = []

    def _commit_batch(self, batch):
        members = {(i.task.name, i.it) for i in batch.members}
        # Check actual future overlap independently of _earliest_start/_frontier.
        for inst in self.instances.values():
            key = (inst.task.name, inst.it)
            if inst.state == 'waiting' and set(inst.task.demand) & batch.resources:
                assert float(self.reference['starts'][key]) >= batch.release - 1e-8, 'future contender crosses commit'
        self.commit_observations.append((batch.release, members))
        super()._commit_batch(batch)

    def _rollback(self, batch):
        members = list(batch.members)
        saved = dict(batch.snapshot)
        done_before = self.done
        result = super()._rollback(batch)
        assert self.done == done_before  # speculative results cannot fire ticks
        for inst in members:
            got = (inst.ready, inst.remaining, inst.holding, inst.start, inst.portions, inst.work)
            assert got == saved[inst]
            assert inst.state == 'queued' and inst.batch is None
        self.rollback_observations.append(batch.id)
        return result


def checked(jobs, *, scheduler_type=AuditScheduler, **options):
    reference = exact_reference(jobs, **options)
    scheduler = scheduler_type(graph_of(jobs, **options), reference)
    trace = scheduler.run()
    for record in trace.records:
        key = (record.name, record.iteration)
        assert record.start_ns == pytest.approx(float(reference['starts'][key]))
        assert record.release_ns == pytest.approx(float(reference['releases'][key]))
        assert record.end_ns == pytest.approx(float(reference['ends'][key]))
    assert dict(trace.memory_peak) == reference['memory_peak']
    assert len(trace.records) == len(reference['ends'])
    return scheduler, trace, reference


def delayed_reader(short, gate, long=11):
    return (Job('buffer', memory=('sram', 32)),
            Job('short', 'pe', short, ('buffer',)),
            Job('long', 'pe', long, ('buffer',)),
            Job('gate', 'gate', gate),
            Job('late', 'pe', 3, ('gate',)),
            Job('tail', parents=('long', 'late'), delay=2))


@pytest.mark.parametrize('short,gate', tuple(itertools.product((1, 2, 4), (1, 3, 7))))
def test_late_contention_commit_observations_and_exact_work_conservation(short, gate):
    jobs = delayed_reader(short, gate)
    scheduler, trace, ref = checked(jobs)
    for j in jobs:
        assert sum(s[3] for s in ref['segments'] if s[0] == (j.name, 0)) == j.work
    assert scheduler.commit_observations
    if gate == 1:
        assert scheduler.rollback_observations  # deterministic late conflict, no seed search
    assert sum(r.portions for r in trace.records if r.name == 'long') >= 2


def test_iteration_distance_edges_and_two_readers_keep_original_buffer_alive():
    jobs = (Job('weights', memory=('dram', 64), static=True),
            Job('buf', memory=('sram', 32)),
            Job('short', 'pe', 2, ('buf', 'weights')),
            Job('late', 'gate', 5, ('buf',)),
            Job('join', parents=('short', 'late')))
    scheduler, _, ref = checked(jobs, iterations=3, distance_edges=(('join', 'short', 1),))
    assert ref['lifetimes']['buf', 0][1] == max(ref['ends']['short', 0], ref['ends']['late', 0])
    assert scheduler.memory_events


def test_memory_free_precedes_same_time_allocation_without_new_peak():
    jobs = (Job('first', memory=('sram', 32)), Job('reader', 'pe', 4, ('first',)),
            Job('second', parents=('reader',), memory=('sram', 32)),
            Job('last', 'pe', 2, ('second',)))
    _, trace, ref = checked(jobs)
    assert ref['lifetimes']['first', 0][1] == ref['lifetimes']['second', 0][0] == 4
    assert trace.memory_peak['sram'] == 32


def test_commit_only_after_actual_future_contender_start_bound():
    # Exact fractions: weighted service exposes incorrect unit-rate bounds.
    jobs = (Job('long', 'pe', 19, weight=2), Job('gate', 'gate', 3),
            Job('late', 'pe', 4, ('gate',), weight=3),
            Job('later', 'pe', 1, ('late',)))
    checked(jobs, capacities={'pe': 2, 'gate': 1})


def test_negative_control_detects_premature_commit():
    class Premature(AuditScheduler):
        def _commit(self):
            # Serving the earliest CSB completion first can itself be a valid
            # conservative strategy. The faulty control instead publishes the
            # latest speculative completion before a pending earlier gate.
            if not self.csb:
                return False
            self._commit_batch(max(self.csb.values(), key=lambda b: b.release))
            return True

    with pytest.raises(AssertionError, match='future contender crosses commit'):
        checked(delayed_reader(4, 1), scheduler_type=Premature)


def test_negative_control_detects_missing_late_conflict_rollback():
    class NoRollback(AuditScheduler):
        def _check_rollback(self, inst):
            pass

    with pytest.raises((AssertionError, RuntimeError)):
        checked(delayed_reader(4, 1), scheduler_type=NoRollback)
