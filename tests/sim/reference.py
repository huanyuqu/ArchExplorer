"""Test-only exact references for the hardware-consistent scheduler.

``global_event_reference`` advances one global clock from event to event and
recomputes every resource share at each event, so no contention can be missed.
It shares the arbitration rule (:func:`allocate`) with the engine; the
scheduler under test differs only in how it finds contention sets and when it
commits. ``time_step_reference`` is an independent fixed-step simulation for
single-resource fair sharing, used to check the arbitration itself.
"""
from collections import defaultdict
import heapq
import math

from arch_explorer.simulator.events.engine import _Engine, _advance, allocate


class _Reference(_Engine):
    def run(self):
        heap, seq = [], 0

        def push(inst):
            nonlocal seq
            heapq.heappush(heap, (inst.ready, seq, inst))
            seq += 1

        for inst in self._sources():
            inst.ready = 0.0
            push(inst)
        present, time = [], 0.0
        while heap or present:
            while heap and heap[0][0] <= time + 1e-9 * max(1.0, time):
                _, _, inst = heapq.heappop(heap)
                inst.arrival = inst.ready
                self._allocate_memory(inst, inst.ready)
                if not inst.task.demand:
                    inst.start = inst.release = inst.ready
                    for succ in self._finish(inst, inst.ready + inst.task.latency_ns):
                        push(succ)
                else:
                    present.append(inst)
            if not present:
                if heap:
                    time = heap[0][0]
                continue
            rates = allocate(present, self.resources, self.normalized)
            arrival = heap[0][0] if heap else math.inf
            time, finished = _advance(rates, time, arrival)
            for inst in sorted(finished, key=lambda v: v.seq):
                present.remove(inst)
                inst.release, inst.portions = time, 1
                for succ in self._finish(inst, time + inst.task.latency_ns):
                    push(succ)
        return self._trace()


def global_event_reference(graph):
    return _Reference(graph).run()


def time_step_reference(graph, dt):
    """Fixed-step fair sharing; every task uses at most one resource, no latency."""
    capacity = {name: r.capacity for name, r in graph.resources.items()}
    remaining, done, ready, start, end = {}, set(), {}, {}, {}
    n = len(graph.tasks)
    for task in graph.tasks:
        assert len(task.demand) <= 1 and task.latency_ns == 0 and not task.static
    keys = [(t, i) for t in range(n) for i in range(graph.iterations)]
    for key in keys:
        remaining[key] = sum(graph.tasks[key[0]].demand.values())

    def preds(key):
        task, it = key
        result = [(task, it - 1)] if it else []
        result += [(src, it - d) for src, d in graph.preds[task] if it - d >= 0]
        return result

    time = 0.0
    while len(done) < len(keys):
        for key in keys:
            if key not in ready and all(p in done for p in preds(key)):
                ready[key] = max([end[p] for p in preds(key)], default=0.0)
        active = defaultdict(list)
        for key, at in ready.items():
            if key in done or at > time + 1e-12:
                continue
            if not graph.tasks[key[0]].demand:
                done.add(key)
                start[key] = end[key] = at
                continue
            (resource,) = graph.tasks[key[0]].demand
            active[resource].append(key)
        for resource, members in active.items():
            share = capacity[resource] / len(members) * dt
            for key in members:
                start.setdefault(key, time)
                remaining[key] -= share
                if remaining[key] <= 1e-12:
                    done.add(key)
                    end[key] = time + dt
        time += dt
        if time > 1e9:
            raise RuntimeError("time-step reference did not terminate")
    return start, end
