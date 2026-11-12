"""Exact, globally chronological single-resource processor-sharing oracle.

This QA model has no speculative batches, stash, commit or rollback and imports
no production scheduler/allocation code. Fractions avoid time-step tolerances.
It intentionally does not model native state, FIFO or multi-resource DRF.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F


@dataclass(frozen=True)
class Job:
    name: str
    resource: str = ''
    work: int = 0
    parents: tuple = ()
    delay: int = 0
    weight: int = 1
    memory: tuple = ()
    static: bool = False


def exact_reference(jobs, *, iterations=1, capacities=None, distance_edges=()):
    capacities = capacities or {'pe': 1, 'gate': 1}
    by_name = {j.name: j for j in jobs}
    keys = [(j.name, i) for j in jobs for i in range(1 if j.static else iterations)]
    deps = {}
    for name, it in keys:
        j = by_name[name]
        parents = {(p, 0 if by_name[p].static else it) for p in j.parents}
        if it:
            parents.add((name, it - 1))
        parents.update((src, it - distance) for src, dst, distance in distance_edges
                       if dst == name and it >= distance)
        deps[name, it] = parents
    starts, releases, ends, segments = {}, {}, {}, []
    active, tails = {}, {}
    t = F(0)
    while len(ends) != len(keys):
        # Complete tails and all newly enabled zero-work nodes at this instant.
        while True:
            changed = False
            for key, end in list(tails.items()):
                if end == t:
                    ends[key] = end
                    del tails[key]
                    changed = True
            for key in keys:
                if key in starts or not deps[key].issubset(ends):
                    continue
                j = by_name[key[0]]
                starts[key] = t
                if j.work:
                    active[key] = F(j.work)
                else:
                    releases[key] = t
                    tails[key] = t + j.delay
                changed = True
            if not changed:
                break
        if len(ends) == len(keys):
            break
        groups = defaultdict(list)
        for key in active:
            groups[by_name[key[0]].resource].append(key)
        rates = {key: F(capacities[r]) * by_name[key[0]].weight /
                 sum(by_name[k[0]].weight for k in group)
                 for r, group in groups.items() for key in group}
        events = [t + work / rates[key] for key, work in active.items()] + list(tails.values())
        if not events:
            raise ValueError('oracle dependency cycle')
        nxt = min(events)
        assert nxt > t
        for key in list(active):
            consumed = rates[key] * (nxt - t)
            segments.append((key, t, nxt, consumed))
            active[key] -= consumed
            if active[key] == 0:
                releases[key] = nxt
                tails[key] = nxt + by_name[key[0]].delay
                del active[key]
        t = nxt
    memory_events = []
    lifetimes = {}
    for key in keys:
        j = by_name[key[0]]
        if not j.memory:
            continue
        readers = [k for k in keys if key in deps[k]]
        end = None if j.static else max((ends[k] for k in readers), default=starts[key])
        lifetimes[key] = (starts[key], end)
        owner, size = j.memory
        memory_events.append((starts[key], 1, owner, size))
        if end is not None:
            memory_events.append((end, 0, owner, -size))
    peak, current = defaultdict(int), defaultdict(int)
    for _, _, owner, delta in sorted(memory_events):
        current[owner] += delta
        peak[owner] = max(peak[owner], current[owner])
    return dict(starts=starts, releases=releases, ends=ends, deps=deps,
                segments=segments, lifetimes=lifetimes, memory_peak=dict(peak))
