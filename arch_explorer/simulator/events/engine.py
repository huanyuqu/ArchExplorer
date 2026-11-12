"""Task-level discrete-event engine with the hardware-consistent scheduler.

Implements the protocol of the HPCA draft, sections III-B1 and III-D:

* A workload is a graph of computation, storage and communication tasks.
  A task executes once per iteration (one tick per input edge). Iterations are
  injected at the sources and flow through the graph as a pipeline; a task
  processes its iterations in order. ``static`` tasks (weights) run once and
  their output serves every iteration.
* Storage tasks allocate memory when they activate and release it after every
  dependent task instance has finished.
* A task instance occupies a set of resources. ``demand[r]`` is the amount of
  resource ``r`` it consumes (bytes for a link, nanoseconds of exclusive use
  for a compute unit ...); ``capacity`` of ``r`` is the amount per ns. Alone,
  an instance therefore takes ``max_r demand[r] / capacity[r]`` plus its
  ``latency_ns`` tail, during which it holds no resource.
* Concurrent instances on shared resources follow each resource's declared
  arbitration (see :func:`allocate`). The schedulers below only decide which
  instances contend and over which interval.

``ConsistentScheduler`` is the speculative scheduler of section III-D: tasks of
one contention zone are issued together, evaluated up to the first completion
(the others are truncated, v -> v[1], v[2]), buffered in the contention stash
buffer (CSB), committed once no unissued task that might contend can start
before their end, and rolled back when a newly activated task starts earlier.
The optional ``conservative`` driver shares Demand arbitration and additionally
drives stateful component requests through the unified execution protocol.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import functools
import math
from typing import Optional

TASK_KINDS = ("compute", "storage", "comm", "sync", "launch")
SINGLE_OCCUPANCY = ("exclusive", "fifo")
ARBITRATIONS = ("exclusive", "fair_share", "fifo", "priority")
EPS = 1e-9


def _tol(value):
    return EPS * max(1.0, abs(value))


@dataclass
class Resource:
    """A shareable hardware quantity: a compute unit, a memory port, a link."""
    name: str
    capacity: float
    arbitration: str = "fair_share"
    unit: str = "B/ns"
    owner: str = ""

    def __post_init__(self):
        if not (isinstance(self.capacity, (int, float)) and math.isfinite(self.capacity) and self.capacity > 0):
            raise ValueError(f"Resource {self.name}: capacity must be finite and positive")
        if self.arbitration not in ARBITRATIONS:
            raise ValueError(f"Resource {self.name}: unknown arbitration {self.arbitration!r}")


@dataclass
class Task:
    """One node of the event task graph (executed once per iteration).

    ``weight`` is the number of identical hardware work items the task stands
    for (e.g. the CTAs one SM runs in a kernel); fair sharing gives it that
    many shares, exactly as if the items were separate tasks.
    """
    name: str
    kind: str = "compute"
    demand: dict = field(default_factory=dict)
    latency_ns: float = 0.0
    operator: str = ""
    static: bool = False
    memory: Optional[tuple] = None
    meta: dict = field(default_factory=dict)
    weight: float = 1.0
    id: int = -1
    priority: float = 0.0
    # Pure absolute-time calendar for resource-free waits. Called only after
    # predecessors commit; it must not mutate shared simulation state.
    end_time: object = None
    execution: object = None  # ExecutionRequest template; per-iteration id is suffixed

    def __post_init__(self):
        if self.execution is not None and (self.demand or self.end_time is not None or self.latency_ns):
            raise ValueError("State requests cannot also charge Demand/calendar/tail")
        if self.kind not in TASK_KINDS:
            raise ValueError(f"Task {self.name}: kind must be one of {TASK_KINDS}")
        if not (math.isfinite(self.weight) and self.weight > 0):
            raise ValueError(f"Task {self.name}: weight must be finite and positive")
        if not math.isfinite(self.latency_ns) or self.latency_ns < 0:
            raise ValueError(f"Task {self.name}: latency must be finite and nonnegative")
        for resource, amount in self.demand.items():
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(f"Task {self.name}: demand on {resource} must be finite and nonnegative")
        self.demand = {r: a for r, a in self.demand.items() if a > 0}
        if self.end_time is not None and (not callable(self.end_time) or self.demand):
            raise ValueError("calendar nodes must be callable and resource-free")

    def calendar_end(self, start):
        end = self.end_time(start) if self.end_time is not None else start + self.latency_ns
        if not math.isfinite(end) or end < start:
            raise ValueError(f"Task {self.name}: invalid calendar end {end} for start {start}")
        return end


class TaskGraph:
    """Tasks, dependency edges and the iteration count of one simulation."""

    def __init__(self, iterations: int = 1):
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.iterations = iterations
        self.tasks: list = []
        self.preds: list = []
        self.succs: list = []
        self.resources: dict = {}
        self.execution_models: dict = {}
        self.execution_owners: dict = {}

    def bind_execution(self, name, factory, resources=()):
        """Bind one fresh model factory per evaluation and exclusive physical owners."""
        if name in self.execution_models or not callable(factory):
            raise ValueError("Execution binding requires a unique name and model factory")
        for resource in resources:
            if resource in self.execution_owners:
                raise ValueError(f"Duplicate execution resource owner: {resource}")
        self.execution_models[name] = factory
        self.execution_owners.update({resource: name for resource in resources})

    def add_resource(self, resource: Resource) -> Resource:
        existing = self.resources.get(resource.name)
        if existing is not None:
            if (existing.capacity, existing.arbitration) != (resource.capacity, resource.arbitration):
                raise ValueError(f"Resource {resource.name} redefined with different capacity/arbitration")
            return existing
        self.resources[resource.name] = resource
        return resource

    def add(self, task: Task, deps=()) -> int:
        for resource in task.demand:
            if resource not in self.resources:
                raise KeyError(f"Task {task.name} uses unknown resource {resource!r}")
        task.id = len(self.tasks)
        self.tasks.append(task)
        self.preds.append([])
        self.succs.append([])
        for dep in deps:
            self.add_edge(dep, task.id)
        return task.id

    def add_edge(self, src: int, dst: int, distance: int = 0) -> None:
        """``dst`` iteration i depends on ``src`` iteration ``i - distance``."""
        if not (0 <= src < len(self.tasks) and 0 <= dst < len(self.tasks)):
            raise IndexError("Edge endpoints must be existing task ids")
        if type(distance) is not int or distance < 0:
            raise ValueError("Edge distance must be a nonnegative integer")
        if self.tasks[dst].static and not self.tasks[src].static:
            raise ValueError(f"Static task {self.tasks[dst].name} cannot depend on per-iteration tasks")
        if src == dst and distance == 0:
            raise ValueError("Self edge needs a positive distance")
        self.preds[dst].append((src, distance))
        self.succs[src].append((dst, distance))

    def alone_ns(self, task: Task) -> float:
        work = max((amount / self.resources[r].capacity for r, amount in task.demand.items()), default=0.0)
        return work + task.latency_ns

    def topological_order(self):
        indegree = [sum(1 for _, d in preds if d == 0) for preds in self.preds]
        order = [t for t, n in enumerate(indegree) if n == 0]
        for task in order:
            for succ, distance in self.succs[task]:
                if distance == 0:
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        order.append(succ)
        if len(order) != len(self.tasks):
            raise ValueError("Task graph has a cycle within one iteration")
        return order

    def assign_priorities(self):
        """Priority = alone time of the task plus its longest downstream chain."""
        for task in reversed(self.topological_order()):
            below = max((self.tasks[s].priority for s, d in self.succs[task] if d == 0), default=0.0)
            self.tasks[task].priority = self.alone_ns(self.tasks[task]) + below


class _Instance:
    __slots__ = ("task", "it", "seq", "pending", "ready", "arrival", "remaining", "holding", "start",
                 "end", "release", "state", "blocked_by", "batch", "portions", "work", "anchor_ready",
                 "anchor_remaining", "waiters", "freed", "last_consumer_end")

    def __init__(self, task, it, seq):
        self.task, self.it, self.seq = task, it, seq
        self.pending = 0
        self.ready = self.arrival = self.start = self.end = self.release = None
        self.remaining = 1.0
        self.holding = frozenset()
        self.state = "waiting"
        self.blocked_by = None
        self.batch = None
        self.portions = 0
        self.work = 0.0
        self.anchor_ready = None
        self.anchor_remaining = 1.0
        self.waiters = 0
        self.last_consumer_end = None
        self.freed = False

    @property
    def key(self):
        return (self.task.id, self.it)


def _compare_arrival(left, right):
    """Arrival order; arrivals equal within floating-point tolerance keep task order."""
    if abs(left.arrival - right.arrival) > _tol(max(abs(left.arrival), abs(right.arrival))):
        return -1 if left.arrival < right.arrival else 1
    return (left.seq > right.seq) - (left.seq < right.seq)


_ARRIVAL_ORDER = functools.cmp_to_key(_compare_arrival)


def allocate(active, resources, normalized):
    """Progress rates (task fraction per ns) for instances sharing resources.

    ``fifo``/``exclusive`` resources serve one instance at a time without
    preemption, in arrival order; an instance acquires all of its
    single-occupancy resources atomically, so hold-and-wait cannot deadlock.
    ``priority`` resources serve only the instances with the highest priority
    (most downstream work), preempting lower ones. Remaining instances share
    ``fair_share`` resources by progressive filling of dominant shares
    (dominant resource fairness, weighted by ``Task.weight``), which gives
    every one of n instances on a single resource 1/n of its capacity. An
    instance using several resources (operands streamed while computing)
    progresses at the rate of its most loaded resource. Holding state is
    updated in place.
    """
    rates = {inst: 0.0 for inst in active}
    held = {}
    for inst in active:
        for r in inst.holding:
            held[r] = inst
    for inst in sorted(active, key=_ARRIVAL_ORDER):
        needed = [r for r in inst.task.demand if resources[r].arbitration in SINGLE_OCCUPANCY]
        if needed and all(held.get(r, inst) is inst for r in needed):
            for r in needed:
                held[r] = inst
            inst.holding = frozenset(needed)
    candidates = [inst for inst in active
                  if all(held.get(r) is inst for r in inst.task.demand
                         if resources[r].arbitration in SINGLE_OCCUPANCY)]
    eligible, top = [], {}
    for inst in sorted(candidates, key=lambda v: (-v.task.priority, v.arrival, v.seq)):
        prioritized = [r for r in inst.task.demand if resources[r].arbitration == "priority"]
        if all(inst.task.priority >= top.get(r, -math.inf) - _tol(inst.task.priority) for r in prioritized):
            eligible.append(inst)
            for r in prioritized:
                top.setdefault(r, inst.task.priority)
    residual = {}
    unfrozen = set(eligible)
    for inst in eligible:
        for r in inst.task.demand:
            residual[r] = 1.0
    while unfrozen:
        coef = defaultdict(float)
        for inst in unfrozen:
            shares, alone = normalized[inst.task.id]
            for r, share in shares.items():
                coef[r] += inst.task.weight * share / alone
        step = min(residual[r] / c for r, c in coef.items())
        for inst in unfrozen:
            rates[inst] += step * inst.task.weight / normalized[inst.task.id][1]
        saturated = set()
        for r, c in coef.items():
            residual[r] -= step * c
            if residual[r] <= EPS:
                saturated.add(r)
        frozen = {inst for inst in unfrozen if saturated.intersection(inst.task.demand)}
        if not frozen:  # numerical safety: the minimum ratio must saturate a resource
            break
        unfrozen -= frozen
    return rates


@dataclass
class TaskRecord:
    task: int
    name: str
    kind: str
    operator: str
    iteration: int
    start_ns: float
    release_ns: float
    end_ns: float
    portions: int
    resources: tuple


@dataclass
class Trace:
    records: list
    latency_ns: float
    resource_usage: dict
    memory_peak: dict
    stats: dict


class _Engine:
    """State shared by both schedulers: instances, activation, memory tracking."""

    def __init__(self, graph: TaskGraph):
        graph.assign_priorities()
        self.graph = graph
        self.resources = graph.resources
        self.normalized = {}
        for task in graph.tasks:
            shares = {r: amount / self.resources[r].capacity for r, amount in task.demand.items()}
            self.normalized[task.id] = (shares, max(shares.values(), default=0.0))
        self.instances = {}
        seq = 0
        for task in graph.tasks:
            for it in range(1 if task.static else graph.iterations):
                self.instances[(task.id, it)] = _Instance(task, it, seq)
                seq += 1
        for inst in self.instances.values():
            inst.pending = len(self._pred_keys(inst))
            if inst.task.kind == "storage":
                inst.waiters = len(self._succ_keys(inst))
        self.memory_events = []
        self.done = 0
        self.stats = defaultdict(int)

    def _pred_keys(self, inst):
        keys = []
        if not inst.task.static and inst.it > 0:
            keys.append((inst.task.id, inst.it - 1))
        for src, distance in self.graph.preds[inst.task.id]:
            source = self.graph.tasks[src]
            if source.static:
                keys.append((src, 0))
            elif inst.it - distance >= 0:
                keys.append((src, inst.it - distance))
        return keys

    def _succ_keys(self, inst):
        keys = []
        iterations = self.graph.iterations
        if not inst.task.static and inst.it + 1 < iterations:
            keys.append((inst.task.id, inst.it + 1))
        for dst, distance in self.graph.succs[inst.task.id]:
            if inst.task.static:
                target = self.graph.tasks[dst]
                keys.extend((dst, it) for it in range(1 if target.static else iterations))
            elif inst.it + distance < iterations:
                keys.append((dst, inst.it + distance))
        return keys

    def _finish(self, inst, end):
        """Mark an instance complete and return successors that became ready."""
        inst.end, inst.state = end, "done"
        self.done += 1
        ready = []
        for key in self._succ_keys(inst):
            succ = self.instances[key]
            succ.pending -= 1
            succ.ready = end if succ.ready is None else max(succ.ready, end)
            if succ.pending == 0:
                ready.append(succ)
        for key in self._pred_keys(inst):
            pred = self.instances[key]
            if pred.task.kind == "storage" and not pred.task.static:
                # Resource-free/calendar tasks may be committed in an order
                # different from their actual completion times. Every direct
                # consumer owns this allocation until its real end, regardless
                # of which _finish call happens to consume the final waiter.
                pred.last_consumer_end = (end if pred.last_consumer_end is None
                                          else max(pred.last_consumer_end, end))
                pred.waiters -= 1
                if pred.waiters == 0:
                    self._free(pred, pred.last_consumer_end)
        return ready

    def _allocate_memory(self, inst, time):
        if inst.task.kind == "storage" and inst.task.memory is not None:
            owner, size = inst.task.memory
            self.memory_events.append((time, 1, owner, size))
            if not inst.task.static and inst.waiters == 0:
                self._free(inst, time)

    def _free(self, inst, time):
        if inst.task.memory is not None and not inst.freed:
            owner, size = inst.task.memory
            self.memory_events.append((time, 0, owner, -size))
            inst.freed = True

    def _sources(self):
        return [inst for inst in self.instances.values() if inst.pending == 0]

    def _trace(self):
        if self.done != len(self.instances):
            stuck = [inst.task.name for inst in self.instances.values() if inst.state != "done"][:5]
            raise RuntimeError(f"Simulation stopped with unfinished tasks (dependency cycle?): {stuck}")
        records, usage = [], defaultdict(float)
        for inst in self.instances.values():
            records.append(TaskRecord(inst.task.id, inst.task.name, inst.task.kind, inst.task.operator, inst.it,
                                      inst.start, inst.release, inst.end, inst.portions,
                                      tuple(sorted(inst.task.demand))))
            for r, amount in inst.task.demand.items():
                usage[r] += amount
        peak, current = defaultdict(float), defaultdict(float)
        for time, _, owner, delta in sorted(self.memory_events, key=lambda e: (e[0], e[1])):
            current[owner] += delta
            peak[owner] = max(peak[owner], current[owner])
        latency = max((inst.end for inst in self.instances.values()), default=0.0)
        return Trace(records, latency, dict(usage), dict(peak), dict(self.stats))


class _Batch:
    __slots__ = ("id", "members", "snapshot", "release", "resources", "finished", "cut")

    def __init__(self, ident):
        self.id = ident
        self.members, self.snapshot = [], {}
        self.release = None
        self.resources = set()
        self.finished, self.cut = [], []


class ConsistentScheduler(_Engine):
    """Speculative hardware-consistent scheduler (HPCA draft section III-D)."""

    def run(self) -> Trace:
        self.queue = {}                       # schedule queue S (ordered set)
        self.queue_by_resource = defaultdict(set)
        self.csb = {}                         # contention stash buffer: id -> batch
        self.csb_by_resource = defaultdict(set)
        self.tasks_by_resource = defaultdict(list)
        for task in self.graph.tasks:
            for r in task.demand:
                self.tasks_by_resource[r].append(task.id)
        # Fully activated tasks have an infinite _earliest_start bound. Keep
        # the remaining IDs in their original graph order without repeatedly
        # visiting all historical contenders on every commit query.
        self._future_tasks_by_resource = {
            r: dict.fromkeys(ids) for r, ids in self.tasks_by_resource.items()
        }
        self._commit_bounds = None
        self.next_unactivated = {task.id: 0 for task in self.graph.tasks}
        self.batch_ids = 0
        for inst in self._sources():
            self._activate(inst, 0.0)
        while self.queue or self.csb:
            progressed = self._issue()
            progressed |= self._commit()
            if not progressed:
                raise RuntimeError("Consistent scheduler made no progress; this indicates a scheduler bug")
        return self._trace()

    # -- activation and the schedule queue ---------------------------------
    def _activate(self, inst, time):
        work = [(inst, time)]
        while work:
            inst, time = work.pop()
            inst.ready = inst.arrival = inst.anchor_ready = time
            inst.state = "queued"
            pointer = self.next_unactivated
            if not inst.task.static or inst.it == 0:
                pointer[inst.task.id] = max(pointer[inst.task.id], inst.it + 1)
            if pointer[inst.task.id] >= (1 if inst.task.static else self.graph.iterations):
                for r in inst.task.demand:
                    self._future_tasks_by_resource[r].pop(inst.task.id, None)
            self._allocate_memory(inst, time)
            if not inst.task.demand:
                inst.start = inst.release = time
                work.extend((succ, succ.ready) for succ in self._finish(inst, inst.task.calendar_end(time)))
                continue
            self._enqueue(inst)

    def _enqueue(self, inst):
        self.queue[inst] = None
        for r in inst.task.demand:
            self.queue_by_resource[r].add(inst)
        self._check_rollback(inst)

    def _dequeue(self, inst):
        del self.queue[inst]
        for r in inst.task.demand:
            self.queue_by_resource[r].discard(inst)

    # -- rollback ------------------------------------------------------------
    def _check_rollback(self, inst):
        """Roll back stashed batches that a newly queued task overlaps."""
        work = [inst]
        while work:
            current = work.pop()
            if current not in self.queue:
                continue
            for r in current.task.demand:
                for batch in list(self.csb_by_resource[r]):
                    if batch.id in self.csb and current.ready < batch.release - _tol(batch.release):
                        work.extend(self._rollback(batch))

    def _rollback(self, batch):
        self.stats["rollbacks"] += 1
        self._drop(batch)
        restored = []
        for inst in batch.members:
            ready, remaining, holding, start, portions, work = batch.snapshot[inst]
            inst.ready, inst.remaining, inst.holding = ready, remaining, holding
            inst.start, inst.portions, inst.work = start, portions, work
            inst.blocked_by, inst.batch, inst.state = None, None, "queued"
            if inst not in self.queue:
                self.queue[inst] = None
                for r in inst.task.demand:
                    self.queue_by_resource[r].add(inst)
            restored.append(inst)
        return restored

    def _drop(self, batch):
        del self.csb[batch.id]
        for r in batch.resources:
            self.csb_by_resource[r].discard(batch)

    # -- issue: evaluate contention zones -----------------------------------
    def _zones(self):
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Queued instances always demand at least one resource. Connecting
        # their resources gives the same components as the bipartite graph,
        # without allocating a union-find node for every queued instance.
        for inst in self.queue:
            resources = iter(inst.task.demand)
            first = next(resources)
            parent.setdefault(first, first)
            for r in resources:
                parent.setdefault(r, r)
                a, b = find(first), find(r)
                if a != b:
                    parent[a] = b
        zones = defaultdict(list)
        for inst in self.queue:
            zones[find(next(iter(inst.task.demand)))].append(inst)
        return list(zones.values())

    def _issue(self):
        """Issue every zone of the schedule queue that can advance.

        Remainders of uncommitted batches are blocked until their batch
        commits. The other tasks of their zone are still evaluated, but only up
        to the earliest blocked remainder's start, before which no blocked task
        can share a resource with them.
        """
        issued = False
        for zone in self._zones():
            free = [inst for inst in zone if inst.blocked_by is None]
            horizon = min((inst.ready for inst in zone if inst.blocked_by is not None), default=math.inf)
            if not free or min(inst.ready for inst in free) >= horizon - _tol(horizon):
                continue
            self._evaluate(free, horizon)
            issued = True
        return issued

    def _evaluate(self, zone, horizon=math.inf):
        """Evaluate a zone from its earliest ready time to the first completion (or the horizon)."""
        batch = _Batch(self.batch_ids)
        self.batch_ids += 1
        self.stats["batches"] += 1
        pending = sorted(zone, key=lambda v: (v.ready, v.seq))
        time = pending[0].ready
        present, index = [], 0
        while index < len(pending) and pending[index].ready <= time + _tol(time):
            inst = pending[index]
            batch.snapshot[inst] = (inst.ready, inst.remaining, inst.holding, inst.start, inst.portions, inst.work)
            present.append(inst)
            index += 1
        while True:
            rates = allocate(present, self.resources, self.normalized)
            arrival = pending[index].ready if index < len(pending) else math.inf
            time, finished = _advance(rates, time, min(arrival, horizon))
            if finished or time >= horizon - _tol(horizon):
                break
            while index < len(pending) and pending[index].ready <= time + _tol(time):
                # Pending instances have not entered allocate/_advance and
                # cannot have changed. Capture each one immediately before
                # its first participation, retaining every rollback field.
                inst = pending[index]
                batch.snapshot[inst] = (inst.ready, inst.remaining, inst.holding, inst.start, inst.portions, inst.work)
                present.append(inst)
                index += 1
        batch.release = time
        for inst in present:
            done = inst in finished
            inst.portions += 1
            batch.members.append(inst)
            batch.resources.update(inst.task.demand)
            if done:
                inst.remaining = 0.0
                batch.finished.append(inst)
                self._dequeue(inst)
                inst.state = "issued"
            else:
                inst.ready = time
                inst.blocked_by = batch
                batch.cut.append(inst)
            inst.batch = batch
        self.csb[batch.id] = batch
        for r in batch.resources:
            self.csb_by_resource[r].add(batch)

    # -- commit ----------------------------------------------------------------
    def _frontier(self):
        """No task can be activated before this time (global lower bound)."""
        bound = math.inf
        for inst in self.queue:
            bound = min(bound, inst.ready)
        for batch in self.csb.values():
            bound = min(bound, batch.release)
        return bound

    def _end_lower_bound(self, inst, frontier):
        """Contention only slows a task, so its committed progress at full speed bounds its end."""
        if inst.state == "done":
            return inst.end
        task = inst.task
        alone = self.normalized[task.id][1]
        if inst.state == "waiting":
            return frontier + alone + task.latency_ns
        return max(frontier, inst.anchor_ready + inst.anchor_remaining * alone) + task.latency_ns

    def _earliest_start(self, task_id, frontier):
        it = self.next_unactivated[task_id]
        task = self.graph.tasks[task_id]
        if it >= (1 if task.static else self.graph.iterations):
            return math.inf
        inst = self.instances[(task_id, it)]
        bound = frontier
        for key in self._pred_keys(inst):
            bound = max(bound, self._end_lower_bound(self.instances[key], frontier))
        return bound

    def _committable(self, batch, frontier):
        release = batch.release
        for r in batch.resources:
            for other in self.csb_by_resource[r]:
                if other is not batch and other.release < release - _tol(release):
                    return False
        if release <= frontier + _tol(frontier):
            return True
        scoped = self._commit_bounds
        bounds = scoped[1] if scoped is not None and scoped[0] == frontier else None
        for r in batch.resources:
            for task_id in self._future_tasks_by_resource[r]:
                if bounds is None:
                    earliest = self._earliest_start(task_id, frontier)
                else:
                    if task_id not in bounds:
                        bounds[task_id] = self._earliest_start(task_id, frontier)
                    earliest = bounds[task_id]
                if earliest < release - _tol(release):
                    return False
        return True

    def _commit(self):
        committed = False
        while True:
            frontier = self._frontier()
            # Queries in this sweep do not change scheduler state. Clear all
            # bounds before a commit/activation can change that state, and on
            # exceptions; no cache survives into another sweep or run.
            self._commit_bounds = (frontier, {})
            try:
                batch = next((b for b in sorted(self.csb.values(), key=lambda b: (b.release, b.id))
                              if self._committable(b, frontier)), None)
            finally:
                self._commit_bounds = None
            if batch is None:
                return committed
            self._commit_batch(batch)
            committed = True

    def _commit_batch(self, batch):
        self.stats["commits"] += 1
        self._drop(batch)
        activated = []
        cut = set(batch.cut)  # membership only; preserve ordered member processing
        for inst in batch.members:
            inst.batch = None
            if inst in cut:
                inst.blocked_by = None
                inst.anchor_ready, inst.anchor_remaining = inst.ready, inst.remaining
            else:
                inst.release = batch.release
                activated.extend(self._finish(inst, batch.release + inst.task.latency_ns))
        for inst in activated:
            self._activate(inst, inst.ready)


def _advance(rates, time, arrival):
    """Advance progress to the next completion or arrival.

    Returns the new time and the set of instances that completed there (empty
    when an arrival comes first). Completions within floating-point tolerance
    of the earliest one are simultaneous.
    """
    finish = min((time + inst.remaining / rate for inst, rate in rates.items() if rate > 0), default=math.inf)
    if math.isinf(arrival) and math.isinf(finish):
        raise RuntimeError("Zone cannot progress: every task is blocked")
    completes = not arrival < finish - _tol(finish)
    end = finish if completes else arrival
    finished = set()
    for inst, rate in rates.items():
        if rate <= 0:
            continue
        if inst.start is None:
            inst.start = time
        if completes and time + inst.remaining / rate <= finish + _tol(finish):
            inst.remaining = 0.0
            finished.add(inst)
        else:
            inst.remaining -= rate * (end - time)
    return end, finished


SCHEDULERS = {"consistent": ConsistentScheduler}


def simulate(graph: TaskGraph, scheduler: str = "consistent") -> Trace:
    if scheduler == "conservative":
        from .conservative import ConservativeScheduler
        return ConservativeScheduler(graph).run()
    if any(t.execution is not None for t in graph.tasks):
        raise ValueError("State requests require scheduler=conservative")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler {scheduler!r}; choose from {sorted(SCHEDULERS)}")
    return SCHEDULERS[scheduler](graph).run()
