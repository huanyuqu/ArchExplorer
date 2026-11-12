"""Mapping context: the DSE primitives templates use to build the event task graph.

``map_node`` places a compute task on a compute point, ``map_edge`` a tensor
transfer between memories (routed through every level it crosses),
``storage`` a tensor allocation, ``sync`` a synchronization point and
``allreduce`` a ring or tree collective composed of ``map_edge`` transfers
and reduction tasks (section III-C1). Each primitive converts the placement
into resource demands with the component models of the built simulator.
"""
from __future__ import annotations

import math

from arch_explorer.components.base import Demand, Kernel, dtype_bytes
from arch_explorer.simulator.events.engine import Task, TaskGraph
from arch_explorer.simulator.plan import ExecutionPlan, PlanConstraints
from arch_explorer.workload.operators import VECTOR_COSTS

COLLECTIVE_ALGORITHMS = ("ring", "multiring", "tree")


def ring_orders(p):
    """Directed rings used by ``multiring``: the data is split evenly across them.

    For up to four ranks every Hamiltonian cycle of the complete graph in both
    directions, so a fully connected group uses every directed link equally;
    for more ranks the stride-s rings ``r -> r + s`` with gcd(s, p) = 1.
    """
    import itertools
    if p <= 4:
        cycles = []
        for perm in itertools.permutations(range(1, p)):
            if perm[0] < perm[-1] or p < 3:
                cycles.append([0, *perm])
        rings = cycles + [[order[0], *reversed(order[1:])] for order in cycles if p > 2]
        return rings or [list(range(p))]
    rings = []
    for stride in range(1, p):
        if math.gcd(stride, p) == 1:
            rings.append([(i * stride) % p for i in range(p)])
    return rings


class MappingContext:
    def __init__(self, simulator, iterations=1):
        self.sim = simulator
        self.graph = TaskGraph(iterations)
        self.plan = ExecutionPlan()
        self._resources = {}
        self.details = {}  # JSON-compatible mapping evidence, per evaluation

    def _resource(self, name):
        if name not in self._resources:
            self._resources = self.sim.resources
            if name not in self._resources:
                raise KeyError(f"Unknown resource {name}")
        return self._resources[name]

    def task(self, name, demand: Demand, deps=(), kind="compute", operator="", label="", weight=1.0,
             static=False, memory=None, latency_ns=None) -> int:
        for resource in demand.resources:
            self.graph.add_resource(self._resource(resource))
        task = Task(name, kind, dict(demand.resources), demand.latency_ns if latency_ns is None else latency_ns,
                    operator, static, memory, {"label": label or operator}, weight)
        return self.graph.add(task, list(dict.fromkeys(deps)))

    def bind_execution(self, name, factory, resources=()):
        """Bind one evaluation-local physical owner, shared by request nodes."""
        self.plan.require(PlanConstraints(scheduler='conservative', reason='stateful execution'))
        def create():
            model = factory()
            self.sim.execution_instances[name] = model
            return model
        self.graph.bind_execution(name, create, resources)

    def program(self, name, component, payload, deps=(), operator="", label="", available_time=0):
        from arch_explorer.simulator.events.protocol import ExecutionRequest
        request = ExecutionRequest(name, component, available_time, 'program', payload, operator)
        return self.graph.add(Task(name, 'compute', operator=operator, meta={'label': label or operator},
                                   execution=request), list(dict.fromkeys(deps)))

    # -- primitives --------------------------------------------------------------
    def calendar(self, name, deps, end_time, operator="", label=""):
        """Pure resource-free wait evaluated at committed absolute dependency time."""
        task = Task(name, "sync", operator=operator, meta={"label": label or operator}, end_time=end_time)
        return self.graph.add(task, list(dict.fromkeys(deps)))

    def map_node(self, name, compute, kernels, deps=(), streams=(), operator="", label="", weight=1.0,
                 latency_ns=0.0, extra=None) -> int:
        """Compute task on ``compute`` running ``kernels`` while reading/writing ``streams``.

        ``kernels`` is a list of ``(Kernel, count)``; ``streams`` a list of
        ``(memory path, bytes)`` read by the compute point while it computes
        (operands are streamed, not staged), or ``(memory path, bytes,
        "write")`` for results it writes. ``extra`` adds demand such as DRAM
        traffic behind a cache.
        """
        model = self.sim.points[compute]
        if getattr(model, 'includes_boundary_traffic', False) and (streams or extra is not None):
            raise ValueError('component accounts its own IO boundary; duplicate external streams/extra refused')
        demand = Demand()
        for kernel, count in kernels:
            demand = demand.add(model.kernel_demand(kernel).scaled(count))
        for memory, size, *direction in streams:
            if size > 0:
                demand = demand.add(self.sim.stream(memory, compute, size, write=direction == ["write"]))
        if extra is not None:
            demand = demand.add(extra)
        demand.latency_ns = latency_ns
        return self.task(name, demand, deps, "compute", operator, label, weight)

    def map_edge(self, name, src_memory, dst_memory, size, deps=(), operator="", label="") -> int:
        return self.task(name, self.sim.transfer(src_memory, dst_memory, size), deps, "comm", operator, label)

    def storage(self, name, memory, size, deps=(), operator="", label="", static=False) -> int:
        return self.task(name, Demand(), deps, "storage", operator, label, static=static,
                         memory=(memory, float(size)))

    def map_multicast(self,name,src_memory,dst_memories,size,deps=(),operator='',label=''):
        return self.task(name,self.sim.multicast(src_memory,dst_memories,size),deps,'comm',operator,label)

    def sync(self, name, deps, operator="", label="", latency_ns=0.0) -> int:
        return self.task(name, Demand(), deps, "sync", operator, label, latency_ns=latency_ns)

    def spread(self, name, computes, kernel, deps=(), streams=(), operator="", label="") -> int:
        """One task dividing a vector kernel evenly over several compute points in lockstep."""
        rows, cols = kernel.shape
        share = Kernel(kernel.op, (rows / len(computes), cols), kernel.precision, kernel.simt, kernel.sfu,
                       kernel.row_simt, kernel.row_sfu)
        demand = Demand()
        for compute in computes:
            if getattr(self.sim.points[compute], 'includes_boundary_traffic', False):
                raise ValueError('explicit component IO work cannot use implicit vector spread/streams')
            demand = demand.add(self.sim.points[compute].kernel_demand(share))
            for memory, size, *direction in streams:
                demand = demand.add(self.sim.stream(memory, compute, size / len(computes),
                                                    write=direction == ["write"]))
        demand.latency_ns = 0.0
        return self.task(name, demand, deps, "compute", operator, label, weight=len(computes))

    def allreduce(self, name, ranks, size, deps, algorithm="ring", operator="", label="AllReduce", dtype="fp16"):
        """All-reduce ``size`` bytes held in each rank's memory.

        ``ranks`` is a list of ``(memory path, [compute paths])``; ``deps`` the
        per-rank tasks producing the data. Returns per-rank completion task ids.
        Ring: reduce-scatter then all-gather, 2(P-1) steps of size/P chunks.
        Multiring: the same on several directed rings (:func:`ring_orders`),
        each carrying an equal share of the data concurrently.
        Tree: binary-tree reduce to rank 0, then broadcast down the same tree.
        """
        if algorithm not in COLLECTIVE_ALGORITHMS:
            raise ValueError(f"collective must be one of {COLLECTIVE_ALGORITHMS}")
        p = len(ranks)
        last = [list(d) for d in deps]
        if p == 1:
            return [self.sync(f"{name}.done0", last[0], operator, label)]
        element = dtype_bytes(dtype)

        def reduce(rank, chunk, after, tag):
            memory, computes = ranks[rank]
            kernel = Kernel("vector", (1, max(1.0, chunk / element)), dtype, **VECTOR_COSTS["add"])
            # Reads both operands from and writes the sum to the rank's memory.
            return self.spread(f"{name}.{tag}.reduce{rank}", computes, kernel, after,
                               [(memory, 2 * chunk), (memory, chunk, "write")], operator, label)

        if algorithm in ("ring", "multiring"):
            orders = [list(range(p))] if algorithm == "ring" else ring_orders(p)
            finals = [[] for _ in range(p)]
            for index, order in enumerate(orders):
                chunk = size / len(orders) / p
                ring_last = [list(d) for d in last]
                nxt = {order[i]: order[(i + 1) % p] for i in range(p)}
                for phase in ("rs", "ag"):
                    for step in range(p - 1):
                        sent = []
                        for r in range(p):
                            dst = nxt[r]
                            sent.append((dst, self.map_edge(f"{name}.r{index}{phase}{step}.{r}->{dst}", ranks[r][0],
                                                            ranks[dst][0], chunk, ring_last[r], operator, label)))
                        for dst, transfer in sent:
                            if phase == "rs":
                                ring_last[dst] = [reduce(dst, chunk, ring_last[dst] + [transfer],
                                                         f"r{index}rs{step}")]
                            else:
                                ring_last[dst] = ring_last[dst] + [transfer]
                for r in range(p):
                    finals[r] += ring_last[r]
            last = finals
        else:
            levels = math.ceil(math.log2(p))
            for level in range(levels):
                stride = 2 ** level
                for r in range(stride, p, 2 * stride):
                    parent = r - stride
                    transfer = self.map_edge(f"{name}.up{level}.{r}->{parent}", ranks[r][0], ranks[parent][0], size,
                                             last[r], operator, label)
                    last[parent] = [reduce(parent, size, last[parent] + [transfer], f"up{level}")]
            for level in reversed(range(levels)):
                stride = 2 ** level
                for r in range(stride, p, 2 * stride):
                    parent = r - stride
                    last[r] = [self.map_edge(f"{name}.down{level}.{parent}->{r}", ranks[parent][0], ranks[r][0], size,
                                             last[parent], operator, label)]
        # The in-place buffers stay allocated until the last step on each rank.
        return [self.sync(f"{name}.done{r}", last[r] + list(deps[r]), operator, label) for r in range(p)]
