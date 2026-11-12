"""Conservative scalar, output-stationary CGRA DFG; not an RTL control-word compiler.

All schedule choices are independent of reference timings. See docs/cgra.md
for configuration-packet assumptions and the boundary of this timing model.
"""
from arch_explorer.adf.registry import MappingTemplate, register_template
from arch_explorer.components.base import Demand, Kernel
from arch_explorer.simulator.plan import PlanConstraints


def layers(workload):
    if workload.kind != "operator":
        raise ValueError("CGRA requires an operator workload")
    p = workload.params
    op = p.get("op", p.get("operator", "matmul"))
    if (p.get("dtype", p.get("precision", "uint32")) != "uint32"
            or type(p.get("batch", 1)) is not int or p.get("batch", 1) != 1):
        raise ValueError("CGRA requires uint32 and batch=1")
    m, k, n = p.get("m"), p.get("k"), p.get("n")
    if op == "mvm" and n is None:
        # Public MVM API: A[m,k] times x[k]; paper triples use [1,k] times B[k,n].
        n = 1
    if op == "linear2":
        if not isinstance(n, (list, tuple)) or len(n) != 2:
            raise ValueError("linear2 needs n=[hidden, output]; no bias or activation")
        result = [(m, k, n[0]), (m, n[0], n[1])]
    elif op in ("matmul", "gemm", "mvm"):
        result = [(m, k, n)]
    else:
        raise ValueError(f"CGRA does not support {op}")
    if any(type(x) is not int or x <= 0 for shape in result for x in shape):
        raise ValueError("CGRA matrix dimensions must be positive integers")
    return result


def _points(node):
    if hasattr(node, "component"):
        yield node
    else:
        for element in node.elements:
            yield from _points(element)


class CgraSpatial(MappingTemplate):
    name = "cgra_spatial"

    def applicable(self, adf, workload):
        try:
            layers(workload)
        except (ValueError, TypeError):
            return False
        return any(p.component == "cgra_pe" for p in _points(adf))

    def space(self, adf, workload):
        # II is minimum scalar issue spacing, NOT the unpublished compiler loop II.
        return {"ii": [1, 2, 4], "placement": ["row_major", "column_major"],
                "routing": ["xy", "yx"], "memory_port": ["striped", "nearest"],
                "config_words": [17]}

    def apply(self, simulator, workload, params):
        simulator.mapping.plan.require(PlanConstraints(iterations=1, reason='CGRA cycle boundaries'))
        choices = dict(ii=1, placement="row_major", routing="xy", memory_port="striped", config_words=17)
        choices.update(params)
        if type(choices["ii"]) is not int or type(choices["config_words"]) is not int:
            raise ValueError("ii/config_words must be integers")
        schedule = _Schedule(simulator, choices)
        schedule.build(layers(workload))
        simulator.mapping.cycle_report = schedule.report
        simulator.mapping.plan.detail_reporters.append(lambda sim, result: schedule.report(result))


class _Schedule:
    def __init__(self, sim, choices):
        self.sim, self.ctx, self.p = sim, sim.mapping, choices
        self.pes = sim.find_points(component="cgra_pe")
        self.banks = sim.find_points(component="sram")
        if (len(self.pes) != 16 or len(self.banks) != 2
                or len(sim.find_points(role="compute")) != 16):
            raise ValueError("cgra_spatial requires exactly 16 PEs and two SRAM banks")
        parent = sim.container(self.pes[0].path.rsplit("/", 1)[0])
        if any(pe.path.rsplit("/", 1)[0] != parent.path for pe in self.pes) or len(parent.networks) != 1:
            raise ValueError("CGRA PEs must share a single array/mesh container")
        self.mesh = parent.networks[0]
        if self.mesh.component.name != "mesh2d" or (self.mesh.rows, self.mesh.cols) != (4, 4):
            raise ValueError("cgra_spatial requires a 4x4 mesh")
        if [pe.endpoint for pe in self.pes] != list(range(16)):
            raise ValueError("CGRA mesh endpoints must be the 16 row-major PEs only")
        self.clock = self.pes[0].clock_ghz
        if any(pe.clock_ghz != self.clock for pe in self.pes):
            raise ValueError("CGRA requires one clock domain")
        if self.mesh.link_bandwidth != 4 * self.clock or self.mesh.hop_latency != 1 / self.clock:
            raise ValueError("CGRA mesh must carry one uint32 per tick with one tick per hop")
        self.bank_words = int(self.banks[0].capacity / 4)
        if (self.bank_words < 1 or self.bank_words & (self.bank_words - 1)
                or any(b.capacity != 4 * self.bank_words for b in self.banks)):
            raise ValueError("CGRA banks need equal power-of-two word capacities")
        if any(b.port.capacity != 4 * self.clock or b.latency not in (0, 1 / self.clock) for b in self.banks):
            raise ValueError("CGRA SRAM needs one word/tick and 0/1 tick latency")
        self.ports = {(i, direction): b.resource(f"cgra_{direction}", 4 * self.clock, "B/ns", arbitration="fifo")
                      for i, b in enumerate(self.banks) for direction in ("read", "write")}
        self.cpu = self.banks[0].resource("cgra_cpu_packet", self.clock, "packet/ns", arbitration="fifo")
        self.last_pe, self.last_link = {}, {}
        self.counts = dict(add=0, mul=0, load=0, store=0, routed_words=0, link_hops=0)
        self.stages = []
        self.serial = 0
        sim.warnings.append("CGRA is an analytic scalar DFG scenario, not the paper's unpublished compiled schedule; "
                            "configuration assumes ready sinks and serialized stage barriers (docs/cgra.md).")
        if any("unknown" in pe.point.params["clock_ghz"].source.lower() for pe in self.pes):
            sim.warnings.append("CGRA physical clock unknown: latency_ns uses cycle-normalized reporting units.")

    def task(self, name, demand=Demand(), deps=(), kind="comm", phase="compute"):
        self.serial += 1
        return self.ctx.task(f"cgra.{self.serial}.{name}", demand, deps, kind, phase, phase)

    def issue_deps(self, tile, deps):
        deps = list(deps)
        if tile in self.last_pe:
            last = self.last_pe[tile]
            if self.p["ii"] > 1:
                last = self.task("issue_spacing", Demand({}, (self.p["ii"] - 1) / self.clock), [last], "sync")
            deps.append(last)
        return deps

    def operation(self, tile, op, deps, address=None):
        pe = self.pes[tile]
        demand = pe.kernel_demand(Kernel(op, (1,), "uint32"))
        if address is not None:
            bank = address // self.bank_words
            direction = "read" if op == "load" else "write"
            demand.resources[self.ports[bank, direction].name] = 4
            demand.latency_ns = self.banks[bank].latency
        ident = self.task(f"tile{tile}.{op}", demand, self.issue_deps(tile, deps), "compute")
        self.ctx.graph.tasks[ident].meta.update(tile=tile, op=op, address=address)
        self.last_pe[tile] = ident
        self.counts[op] += 1
        return ident

    def route(self, src, dst, dep):
        r, c = divmod(src, 4)
        rr, cc = divmod(dst, 4)
        self.counts["routed_words"] += int(src != dst)
        while (r, c) != (rr, cc):
            nr, nc = r, c
            if self.mesh.diagonal and r != rr and c != cc:
                nr, nc = r + (1 if rr > r else -1), c + (1 if cc > c else -1)
            elif (self.p["routing"] == "xy" and c != cc) or r == rr:
                nc += 1 if cc > c else -1
            else:
                nr += 1 if rr > r else -1
            link = self.mesh.link((r, c), (nr, nc))
            deps = [dep] + ([self.last_link[link.name]] if link.name in self.last_link else [])
            # Word serialization and the one-tick registered input are the SAME tick.
            dep = self.task(f"link{(r,c)}->{(nr,nc)}", Demand({link.name: 4}), deps)
            self.ctx.graph.tasks[dep].meta["value_task"] = deps[0]
            self.last_link[link.name] = dep
            self.counts["link_hops"] += 1
            r, c = nr, nc
        return dep

    def edge(self, tile, address):
        if self.p["memory_port"] == "striped":
            return (0, 1, 2, 3, 4, 8, 12)[address % 7]
        r, c = divmod(tile, 4)
        # RTL connects row y=0 and column x=0; tie goes to row y=0.
        return c if r <= c else 4 * r

    def build(self, shapes):
        # Disjoint contiguous resident buffers. Layer 2 reads layer 1's exact SRAM addresses.
        m, k, _ = shapes[0]
        cursor, a = m * k, 0
        buffers = []
        preload_words = cursor
        for m, k, n in shapes:
            b, c = cursor, cursor + k * n
            cursor = c + m * n
            buffers.append((a, b, c))
            preload_words += k * n
            a = c
        if cursor > 2 * self.bank_words:
            raise ValueError(f"CGRA resident buffers need {cursor} words, available {2*self.bank_words}; "
                             "select an explicit larger words_per_bank scenario")
        storage = [self.ctx.storage(f"cgra.bank{i}.resident", bank.path,
                                   4 * max(0, min(self.bank_words, cursor - i*self.bank_words)), static=True)
                   for i, bank in enumerate(self.banks)]
        preload = self.task("host_preload", Demand({self.cpu.name: preload_words}), storage, phase="data_load")
        self.stages.append(("data_load", preload))
        injected, received = preload, []
        # Explicit configuration scenario: two count packets, N control words, launch per tile.
        # Constants/prologues are not inferred from arithmetic; see limitations in docs/cgra.md.
        for tile in range(16):
            for _ in range(self.p["config_words"] + 3):
                injected = self.task("control.inject", Demand({self.cpu.name: 1}), [injected], phase="config")
                distance = min(tile + 1, 16 - tile)  # bidirectional shortest path from controller 16
                received.append(self.task("control.deliver", Demand({}, distance / self.clock),
                                          [injected], phase="config"))
        ready = self.task("configured", deps=received, kind="sync", phase="config")
        self.stages.append(("config", ready))
        for layer, ((m, k, n), (a, b, c)) in enumerate(zip(shapes, buffers)):
            finished = []
            # Serialize outputs per tile, so an accumulator and two operands fit the RTL registers.
            # Memory request issue shares the boundary tile's scalar FU with arithmetic there.
            for i in range(m):
                for j in range(n):
                    ordinal = i*n+j if self.p["placement"] == "row_major" else j*m+i
                    tile = ordinal % 16
                    previous = self.last_pe.get(tile, ready)
                    for t in range(k):
                        ax, by = a + i*k+t, b + t*n+j
                        ex, ey = self.edge(tile, ax), self.edge(tile, by)
                        x = self.operation(ex, "load", [ready, previous], ax)
                        x = self.route(ex, tile, x)
                        y = self.operation(ey, "load", [ready, previous], by)
                        y = self.route(ey, tile, y)
                        product = self.operation(tile, "mul", [x, y])
                        self.ctx.graph.tasks[product].meta["operands"] = [x, y]
                        acc = self.operation(tile, "add", [product, previous])
                        self.ctx.graph.tasks[acc].meta["operands"] = [product, previous if t else None]
                        previous = acc
                    edge = self.edge(tile, c + i*n+j)
                    data = self.route(tile, edge, previous)
                    stored = self.operation(edge, "store", [data], c + i*n+j)
                    self.ctx.graph.tasks[stored].meta["value_task"] = data
                    finished.append(stored)
            ready = self.task(f"layer{layer}.done", deps=finished, kind="sync")
        self.stages.append(("compute", ready))
        # A mapped return at tile 0: one injection tick + one shortest-path hop to controller 16.
        done = self.task("complete", Demand({}, 2 / self.clock), [ready], phase="completion")
        self.stages.append(("completion", done))
        self.metadata = {"mapping": self.p, "operations": self.counts, "resident_words": cursor,
                         "preload_words": preload_words, "bank_words": self.bank_words,
                         "buffers": [{"a": a, "b": b, "c": c, "shape": list(shape)}
                                     for (a, b, c), shape in zip(buffers, shapes)],
                         "control_packets": 16*(self.p["config_words"]+3),
                         "clock_ghz": self.clock, "paper_configuration_confirmed": False}

    def report(self, result):
        ends = {record.task: record.end_ns for record in result.records}
        previous, cycles = 0.0, {}
        for phase, ident in self.stages:
            now = ends[ident] * self.clock
            cycles[phase] = now - previous
            previous = now
        cycles["end_to_end"] = previous
        return {"cycles": cycles, "cgra": self.metadata}


register_template(CgraSpatial)
