"""Explicit resident tensors, tiled compute waves and BSP exchange boundaries.

This is an analytical schedule, not Poplar or a TianjicX instruction compiler.
Operands/results reside in tile-private SRAM, with contiguous legacy shards
or capacity-aware blocked layouts. Each wave scatters disjoint output blocks.
Blocked K partials accumulate on the same tile; no cross-tile reduction is used.
"""
import math
from dataclasses import replace
from itertools import islice
import heapq

from arch_explorer.adf.registry import MappingTemplate, register_template
from arch_explorer.components.base import Demand, Kernel, dtype_bytes
from arch_explorer.workload.llm import workload_from_spec
from arch_explorer.workload.operators import VECTOR_COSTS


def _points(node):
    if hasattr(node, "component"):
        yield node
    else:
        for child in node.elements:
            yield from _points(child)


def matrix_jobs(op, tm, tn):
    """Element intervals, output coordinates and MAC shape; independent of hardware."""
    a = op.attrs
    b, k = a.get("batch", 1), a["k"]
    m, n = (a["m"], a["n"]) if op.op == "matmul" else (1, a["m"])
    for batch in range(b):
        for row in range(0, m, tm):
            r = min(tm, m-row)
            for col in range(0, n, tn):
                c = min(tn, n-col)
                if op.op == "matmul":
                    bn = 0 if a.get("shared_b", False) else batch
                    inputs = [(op.inputs[0], [(batch*m*k+row*k, r*k)]),
                              (op.inputs[1], [(bn*k*n+i*n+col, c) for i in range(k)])]
                else:
                    inputs = [(op.inputs[0], [(col*k, c*k)]), (op.inputs[1], [(batch*k, k)])]
                outputs = [(batch*m*n+i*n+col, c) for i in range(row, row+r)]
                yield {"batch": batch, "row": row, "col": col, "rows": r, "cols": c,
                       "shape": (1, r, k, c), "inputs": inputs, "outputs": outputs}


class _Schedule:
    def __init__(self, sim, workload, params, component):
        self.sim, self.ctx, self.w, self.p = sim, sim.mapping, workload, params
        if workload.iterations != 1:
            raise ValueError("manycore explicit program currently requires iterations=1")
        if any(op.op not in ("matmul", "mvm", "scale", "softmax") for op in workload.operators):
            raise NotImplementedError("manycore supports GEMM/MVM/scale/softmax; other operators and collectives are unsupported")
        if component == "ipu_tile" and any(
                op.kernel(workload.dtype).sfu or op.kernel(workload.dtype).row_sfu
                for op in workload.operators if op.op in VECTOR_COSTS):
            raise NotImplementedError(
                "IPU nonlinear/Softmax timing requires a sourced exponential, reciprocal and "
                "FP16 conversion execution cost contract; incomplete arithmetic-only timing refused")
        self.units = sim.find_points(component=component)
        if not self.units or len(sim.find_points(role="compute")) != len(self.units):
            raise ValueError("manycore template requires homogeneous tile compute")
        self.mem = []
        for index, unit in enumerate(self.units):
            parent = unit.path.rsplit("/", 1)[0]
            container = sim.container(parent)
            if container.parent is not sim.root or container.endpoint != index:
                raise ValueError("compute tiles must be ordered direct children of the root exchange")
            memories = sim.find_points(component="sram", within=parent)
            if len(memories) != 1:
                raise ValueError("each compute tile needs exactly one private SRAM")
            self.mem.append(memories[0])
        self.net = sim.root.networks[0] if len(sim.root.networks) == 1 else None
        if self.net is None or self.net.component.name != "tile_exchange" or self.net.tiles != len(self.units):
            raise ValueError("one root tile_exchange must match the private tile count")
        self.remote = sim.find_points(component="dram")
        if workload.meta.get("kv_storage") == "remote" and len(self.remote) != 1:
            raise NotImplementedError("remote KV requires an explicitly sourced remote bandwidth/capacity attachment; target path unknown")
        if component == "tianjic_core":
            if workload.meta.get("attention") or any(op.op not in ("matmul", "mvm") for op in workload.operators):
                raise NotImplementedError("TianjicX template supports int8 GEMM/MVM only")
            for op in workload.operators:
                for name in op.outputs:
                    workload.tensors[name] = replace(workload.tensors[name], dtype="int32")
        if workload.dtype != self.units[0].param("precision"):
            raise NotImplementedError(f"{component} cannot run dtype={workload.dtype}")
        self.count = len(self.units)
        self.bytes = [dict(local_read=0, local_write=0, sent=0, received=0, remote_read=0) for _ in self.units]
        self.resident, self.shards, self.holds = [0]*self.count, {}, []
        self.events, self.jobs = [], []
        self.detail = {
            "estimate": "partial analytical schedule; not full-device latency or paper reproduction",
            "mapping": dict(params), "workload": dict(workload.meta), "stages": self.events,
            "tiles": self.bytes, "jobs": self.jobs,
            "unmodeled": ["instruction issue, compiler layout, SRAM startup, exchange propagation and code/stack allocation"],
            "input_boundary": "ordinary inputs resident and pre-sharded before timing; output remains sharded",
            "network": self.net.topology, "paper_configuration_confirmed": False,
        }
        if self.net.bandwidth is None:
            self.detail["unmodeled"].append("mesh link bandwidth, routing contention and sync latency; communication only pays SRAM service")
        if any(op.kernel(workload.dtype).sfu or op.kernel(workload.dtype).row_sfu
               for op in workload.operators if op.op in VECTOR_COSTS):
            self.detail["unmodeled"].append("softmax exponential and reciprocal timing, FP16 conversions; no fused-kernel equivalence")
        if workload.meta.get("kv_storage") == "remote":
            self.detail["unmodeled"].append("remote command startup and gateway topology; explicit attachment scenario only")
        sim.warnings.append("manycore: " + self.detail["estimate"] + "; " + "; ".join(self.detail["unmodeled"]))
        self.ctx.details["manycore"] = self.detail
        from arch_explorer.simulator.plan import PlanConstraints
        self.ctx.plan.require(PlanConstraints(iterations=1, reason='manycore explicit program'))

    def storage(self):
        for name, tensor in self.w.tensors.items():
            if tensor.kind == "kv" and self.w.meta.get("kv_storage") == "remote":
                self.shards[name] = (tensor.bytes, dtype_bytes(tensor.dtype), True)
                continue
            elements = math.prod(tensor.shape)
            chunk = (elements + self.count-1)//self.count
            width = dtype_bytes(tensor.dtype)
            self.shards[name] = (chunk, width, False)
            for tile in range(self.count):
                size = max(0, min(chunk, elements-tile*chunk))*width
                self.resident[tile] += size
        for tile, size in enumerate(self.resident):
            if size > self.mem[tile].capacity:
                raise ValueError(f"tile {tile} resident capacity exceeded: {size} > {self.mem[tile].capacity} bytes")
            self.holds.append(self.ctx.storage(f"resident.{tile}", self.mem[tile].path, size, static=True))
        if any(remote for _, _, remote in self.shards.values()):
            total = sum(t.bytes for t in self.w.tensors.values() if t.kind == "kv")
            if total > self.remote[0].capacity:
                raise ValueError(f"remote KV capacity exceeded: {total} > {self.remote[0].capacity}")
            self.holds.append(self.ctx.storage("remote.KV", self.remote[0].path, total, static=True))

    def pieces(self, name, ranges):
        chunk, width, remote = self.shards[name]
        if remote:
            return {-1: sum(size for _, size in ranges)*width}
        pieces = {}
        limit = math.prod(self.w.tensors[name].shape)
        for start, size in ranges:
            if start < 0 or size < 0 or start+size > limit:
                raise ValueError("mapping tensor interval outside its shape")
            end = start+size
            while start < end:
                tile = start//chunk
                stop = min(end, (tile+1)*chunk)
                pieces[tile] = pieces.get(tile, 0)+(stop-start)*width
                start = stop
        return pieces

    def move(self, src, dst, size, deps, op, stage):
        if size == 0:
            return None
        if len(self.ctx.graph.tasks) >= 200000:
            raise ValueError("explicit manycore task budget exceeded (200000); choose larger tiles")
        source = self.remote[0] if src == -1 else self.mem[src]
        target = self.mem[dst]
        # Same-tile staging still performs a read and a write on its one SRAM port.
        demand = source.access(size).add(target.access(size))
        if src != dst:
            demand = demand.add(self.net.route(self.count if src == -1 else src, dst).scaled(size))
        task = self.ctx.task(f"{op}.{stage}.{len(self.ctx.graph.tasks)}", demand, deps, "comm", op, stage)
        self.bytes[dst]["local_write"] += size
        if src == -1:
            self.bytes[dst]["remote_read"] += size
        else:
            self.bytes[src]["local_read"] += size
            if src != dst:
                self.bytes[src]["sent"] += size
        if src != dst:
            self.bytes[dst]["received"] += size
        return task

    def barrier(self, deps, op, phase):
        delay = self.units[0].sync_cycles/self.units[0].clock_ghz
        task = self.ctx.sync(f"{op}.{phase}.{len(self.events)}", deps, op, "Sync", latency_ns=delay)
        self.events.append({"phase": phase, "operator": op, "tasks": [task], "fixed_ns": delay})
        return task

    def run(self):
        self.ctx.graph.iterations = self.w.iterations
        self.storage()
        previous = self.barrier(self.holds, "program", "initial_sync")
        for op in self.w.operators:
            if op.op in ("matmul", "mvm"):
                a = op.attrs
                n = a.get("n", a["m"])
                m = a["m"] if op.op == "matmul" else 1
                number = a.get("batch", 1)*math.ceil(m/self.p["tile_m"])*math.ceil(n/self.p["tile_n"])
                if number > 8192:
                    raise ValueError("explicit manycore schedule exceeds 8192 output blocks; use larger tiles")
                jobs = matrix_jobs(op, self.p["tile_m"], self.p["tile_n"])
            else:
                rows, cols = op.attrs["rows"], op.attrs["cols"]
                jobs = ({"row": row, "rows": 1, "cols": cols, "shape": (1, cols),
                         "inputs": [(op.inputs[0], [(row*cols, cols)])], "outputs": [(row*cols, cols)]}
                        for row in range(rows))
            offset = 0
            while wave := list(islice(jobs, self.count)):
                loads, held, per_job = [], [], []
                for tile, job in enumerate(wave):
                    parts = [(name, self.pieces(name, ranges)) for name, ranges in job["inputs"]]
                    read = sum(sum(p.values()) for _, p in parts)
                    write = sum(size for _, size in job["outputs"])*dtype_bytes(self.w.tensors[op.outputs[0]].dtype)
                    if self.resident[tile]+read+write > self.mem[tile].capacity:
                        raise ValueError(f"tile {tile} staging capacity exceeded: resident {self.resident[tile]} + buffers {read+write} > {self.mem[tile].capacity}")
                    hold = self.ctx.storage(f"buffer.{op.name}.{offset+tile}", self.mem[tile].path,
                                            read+write, [previous], op.name)
                    held.append(hold)
                    for _, by_owner in parts:
                        loads += [self.move(src, tile, size, [previous, hold], op.name, "Gather") for src, size in by_owner.items()]
                    per_job.append((tile, job, read, write))
                loads = [task for task in loads if task is not None]
                self.events.append({"phase": "gather", "operator": op.name, "tasks": loads})
                ready = self.ctx.sync(f"{op.name}.gather_done.{offset}", loads or [previous], op.name, "Exchange")
                computes = []
                for tile, job, read, write in per_job:
                    kernel = (Kernel("gemm", job["shape"], self.w.dtype) if op.op in ("matmul", "mvm")
                              else op.kernel(self.w.dtype, rows=1))
                    demand = self.units[tile].kernel_demand(kernel).add(self.mem[tile].access(read+write))
                    task = self.ctx.task(f"{op.name}.compute.{offset+tile}", demand,
                                         [ready, held[tile]], "compute", op.name, "Compute")
                    computes.append(task)
                    self.bytes[tile]["local_read"] += read
                    self.bytes[tile]["local_write"] += write
                    self.jobs.append({"operator": op.name, "tile": tile, "shape": list(job["shape"]),
                                      "input_bytes": read, "output_bytes": write, "task": task,
                                      "isolated_compute_ns": self.units[tile].kernel_demand(kernel).resources[self.units[tile].unit.name]})
                self.events.append({"phase": "compute", "operator": op.name, "tasks": computes})
                compute_done = self.barrier(computes, op.name, "compute_sync")
                scatters = []
                for tile, job, _, _ in per_job:
                    for dst, size in self.pieces(op.outputs[0], job["outputs"]).items():
                        scatters.append(self.move(tile, dst, size, [compute_done, held[tile]], op.name, "Scatter"))
                self.events.append({"phase": "scatter", "operator": op.name, "tasks": scatters})
                previous = self.barrier(scatters or [compute_done], op.name, "exchange_sync")
                if len(self.ctx.graph.tasks) > 200000:
                    raise ValueError("explicit manycore task budget exceeded (200000); choose larger tiles")
                offset += len(wave)
        self.detail["resident_bytes"] = self.resident
        self.detail["isolated_service_ns"] = [
            {"sram": (b["local_read"]+b["local_write"])/self.mem[i].port.capacity,
             "send": b["sent"]/self.net.bandwidth if self.net.bandwidth else None,
             "receive": b["received"]/self.net.bandwidth if self.net.bandwidth else None}
            for i, b in enumerate(self.bytes)]
        self.detail["remote_service_ns"] = (sum(b["remote_read"] for b in self.bytes)/self.remote[0].port.capacity
                                             if self.w.meta.get("kv_storage") == "remote" else None)
        self.detail["parameters"] = {u.type_path: {k: v.to_dict() for k, v in u.point.params.items()}
                                     for u in [self.units[0], self.mem[0], self.net]}


class IpuBsp(MappingTemplate):
    name = "ipu_bsp"
    component = "ipu_tile"

    def applicable(self, adf, workload):
        return workload.kind == "operator" and any(p.component == self.component for p in _points(adf))

    def space(self, adf, workload):
        return {"tile_m": [1, 4, 16], "tile_n": [32, 128, 256],
                "tile_k": [32, 64, 128], "schedule": ["blocked"]}

    def apply(self, simulator, workload, params):
        if params.get("schedule", "legacy") == "blocked":
            p = dict(tile_m=16, tile_n=128, tile_k=128, schedule="blocked")
            p.update(params)
            if any(type(p[k]) is not int or p[k] < 1 for k in ("tile_m", "tile_n", "tile_k")):
                raise ValueError("manycore tile sizes must be positive integers")
            BlockedSchedule(simulator, workload_from_spec(workload), p, self.component).run()
            return
        if "tile_k" in params:
            raise ValueError("tile_k requires schedule='blocked'; legacy replay is unchanged")
        p = dict(tile_m=16, tile_n=128)
        p.update({k: v for k, v in params.items() if k != "schedule"})
        if any(type(v) is not int or v < 1 for v in p.values()):
            raise ValueError("manycore tile sizes must be positive integers")
        _Schedule(simulator, workload_from_spec(workload), p, self.component).run()


class TianjicSpatial(IpuBsp):
    name = "tianjic_spatial"
    component = "tianjic_core"


register_template(IpuBsp)
register_template(TianjicSpatial)


class IpuSplitK(IpuBsp):
    """Explicit FP32 tree reduction; separate template preserves older spaces."""
    name = "ipu_split_k"

    def space(self, adf, workload):
        w = workload_from_spec(workload)
        blocks = max((math.ceil(op.attrs["k"]/32) for op in w.operators
                      if op.op in ("matmul", "mvm")), default=1)
        tiles = adf.networks[0].params["tiles"].value
        return {"tile_m": [1, 4, 16], "tile_n": [32, 128, 256], "tile_k": [32, 64, 128],
                "partitions": [p for p in (1, 2, 3, 4, 8, 16, 32) if p <= min(tiles, blocks)],
                "reduction": ["tree"]}

    def apply(self, simulator, workload, params):
        from arch_explorer.mapping.manycore_split import SplitSchedule
        p = dict(tile_m=16, tile_n=128, tile_k=128, partitions=1, reduction="tree")
        p.update(params)
        if any(type(p[k]) is not int or p[k] < 1 for k in ("tile_m", "tile_n", "tile_k", "partitions")):
            raise ValueError("split-K sizes and partitions must be positive integers")
        if p["reduction"] != "tree":
            raise ValueError("split-K supports only explicit binary tree reduction")
        SplitSchedule(simulator, workload_from_spec(workload), p, self.component).run()


register_template(IpuSplitK)


def output_blocks(op, tm, tn):
    """Logical output rectangles; no K-sized interval lists."""
    a = op.attrs
    m, n = (a["m"], a["n"]) if op.op == "matmul" else (1, a["m"])
    for b in range(a.get("batch", 1)):
        for r in range(0, m, tm):
            for c in range(0, n, tn):
                yield (b, r, min(tm, m-r), c, min(tn, n-c))


def block_inputs(op, job, start, length):
    """(tensor, (batch,row,rows,col,cols)) source rectangles for a partial dot."""
    b, r, nr, c, nc = job
    if op.op == "matmul":
        return [(op.inputs[0], (b, r, nr, start, length)),
                (op.inputs[1], (0 if op.attrs.get("shared_b", False) else b, start, length, c, nc))]
    return [(op.inputs[0], (0, c, nc, start, length)),
            (op.inputs[1], (0, b, 1, start, length))]


def block_output(op, job):
    b, r, nr, c, nc = job
    return job if op.op == "matmul" else (0, b, 1, c, nc)


class BlockLayout:
    """Rectangular logical tiles, with exact element-aligned physical segments.

    Segments may split a logical tile when the free space is fragmented. Prefix
    rectangle intersection avoids enumerating strided rows/elements.
    """
    def __init__(self, tensor, bm, bn):
        self.b = math.prod(tensor.shape[:-2])
        self.m, self.n = tensor.shape[-2:]
        self.bm, self.bn, self.width = bm, bn, dtype_bytes(tensor.dtype)
        self.blocks = {}

    def specs(self):
        for b in range(self.b):
            for r in range(0, self.m, self.bm):
                for c in range(0, self.n, self.bn):
                    rows, cols = min(self.bm, self.m-r), min(self.bn, self.n-c)
                    yield (b, r//self.bm, c//self.bn), rows*cols*self.width

    @staticmethod
    def _prefix(pos, stride, r0, r1, c0, c1):
        row, col = divmod(pos, stride)
        return max(0, min(row, r1)-r0)*(c1-c0) + (max(0, min(col, c1)-c0) if r0 <= row < r1 else 0)

    def pieces(self, rect):
        b, r, nr, c, nc = rect
        if b < 0 or b >= self.b or min(r, c, nr, nc) < 0 or r+nr > self.m or c+nc > self.n:
            raise ValueError("blocked tensor rectangle outside its shape")
        if not nr or not nc:
            return {}
        result = {}
        for br in range(r//self.bm, (r+nr+self.bm-1)//self.bm):
            rr = br*self.bm
            r0, r1 = max(r, rr)-rr, min(r+nr, rr+self.bm)-rr
            for bc in range(c//self.bn, (c+nc+self.bn-1)//self.bn):
                cc = bc*self.bn
                stride = min(self.bn, self.n-cc)
                c0, c1 = max(c, cc)-cc, min(c+nc, cc+self.bn)-cc
                parts = self.blocks[b, br, bc]
                if len(parts) == 1:
                    tile = parts[0][2]
                    result[tile] = result.get(tile, 0)+(r1-r0)*(c1-c0)*self.width
                else:
                    for start, length, tile in parts:
                        cells = (self._prefix((start+length)//self.width, stride, r0, r1, c0, c1)
                                 - self._prefix(start//self.width, stride, r0, r1, c0, c1))
                        if cells:
                            result[tile] = result.get(tile, 0)+cells*self.width
        return result


class BlockedSchedule(_Schedule):
    """Exact explicit events, tiled placement and same-tile SRAM partial sums."""
    def __init__(self, sim, workload, params, component):
        super().__init__(sim, workload, params, component)
        # Ports are lazy in the common builder. Materialize this dedicated
        # network once, avoiding O(tiles^2) whole-simulator resource rebuilds.
        if self.net.bandwidth is not None:
            for i in range(self.count):
                self.net.port(i, "tx")
                self.net.port(i, "rx")
        self.ctx._resources = sim.resources
        self.detail.update(schedule="blocked", placement="rectangular tiles; capacity-aware physical segments",
                           accumulator="fp32" if component == "ipu_tile" else "int32",
                           partial_blocks=0, macs=0, accumulator_bytes_written=0,
                           accumulator_bytes_read=0, capacity_peak_bytes=[0]*self.count,
                           fp32_store_rate="GC200 HC2021 slide13: f16 multiply/f32 memory half f16-memory rate" if component == "ipu_tile" else None)
        self.detail["unmodeled"].append("final accumulator conversion instruction timing; explicit SRAM bytes only")
        sim.warnings.append("manycore blocked: final accumulator conversion instruction timing omitted; no intermediate fp16 rounding")

    def matrix_workspace(self, nr, nc, kk):
        return (nr+nc)*kk*dtype_bytes(self.w.dtype)+nr*nc*4

    def _layouts(self):
        tm, tn, tk = (self.p[k] for k in ("tile_m", "tile_n", "tile_k"))
        shapes = {}
        self.workspace = 0
        for op in self.w.operators:
            if op.op in ("matmul", "mvm"):
                a = op.attrs
                nr = min(tm, a["m"]) if op.op == "matmul" else 1
                nc = min(tn, a["n"] if op.op == "matmul" else a["m"])
                kk = min(tk, a["k"])
                # Reuse the accumulator's buffer for its final equal/narrower result.
                self.workspace = max(self.workspace, self.matrix_workspace(nr, nc, kk))
                shapes.setdefault(op.inputs[0], (tm, tk) if op.op == "matmul" else (tn, tk))
                shapes.setdefault(op.inputs[1], (tk, tn) if op.op == "matmul" else (1, tk))
                shapes.setdefault(op.outputs[0], (tm, tn) if op.op == "matmul" else (1, tn))
            else:
                self.workspace = max(self.workspace, op.attrs["cols"]*2*dtype_bytes(self.w.dtype))
                shapes.setdefault(op.inputs[0], (1, tn))
                shapes.setdefault(op.outputs[0], (1, tn))
        self.workspace = math.ceil(self.workspace/4)*4
        layouts, specs, remote_bytes = {}, [], 0
        for name, tensor in self.w.tensors.items():
            if tensor.kind == "kv" and self.w.meta.get("kv_storage") == "remote":
                remote_bytes += tensor.bytes
                layouts[name] = None
                continue
            layout = layouts[name] = BlockLayout(tensor, *shapes[name])
            for key, size in layout.specs():
                specs.append((name, key, size, layout.width))
        total = sum(size for _, _, size, _ in specs)
        capacity = sum(int(m.capacity) for m in self.mem)
        if total > capacity:
            raise ValueError(f"physical resident tensor capacity exceeded: {total} > {capacity} bytes; all-tensor resident policy")
        # Largest/type-aligned blocks first. Physical segmentation prevents
        # logical tile granularity from masquerading as physical capacity.
        specs.sort(key=lambda item: (-item[3], -item[2], item[0], item[1]))
        active = min(self.count, (capacity-total)//self.workspace)
        if active < 1:
            raise ValueError(f"resident plus one blocked workspace exceeds total capacity: {total} + {self.workspace} > {capacity}; choose smaller tile sizes")
        while active:
            free = [int(m.capacity)-(self.workspace if i < active else 0) for i, m in enumerate(self.mem)]
            if min(free) < 0:
                raise ValueError("blocked workspace exceeds per-tile capacity; choose smaller tile sizes")
            heap = [(-n, i) for i, n in enumerate(free)]
            heapq.heapify(heap)
            resident = [0]*self.count
            success = True
            for name, key, size, width in specs:
                parts, offset = [], 0
                while offset < size:
                    negative, tile = heapq.heappop(heap)
                    take = min(-negative, size-offset)//width*width
                    if take <= 0:
                        success = False
                        break
                    parts.append((offset, take, tile))
                    offset += take
                    resident[tile] += take
                    heapq.heappush(heap, (negative+take, tile))
                layouts[name].blocks[key] = parts
                if not success:
                    break
            if success:
                break
            active //= 2
        if not active:
            raise ValueError("capacity-aware layout cannot place resident tensors plus one workspace")
        self.layouts, self.active, self.resident = layouts, active, resident
        self.detail.update(active_tiles=active, workspace_bytes_per_active_tile=self.workspace,
                           resident_bytes=resident, layout_blocks=len(specs),
                           physical_segments=sum(len(x.blocks[k]) for x in layouts.values() if x for k in x.blocks))
        self.holds = [self.ctx.storage(f"resident.{i}", self.mem[i].path, size, static=True) for i, size in enumerate(resident)]
        if remote_bytes:
            if remote_bytes > self.remote[0].capacity:
                raise ValueError("remote KV capacity exceeded")
            self.holds.append(self.ctx.storage("remote.KV", self.remote[0].path, remote_bytes, static=True))

    def rect(self, name, rect):
        layout = self.layouts[name]
        return layout.pieces(rect) if layout else {-1: rect[2]*rect[4]*dtype_bytes(self.w.tensors[name].dtype)}

    def local(self, tile, read, write, deps, op, label, kernel=None):
        demand = self.mem[tile].access(read+write)
        if kernel is not None:
            demand = demand.add(self.units[tile].kernel_demand(kernel))
        self.bytes[tile]["local_read"] += read
        self.bytes[tile]["local_write"] += write
        task = self.ctx.task(f"{op}.{label}.{len(self.ctx.graph.tasks)}", demand, deps,
                             "compute" if kernel else "comm", op, label)
        return task

    def run(self):
        self._layouts()
        previous = self.barrier(self.holds, "program", "initial_sync")
        tm, tn, tk = (self.p[k] for k in ("tile_m", "tile_n", "tile_k"))
        for op in self.w.operators:
            matrix = op.op in ("matmul", "mvm")
            jobs = list(output_blocks(op, tm, tn)) if matrix else [(r, 0, 1, 0, op.attrs["cols"]) for r in range(op.attrs["rows"])]
            if len(jobs) > 8192:
                raise ValueError("explicit manycore schedule exceeds 8192 output blocks; use larger tiles")
            for offset in range(0, len(jobs), self.active):
                wave = jobs[offset:offset+self.active]
                held = [self.ctx.storage(f"buffer.{op.name}.{offset+i}", self.mem[i].path, self.workspace,
                                         [previous], op.name) for i in range(len(wave))]
                for i in range(len(wave)):
                    self.detail["capacity_peak_bytes"][i] = max(self.detail["capacity_peak_bytes"][i], self.resident[i]+self.workspace)
                if matrix:
                    init = [self.local(i, 0, j[2]*j[4]*4, [previous, held[i]], op.name, "AccumulatorInit") for i, j in enumerate(wave)]
                    previous = self.ctx.sync(f"{op.name}.init.{offset}", init, op.name, "Init")
                    self.detail["accumulator_bytes_written"] += sum(j[2]*j[4]*4 for j in wave)
                starts = range(0, op.attrs["k"], tk) if matrix else [0]
                for start in starts:
                    length = min(tk, op.attrs["k"]-start) if matrix else 0
                    loads, read_bytes = [], []
                    for tile, job in enumerate(wave):
                        if matrix:
                            inputs = block_inputs(op, job, start, length)
                        else:
                            tensor = self.w.tensors[op.inputs[0]]
                            row = job[0]
                            rect = (row//tensor.shape[-2], row % tensor.shape[-2], 1, 0, job[4])
                            inputs = [(op.inputs[0], rect)]
                        read = 0
                        for name, rect in inputs:
                            for src, size in self.rect(name, rect).items():
                                read += size
                                loads.append(self.move(src, tile, size, [previous, held[tile]], op.name, "Gather"))
                        read_bytes.append(read)
                    self.events.append({"phase": "gather", "operator": op.name, "tasks": loads})
                    ready = self.ctx.sync(f"{op.name}.gather.{offset}.{start}", loads, op.name, "Exchange")
                    computes = []
                    for tile, job in enumerate(wave):
                        cells = job[2]*job[4]
                        if matrix:
                            accumulator = cells*4
                            kernel = Kernel("gemm_accumulate", (1, job[2], length, job[4]), self.w.dtype)
                            read, write = read_bytes[tile]+accumulator, accumulator
                            self.detail["partial_blocks"] += 1
                            self.detail["macs"] += cells*length
                            self.detail["accumulator_bytes_read"] += accumulator
                            self.detail["accumulator_bytes_written"] += accumulator
                        else:
                            kernel = op.kernel(self.w.dtype, rows=1)
                            read, write = read_bytes[tile], cells*dtype_bytes(self.w.dtype)
                        task = self.local(tile, read, write, [ready, held[tile]], op.name, "Compute", kernel)
                        computes.append(task)
                        self.jobs.append({"operator": op.name, "tile": tile, "shape": list(kernel.shape),
                                          "k_start": start, "input_bytes": read_bytes[tile], "output_bytes": write,
                                          "accumulator_read_bytes": cells*4 if matrix else 0, "task": task,
                                          "isolated_compute_ns": self.units[tile].kernel_demand(kernel).resources[self.units[tile].unit.name]})
                    self.events.append({"phase": "compute", "operator": op.name, "tasks": computes})
                    previous = self.barrier(computes, op.name, "compute_sync")
                if matrix and dtype_bytes(self.w.tensors[op.outputs[0]].dtype) < 4:
                    # Forward in-place narrowing after every K block has completed.
                    finals = [self.local(i, j[2]*j[4]*4, j[2]*j[4]*2, [previous, held[i]], op.name, "FinalStore") for i, j in enumerate(wave)]
                    self.detail["accumulator_bytes_read"] += sum(j[2]*j[4]*4 for j in wave)
                    previous = self.ctx.sync(f"{op.name}.final.{offset}", finals, op.name, "FinalStore")
                scatters = []
                for tile, job in enumerate(wave):
                    if matrix:
                        rect = block_output(op, job)
                    else:
                        t = self.w.tensors[op.outputs[0]]
                        rect = (job[0]//t.shape[-2], job[0] % t.shape[-2], 1, 0, job[4])
                    for dst, size in self.rect(op.outputs[0], rect).items():
                        scatters.append(self.move(tile, dst, size, [previous, held[tile]], op.name, "Scatter"))
                self.events.append({"phase": "scatter", "operator": op.name, "tasks": scatters})
                previous = self.barrier(scatters, op.name, "exchange_sync")
                if len(self.ctx.graph.tasks) > 200000:
                    raise ValueError("explicit manycore task budget exceeded (200000); choose larger tiles")
        self.detail["capacity_peak_bytes"] = [max(p, r) for p, r in zip(self.detail["capacity_peak_bytes"], self.resident)]
        self.detail["isolated_service_ns"] = [
            {"sram": (b["local_read"]+b["local_write"])/self.mem[i].port.capacity,
             "send": b["sent"]/self.net.bandwidth if self.net.bandwidth else None,
             "receive": b["received"]/self.net.bandwidth if self.net.bandwidth else None}
            for i, b in enumerate(self.bytes)]
        self.detail["remote_service_ns"] = (sum(b["remote_read"] for b in self.bytes)/self.remote[0].port.capacity
                                             if self.w.meta.get("kv_storage") == "remote" else None)
        self.detail["parameters"] = {u.type_path: {k: v.to_dict() for k, v in u.point.params.items()}
                                     for u in [self.units[0], self.mem[0], self.net, *self.remote]}
