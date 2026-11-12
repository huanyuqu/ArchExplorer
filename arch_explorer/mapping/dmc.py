"""``dmc_partition``: distributed many-core mapping (output partitioning, mesh traffic).

A core is a compute point together with the SRAM in its own container. Every
operator is partitioned across cores along the rows (flattened leading
dimensions) or columns of its output:

* ``split = "n"``: core c computes a contiguous slice of output columns of a
  GEMM and needs every input row; ``split = "m"``: core c computes a slice of
  output rows and needs the full weight. MVM and vector operators split rows.
* ``weights = "local"``: each core keeps the weight part it uses resident in
  its SRAM for the whole run; ``weights = "stream"``: weights stream from DRAM
  over the network while the core computes; ``weights = "bonded"`` (offered
  when the ADF has ``bonded`` DRAM, e.g. from ``bond_memory``): each core's
  weight part stays in the bonded DRAM shard in its own container and streams
  from there, through that shard's port only.
* ``activations = "local"``: outputs stay in the producing cores' SRAM and
  consumers fetch the pieces they need from those cores over the network;
  ``activations = "dram"``: outputs are written back to DRAM and consumers read
  them from there.

Fetched pieces are copied into a staging buffer in the consumer's SRAM before
it computes (``map_edge``; the buffer is a storage task, so it counts toward
the SRAM capacity until the compute finishes); the compute task then streams
its SRAM traffic. For systolic arrays that traffic is the element accesses
SCALE-Sim reports; for other units it is the operand and result bytes.

Which part of a producer's output a consumer needs is matched by position in
the flattened tensor: a consumer's row range and a producer's row range are
compared as fractions of their tensors (the two may index rows differently,
e.g. tokens vs. batch x heads); a column-partitioned producer contributes its
column share of every needed row. A K/V cache operand is half K and half V;
each core fetches only the half and the part (columns or batch entries) it
uses.
"""
from __future__ import annotations

from arch_explorer.adf.model import SpaceMatrix, SpacePoint
from arch_explorer.adf.registry import COMPONENTS, register_template
from arch_explorer.components.base import Kernel, dtype_bytes
from arch_explorer.components.compute import SystolicArrayModel
from arch_explorer.mapping.base import DeviceState, PlacementTemplate, is_memory_group
from arch_explorer.workload.operators import VECTOR_COSTS


def slices(total, parts):
    """Contiguous near-equal ``(start, size)`` slices; parts beyond ``total`` get none."""
    parts = min(parts, total)
    base, extra = divmod(total, parts)
    result, start = [], 0
    for i in range(parts):
        size = base + (1 if i < extra else 0)
        result.append((start, size))
        start += size
    return result


def _cores_have_local_sram(node):
    if isinstance(node, SpacePoint) or is_memory_group(node):     # PIM units in a DRAM group are not cores
        return True
    points = [e for e in node.elements if isinstance(e, SpacePoint)]
    roles = {COMPONENTS.get(p.component).role for p in points}
    if "compute" in roles and not any(p.component == "sram" for p in points):
        return False
    return all(_cores_have_local_sram(e) for e in node.elements if isinstance(e, SpaceMatrix))


def _has_bonded(node):
    if isinstance(node, SpacePoint):
        param = node.params.get("technology")
        return node.component == "dram" and param is not None and param.value == "bonded"
    return any(_has_bonded(e) for e in node.elements)


class DmcState(DeviceState):
    """Cores are the containers holding compute points; ``cores[i] = (first compute, local SRAM)``."""

    def __init__(self, sim, container):
        super().__init__(sim, container)
        groups = {}
        for compute in self.computes:
            groups.setdefault(compute.rsplit("/", 1)[0], []).append(compute)
        self.cores, self.units, self.bonded = [], [], []
        for parent, computes in groups.items():
            local = [p for p in sim.find_points(component="sram", within=parent)
                     if p.path.rsplit("/", 1)[0] == parent]
            if not local:
                raise ValueError(f"{parent} has compute but no SRAM of its own")
            self.cores.append((computes[0], local[0].path))
            self.units.append(computes)
            bonded = [p.path for p in sim.find_points(component="dram", within=parent)
                      if p.path.rsplit("/", 1)[0] == parent and p.technology == "bonded"]
            self.bonded.append(bonded[0] if bonded else None)
        self.location = {}     # tensor -> ("dram",) or ("local", axis, [(core, start, size)], total)
        self.resident = set()

    def unit(self, core, kernel):
        """The first compute point of the core that can execute the kernel."""
        errors = []
        for compute in self.units[core]:
            try:
                self.sim.points[compute].kernel_demand(kernel)
                return compute
            except NotImplementedError as error:
                errors.append(str(error))
        raise NotImplementedError("; ".join(errors))


class DmcPartition(PlacementTemplate):
    name = "dmc_partition"

    def space(self, adf, workload) -> dict:
        weights = ["local", "stream"] + (["bonded"] if _has_bonded(adf) else [])
        return {"split": ["n", "m"], "weights": weights, "activations": ["local", "dram"]}

    def defaults(self):
        return {"split": "n", "weights": "local", "activations": "local"}

    def applicable(self, adf, workload) -> bool:
        if not (super().applicable(adf, workload) and _cores_have_local_sram(adf)):
            return False
        # DMC's partition/weight-cache contract has explicit matrix/vector
        # layouts. A registered semantic name does not establish such a layout;
        # use the public execution_graph policy for selected target programs.
        from arch_explorer.workload.api import workload_from_spec
        ir = workload_from_spec(workload)
        return all(op.op in ('matmul', 'mvm', 'allreduce', *VECTOR_COSTS) for op in ir.operators)

    def apply(self, simulator, workload, params):
        from arch_explorer.workload.api import workload_from_spec
        ir = workload_from_spec(workload)
        unsupported = [op.name for op in ir.operators
                       if op.op not in ('matmul', 'mvm', 'allreduce', *VECTOR_COSTS)]
        if unsupported:
            raise NotImplementedError('dmc_partition has no declared distributed layout for nodes '
                                      +str(unsupported)+'; select execution_graph or an explicit placement policy')
        return super().apply(simulator, workload, params)

    def device_state(self, sim, container):
        return DmcState(sim, container)

    def fetch(self, ctx, state, workload, name, core, rows=None, fraction=None, op=None):
        """Pieces ``(source memory, bytes)`` of a tensor that ``core`` must copy into its SRAM.

        ``rows = (start, size, total)`` asks for a slice of the tensor's rows;
        ``fraction`` for that fraction of every part; neither for all of it.
        Pieces the core already holds are skipped.
        """
        tensor = workload.tensors[name]
        location = state.location.get(name, ("dram",))
        if fraction is None:
            fraction = 1.0 if rows is None else rows[1] / rows[2]
        if location[0] == "dram":
            return [(state.dram, tensor.bytes * fraction)]
        _, axis, pieces, total = location
        result = []
        for holder, start, size in pieces:
            if holder == core:
                continue
            if rows is not None and axis == "m":
                low, high = rows[0] / rows[2], (rows[0] + rows[1]) / rows[2]
                share = max(0.0, min(high, (start + size) / total) - max(low, start / total))
            else:
                share = fraction * size / total
            if share > 0:
                result.append((state.cores[holder][1], tensor.bytes * share))
        return result

    def stage(self, ctx, state, op, core, pieces, deps):
        """Copy fetched pieces into a staging buffer of ``core``'s SRAM; return ids the compute waits for."""
        size = sum(amount for _, amount in pieces)
        if size <= 0:
            return []
        sram = state.cores[core][1]
        buffer = ctx.storage(f"{state.path}:{op.name}.stage@{core}", sram, size, deps, op.name, op.label)
        copies = [ctx.map_edge(f"{op.name}:{i}->{core}", source, sram, amount, list(deps) + [buffer], op.name,
                               op.label) for i, (source, amount) in enumerate(pieces)]
        return [buffer] + copies

    def layout(self, op, params, cores):
        """Output partition ``(axis, [(start, size)], total)`` of an operator."""
        a = op.attrs
        if op.op in VECTOR_COSTS:
            return "m", slices(a["rows"], cores), a["rows"]
        if op.op == "mvm":
            return "m", slices(a["m"], cores), a["m"]
        if op.op == "matmul":
            if params["split"] == "n":
                return "n", slices(a["n"], cores), a["n"]
            rows = a["batch"] * a["m"]
            return "m", slices(rows, cores), rows
        raise NotImplementedError(f"dmc_partition cannot place {op.op}")

    def allocate_outputs(self, ctx, state, workload, op, params, deps):
        if op.op == "allreduce":   # collective results live in the rank's DRAM
            return super().allocate_outputs(ctx, state, workload, op, params, deps)
        params = {**self.defaults(), **params}
        axis, parts, total = self.layout(op, params, len(state.cores))
        allocations = {}
        for name in op.outputs:
            size = workload.tensors[name].bytes
            if params["activations"] == "dram":
                state.location[name] = ("dram",)
                allocations[name] = [ctx.storage(f"{state.path}:{name}", state.dram, size, deps, op.name, op.label)]
                continue
            state.location[name] = ("local", axis, [(c, s, z) for c, (s, z) in enumerate(parts)], total)
            allocations[name] = [ctx.storage(f"{state.path}:{name}@{core}", state.cores[core][1], size * z / total,
                                             deps, op.name, op.label) for core, (_, z) in enumerate(parts)]
        return allocations

    def place_operator(self, ctx, state, workload, op, params, deps):
        params = {**self.defaults(), **params}
        axis, parts, total = self.layout(op, params, len(state.cores))
        if op.op in VECTOR_COSTS:
            return self._vector(ctx, state, workload, op, params, deps, parts, total)
        eb = dtype_bytes(workload.dtype)
        a = op.attrs
        batch, m, k = a.get("batch", 1), a["m"], a["k"]
        if op.op == "matmul":
            n, shared = a["n"], a.get("shared_b", False)
            activation, weight = op.inputs
        else:
            n, shared = 1, True
            weight, activation = op.inputs
        weight_tensor = workload.tensors[weight]
        half = 0.5 if weight_tensor.kind == "kv" else 1.0     # attention reads either K or V
        outputs = sum(workload.tensors[name].bytes for name in op.outputs)
        tasks = []
        for core, (start, size) in enumerate(parts):
            sram = state.cores[core][1]
            if op.op == "mvm":
                kernel = Kernel("mvm", (batch, size, k), workload.dtype)
                need_act, weight_share = None, size / m
            elif axis == "n":
                kernel = Kernel("gemm", (batch, m, k, size), workload.dtype)
                need_act, weight_share = None, size / n
            else:
                kernel = Kernel("gemm", (1, size, k, n), workload.dtype)
                need_act, weight_share = (start, size, total), 1.0 if shared else size / total
            weight_bytes = weight_tensor.bytes * half * weight_share
            pieces = self.fetch(ctx, state, workload, activation, core, need_act, op=op)
            streams = []
            if weight_tensor.kind != "weight":
                pieces += self.fetch(ctx, state, workload, weight, core, fraction=half * weight_share, op=op)
            elif params["weights"] == "local":
                if (weight, core) not in state.resident:
                    state.resident.add((weight, core))
                    ctx.storage(f"{state.path}:{weight}@{core}", sram, weight_bytes, (), op.name, "Storage",
                                static=True)
            elif params["weights"] == "bonded":
                bonded = state.bonded[core]
                if bonded is None:
                    raise ValueError(f"weights='bonded': core {state.cores[core][0]} has no bonded DRAM")
                if (weight, core) not in state.resident:
                    state.resident.add((weight, core))
                    ctx.storage(f"{state.path}:{weight}@{core}", bonded, weight_bytes, (), op.name, "Storage",
                                static=True)
                streams.append((bonded, weight_bytes))
            else:
                streams.append((state.dram, weight_bytes))
            staged = self.stage(ctx, state, op, core, pieces, deps)
            local = self._sram_bytes(ctx, state.unit(core, kernel), kernel, eb)
            tasks.append(self._compute(ctx, state, op, core, kernel, local, deps + staged, params,
                                       outputs * size / total, streams))
        return tasks

    def _vector(self, ctx, state, workload, op, params, deps, parts, total):
        read = sum(workload.tensors[name].bytes for name in op.inputs)
        written = sum(workload.tensors[name].bytes for name in op.outputs)
        tasks = []
        for core, (start, size) in enumerate(parts):
            pieces = []
            for name in op.inputs:
                pieces += self.fetch(ctx, state, workload, name, core, (start, size, total), op=op)
            staged = self.stage(ctx, state, op, core, pieces, deps)
            kernel = op.kernel(workload.dtype, rows=size)
            share = size / total
            tasks.append(self._compute(ctx, state, op, core, kernel, (read + written) * share, deps + staged, params,
                                       written * share))
        return tasks

    def _sram_bytes(self, ctx, compute, kernel, eb):
        model = ctx.sim.points[compute]
        if isinstance(model, SystolicArrayModel):
            return sum(model.sram_accesses(kernel)) * eb
        if kernel.op == "gemm":
            b, m, k, n = kernel.shape
            return b * (m * k + k * n + m * n) * eb
        b, m, k = kernel.shape
        return (m * k + b * k + b * m) * eb

    def _compute(self, ctx, state, op, core, kernel, local_bytes, deps, params, out_bytes, streams=()):
        sram = state.cores[core][1]
        compute = state.unit(core, kernel)
        streams = [(sram, local_bytes), *streams]
        if params["activations"] == "dram":
            streams.append((state.dram, out_bytes, "write"))
        return ctx.map_node(f"{op.name}@{core}", compute, [(kernel, 1)], deps, streams, op.name, op.label,
                            latency_ns=ctx.sim.access_latency(sram, compute))


register_template(DmcPartition)

__all__ = ["DmcPartition", "slices"]
