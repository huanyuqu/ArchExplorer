"""Shared-memory tiled placement for non-GPU event models.

GEMM panels are assigned round-robin to systolic/vector compute units.
An optional shared SRAM caches complete operands or shares panels within a
wave. Vector/MVM operators stream explicit traffic. Operators use one ordered
stream. GPU timing belongs to the native Accel-Sim backend; this template has
no SM residency, warp tiling, GPU launch overhead or GPU fallback.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

from arch_explorer.adf.model import SpaceMatrix
from arch_explorer.adf.registry import register_template
from arch_explorer.components.base import Kernel, dtype_bytes
from arch_explorer.mapping.base import PlacementTemplate, adf_devices
from arch_explorer.workload.operators import VECTOR_COSTS

TILES = (64, 128, 256)
ROWS_PER_CTA = (1, 4, 16, 64)


def _has_gpu(node):
    if isinstance(node, SpaceMatrix):
        return any(_has_gpu(child) for child in (*node.elements, *node.networks))
    return node.component.startswith("accelsim_") or node.component in {
        "gpu_sm", "gpu_kernel_service", "gpu_kernel_service_v2"}


class GsmTiled(PlacementTemplate):
    name = "gsm_tiled"

    def apply(self, simulator, workload, params):
        if _has_gpu(simulator.adf):
            raise ValueError("gsm_tiled is a non-GPU event mapping; use backend='accelsim' for GPU programs")
        unknown = set(params) - set(self.defaults())
        if unknown:
            raise ValueError(f"gsm_tiled: unsupported mapping parameters {sorted(unknown)}")
        if workload.kind in ("vit", "clip", "pi05"):
            from arch_explorer.mapping.full_models import apply_full_model
            return apply_full_model(self, simulator, workload, params)
        if workload.kind not in ("operator", "transformer"):
            from arch_explorer.mapping.graph import apply_graph
            return apply_graph(self, simulator, workload, params)
        return super().apply(simulator, workload, params)

    def applicable(self, adf, workload):
        if _has_gpu(adf) or not super().applicable(adf, workload):
            return False
        if workload.kind in ("vit", "clip", "pi05"):
            from arch_explorer.mapping.full_models import full_model_applicable
            return full_model_applicable(self, adf, workload)
        if workload.kind not in ("operator", "transformer"):
            from arch_explorer.mapping.graph import graph_applicable
            return graph_applicable(self, adf, workload)
        return True

    def space(self, adf, workload) -> dict:
        return {"tile_m": list(TILES) + ["auto"], "tile_n": list(TILES) + ["auto"],
                "rows_per_cta": list(ROWS_PER_CTA)}

    def defaults(self):
        return {"tile_m": "auto", "tile_n": "auto", "rows_per_cta": 4}

    def device_state(self, sim, container):
        if _has_gpu(sim.adf):
            raise ValueError("GPU components require backend='accelsim'; no GSM event fallback")
        state = super().device_state(sim, container)
        shared = [child for child in container.children if getattr(child, "role", None) == "memory"
                  and child.component.name == "sram"]
        state.l2 = shared[0].path if shared else None
        state.l2_capacity = sim.points[state.l2].capacity if shared else 0
        state.stream = []          # the previous kernel's completion (one in-order stream per device)
        return state

    def kernel_start(self, ctx, state, op, deps):
        """Preserve dependency and single-device stream order."""
        return list(deps) + state.stream

    def kernel_end(self, state, done):
        state.stream = [done]

    def place_operator(self, ctx, state, workload, op, params, deps):
        params = {**self.defaults(), **params}
        dtype = op.attrs.get("dtype", workload.dtype)
        if getattr(state, "materialized_graph", False):
            from arch_explorer.mapping.graph import operator_dtype
            dtype = operator_dtype(workload, op)
        if op.op == "matmul":
            tile_m, tile_n = params["tile_m"], params["tile_n"]
            if "auto" in (tile_m, tile_n):
                tile_m, tile_n = self.best_tile(ctx, state, op, dtype, tile_m, tile_n)
            return self.gemm(ctx, state, op, dtype, tile_m, tile_n, deps)
        if op.op == "mvm":
            return self.mvm(ctx, state, op, dtype, params["rows_per_cta"], deps)
        if op.op in VECTOR_COSTS:
            return self.vector(ctx, state, workload, op, dtype, params["rows_per_cta"], deps)
        from arch_explorer.mapping.graph import place_registered
        return place_registered(ctx, state, workload, op, dtype, deps)

    # -- GEMM ----------------------------------------------------------------------
    def best_tile(self, ctx, state, op, dtype, tile_m, tile_n):
        from arch_explorer.simulator.plan import require_planning_execution
        require_planning_execution()
        from arch_explorer.mapping.context import MappingContext
        from arch_explorer.simulator.events.engine import simulate
        a = op.attrs
        key = (state.container.type_path, a.get("batch", 1), a["m"], a["k"], a["n"], a.get("shared_b", False),
               tile_m, tile_n, dtype, a.get("kv_group", 1))
        cache = ctx.sim.__dict__.setdefault("gsm_tile_cache", {})
        if key not in cache:
            best = None
            for tm in (TILES if tile_m == "auto" else [tile_m]):
                for tn in (TILES if tile_n == "auto" else [tile_n]):
                    trial = MappingContext(ctx.sim)
                    self.gemm(trial, state, op, dtype, tm, tn, [])
                    latency = simulate(trial.graph).latency_ns
                    if best is None or latency < best[0] - 1e-9 * latency:
                        best = (latency, tm, tn)
            cache[key] = best[1:]
        return cache[key]

    def gemm(self, ctx, state, op, dtype, tm, tn, deps):
        a = op.attrs
        b, m, k, n, shared = a.get("batch", 1), a["m"], a["k"], a["n"], a.get("shared_b", False)
        group = a.get("kv_group", 1)          # GQA: consecutive batch entries share one B operand
        eb = dtype_bytes(dtype)
        output_eb = dtype_bytes(a.get("output_dtype", dtype))
        units = state.units_for(op.kernel(dtype))
        mt, nt = math.ceil(m / tm), math.ceil(n / tn)
        total = b * mt * nt
        unique = (b * m * k + (1 if shared else math.ceil(b / group)) * k * n ) * eb + b * m * n * output_eb
        fits_all = state.l2 is not None and unique <= state.l2_capacity
        seen = set()
        per_unit = defaultdict(lambda: {"count": 0, "read": 0.0, "write": 0.0, "dram": 0.0})
        for first in range(0, total, len(units)):
            tiles, a_panels, b_panels = [], {}, {}
            for t in range(first, min(total, first + len(units))):
                bi, rest = divmod(t, mt * nt)
                i, j = divmod(rest, nt)
                rows, cols = min(tm, m - i * tm), min(tn, n - j * tn)
                a_panels[("a", bi, i)] = rows * k * eb
                b_panels[("b", 0 if shared else bi // group, j)] = k * cols * eb
                tiles.append((t, rows, cols))
            outputs = sum(rows * cols for _, rows, cols in tiles) * output_eb
            own = sum((rows * k + k * cols) * eb for _, rows, cols in tiles)
            if state.l2 is None:
                dram = own + outputs
            elif fits_all:
                fresh = {key: size for key, size in {**a_panels, **b_panels}.items() if key not in seen}
                seen.update(fresh)
                dram = sum(fresh.values()) + outputs
            else:
                panels = sum(a_panels.values()) + sum(b_panels.values())
                # Full-model streaming reuse needs the concurrent K-slice
                # resident. When it cannot fit, every CTA fetches its panels;
                # the legacy path retains its established convention.
                slice_bytes = panels * min(32, k) / k
                no_reuse = getattr(state, "full_model", False) and slice_bytes > state.l2_capacity
                dram = (own if no_reuse else panels) + outputs
            for t, rows, cols in tiles:
                record = per_unit[t % len(units)]
                record["count"] += 1
                record["read"] += (rows * k + k * cols) * eb
                record["write"] += rows * cols * output_eb
                record["dram"] += dram / len(tiles)
        kernel = Kernel("gemm", (1, min(tm, m), k, tn), dtype)
        tasks = []
        for u, r in sorted(per_unit.items()):
            tasks.append(self._unit_task(ctx, state, op, units[u], [(kernel, r["count"])], r["read"], r["write"],
                                         r["dram"], r["count"], deps))
        return tasks

    def _unit_task(self, ctx, state, op, unit, kernels, read_bytes, write_bytes, dram_bytes, count, deps):
        """``read``/``write``: bytes between the unit and L2; ``dram``: traffic behind L2."""
        if getattr(state, "full_model", False) and op.attrs.get("output_dtype", state.full_dtype) in ("fp16", "bf16") and write_bytes:
            # Fused FP32-accumulator -> FP16 destination conversion on the
            # SAME unit that writes the output. Never migrate it to another
            # unit without accounting for communication.
            cast = Kernel("vector", (1, write_bytes / 2), "fp32", simt=1)
            try:
                ctx.sim.points[unit].kernel_demand(cast)
            except NotImplementedError as error:
                raise ValueError("full-model FP16 matrix epilogue requires a unit with vector conversion support") from error
            kernels = [*kernels, (cast, 1)]
        if state.l2 is None:
            streams, extra = [(state.dram, read_bytes), (state.dram, write_bytes, "write")], None
        else:
            streams = [(state.l2, read_bytes), (state.l2, write_bytes, "write")]
            if getattr(state, "full_model", False):
                # Full-model path charges the actual DRAM <-> cache fabric
                # as well as both endpoint ports. Legacy paths unchanged.
                write = min(write_bytes, dram_bytes)
                extra = ctx.sim.transfer(state.dram, state.l2, dram_bytes - write)
                extra = extra.add(ctx.sim.transfer(state.l2, state.dram, write))
            else:
                extra = ctx.sim.port_demand(state.dram, dram_bytes)
        return ctx.map_node(f"{op.name}@{unit}", unit, kernels, deps, streams, op.name, op.label, weight=count,
                            latency_ns=ctx.sim.access_latency(state.dram, unit), extra=extra)

    # -- vector and MVM kernels ----------------------------------------------------
    def vector(self, ctx, state, workload, op, dtype, rows_per_cta, deps):
        rows = op.attrs["rows"]
        read = sum(workload.tensors[name].bytes for name in op.inputs)
        written = sum(workload.tensors[name].bytes for name in op.outputs)
        per_unit = defaultdict(Counter)
        units = state.units_for(op.kernel(dtype))
        for cta in range(math.ceil(rows / rows_per_cta)):
            per_unit[cta % len(units)]["rows"] += min(rows_per_cta, rows - cta * rows_per_cta)
            per_unit[cta % len(units)]["ctas"] += 1
        tasks = []
        for u, record in sorted(per_unit.items()):
            share = record["rows"] / rows
            kernel = op.kernel(dtype, rows=record["rows"])
            tasks.append(self._unit_task(ctx, state, op, units[u], [(kernel, 1)], read * share, written * share,
                                         (read + written) * share, record["ctas"], deps))
        return tasks

    def mvm(self, ctx, state, op, dtype, rows_per_cta, deps):
        a = op.attrs
        batch, m, k = a.get("batch", 1), a["m"], a["k"]
        eb = dtype_bytes(dtype)
        units = state.units_for(Kernel("mvm", (batch, m, k), dtype))
        ctas = math.ceil(m / rows_per_cta)
        per_unit = defaultdict(Counter)
        for cta in range(ctas):
            per_unit[cta % len(units)]["rows"] += min(rows_per_cta, m - cta * rows_per_cta)
            per_unit[cta % len(units)]["ctas"] += 1
        vector_bytes = batch * k * eb
        tasks = []
        for u, record in sorted(per_unit.items()):
            rows = record["rows"]
            read = rows * k * eb + record["ctas"] * vector_bytes
            dram = rows * k * eb + vector_bytes * record["ctas"] / ctas + batch * rows * eb
            kernel = Kernel("mvm", (batch, rows, k), dtype)
            tasks.append(self._unit_task(ctx, state, op, units[u], [(kernel, 1)], read, batch * rows * eb, dram,
                                         record["ctas"], deps))
        return tasks


register_template(GsmTiled)

__all__ = ["GsmTiled", "TILES", "ROWS_PER_CTA", "adf_devices"]
