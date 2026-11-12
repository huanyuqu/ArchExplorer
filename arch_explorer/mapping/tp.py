"""``tensor_parallel``: Megatron-style tensor parallelism over several devices.

The workload is built for ``tp`` ranks (heads and FFN columns split, see
:mod:`arch_explorer.workload.llm`). Rank ``r`` runs on the ``r``-th device of
the ADF and places its operators with the ``inner`` device template
(``gsm_tiled`` or ``dmc_partition``) using the ``inner.*`` parameters. Each
all-reduce becomes a ring or tree collective of ``map_edge`` transfers between
the ranks' DRAMs plus reduction tasks on the ranks' compute units.
"""
from __future__ import annotations

from dataclasses import replace

from arch_explorer.adf.model import SpaceMatrix
from arch_explorer.adf.registry import (ConditionalMappingSpace, MAPPING_TEMPLATES, MappingTemplate,
                                       mapping_spaces, register_template, validate_mapping_params)
from arch_explorer.components.base import Kernel
from arch_explorer.mapping.base import PlacementTemplate, adf_devices, sim_devices
from arch_explorer.mapping.context import COLLECTIVE_ALGORITHMS
from arch_explorer.workload.llm import model_config, workload_from_spec

INNER = ("gsm_tiled", "dmc_partition")


def _rank_views(adf):
    """Single-instance device subtrees in the builder's endpoint order.

    Do not ask a single-device inner template to accept the entire server.
    Walking nested repeats preserves rank order (type counts alone do not).
    Only the device root's repeat is removed; its cores and parameters stay intact.
    """
    types = {path for path, _ in adf_devices(adf)}

    def visit(node, path):
        if not isinstance(node, SpaceMatrix):
            return
        for _ in range(node.repeat):
            if path in types:
                yield replace(node, repeat=1)
            else:
                for child in node.elements:
                    yield from visit(child, f"{path}/{child.name}")

    return list(visit(adf, adf.name))


class TensorParallel(MappingTemplate):
    name = "tensor_parallel"

    def _degrees(self, adf, workload):
        devices = sum(count for _, count in adf_devices(adf))
        degrees = [d for d in range(1, devices + 1) if devices % d == 0]
        if workload.kind == "transformer":
            config = model_config(workload.params["model"])
            degrees = [d for d in degrees if config.heads % d == 0 and config.intermediate % d == 0
                       and (config.kv_heads % d == 0 or d % config.kv_heads == 0)]
        else:
            degrees = [1]
        return degrees

    def applicable(self, adf, workload) -> bool:
        return bool(self.space(adf, workload).branches)

    def space(self, adf, workload) -> ConditionalMappingSpace:
        if workload.kind not in ("operator", "transformer"):
            return ConditionalMappingSpace([], {}, {}, [
                {"reason": "tensor_parallel supports operator/transformer only; full models require their own mapping"}])
        degrees = self._degrees(adf, workload)
        ranks = _rank_views(adf)
        defaults = {"tp": max(degrees, default=1), "collective": "ring", "inner": "gsm_tiled"}
        branches, unavailable = [], []
        for tp in degrees:
            for name in INNER:
                if name not in MAPPING_TEMPLATES.names():
                    continue
                inner = MAPPING_TEMPLATES.get(name)
                rejected = [r for r, rank in enumerate(ranks[:tp]) if not inner.applicable(rank, workload)]
                if rejected:
                    unavailable.append({"tp": tp, "inner": name, "ranks": rejected,
                                        "reason": "inner template does not apply to these rank subtrees"})
                    continue
                spaces = [inner.space(rank, workload) for rank in ranks[:tp]]
                if not spaces:
                    continue
                # One shared inner mapping must be valid on every selected rank.
                # Optional local/bonded choices on unused or other ranks cannot leak in.
                common = {key: [v for v in values if all(key in s and v in s[key] for s in spaces[1:])]
                          for key, values in spaces[0].items() if all(key in s for s in spaces[1:])}
                if any(not values for values in common.values()):
                    unavailable.append({"tp": tp, "inner": name, "reason": "rank parameter domains do not intersect"})
                    continue
                branches.append({"tp": [tp], "collective": list(COLLECTIVE_ALGORITHMS), "inner": [name],
                                 **{f"inner.{key}": values for key, values in common.items()}})
        return ConditionalMappingSpace(branches, defaults, {}, unavailable)

    def request_space(self, adf, workload, params=None):
        space = self.space(adf, workload)
        if params is None:
            return space
        # Filters branches without filling defaults or dropping inactive selectors.
        try:
            branches = mapping_spaces(space, params)
        except ValueError as error:
            raise ValueError(f"{self.name}: requested combination is not in its domain; {error}") from error
        return ConditionalMappingSpace(branches, space.defaults, {}, space.unavailable)

    def apply(self, simulator, workload, params) -> None:
        params = validate_mapping_params(self.request_space(simulator.adf, workload, params), params)
        tp = params["tp"]
        algorithm = params["collective"]
        inner = MAPPING_TEMPLATES.get(params["inner"])
        inner_params = {key.split(".", 1)[1]: value for key, value in params.items() if key.startswith("inner.")}
        devices = sim_devices(simulator)
        if tp > len(devices):
            raise ValueError(f"tp={tp} exceeds the {len(devices)} devices of the ADF")
        ir = workload_from_spec(workload, tp=tp)
        ctx = simulator.mapping
        ctx.graph.iterations = ir.iterations
        states = [inner.device_state(simulator, device) for device in devices[:tp]]
        for op in ir.operators:
            if op.op == "allreduce" and tp > 1:
                deps = [state.deps(ctx, ir, op.inputs, op.name) for state in states]
                allocations = [PlacementTemplate.allocate_outputs(inner, ctx, state, ir, op, inner_params, d)
                               for state, d in zip(states, deps)]
                held = [d + [i for ids in a.values() for i in ids] for d, a in zip(deps, allocations)]
                started = [inner.kernel_start(ctx, state, op, h) for state, h in zip(states, held)]
                ranks = [(state.dram, reducers(simulator, state.computes)) for state in states]
                done = ctx.allreduce(op.name, ranks, op.attrs["bytes"], started, algorithm, op.name, op.label,
                                     ir.dtype)
                for state, allocation, finished in zip(states, allocations, done):
                    inner.kernel_end(state, finished)
                    inner.publish(state, allocation, finished)
            else:
                for state in states:
                    inner.run_device(ctx, state, ir, op, inner_params)


def reducers(simulator, computes):
    """Compute points of a rank that can run the vector add of an all-reduce."""
    probe = Kernel("vector", (1, 1), simt=1)
    capable = []
    for compute in computes:
        try:
            simulator.points[compute].kernel_demand(probe)
            capable.append(compute)
        except NotImplementedError:
            continue
    if not capable:
        raise NotImplementedError("No compute point of the rank can run the all-reduce reduction")
    return capable


register_template(TensorParallel)
