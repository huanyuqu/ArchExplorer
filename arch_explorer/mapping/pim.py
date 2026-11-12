"""``pim_offload``: decoder attention on in-bank PIM, everything else on the host compute units.

The device is one host (an NPU with systolic arrays and vector units, or
SRAM-PNM cores) whose main memory is a DRAM group with a ``pim_unit`` per bank
(:func:`arch_explorer.structure.transforms.pim_transform`). The template
follows the NeuPIMs execution model (Heo et al., ASPLOS'24, §5.3 and §6):

* Operators other than attention are placed by the ``inner`` device template
  (``gsm_tiled`` or ``dmc_partition``, with its ``inner.*`` parameters) on the
  host units, in the host's order.
* With ``attention = "pim"``, the two attention GEMVs of decoding,
  ``q . K^T`` and ``p . V``, run on the PIM units: each request's K/V cache
  lives in one channel (requests are assigned round-robin, which is the
  NeuPIMs greedy min-load assignment when all requests have the same length);
  every bank of that channel works in lockstep for
  :meth:`~arch_explorer.components.pim.PimUnitModel.gemv_time`. A channel
  serving several requests runs their GEMVs back to back. Softmax runs on the
  host between the two GEMVs. ``attention = "inner"`` leaves attention to the
  inner template (an NPU-only baseline on the same hardware).
* ``sub_batches = 2`` is NeuPIMs sub-batch interleaving: the batch is split
  into two halves with their own activations and K/V caches (weights are
  shared). Per layer the host runs ``QKV(1), QKV(2), [softmax(1)],
  Proj/FFN(1), [softmax(2)], Proj/FFN(2)`` while the PIM attention of one
  half overlaps the host work of the other. ``sub_batches = 1`` serializes
  QKV, attention and Proj/FFN of the whole batch.
* ``tp``: the workload is built for ``tp`` tensor-parallel ranks and the
  device runs one rank; its all-reduces are not timed (a single device has
  no peer to reduce with).

Breakdown labels name the three segments of a decoder layer:
``LayerNorm+QKV`` (``norm1``, ``qkv``), ``PIM attention`` (``attn_qk``,
``softmax``, ``attn_av`` when attention runs on PIM; ``Attention`` otherwise)
and ``Projection+FFN`` (``o_proj`` to ``residual2``).

Optional pim_unit modes use the explicit command/head program in
``pim_attention``. Defaults retain whole-GEMV demands and a softmax barrier.
PIM/host command-bus arbitration and variable-length sub-batches remain
unmodeled (all requests here have one length).
"""
from __future__ import annotations

from dataclasses import replace

from arch_explorer.adf.registry import COMPONENTS, ConditionalMappingSpace, MAPPING_TEMPLATES, register_template
from arch_explorer.components.base import Demand, Kernel
from arch_explorer.adf.model import SpacePoint
from arch_explorer.mapping.base import PlacementTemplate, adf_devices, check_params, sim_devices
from arch_explorer.workload.llm import model_config, workload_from_spec
from arch_explorer.workload.operators import Workload

INNER = ("gsm_tiled", "dmc_partition")
SEGMENTS = {"norm1": "LayerNorm+QKV", "qkv": "LayerNorm+QKV", "attn_qk": "attention", "softmax": "attention",
            "attn_av": "attention"}
PIM_OPS = ("attn_qk", "attn_av")


def _has_pim(node):
    if isinstance(node, SpacePoint):
        return node.component == "pim_unit"
    return any(_has_pim(e) for e in node.elements)


def segment(op_name):
    """``LayerNorm+QKV``, ``attention`` or ``Projection+FFN`` for an operator ``l<i>.<name>``."""
    return SEGMENTS.get(op_name.split(".")[-1], "Projection+FFN")


def split_batch(workload: Workload, batch: int, parts: int, build):
    """Sub-batch workloads named ``sb<i>.``; weight tensors keep their names so the halves share them."""
    sizes = [batch // parts + (1 if i < batch % parts else 0) for i in range(parts)]
    result = []
    for i, size in enumerate(sizes):
        sub = build(size)
        prefix = f"sb{i}."
        rename = {name: name if tensor.kind == "weight" else prefix + name for name, tensor in sub.tensors.items()}
        renamed = Workload(sub.name, {rename[n]: replace(t, name=rename[n]) for n, t in sub.tensors.items()},
                           [replace(op, name=prefix + op.name, inputs=tuple(rename[n] for n in op.inputs),
                                    outputs=tuple(rename[n] for n in op.outputs)) for op in sub.operators],
                           sub.dtype, sub.iterations, dict(sub.meta, batch=size))
        result.append(renamed)
    return result


class PimOffload(PlacementTemplate):
    name = "pim_offload"

    def _tp_degrees(self, workload):
        if workload.kind != "transformer":
            return [1]
        config = model_config(workload.params["model"])
        return [d for d in (1, 2, 4, 8, 16) if config.heads % d == 0 and config.intermediate % d == 0
                and (config.kv_heads % d == 0 or d % config.kv_heads == 0)]

    def applicable(self, adf, workload) -> bool:
        devices = adf_devices(adf)
        return (len(devices) == 1 and devices[0][1] == 1 and _has_pim(adf) and workload.kind == "transformer"
                and workload.params.get("phase") == "decode")

    def space(self, adf, workload) -> ConditionalMappingSpace:
        branches, unavailable = [], []
        if not self.applicable(adf, workload):
            return ConditionalMappingSpace([], self.defaults(), self.execution_defaults(),
                                           [{"reason": "pim_offload requires one PIM device and transformer decode"}])
        degrees = self._tp_degrees(workload)
        for name in INNER:
            inner = MAPPING_TEMPLATES.get(name)
            if inner.applicable(adf, workload):
                branches.append({"attention": ["pim", "inner"], "sub_batches": [1, 2], "tp": degrees,
                    "npu_execution": ["aggregate"], "memory_execution": ["aggregate"], "inner": [name],
                    **{f"inner.{k}": v for k, v in inner.space(adf, workload).items()}})
        # Only bounded profile validation, never execute/simulate a program to discover it.
        from arch_explorer.components.stateful.neupims_program import prepare_program
        for memory in ("aggregate", "transactions"):
            for tp in degrees:
                params = {**self.defaults(), "tp": tp, "npu_execution": "instructions", "memory_execution": memory}
                try:
                    prepare_program(adf, workload, params, registry=COMPONENTS)
                except (ValueError, TypeError, KeyError, NotImplementedError) as error:
                    unavailable.append({"npu_execution": "instructions", "memory_execution": memory,
                                        "tp": tp, "reason": str(error)})
                else:
                    branches.append({k: [v] for k, v in params.items()})
        return ConditionalMappingSpace(branches, self.defaults(), self.execution_defaults(), unavailable)

    @staticmethod
    def execution_defaults():
        return {"npu_execution": "aggregate", "memory_execution": "aggregate"}

    def defaults(self):
        return {"attention": "pim", "sub_batches": 1, "tp": 1, "inner": "gsm_tiled",
                **self.execution_defaults()}

    # -- PIM placement -----------------------------------------------------------
    @staticmethod
    def channels(sim, state):
        """Lockstep PIM channels of the device: ``[[pim unit paths of one channel]]``.

        A channel is the container directly above the bank containers (the
        containers holding one ``pim_unit`` each).
        """
        channels = {}
        for group in state.pim_groups:
            for pim in group.pims:
                bank = pim.rsplit("/", 1)[0]
                channels.setdefault(bank.rsplit("/", 1)[0], []).append(pim)
        return [channels[key] for key in sorted(channels, key=_natural)]

    def place_pim(self, ctx, state, workload, op, deps):
        """One task per busy channel; each request's heads run on its channel."""
        a = op.attrs
        heads = a["batch"] // workload.meta["batch"]
        requests = workload.meta["batch"]
        # The workload states both as matmuls with one query row (m = 1): q . K^T has k = head_dim,
        # n = seq, so the bank matrix K is seq x head_dim; p . V has k = seq, n = head_dim, so the
        # bank matrix V^T is head_dim x seq. Either way the GEMV matrix is n x k.
        m, k = a["n"], a["k"]
        channels = state.pim_channels
        load = [0] * len(channels)
        for r in range(requests):
            load[r % len(channels)] += 1
        tasks = []
        for c, count in enumerate(load):
            if not count:
                continue
            units = channels[c]
            kernel = Kernel("mvm", (count * heads, m, k), workload.dtype)
            model = ctx.sim.points[units[0]]
            duration = model.gemv_time(kernel, len(units))
            demand = {ctx.sim.points[u].unit.name: duration for u in units}
            if model.row_buffer == "shared":     # PIM work holds the banks: host accesses wait
                for u in units:
                    port = ctx.sim.points[u].bank.port
                    demand[port.name] = demand.get(port.name, 0.0) + duration * port.capacity
            tasks.append(ctx.task(f"{state.path}:{op.name}@pim{c}", Demand(demand), deps, "compute", op.name,
                                  op.label))
        return tasks

    def run_pim(self, ctx, state, workload, op):
        deps = state.deps(ctx, workload, op.inputs, op.name)
        allocations = PlacementTemplate.allocate_outputs(self, ctx, state, workload, op, {}, deps)
        held = [i for ids in allocations.values() for i in ids]
        tasks = self.place_pim(ctx, state, workload, op, deps + held)
        done = ctx.sync(f"{state.path}:{op.name}.done", tasks, op.name, op.label)
        self.publish(state, allocations, done)
        return done

    # -- whole workload ------------------------------------------------------------
    def apply(self, simulator, workload, params) -> None:
        params = {**self.defaults(), **params}
        if params.get('npu_execution')=='instructions' or params.get('memory_execution')=='transactions':
            from .instruction_program import map_program
            map_program(simulator,workload,params)
            return
        params = check_params(self.name, self.space(simulator.adf, workload), params)
        if workload.kind != "transformer" or workload.params.get("phase") != "decode":
            raise ValueError("pim_offload maps transformer decode workloads")
        devices = sim_devices(simulator)
        if len(devices) != 1:
            raise ValueError(f"pim_offload maps onto one device; the ADF has {len(devices)}")
        inner = MAPPING_TEMPLATES.get(params["inner"])
        inner_params = {key.split(".", 1)[1]: value for key, value in params.items() if key.startswith("inner.")}
        state = inner.device_state(simulator, devices[0])
        state.pim_channels = self.channels(simulator, state)
        on_pim = params["attention"] == "pim"
        if on_pim and not state.pim_channels:
            raise ValueError(f"{state.path} has no pim_unit in its DRAM")
        spec = dict(workload.params)
        batch, tp = spec.get("batch", 1), params["tp"]
        parts = min(params["sub_batches"], batch)
        whole = workload_from_spec(workload, tp=tp)
        simulator.mapping.graph.iterations = whole.iterations
        if parts == 1:
            subs = [whole]
        else:
            from arch_explorer.adf.evaluation import WorkloadSpec

            def build(size):
                return workload_from_spec(WorkloadSpec("transformer", dict(spec, batch=size)), tp=tp)
            subs = split_batch(whole, batch, parts, build)
        attention_label = "PIM attention" if on_pim else "Attention"

        def label(op):
            seg = segment(op.name)
            return replace(op, label=attention_label if seg == "attention" else seg)

        for sub in subs:
            sub.operators = [label(op) for op in sub.operators]
        handled = set()
        command_mapping = on_pim and any(simulator.points[ch[0]].command_mapping for ch in state.pim_channels)
        for op in self._order(subs):
            if op.name in handled:
                continue
            sub = next(s for s in subs if op in s.operators)
            if command_mapping and op.name.split(".")[-1] == "attn_qk":
                from .pim_attention import AttentionProgram
                prefix = op.name.rsplit('.',1)[0]
                softmax = next(o for o in sub.operators if o.name == prefix+'.softmax')
                av = next(o for o in sub.operators if o.name == prefix+'.attn_av')
                AttentionProgram(simulator.mapping,state,sub,op,softmax,av).run()
                handled.update((softmax.name,av.name))
            elif on_pim and op.name.split(".")[-1] in PIM_OPS:
                self.run_pim(simulator.mapping, state, sub, op)
            else:
                inner.run_device(simulator.mapping, state, sub, op, inner_params)

    @staticmethod
    def _order(subs):
        """Host order: per layer, every sub-batch's QKV, then each sub-batch's attention and Proj/FFN."""
        if len(subs) == 1:
            return list(subs[0].operators)
        layers = {}
        for i, sub in enumerate(subs):
            for op in sub.operators:
                layer = op.name.split(".")[1]
                stage = 0 if segment(op.name) == "LayerNorm+QKV" else 1
                layers.setdefault(layer, {}).setdefault((stage, 0 if stage == 0 else i), []).append(op)
        order = []
        for layer in sorted(layers, key=_natural):
            for key in sorted(layers[layer]):
                order.extend(layers[layer][key])
        return order


def _natural(text):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


register_template(PimOffload)

# Explicit, separate GDDR6 shared-controller mapping; defaults above are unchanged.
from arch_explorer.mapping.pim_aim import AimShared  # noqa: E402,F401

__all__ = ["PimOffload", "segment", "split_batch"]
