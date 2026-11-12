"""Non-GPU single-device full graph execution using GSM matrix placement.

One ordered stream; all weights are resident once at entry. A view holds its
base storage through all consumers. Every transform except a legal reshape or
terminal transpose is materialized. Outputs are held through the return event.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from functools import partial
import math

from arch_explorer.components.base import Kernel
from arch_explorer.mapping.base import sim_devices
from arch_explorer.workload.llm import workload_from_spec
from arch_explorer.workload.full_primitives import FullOperator
from arch_explorer.simulator.plan import PlanConstraints


def report_full_model(kind, simulator, result):
    """Mapping-owned tensor residency and model request accounting."""
    fm = deepcopy(simulator.mapping.details['full_model'])
    fm['memory_peak'] = dict(result.memory_peak)
    fm['memory_capacity'] = {name: simulator.points[name].capacity for name in result.memory_peak}
    if kind == 'pi05':
        graph, latency = simulator.mapping.graph, result.latency_ns
        by_task = {record.task: record for record in result.records}
        for tensor in fm['tensors'].values():
            records = [by_task[i] for i in tensor['storage_tasks']]
            tensor['allocation_ns'] = min(r.start_ns for r in records)
            tensor['free_ns'] = max(latency if graph.tasks[r.task].static else
                max((by_task[dst].end_ns for dst, distance in graph.succs[r.task]), default=r.end_ns)
                for r in records)
            tensor['ready_ns'] = by_task[tensor['ready_task']].end_ns
        boundaries = [0.0]
        for i in range(fm['iterations']):
            boundaries.append(max(r.end_ns for r in result.records if r.operator.startswith(f'request{i}.')))
        fm['request_timing'] = [dict(request=i, start_ns=boundaries[i], end_ns=boundaries[i+1],
            latency_ns=boundaries[i+1]-boundaries[i]) for i in range(fm['iterations'])]
        fm['total_latency_ns'] = latency
        fm['timing_policy'] = 'sequential independent requests; first-use weight conversion included; shared weight allocations at t=0; all prepared inputs ready'
    return {'full_model': fm}


def _checked_units(state, kernel):
    units = state.units_for(kernel)
    # The one-type selection path returns all units without a capability query.
    # Validate with the same component method that actual placement consumes.
    for unit in units:
        state.sim.points[unit].kernel_demand(kernel)
    return units


def _matrix_cast_needed(state, op, dtype):
    if op.op != "matmul" or op.attrs.get("output_dtype", dtype) not in ("bf16", "fp16"):
        return False
    units = _checked_units(state, op.kernel(op.attrs.get("dtype", dtype)))
    cast = Kernel("vector", (1, 1), "fp32", simt=1)
    try:
        for unit in units:
            state.sim.points[unit].kernel_demand(cast)
    except NotImplementedError:
        _checked_units(state, cast)
        return True
    return False


def full_model_applicable(template, adf, spec):
    """Check actual operator precision/capabilities, without mapping or scheduling."""
    from arch_explorer.simulator.events.builder import build_simulator
    try:
        sim = build_simulator(adf, options={"systolic_model": "native"})
        devices = sim_devices(sim)
        if len(devices) != 1:
            return False
        state = template.device_state(sim, devices[0])
        ir = workload_from_spec(spec)
        checked = set()
        for op in ir.operators:
            if op.op == "view":
                continue
            kernel = op.kernel(op.attrs.get("dtype", ir.dtype))
            if kernel not in checked:
                _checked_units(state, kernel)
                checked.add(kernel)
            _matrix_cast_needed(state, op, ir.dtype)
        return True
    except (ValueError, NotImplementedError):
        return False


def _matrix_output_cast(ctx, state, op, elements, deps):
    """Materialize FP32 accumulators, then convert on capable vector hardware."""
    units = _checked_units(state, Kernel("vector", (1, elements), "fp32", simt=1))
    tasks = []
    for index, unit in enumerate(units):
        count = elements // len(units) + (index < elements % len(units))
        if not count:
            continue
        tasks.append(ctx.map_node(op.name+".output_cast@"+unit, unit,
                     [(Kernel("vector", (1, count), "fp32", simt=1), 1)], deps,
                     [(state.dram, 4*count), (state.dram, 2*count, "write")], op.name, op.label))
    return tasks


class MatrixOutputPlan:
    """One local matrix output, shared by single-device and ranked placement.

    Determine its physical dtype/size before allocating outputs or cache.
    The plan owns the FP32 temporary and its conversion consumers; rank and
    tensor shapes come from the caller's local state/IR, never global shapes.
    """

    def __init__(self, state, ir, op):
        self.state, self.op = state, op
        self.elements = sum(math.prod(ir.tensors[n].shape) for n in op.outputs)
        self.logical_bytes = sum(ir.tensors[n].bytes for n in op.outputs)
        self.explicit = getattr(state, "full_model", False) and _matrix_cast_needed(state, op, ir.dtype)
        self.scratch_bytes = 4*self.elements if self.explicit else 0
        self.output_bytes = self.scratch_bytes or self.logical_bytes
        self.placement_op = replace(op, attrs=dict(op.attrs, output_dtype="fp32")) if self.explicit else op
        self.casts = (self.elements if op.op == "matmul" and getattr(state, "full_model", False)
                      and op.attrs.get("output_dtype", ir.dtype) in ("fp16", "bf16") else 0)
        self.scratch_task = None
        self.cast_tasks = []

    def allocate(self, ctx, deps):
        if not self.explicit:
            return []
        self.scratch_task = ctx.storage(self.op.name+".fp32_matrix_scratch@"+self.state.path,
            self.state.dram, self.scratch_bytes, deps, self.op.name, self.op.label)
        return [self.scratch_task]

    def finish(self, ctx, tasks, allocations):
        if not self.explicit:
            return tasks
        if self.scratch_task is None:
            raise RuntimeError("matrix output conversion requires its FP32 temporary")
        done = ctx.sync(self.op.name+".matrix.done@"+self.state.path,
                        [*tasks, *allocations], self.op.name, self.op.label)
        self.cast_tasks = _matrix_output_cast(ctx, self.state, self.op, self.elements,
                                              [done, *allocations])
        return self.cast_tasks

    def report(self):
        return dict(fused_output_casts=0 if self.explicit else self.casts,
                    explicit_output_casts=self.casts if self.explicit else 0,
                    fp32_matrix_scratch_bytes=self.scratch_bytes,
                    output_cast_read_bytes=self.scratch_bytes,
                    output_cast_write_bytes=self.logical_bytes if self.explicit else 0,
                    fp32_matrix_scratch_task=self.scratch_task, output_cast_tasks=self.cast_tasks,
                    matrix_output_bytes=self.output_bytes)


def apply_full_model(template, sim, spec, params):
    ir = workload_from_spec(spec)
    ctx = sim.mapping
    ctx.plan.require(PlanConstraints(iterations=1, check_capacity=spec.kind == 'pi05', reason='full model graph'))
    ctx.plan.detail_reporters.append(partial(report_full_model, spec.kind))
    ctx.graph.iterations = 1
    devices = sim_devices(sim)
    if len(devices) != 1:
        raise ValueError("full models require one device")
    state = template.device_state(sim, devices[0])
    state.full_model = True
    state.full_dtype = ir.dtype
    producer = ir.producers()
    storage, ready, initial = {}, {}, []
    for name, tensor in ir.tensors.items():
        if name not in producer:
            task = ctx.storage(state.path+":"+name, state.dram, tensor.bytes,
                               static=tensor.kind == "weight")
            storage[name], ready[name] = [task], task
            initial.append(task)
    entry = ctx.sync("full_model.entry", initial)
    state.stream = [entry]
    cache_holds = {}
    for output, layers in zip(ir.meta["full_model"]["outputs"], ir.meta["full_model"].get("prefix_cache_ids", [])):
        cache_holds[output] = [n for pair in layers for n in pair]
    ledger = []
    tensors = {}
    for name, t in ir.tensors.items():
        tensors[name] = dict(shape=list(t.shape), dtype=t.dtype, kind=t.kind,
                             logical_bytes=t.bytes, allocated_bytes=0 if t.alias_of else t.bytes,
                             alias_of=t.alias_of, strides=t.strides, layout="strided" if t.strides else "contiguous")
    for op in ir.operators:
        held = [s for name in op.inputs for s in storage[name]]
        held += [s for name in cache_holds.get(op.name, []) for s in storage[name]]
        deps = list(dict.fromkeys([*state.stream, *held, *(ready[n] for n in op.inputs)]))
        if op.op == "view":
            out = op.outputs[0]
            source = ir.tensors[out].alias_of
            storage[out] = storage[source]
            ready[out] = ctx.sync(op.name, deps, op.name, op.label)
            state.stream = [ready[out]]
            ledger.append(dict(operator=op.name, op="view", read_bytes=0, write_bytes=0,
                               scratch_bytes=0, alias_of=source, inputs=list(op.inputs), outputs=list(op.outputs)))
            continue
        for name in op.inputs:
            if ir.tensors[name].strides is not None:
                raise ValueError(f"{op.name}: strided consumer requires materialization")
        output_plan = MatrixOutputPlan(state, ir, op)
        output_storage = []
        for name in op.outputs:
            task = ctx.storage(state.path+":"+name, state.dram, ir.tensors[name].bytes,
                               deps, op.name, op.label)
            storage[name] = [task]
            output_storage.append(task)
        output_storage.extend(output_plan.allocate(ctx, deps))
        scratch = op.attrs.get("scratch_bytes", 0)
        if scratch:
            output_storage.append(ctx.storage(op.name+".fp32_scratch", state.dram, scratch,
                                               deps, op.name, op.label))
        cache = 0
        if op.op == "matmul" and state.l2:
            cache = min(state.l2_capacity, sum(ir.tensors[n].bytes for n in op.inputs) + output_plan.output_bytes)
            output_storage.append(ctx.storage(op.name+".cache", state.l2, cache, deps, op.name, op.label))
        state.full_dtype = op.attrs.get("output_dtype", ir.dtype)
        start = template.kernel_start(ctx, state, op, [*deps, *output_storage])
        if isinstance(op, FullOperator):
            rows = op.attrs["rows"]
            units = state.units_for(op.kernel())
            rpc = params.get("rows_per_cta", template.defaults()["rows_per_cta"])
            assignment = Counter()
            for first in range(0, rows, rpc):
                assignment[(first//rpc) % len(units)] += min(rpc, rows-first)
            read, write = op.attrs["read_bytes"], op.attrs["write_bytes"]
            tasks = []
            for u, count in sorted(assignment.items()):
                share = count / rows
                # Explicit global reduction scratch written and read once.
                streams = [(state.dram, (read + scratch)*share),
                           (state.dram, (write + scratch)*share, "write")]
                tasks.append(ctx.map_node(op.name+"@"+units[u], units[u],
                             [(op.kernel(rows=count), 1)], start, streams, op.name, op.label))
            counts = op.kernel()
            ledger.append(dict(operator=op.name, op=op.op, read_bytes=read, write_bytes=write,
                               scratch_bytes=scratch, scratch_traffic_bytes=2*scratch,
                               simt_instructions=rows*(counts.shape[1]*counts.simt + counts.row_simt),
                               sfu_results=rows*(counts.shape[1]*counts.sfu + counts.row_sfu)))
        else:
            tasks = template.place_operator(ctx, state, ir, output_plan.placement_op, params, start)
            tasks = output_plan.finish(ctx, tasks, output_storage)
            ledger.append(dict(operator=op.name, op=op.op, logical_input_bytes=sum(ir.tensors[n].bytes for n in op.inputs),
                               write_bytes=sum(ir.tensors[n].bytes for n in op.outputs),
                               flops=op.kernel(op.attrs.get("dtype", ir.dtype)).flops, cache_reservation_bytes=cache,
                               **output_plan.report()))
        ledger[-1].update(inputs=list(op.inputs), outputs=list(op.outputs),
                          input_dtypes=[ir.tensors[n].dtype for n in op.inputs],
                          output_dtypes=[ir.tensors[n].dtype for n in op.outputs],
                          attributes=dict(op.attrs))
        # Direct storage predecessors are necessary for the engine's lifetime
        # rules; the barrier owns input/output buffers until all tasks finish.
        done = ctx.sync(op.name+".done", [*tasks, *held, *output_storage], op.name, op.label)
        template.kernel_end(state, done)
        for name in op.outputs:
            ready[name] = done
    outputs = ir.meta["full_model"]["outputs"]
    consumers = {}
    for op in ir.operators:
        for name in op.inputs: consumers.setdefault(name, []).append(op.name)
    for name, tensor in tensors.items():
        tensor['storage_tasks']=storage[name]
        tensor['ready_task']=ready[name]
        tensor['producer']=producer[name].name if name in producer else None
        tensor['consumers']=consumers.get(name, [])
    ctx.sync("full_model.return", [*state.stream, *(ready[n] for n in outputs),
                                  *(s for n in outputs for s in storage[n]),
                                  *(s for n,t in ir.tensors.items() if t.kind=="weight" for s in storage[n])])
    report = deepcopy(ir.meta["full_model"])
    report.update(tensors=tensors, operators=ledger,
                  storage_policy="source weights resident at entry; converted weights retained after generation; activations held to last consumer; alias shares base; outputs retained",
                  cache_policy="GSM bounded working set; per-kernel L2 reservation; K-slice too large disables wave reuse; vectors stream DRAM directly")
    # Resource demand units are given by the graph resources. Memory/network
    # entries are bytes; compute entries are service ns. Never sum both as bytes.
    totals = Counter()
    for task in ctx.graph.tasks:
        totals.update(task.demand)
    report["resource_demands"] = dict(totals)
    report["resource_units"] = {name: ctx.graph.resources[name].unit for name in totals}
    sim.warnings.append("Full-model arithmetic is an explicit analytical variant; SFU/erf rates and framework fusion are not calibrated")
    ctx.details["full_model"] = report
