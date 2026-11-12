"""One full tensor DAG, explicit rank placement and directed communication.

Local execution reuses GSM and the existing component demand models. K-axis
TP emits true per-rank GEMMs, FP32 partials and explicit sum communication;
matrix output is cast only after the collective. Selectors and shard ranges
are software mapping choices, never fitted hardware parameters.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import math

from arch_explorer.adf import find
from arch_explorer.adf.multi_device import validate_request, request_space
from arch_explorer.adf.registry import MappingTemplate, register_template
from arch_explorer.components.base import Kernel, dtype_bytes
from arch_explorer.mapping.base import check_params, sim_devices
from arch_explorer.mapping.collective_plan import collective
from arch_explorer.mapping.directed import Copy, DirectedTransport
from arch_explorer.mapping.gsm import GsmTiled
from arch_explorer.mapping.full_models import MatrixOutputPlan
from arch_explorer.workload.full_primitives import FullOperator
from arch_explorer.workload.llm import workload_from_spec


def select(placements, name):
    if name in placements:
        return placements[name]
    prefixes = [p for p in placements if p.endswith('.*') and name.startswith(p[:-1])]
    if prefixes:
        return placements[max(prefixes, key=len)]
    if '*' in placements:
        return placements['*']
    raise ValueError(f'No placement for operator {name}')


class Plan:
    def __init__(self, sim, spec, params):
        self.sim, self.spec = sim, spec
        self.p = validate_request(sim.adf, params)
        if params['routing'] != 'shortest_hops':
            raise NotImplementedError('routing supports shortest_hops with input-edge-order ties')
        self.ir = workload_from_spec(spec)
        if type(self.ir.iterations) is not int or self.ir.iterations != 1:
            raise ValueError('multi_device requires an explicit single iteration DAG')
        if any(type(d) is not int or d <= 0 for t in self.ir.tensors.values() for d in t.shape):
            raise ValueError('tensor shape dimensions must be positive integers')
        self.paths = params['device_paths']
        self.local = GsmTiled()
        self.states = [self.local.device_state(sim, sim.container(p)) for p in self.paths]
        for state in self.states:
            if state.pim_groups:
                raise NotImplementedError('multi_device does not extend single-controller PIM execution')
            state.full_model = spec.kind in ('vit', 'clip')
            state.full_dtype = self.ir.dtype
        self.shards = {s['tensor']: s for s in params['shards']}
        for name, shard in self.shards.items():
            if name not in self.ir.tensors:
                raise ValueError(f'Unknown shard tensor {name}')
            t = self.ir.tensors[name]
            if shard['axis'] >= len(t.shape) or t.shape[shard['axis']] != shard['size']:
                raise ValueError(f'Shard {name} axis/size does not match tensor shape {t.shape}')
        names = {op.name for op in self.ir.operators}
        for selector in params['placements']:
            if selector != '*' and not any(n == selector or selector.endswith('.*') and n.startswith(selector[:-1]) for n in names):
                raise ValueError(f'Unused placement selector {selector}')
        self.collectives = {}
        for c in params['collectives']:
            if c['after'] not in names or c['tensor'] not in self.ir.tensors:
                raise ValueError('collective must reference an existing producer and tensor')
            if c['after'] in self.collectives:
                raise ValueError('one collective per producer is currently supported')
            if c['operation'] not in ('allreduce', 'allgather', 'reduce_scatter'):
                raise NotImplementedError('unknown collective operation')
            if c['algorithm'] != dict(allreduce='tree', allgather='all_pairs', reduce_scatter='direct')[c['operation']]:
                raise NotImplementedError('collective algorithm not implemented; no fallback')
            dtype_bytes(c['dtype'])
            self.collectives[c['after']] = c
        self.ops = {}
        self.output_stages = {}
        used_shards = set()
        self.memory_minimum = Counter()
        initial_seen = set()
        producers = self.ir.producers()
        for op in self.ir.operators:
            placement = select(params['placements'], op.name)
            if placement['mapping']['template'] != 'gsm_tiled':
                raise NotImplementedError('multi_device local execution currently supports gsm_tiled')
            for r in placement['ranks']:
                node = find(sim.adf, sim.container(self.paths[r]).type_path)
                check_params('gsm_tiled', self.local.space(node, spec), placement['mapping']['params'])
            c = self.collectives.get(op.name)
            if c and (c['tensor'] not in op.outputs or c['ranks'] != placement['ranks']):
                raise ValueError('collective producer/output/rank order must match placement')
            affected = [n for n in op.inputs if n in self.shards]
            partial = bool(affected)
            if partial:
                if op.op != 'matmul' or len(affected) != 2 or len(op.inputs) != 2:
                    raise NotImplementedError('shards currently implement paired GEMM K partitions only')
                a, b = [self.shards[n] for n in op.inputs]
                ta, tb = [self.ir.tensors[n] for n in op.inputs]
                if (a['axis'] != len(ta.shape)-1 or b['axis'] != len(tb.shape)-2
                        or a['size'] != op.attrs['k'] or b['size'] != op.attrs['k']
                        or a['ranges'] != b['ranges']
                        or [v['rank'] for v in a['ranges']] != placement['ranks']):
                    raise ValueError('K shards must pair matching A-last/B-penultimate intervals and rank order')
                if not c or c['operation'] not in ('allreduce', 'reduce_scatter') or c['dtype'] != 'fp32':
                    raise ValueError('K partials require explicit FP32 sum collective')
                used_shards.update(affected)
            if c:
                elements = math.prod(self.ir.tensors[c['tensor']].shape)
                expected_dtype = 'fp32' if partial else self.ir.tensors[c['tensor']].dtype
                if c['elements'] != elements or c['dtype'] != expected_dtype:
                    raise ValueError('collective shape/dtype differs from per-rank producer payload')
                if c['operation'] != 'allgather' and c['dtype'] != 'fp32':
                    raise ValueError('sum collectives require FP32 payloads')
                if c['operation'] != 'allreduce' and any(c['tensor'] in x.inputs for x in self.ir.operators):
                    raise NotImplementedError('shape-changing collectives require terminal outputs')
            for r in placement['ranks']:
                tensors = dict(self.ir.tensors)
                attrs = dict(op.attrs)
                if partial:
                    for n in affected:
                        shard = self.shards[n]
                        interval = next(v for v in shard['ranges'] if v['rank'] == r)
                        shape = list(tensors[n].shape)
                        shape[shard['axis']] = interval['stop'] - interval['start']
                        tensors[n] = replace(tensors[n], shape=tuple(shape))
                    attrs['k'] = tensors[op.inputs[0]].shape[-1]
                    for n in op.outputs:
                        tensors[n] = replace(tensors[n], dtype='fp32')
                    attrs['output_dtype'] = 'fp32'
                local_op = replace(op, attrs=attrs)
                local_ir = replace(self.ir, tensors=tensors)
                self.ops[(op.name, r)] = (local_op, local_ir, partial, placement['mapping']['params'])
                output_stage = MatrixOutputPlan(self.states[r], local_ir, local_op)
                self.output_stages[(op.name, r)] = output_stage
                if op.op not in ('view', 'allreduce'):
                    kernel = local_op.kernel(self.ir.dtype)
                    for unit in self.states[r].units_for(kernel):
                        sim.points[unit].kernel_demand(kernel)  # actual arithmetic capability
                for n in op.inputs:
                    if n not in producers and (n, r) not in initial_seen:
                        initial_seen.add((n, r))
                        self.memory_minimum[r] += tensors[n].bytes
                working = (sum(tensors[n].bytes for n in (*op.inputs, *op.outputs))
                           + op.attrs.get('scratch_bytes', 0) + output_stage.scratch_bytes)
                if working > sim.memory(self.states[r].dram).capacity:
                    raise ValueError(f'Rank {r} operator working set exceeds capacity')
        if used_shards != set(self.shards):
            raise ValueError('Shard has no supported consuming operation')
        for r, size in self.memory_minimum.items():
            if size > sim.memory(self.states[r].dram).capacity:
                raise ValueError(f'Rank {r} resident initial tensors exceed capacity')


class MultiDevice(MappingTemplate):
    name = 'multi_device'

    def applicable(self, adf, workload):
        return workload.kind in ('operator', 'transformer', 'vit', 'clip')

    def space(self, adf, workload):
        # No topology or rank assignment is invented by a default Cartesian space.
        return {}

    def defaults(self):
        return {}

    def request_space(self, adf, workload, params):
        from arch_explorer.simulator.events.builder import build_simulator
        from arch_explorer.mapping.context import MappingContext
        sim = build_simulator(adf)
        sim.mapping = MappingContext(sim)
        plan = Plan(sim, workload, params)
        # Build the graph to check required routes, operators and tensor ownership.
        # This performs no evaluation and reads no reference performance values.
        self._apply(plan)
        return request_space(adf, params)

    def apply(self, simulator, workload, params):
        from arch_explorer.simulator.plan import PlanConstraints
        simulator.mapping.plan.require(PlanConstraints(iterations=1, check_capacity=True,
                                                      reason='explicit multi-device graph'))
        plan = Plan(simulator, workload, params)
        # Exact existing single-device behavior for a degenerate unsharded plan.
        if (len(plan.paths) == 1 and len(sim_devices(simulator)) == 1
                and not params['shards'] and not params['collectives']):
            assignments = [select(params['placements'], op.name) for op in plan.ir.operators]
            if all(a == assignments[0] for a in assignments):
                plan.local.apply(simulator, workload, assignments[0]['mapping']['params'])
                return
        self._apply(plan)

    def _apply(self, plan):
        sim, ir, params = plan.sim, plan.ir, plan.p
        ctx, states, paths = sim.mapping, plan.states, plan.paths
        from arch_explorer.simulator.plan import PlanConstraints
        ctx.plan.require(PlanConstraints(iterations=1, check_capacity=True, reason='explicit multi-device graph'))
        ctx.graph.iterations = 1
        memories = {p: states[r].dram for r, p in enumerate(paths)}
        for edge in params['topology_edges']:
            for p in (edge['src'], edge['dst']):
                if p not in memories:
                    choices = sim.find_points(role='memory', within=p)
                    if not choices:
                        raise ValueError(f'Relay {p} has no buffer memory')
                    memories[p] = choices[0].path
        net = DirectedTransport(ctx, params['topology_edges'], memories)
        copies, ledger, collective_ledger = {}, [], []
        produced = ir.producers()

        def obtain(name, rank, tensor):
            key = (name, rank)
            if key in copies and copies[key].size == tensor.bytes:
                return copies[key]
            if name not in produced:
                buf = ctx.storage(paths[rank]+':'+name, states[rank].dram, tensor.bytes,
                                  static=tensor.kind in ('weight', 'kv'))
                value = Copy(buf, [buf], states[rank].dram, tensor.bytes)
            else:
                candidates = [(r, value) for (n, r), value in copies.items() if n == name and value.size >= tensor.bytes]
                if not candidates:
                    raise ValueError(f'{name}: no complete producer payload for rank {rank}')
                choices = []
                for r, value in candidates:
                    try:
                        choices.append((len(net.route(paths[r], paths[rank])), r, value))
                    except ValueError:
                        pass
                if not choices:
                    raise ValueError(f'{name}: no directed route to consuming rank {rank}')
                _, source, original = min(choices, key=lambda x: (x[0], x[1]))
                if source == rank:
                    # A local K slice aliases its producer; GSM counts its reads.
                    value = Copy(original.ready, original.held, original.memory, tensor.bytes)
                else:
                    region = None
                    if tensor.bytes < ir.tensors[name].bytes:
                        shard = plan.shards[name]
                        interval = next(v for v in shard['ranges'] if v['rank'] == rank)
                        region = dict(axis=shard['axis'], start=interval['start'], stop=interval['stop'])
                    value = net.send(paths[source], paths[rank], original, size=tensor.bytes, name=name, region=region)
            if key not in copies or copies[key].size <= value.size:
                copies[key] = value
            return value

        for op in ir.operators:
            placement = select(params['placements'], op.name)
            partials = {}
            for rank in placement['ranks']:
                state = states[rank]
                local_op, local_ir, partial, local_params = plan.ops[(op.name, rank)]
                output_stage = plan.output_stages[(op.name, rank)]
                inputs = [obtain(n, rank, local_ir.tensors[n]) for n in op.inputs]
                held = [h for value in inputs for h in value.held]
                deps = list(dict.fromkeys([*state.stream, *held, *(v.ready for v in inputs)]))
                if op.op in ('view', 'allreduce'):
                    if len(op.outputs) != 1:
                        raise ValueError('view requires one output')
                    done = ctx.sync(op.name+f'.rank{rank}.view', deps, op.name)
                    copies[(op.outputs[0], rank)] = Copy(done, inputs[0].held, state.dram, inputs[0].size)
                    state.stream = [done]
                    partials[rank] = copies[(op.outputs[0], rank)]
                    ledger.append(dict(operator=op.name, rank=rank, op=op.op, partial=False, flops=0))
                    continue
                if any(ir.tensors[n].strides is not None for n in op.inputs):
                    raise ValueError(f'{op.name}: strided consumer needs materialization')
                allocations = []
                for n in op.outputs:
                    buf = ctx.storage(paths[rank]+':'+n, state.dram, local_ir.tensors[n].bytes, deps, op.name)
                    allocations.append(buf)
                allocations.extend(output_stage.allocate(ctx, deps))
                scratch = op.attrs.get('scratch_bytes', 0)
                if scratch:
                    allocations.append(ctx.storage(op.name+'.scratch', state.dram, scratch, deps, op.name))
                if op.op == 'matmul' and state.l2:
                    cache = min(state.l2_capacity, sum(local_ir.tensors[n].bytes for n in op.inputs)
                                + output_stage.output_bytes)
                    allocations.append(ctx.storage(op.name+'.cache', state.l2, cache, deps, op.name))
                start = plan.local.kernel_start(ctx, state, local_op, [*deps, *allocations])
                if isinstance(op, FullOperator):
                    rows = op.attrs['rows']
                    units = state.units_for(op.kernel())
                    rpc = local_params.get('rows_per_cta', 4)
                    counts = Counter()
                    for first in range(0, rows, rpc):
                        counts[first//rpc % len(units)] += min(rpc, rows-first)
                    tasks = [ctx.map_node(op.name+f'.rank{rank}@'+units[u], units[u],
                               [(op.kernel(rows=count), 1)], start,
                               [(state.dram, (op.attrs['read_bytes']+scratch)*count/rows),
                                (state.dram, (op.attrs['write_bytes']+scratch)*count/rows, 'write')], op.name)
                             for u, count in sorted(counts.items())]
                else:
                    tasks = plan.local.place_operator(ctx, state, local_ir, output_stage.placement_op, local_params, start)
                    tasks = output_stage.finish(ctx, tasks, allocations)
                done = ctx.sync(op.name+f'.rank{rank}.done', [*tasks, *held, *allocations], op.name)
                plan.local.kernel_end(state, done)
                for n, buf in zip(op.outputs, allocations):
                    copies[(n, rank)] = Copy(done, [buf], state.dram, local_ir.tensors[n].bytes)
                partials[rank] = copies[(op.outputs[0], rank)]
                ledger.append(dict(operator=op.name, rank=rank, op=op.op, partial=partial,
                                   shape=list(local_op.kernel(ir.dtype).shape),
                                   flops=local_op.kernel(ir.dtype).flops,
                                   input_bytes=sum(v.size for v in inputs),
                                   input_shapes={n:list(local_ir.tensors[n].shape) for n in op.inputs},
                                   output_bytes=sum(local_ir.tensors[n].bytes for n in op.outputs),
                                   scratch_bytes=scratch, completion_task=done,
                                   **output_stage.report()))
            c = plan.collectives.get(op.name)
            if c:
                result, record = collective(ctx, net, paths, states, partials, c)
                collective_ledger.append(record)
                for r, value in result.items():
                    tensor = ir.tensors[c['tensor']]
                    if c['dtype'] != tensor.dtype:
                        elements = value.size//dtype_bytes(c['dtype'])
                        deps = [value.ready, *value.held]
                        size = elements*dtype_bytes(tensor.dtype)
                        buf = ctx.storage(op.name+'.cast.output', states[r].dram, size, deps, op.name)
                        kernel = Kernel('vector', (1, elements), 'fp32', simt=1)
                        unit = states[r].units_for(kernel)[0]
                        cast = ctx.map_node(op.name+'.cast', unit, [(kernel, 1)], [*deps, buf],
                                            [(value.memory, value.size), (states[r].dram, size, 'write')], op.name)
                        done = ctx.sync(op.name+'.cast.ready', [cast, buf, *value.held], op.name)
                        value = Copy(done, [buf], states[r].dram, size)
                    copies[(c['tensor'], r)] = value
                    states[r].stream = [value.ready]
        consumed = {n for op in ir.operators for n in op.inputs}
        outputs = ir.meta.get('full_model', {}).get('outputs', [n for n in produced if n not in consumed])
        final = [v for (name, _), v in copies.items() if name in outputs]
        ctx.sync('multi_device.return', [i for s in states for i in s.stream] +
                 [v.ready for v in final] + [h for v in final for h in v.held])
        totals = Counter()
        for task in ctx.graph.tasks:
            totals.update(task.demand)
        ctx.details['multi_device'] = dict(device_paths=paths, operators=ledger, shards=params['shards'],
            tensors={n:dict(shape=list(t.shape), dtype=t.dtype, bytes=t.bytes,
                            kind=t.kind, alias_of=t.alias_of) for n,t in ir.tensors.items()},
            collectives=collective_ledger, messages=net.messages, outputs=outputs,
            output_placements=[dict(tensor=n, rank=r, bytes=v.size) for (n, r), v in copies.items() if n in outputs],
            resource_demands=dict(totals), resource_units={r:ctx.graph.resources[r].unit for r in totals},
            storage_policy='initial tensors resident once per rank; views alias; received buffers held to consumers; overflow rejects',
            slice_policy='paired K intervals; local slices alias; remote slices gather selected elements through endpoint ports',
            routing='directed shortest hops, ties in input edge order',
            link_arbitration='from ADF component; serialization separate from propagation',
            precision='FP32 K partial sums; cast after collective; analytical component rates',
            calibrated=False)
        if 'full_model' in ir.meta:
            ctx.details['full_model'] = deepcopy(ir.meta['full_model'])
            ctx.details['full_model'].update(operators=ledger, resource_demands=dict(totals))
            from arch_explorer.mapping.full_models import report_full_model
            ctx.plan.detail_reporters.append(lambda sim, result: report_full_model(plan.spec.kind, sim, result))
        sim.warnings.append('Explicit multi-device analytical mapping; no calibrated host/runtime/NCCL overhead')


register_template(MultiDevice)
