"""Place already-adapted component inputs in one dependency/lifetime graph.

This layer restores opaque work through its declared adapter. It never lowers
operators, emits instructions or chooses a compute implementation.
"""
from arch_explorer.adf.registry import MappingTemplate, register_template
from arch_explorer.mapping.base import DeviceState, PlacementTemplate, sim_devices
from arch_explorer.simulator.plan import PlanConstraints


class _Placement(PlacementTemplate):
    def __init__(self, tasks, works, accesses):
        self.tasks, self.works, self.accesses = tasks, works, accesses

    def space(self, adf, workload):
        return {}

    def place_operator(self, ctx, state, workload, op, params, deps):
        item = self.tasks[op.name]
        counts = self.accesses[op.name]
        if op.op=='storage.view':
            return [ctx.sync(op.name+'.alias',deps,op.name,op.label)]
        streams = [] if item['traffic'] == 'included' else [
            (state.dram, counts['read_bytes']), (state.dram, counts['write_bytes'], 'write')]
        return [ctx.map_node(op.name+'@'+item['compute'], item['compute'],
            [(self.works[op.name],1)], deps, streams, op.name, op.label,
            latency_ns=0 if not any(stream[1] for stream in streams) else ctx.sim.access_latency(state.dram,item['compute']))]

    def allocate_outputs(self,ctx,state,workload,op,params,deps):
        result={}
        for name in op.outputs:
            t=workload.tensors[name]
            if t.alias_of is not None:
                result[name]=state.deps(ctx,workload,[t.alias_of],op.name)
            else:
                result[name]=[ctx.storage(state.path+':'+name,state.dram,t.bytes,deps,op.name,op.label)]
        return result


class AdaptedEvents(MappingTemplate):
    name = 'adapted_events'

    def applicable(self, adf, workload):
        from arch_explorer.mapping.base import adf_devices
        devices = adf_devices(adf)
        return len(devices) == 1 and devices[0][1] == 1

    def space(self, adf, workload):
        raise ValueError('adapted mappings require explicitly prepared component inputs')

    def request_space(self, adf, workload, params=None):
        from arch_explorer.simulator.events.builder import build_simulator
        from arch_explorer.mapping.context import MappingContext
        from arch_explorer.simulator.events.backend import validate_capacity_allocations
        sim = build_simulator(adf, options={'systolic_model':'native'})
        sim.mapping = MappingContext(sim)
        self.apply(sim,workload,params)
        validate_capacity_allocations(sim,check_capacity=True)
        return {k:[v] for k,v in params.items()}

    def apply(self, sim, spec, params):
        from arch_explorer.workload.api import workload_from_spec
        from arch_explorer.workload.adapters import get_adapter, AdapterBinding, options_for
        from arch_explorer.mapping.rules import validate_access
        from arch_explorer.workload.graph import finite_json
        graph = workload_from_spec(spec)
        p = finite_json(params)
        if type(p) is not dict or set(p) != {'tasks'} or type(p['tasks']) is not dict:
            raise ValueError('adapted_events requires exactly tasks')
        tasks = p['tasks']
        if set(tasks) != {op.name for op in graph.operators} or graph.iterations != 1:
            raise ValueError('adapted events must cover every task in one graph iteration')
        devices = sim_devices(sim)
        if len(devices) != 1:
            raise ValueError('adapted events requires one shared execution device')
        state = DeviceState(sim,devices[0])
        memories, works, accesses = set(), {}, {}
        for op in graph.operators:
            item = tasks[op.name]
            if type(item) is not dict or set(item) != {'compute','memory','adapter','work','traffic','access'}:
                raise ValueError('adapted task fields differ')
            if item['compute'] not in state.computes:
                raise ValueError('adapted compute must be an actual point in the shared device')
            sim.memory(item['memory'])
            memories.add(item['memory'])
            binding = AdapterBinding.from_dict(item['adapter'])
            adapter = get_adapter(binding.id)
            component = sim.points[item['compute']].component_type
            if (binding.id not in component.implementation.adapters or binding.version != adapter.version
                    or adapter.execution_domain != component.implementation.execution_domain
                    or adapter.restore is None or op.op not in adapter.semantics):
                raise ValueError('adapted task component/adapter contract differs')
            if options_for(adapter,binding.options) != binding.options:
                raise ValueError('adapted options must be normalized')
            if item['traffic'] not in ('external','included'):
                raise ValueError('adapted task requires an explicit traffic boundary')
            work = adapter.restore(item['work'])
            if op.op=='storage.view':
                if work is not None or any(item['access'][k] for k in ('operations','read_bytes','write_bytes')):
                    raise ValueError('readonly view requires explicit zero-service adapter work')
            else:sim.points[item['compute']].kernel_demand(work)
            works[op.name], accesses[op.name] = work, validate_access(item['access'])
        if len(memories) != 1:
            raise NotImplementedError('no implicit cross-memory transfer bridge')
        state.dram = next(iter(memories))
        ctx = sim.mapping
        ctx.plan.require(PlanConstraints(iterations=1,check_capacity=True,reason='adapted materialized semantic DAG'))
        ctx.graph.iterations = 1
        policy = _Placement(tasks,works,accesses)
        done = [policy.run_device(ctx,state,graph,op,{}) for op in graph.operators]
        from arch_explorer.workload.graph import graph_outputs
        held = [i for n in graph_outputs(graph) for i in state.ready[n]]
        ctx.sync('compiled.return',[*done,*held])
        ctx.details['compilation'] = dict(graph=graph.to_dict(),tasks=tasks,accesses=accesses,
            output_hold='all real consumers and return hold shared storage',hardware_accuracy=None)


register_template(AdaptedEvents)
