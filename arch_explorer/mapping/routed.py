"""Place an explicitly located tensor graph using actual streams and copies.

This is a component-domain event mapping. Work is restored by its declared
adapter; numerical operators, access formulas and routes have other owners.
"""
from arch_explorer.adf.registry import MappingTemplate,register_template
from arch_explorer.simulator.plan import PlanConstraints


class AdaptedRoutedEvents(MappingTemplate):
    name='adapted_routed_events'

    def applicable(self,adf,workload):return getattr(workload,'kind',None)=='graph'

    def space(self,adf,workload):raise ValueError('routed events require prepared explicit task products')

    def request_space(self,adf,workload,params=None):
        from arch_explorer.simulator.events.builder import build_simulator
        from arch_explorer.mapping.context import MappingContext
        from arch_explorer.simulator.events.backend import validate_capacity_allocations
        sim=build_simulator(adf,options={'systolic_model':'native'});sim.mapping=MappingContext(sim)
        self.apply(sim,workload,params);validate_capacity_allocations(sim,check_capacity=True)
        return {k:[v] for k,v in params.items()}

    def apply(self,sim,spec,params):
        from arch_explorer.workload.api import workload_from_spec
        from arch_explorer.workload.graph import finite_json,graph_outputs
        from arch_explorer.workload.storage import resolve_layout
        from arch_explorer.workload.adapters import get_adapter,AdapterBinding,options_for
        from arch_explorer.mapping.rules import validate_access
        from .storage_plan import physical_locations
        from arch_explorer.workload.extensions import _record_source
        _record_source(AdaptedRoutedEvents);_record_source(physical_locations)
        graph=workload_from_spec(spec);p=finite_json(params)
        if set(p)!={'tasks'} or type(p['tasks']) is not dict or set(p['tasks'])!={o.name for o in graph.operators} or graph.iterations!=1:
            raise ValueError('routed events need exactly all prepared tasks in one iteration')
        tasks=p['tasks'];where=physical_locations(graph);plan=graph.meta['storage_plan']
        for memory in set(where.values()):
            if memory not in sim.points or sim.points[memory].role!='memory':
                raise ValueError('routed storage requires actual single memory points: '+str(memory))
            sim.memory(memory)
        works={};accesses={}
        for op in graph.operators:
            item=tasks[op.name]
            if type(item) is not dict or set(item)!={'compute','memory','adapter','work','traffic','access'} or item['traffic']!='external':
                raise ValueError('routed task fields/traffic differ')
            if item['compute'] not in sim.points or sim.points[item['compute']].role!='compute':
                raise ValueError('routed compute must be an actual compute point')
            component=sim.points[item['compute']].component_type
            binding=AdapterBinding.from_dict(item['adapter']);adapter=get_adapter(binding.id)
            if (binding.id not in component.implementation.adapters or binding.version!=adapter.version
                    or adapter.execution_domain!=component.implementation.execution_domain or adapter.restore is None
                    or op.op not in adapter.semantics or options_for(adapter,binding.options)!=binding.options):
                raise ValueError('routed task component/adapter contract differs')
            work=adapter.restore(item['work']);counts=validate_access(item['access'])
            if op.op in ('storage.view','storage.copy'):
                if work is not None or counts['operations']!=0:raise ValueError('storage control/copy requires explicit zero arithmetic')
            else:sim.points[item['compute']].kernel_demand(work)
            works[op.name]=work;accesses[op.name]=counts
        ctx=sim.mapping;ctx.graph.iterations=1
        ctx.plan.require(PlanConstraints(iterations=1,check_capacity=True,reason='explicit located storage and real reader lifetimes'))
        produced={n:op.name for op in graph.operators for n in op.outputs}
        ready={};allocations={};done={};transfers=[]
        def root(n):return resolve_layout(graph.tensors,n).storage
        def deps(names):return list(dict.fromkeys(i for n in names for i in ready[n]))
        for n,t in graph.tensors.items():
            if n not in produced:
                if t.alias_of is not None:raise ValueError('routed external roots must own their storage')
                job=ctx.storage('storage:'+n,where[n],t.bytes,static=t.kind=='weight')
                allocations[n]=job;ready[n]=[job]
        for op in graph.operators:
            prerequisites=deps(op.inputs)
            for n in op.outputs:
                t=graph.tensors[n]
                if t.alias_of is None:
                    allocations[n]=ctx.storage('storage:'+n,where[n],t.bytes,prerequisites,op.name,op.label)
                else:allocations[n]=allocations[root(n)]
            allocs=[allocations[n] for n in op.outputs]
            if op.op=='storage.view':
                job=ctx.sync(op.name+'.alias',[*prerequisites,*allocs],op.name,op.label)
            elif op.op=='storage.copy':
                item=plan['transfers'].get(op.name)
                if item is None or (op.inputs!=(item['source'],) or tuple(op.outputs)!=tuple(d['tensor'] for d in item['destinations'])):
                    raise ValueError('storage copy task differs from its declared transfer')
                src=where[op.inputs[0]];targets=[where[n] for n in op.outputs];size=graph.tensors[op.inputs[0]].bytes
                if item['multicast']:
                    job=ctx.map_multicast(op.name,src,targets,size,[*prerequisites,*allocs],op.name)
                    jobs=[job]
                else:
                    jobs=[ctx.map_edge(op.name+'.to.'+str(i),src,d,size,[*prerequisites,allocs[i]],op.name) for i,d in enumerate(targets)]
                    job=ctx.sync(op.name+'.complete',jobs,op.name)
                transfers.append(dict(operator=op.name,source=src,destinations=targets,bytes=size,multicast=item['multicast'],tasks=jobs))
            else:
                item=tasks[op.name];counts=accesses[op.name]
                reads=counts.get('read_by_tensor')
                if reads is None:
                    if len({where[n] for n in op.inputs})!=1:raise ValueError('cross-memory adapter requires authoritative read_by_tensor')
                    reads={op.inputs[0]:counts['read_bytes']} if op.inputs else {}
                if (set(reads)-set(op.inputs) or any(type(v) not in (int,float) or v<0 for v in reads.values())
                        or sum(reads.values())!=counts['read_bytes']):raise ValueError('routed read decomposition differs')
                if counts['write_bytes']!=sum(graph.tensors[n].bytes for n in op.outputs):
                    raise ValueError('routed independent output stores differ from access declaration')
                streams=[(where[n],b) for n,b in reads.items() if b]
                streams.extend((where[n],graph.tensors[n].bytes,'write') for n in op.outputs)
                latency=max((sim.access_latency(m,item['compute']) for m,b,*_ in streams if b),default=0)
                job=ctx.map_node(op.name+'@'+item['compute'],item['compute'],[(works[op.name],1)],
                                 [*prerequisites,*allocs],streams,op.name,op.label,latency_ns=latency)
            done[op.name]=job
            for n in op.outputs:ready[n]=[allocations[n],job]
        if any('has no network' in warning for warning in sim.warnings):
            raise ValueError('routed events require declared networks for every actual storage/compute path')
        ctx.sync('compiled.return',[*done.values(),*deps(graph_outputs(graph))])
        ctx.details['compilation']=dict(graph=graph.to_dict(),tasks=tasks,accesses=accesses,storage_plan=plan,
            transfers=transfers,storage_tasks=allocations,output_hold='all actual root/alias readers and returned roots',hardware_accuracy=None)


register_template(AdaptedRoutedEvents)
