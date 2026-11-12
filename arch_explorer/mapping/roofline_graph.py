"""Explicit per-node F/B on the existing workload dependency DAG.

No external memory/network demand is charged a second time. This bound does
not claim external queueing, resident capacity, or native instruction timing.
"""
from arch_explorer.adf.registry import MappingTemplate, register_template
from arch_explorer.adf.dynamic import instances
from arch_explorer.components.implementations import resolve_component
from arch_explorer.components.sm.implementations.roofline import RooflineWork
from arch_explorer.workload.llm import workload_from_spec


class RooflineGraph(MappingTemplate):
    name = 'roofline_graph'

    def applicable(self, adf, workload):
        return any(resolve_component(p).implementation.input_contract == 'roofline.work.v1'
                   for _, p in instances(adf))

    def request_space(self, adf, workload, params=None):
        params = {} if params is None else params
        if not isinstance(params, dict) or set(params) != {'compute', 'work'}:
            raise ValueError('roofline_graph requires explicit compute and work')
        points = dict(instances(adf))
        if params['compute'] not in points:
            raise ValueError('roofline_graph compute must be an exact physical point path')
        resolved = resolve_component(points[params['compute']], execution_domain='events')
        if resolved.implementation.input_contract != 'roofline.work.v1':
            raise ValueError('selected compute cannot consume explicit roofline work')
        graph = workload_from_spec(workload)
        values = params['work']
        if not isinstance(values, dict) or set(values) != {op.name for op in graph.operators}:
            raise ValueError('explicit F/B required for exactly every graph operator')
        for value in values.values():
            if not isinstance(value, dict) or set(value) != {'operations', 'boundary_bytes', 'boundary'}:
                raise ValueError('exact operations/boundary_bytes/boundary required per node')
            RooflineWork(**value)
        return {k:[v] for k,v in params.items()}

    def space(self, adf, workload):
        raise ValueError('roofline_graph search requires explicit request-sized work')

    def apply(self, simulator, workload, params):
        self.request_space(simulator.adf, workload, params)
        graph = workload_from_spec(workload)
        produced = {}
        for op in graph.operators:
            deps = sorted({produced[n] for n in op.inputs if n in produced})
            work = RooflineWork(**params['work'][op.name])
            task = simulator.mapping.map_node(op.name, params['compute'], [(work,1)], deps,
                                              operator=op.name, label=op.label)
            for name in op.outputs:
                produced[name] = task
        simulator.mapping.details['roofline_scope'] = dict(result_scope='analytical_estimate',
            boundary='component_io', operations_convention='MAC=2',
            external_memory_network='not charged; no queueing or capacity estimate',
            graph='original DAG dependencies; selected component exclusive FIFO service', hardware_accuracy=None)
        simulator.warnings.append('Analytical component-IO roofline; external memory/network timing not modeled')


register_template(RooflineGraph)
