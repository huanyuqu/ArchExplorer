"""Explicit component-IO roofline bound; one MAC is two operations.

No byte-count inference, device calibration or second charge to external
memory resources occurs here. Ordinary Kernel input cannot express this IO
boundary and is deliberately refused.
"""
from dataclasses import dataclass
import math

from arch_explorer.adf.registry import Conventions, ParamSpec
from arch_explorer.components.base import BuildContext, ComputeModel, Demand, Provenance
from arch_explorer.components.implementations import ComponentImplementation


@dataclass(frozen=True)
class RooflineWork:
    operations: float
    boundary_bytes: float
    boundary: str

    def __post_init__(self):
        for value in (self.operations, self.boundary_bytes):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('roofline operations/bytes must be finite nonnegative quantities')
        if self.operations == self.boundary_bytes == 0:
            raise ValueError('roofline requires nonempty work')
        if self.boundary != 'component_io':
            raise ValueError('roofline requires the explicit component_io boundary')


def validate(point):
    for key in ('peak_ops_per_ns', 'bandwidth_bytes_per_ns'):
        param = point.params[key]
        if param.value <= 0 or not isinstance(param.source, str) or not param.source.strip():
            raise ValueError('positive source-attributed roofline '+key+' required')
    if 'precision' not in point.params or point.value('precision') not in ('fp32', 'fp16', 'bf16', 'int8', 'fp64'):
        raise ValueError('roofline requires explicit supported common precision')


class RooflineModel(ComputeModel):
    input_contract = 'roofline.work.v1'
    includes_boundary_traffic = True

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.peak = self.positive('peak_ops_per_ns')
        self.bandwidth = self.positive('bandwidth_bytes_per_ns')
        self.boundary = self.param('boundary')
        self.param('precision')

    def estimate(self, work):
        if not isinstance(work, RooflineWork):
            raise ValueError('roofline requires RooflineWork with explicit F/B; ordinary Kernel has no boundary bytes')
        work.__post_init__()
        if work.boundary != self.boundary:
            raise ValueError('roofline work and implementation boundary differ')
        compute, memory = work.operations/self.peak, work.boundary_bytes/self.bandwidth
        if not math.isfinite(compute) or not math.isfinite(memory):
            raise ValueError('roofline duration overflowed the finite time contract')
        return dict(latency_ns=max(compute, memory), compute_ns=compute, memory_ns=memory,
                    operations=work.operations, boundary_bytes=work.boundary_bytes, boundary=self.boundary,
                    mac_operations=2, result_scope='analytical_estimate',
                    assumptions=['ideal_compute_memory_overlap', 'exclusive_component_service',
                                 'component_io_not_charged_to_external_memory_or_network'],
                    hardware_accuracy=None)

    def kernel_demand(self, work):
        return Demand({self.unit.name: self.estimate(work)['latency_ns']})


def factory(component, point, context):
    return RooflineModel(component, point, context)


DEFINITION = ComponentImplementation(
    component='sm', id='sm.roofline', version='1',
    source={'basis': 'standard roofline bound max(F/P,B/BW)', 'calibration': 'none'},
    factory=factory, validate=validate, execution_domain='events', execution_modes=('demand',), adapters=('roofline',),
    input_contract='roofline.work.v1', result_scope='Analytical estimate under ideal overlap; not native cycles or calibrated timing',
    params={
        'peak_ops_per_ns': ParamSpec('op/ns', extensive=True),
        'bandwidth_bytes_per_ns': ParamSpec('B/ns', extensive=True),
        'boundary': ParamSpec('1', 'str', domain=['component_io']),
    },
    conventions=Conventions(time_base='ns', arbitration='fifo', includes_startup=False),
    capabilities={'operations': 'explicit count; MAC=2', 'boundary': 'component_io',
                  'contained_components': [], 'external_streams': False, 'hardware_accuracy': None})


def evaluate_roofline(point, work):
    """Execute one explicitly accounted component task on the public event engine."""
    from arch_explorer.adf.evaluation import Evaluation
    from arch_explorer.components.implementations import resolve_component
    from arch_explorer.simulator.events.engine import Task, TaskGraph, simulate
    resolved = resolve_component(point, execution_domain='events')
    provenance = Provenance()
    model = resolved.build(point, BuildContext(point.name, point.name, (), provenance))
    if not isinstance(model, RooflineModel):
        raise ValueError('evaluate_roofline requires the roofline implementation')
    demand = model.kernel_demand(work)
    graph = TaskGraph()
    for resource in model.resources:
        graph.add_resource(resource)
    graph.add(Task('work', demand=demand.resources))
    trace = simulate(graph)
    return Evaluation(trace.latency_ns, provenance=provenance.as_list(), backend='events',
                      warnings=['Analytical roofline estimate; hardware accuracy unvalidated'],
                      details={'roofline': model.estimate(work), 'component_implementations': [resolved.identity()]})
