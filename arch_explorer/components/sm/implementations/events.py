"""SM demand implementation reusing the existing vector-lane mechanism.

This is the ordinary event model's declared service abstraction, not a native
SM predictor or an inferred conversion of roofline bandwidth into lanes.
"""
from dataclasses import dataclass
from arch_explorer.adf.registry import Conventions
from arch_explorer.components.implementations import ComponentImplementation
from arch_explorer.components.vector_unit.definition import VectorUnit
from arch_explorer.components.vector_unit.implementations.lanes import VectorUnitModel
from arch_explorer.components.base import Kernel


@dataclass(frozen=True)
class PrecisionWork:
    kernel: Kernel
    conversion_instructions: int

    def __post_init__(self):
        if (type(self.kernel) is not Kernel or self.kernel.precision != 'fp32'
                or type(self.conversion_instructions) is not int or self.conversion_instructions < 0):
            raise ValueError('precision work requires an FP32 Kernel and nonnegative integer scalar conversions')


class EventsModel(VectorUnitModel):
    def kernel_time(self, work):
        if isinstance(work, PrecisionWork):
            work.__post_init__()
            # Explicit serial scalar conversion service, at the component's
            # existing SIMT lanes; storage dtype never implies a faster ALU.
            return super().kernel_time(work.kernel) + work.conversion_instructions/self.positive('lanes')
        return super().kernel_time(work)


def factory(component, point, context):
    return EventsModel(component,point,context)


DEFINITION = ComponentImplementation(component='sm',id='sm.events',version='1',
    source={'basis':'existing VectorUnitModel lane demand mechanism','calibration':'none'},
    factory=factory,execution_domain='events',execution_modes=('demand',),adapters=('events',),
    input_contract='events.kernel-or-precision.v1',hardware_parameters=('precision',),
    params={k:v for k,v in VectorUnit.params.items() if k != 'precision'},
    conventions=Conventions(time_base='cycle',arbitration='fifo',pipelined=False,includes_startup=False),
    result_scope='Event schedule using explicit vector/FMA lanes and memory streams; not native SM timing',
    capabilities={'mechanism':'VectorUnitModel','memory':'external streams','hardware_accuracy':None})
