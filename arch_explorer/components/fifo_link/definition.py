from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel

from arch_explorer.components.bus.definition import Bus

class FifoLink(Bus):
    """Explicit non-preemptive FIFO link, with the same B/ns and ns units.

    This is a selectable queue discipline, not a device bandwidth estimate.
    The mapping releases this resource before propagation begins.
    """
    name = "fifo_link"
    conventions = Conventions(time_base="ns", arbitration="fifo")

