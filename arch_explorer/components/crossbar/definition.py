from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel

from .implementations.ports import CrossbarModel

class Crossbar(ComponentType):
    name = "crossbar"
    role = "network"
    params = {"port_bandwidth": ParamSpec("B/ns", doc="per port per direction"), "latency": ParamSpec("ns")}
    conventions = FAIR

    def build(self, point, context):
        return CrossbarModel(self, point, context)

