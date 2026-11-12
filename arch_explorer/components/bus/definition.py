from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel

from .implementations.port import BusModel

class Bus(ComponentType):
    name = "bus"
    role = "network"
    params = {"bandwidth": ParamSpec("B/ns"), "latency": ParamSpec("ns")}
    conventions = FAIR

    def build(self, point, context):
        return BusModel(self, point, context)

