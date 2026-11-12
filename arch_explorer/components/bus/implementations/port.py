from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel


class BusModel(NetworkModel):
    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.bus = self.resource("bus", self.positive("bandwidth"), "B/ns")
        self.latency = self.param("latency")

    def route(self, src, dst):
        self._check(src, dst)
        if src == dst:
            return Demand()
        return Demand({self.bus.name: 1.0}, self.latency)

