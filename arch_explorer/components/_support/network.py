from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel



FAIR = Conventions(time_base="ns", arbitration="fair_share", pipelined=True, includes_startup=True)


class PortedModel(NetworkModel):
    """Non-blocking switch: a transfer occupies the source egress and destination ingress ports."""

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.latency = self.param("latency")
        self.ports = {}

    def port_capacity(self):
        raise NotImplementedError

    def port(self, direction, endpoint):
        key = (direction, endpoint)
        if key not in self.ports:
            label = "gw" if endpoint == GATEWAY else endpoint
            self.ports[key] = self.resource(f"{direction}{label}", self.port_capacity(), "B/ns")
        return self.ports[key]

    def route(self, src, dst):
        self._check(src, dst)
        if src == dst:
            return Demand()
        return Demand({self.port("out", src).name: 1.0, self.port("in", dst).name: 1.0}, self.latency)

