from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel


class CrossbarModel(PortedModel):
    def port_capacity(self):
        return self.positive("port_bandwidth")

