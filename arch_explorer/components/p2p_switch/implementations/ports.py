from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel


class P2pModel(PortedModel):
    """NVLink/NVSwitch-style links; ``pair_links`` limits each directly connected pair."""

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.link_bandwidth = self.positive("link_bandwidth")
        self.links_per_endpoint = self.positive("links_per_endpoint")
        self.pair_links = self.positive("pair_links", None)
        self.pairs = {}

    def port_capacity(self):
        return self.links_per_endpoint * self.link_bandwidth

    def route(self, src, dst):
        demand = super().route(src, dst)
        if self.pair_links is not None and src != dst:
            key = (src, dst)
            if key not in self.pairs:
                self.pairs[key] = self.resource(f"pair{src}->{dst}", self.pair_links * self.link_bandwidth, "B/ns")
            demand.resources[self.pairs[key].name] = 1.0
        return demand

