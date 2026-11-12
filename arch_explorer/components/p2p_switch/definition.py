from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel

from .implementations.ports import P2pModel

class P2pSwitch(ComponentType):
    name = "p2p_switch"
    role = "network"
    params = {
        "link_bandwidth": ParamSpec("B/ns", doc="per link per direction"),
        "links_per_endpoint": ParamSpec("count", "int"),
        "latency": ParamSpec("ns"),
        "pair_links": ParamSpec("count", "int", required=False,
                                doc="links between each endpoint pair in a direct (switchless) topology"),
    }
    conventions = FAIR

    def build(self, point, context):
        return P2pModel(self, point, context)

