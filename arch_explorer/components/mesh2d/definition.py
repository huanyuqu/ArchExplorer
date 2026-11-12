from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import GATEWAY, Demand, NetworkModel

from arch_explorer.components._support.network import FAIR, PortedModel

from .implementations.xy import MeshModel

class Mesh2d(ComponentType):
    name = "mesh2d"
    role = "network"
    params = {
        "rows": ParamSpec("count", "int"),
        "cols": ParamSpec("count", "int"),
        "link_bandwidth": ParamSpec("B/ns", doc="per direction per link"),
        "hop_latency": ParamSpec("ns"),
        "diagonal": ParamSpec("1", "bool", required=False,
                              doc="KING_MESH: diagonal-first shortest route, then XY; default false"),
        "gateway_tile": ParamSpec("count", "int", required=False,
                                  doc="tile (row-major index) where traffic to the parent level enters; default 0"),
    }
    conventions = FAIR

    def build(self, point, context):
        return MeshModel(self, point, context)

