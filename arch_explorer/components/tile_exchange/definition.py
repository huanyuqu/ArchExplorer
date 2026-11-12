import math
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel, Demand, NetworkModel


from .implementations.exchange import TileExchangeModel

class TileExchange(ComponentType):
    name = "tile_exchange"
    role = "network"
    params = {
        "tiles": ParamSpec("count", "int"), "topology": ParamSpec("1", "str"),
        "endpoint_bandwidth": ParamSpec("B/ns", required=False),
        "rows": ParamSpec("count", "int", required=False), "cols": ParamSpec("count", "int", required=False),
    }
    conventions = Conventions(time_base="ns", arbitration="fair_share", pipelined=True, includes_startup=False)

    def build(self, point, context):
        return TileExchangeModel(self, point, context)

