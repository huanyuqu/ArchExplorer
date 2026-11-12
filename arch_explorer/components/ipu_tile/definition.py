import math
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel, Demand, NetworkModel


from .implementations.tile import TileComputeModel

class IpuTile(ComponentType):
    name = "ipu_tile"
    role = "compute"
    params = {
        "clock_ghz": ParamSpec("GHz"), "precision": ParamSpec("1", "str"),
        "macs_per_cycle": ParamSpec("MAC/cycle"),
        "vector_ops_per_cycle": ParamSpec("ops/cycle", required=False),
        "sync_cycles": ParamSpec("cycle", required=False),
        "identity": ParamSpec("1", "str"), "accumulation": ParamSpec("1", "str"),
        "output_precision": ParamSpec("1", "str"),
    }
    conventions = Conventions(time_base="cycle", arbitration="fifo", pipelined=False, includes_startup=False)

    def build(self, point, context):
        return TileComputeModel(self, point, context)

