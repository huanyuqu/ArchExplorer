import math
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel, Demand, NetworkModel


from arch_explorer.components.ipu_tile.definition import IpuTile

class TianjicCore(IpuTile):
    name = "tianjic_core"

