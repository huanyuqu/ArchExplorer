from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import MemoryModel
from arch_explorer.components.pim import DRAM_REFRESH, DRAM_TIMING, refresh_of



class Sram(ComponentType):
    name = "sram"
    role = "memory"
    params = {
        "capacity": ParamSpec("B", extensive=True),
        "bandwidth": ParamSpec("B/ns", extensive=True),
        "latency": ParamSpec("ns"),
    }
    conventions = Conventions(time_base="ns", arbitration="fair_share", pipelined=True, includes_startup=True)

    def build(self, point, context):
        return MemoryModel(self, point, context)

