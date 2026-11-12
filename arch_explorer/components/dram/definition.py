from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import MemoryModel
from arch_explorer.components.pim import DRAM_REFRESH, DRAM_TIMING, refresh_of


from .implementations.port import DramModel

from .parameters import TECHNOLOGIES


class Dram(ComponentType):
    from arch_explorer.components.banked_memory import assemble_banked_memory
    assembly_hooks = (assemble_banked_memory,)
    execution_modes = ("demand", "stateful")
    execution_profiles = ("neupims-f299af3",)
    name = "dram"
    role = "memory"
    params = {
        "technology": ParamSpec("1", "str", domain=list(TECHNOLOGIES)),
        "capacity": ParamSpec("B", extensive=True),
        "bandwidth": ParamSpec("B/ns", extensive=True, doc="peak bandwidth"),
        "latency": ParamSpec("ns"),
        "sustained_bandwidth": ParamSpec("B/ns", extensive=True, required=False,
                                         doc="published measured sustained bandwidth; replaces the peak when given"),
        **DRAM_TIMING,          # bank command timing, read only by an in-bank pim_unit
        **DRAM_REFRESH,
    }
    conventions = Conventions(time_base="ns", arbitration="fair_share", pipelined=True, includes_startup=True)

    def build(self, point, context):
        return DramModel(self, point, context)

