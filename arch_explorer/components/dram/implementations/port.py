from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import MemoryModel
from arch_explorer.components.dram.parameters import TECHNOLOGIES
from arch_explorer.components.pim import DRAM_REFRESH, DRAM_TIMING, refresh_of



class DramModel(MemoryModel):
    def __init__(self, component, point, context):
        self.technology = None
        super().__init__(component, point, context)
        self.technology = self.param("technology")
        if self.technology not in TECHNOLOGIES:
            raise ValueError(f"{self.type_path}.technology must be one of {TECHNOLOGIES}")

    def port_bandwidth(self):
        peak = self.positive("bandwidth")
        sustained = self.positive("sustained_bandwidth", None)
        if sustained is None:
            refresh = refresh_of(self)
            return peak if refresh is None else peak * (1 - refresh[0] / refresh[1])
        if sustained > peak:
            raise ValueError(f"{self.type_path}: sustained_bandwidth {sustained} exceeds peak bandwidth {peak}")
        return sustained

