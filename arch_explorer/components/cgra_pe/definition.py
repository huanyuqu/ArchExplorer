from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel


from .implementations.pipeline import CgraPeModel

class CgraPe(ComponentType):
    name = "cgra_pe"
    role = "compute"
    params = {"clock_ghz": ParamSpec("GHz"), "precision": ParamSpec("1", "str")}
    conventions = Conventions(time_base="cycle", arbitration="fifo", pipelined=False, includes_startup=False)

    def build(self, point, context):
        return CgraPeModel(self, point, context)

