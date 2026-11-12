from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components import systolic_model
from arch_explorer.components.base import ComputeModel, Kernel


from arch_explorer.components.systolic_array.definition import POWERS_OF_TWO
from .implementations.lanes import VectorUnitModel

class VectorUnit(ComponentType):
    name = "vector_unit"
    role = "compute"
    params = {
        "matrix_kernels": ParamSpec("1", "bool", required=False,
                                    doc="Allow matrix kernels on vector lanes (default true)."),
        "lanes": ParamSpec("count", "int", domain=POWERS_OF_TWO, doc="lane instructions (one FMA each) per cycle"),
        "sfu_lanes": ParamSpec("count", "int", doc="special-function results per cycle"),
        "clock_ghz": ParamSpec("GHz"),
        "precision": ParamSpec("1", "str"),
    }
    conventions = Conventions(time_base="cycle", arbitration="fifo", pipelined=False, includes_startup=False)

    def build(self, point, context):
        return VectorUnitModel(self, point, context)

