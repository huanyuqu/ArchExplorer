from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components import systolic_model
from arch_explorer.components.base import ComputeModel, Kernel


from .implementations.timing import SystolicArrayModel

POWERS_OF_TWO = [8, 16, 32, 64, 128, 256]


class SystolicArray(ComponentType):
    execution_modes = ("demand", "stateful")
    execution_profiles = ("neupims-f299af3",)
    name = "systolic_array"
    role = "compute"
    params = {
        "rows": ParamSpec("count", "int", domain=POWERS_OF_TWO, doc="PE rows"),
        "cols": ParamSpec("count", "int", domain=POWERS_OF_TWO, doc="PE columns"),
        "clock_ghz": ParamSpec("GHz"),
        "dataflow": ParamSpec("1", "str", domain=["os", "ws", "is"], doc="SCALE-Sim dataflow"),
        "precision": ParamSpec("1", "str", doc="operand precision of the MACs, e.g. fp16"),
        "stationary_buffering": ParamSpec("1", "str", required=False, domain=list(systolic_model.BUFFERINGS),
                                          doc="'double': shadow registers load the next stationary tile during a "
                                              "fold (ws/is); default 'single' (SCALE-Sim)"),
    }
    conventions = Conventions(time_base="cycle", arbitration="fifo", pipelined=False, includes_startup=True)

    def build(self, point, context):
        return SystolicArrayModel(self, point, context)

