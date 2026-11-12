from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel



class CgraPeModel(ComputeModel):
    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        if self.param("precision") != "uint32":
            raise ValueError("cgra_pe currently supports only scalar uint32")

    def kernel_time(self, kernel):
        self.precision(kernel)
        if kernel.op not in ("add", "mul", "load", "store") or kernel.shape != (1,):
            raise NotImplementedError("cgra_pe requires one scalar add/mul/load/store, shape (1,)")
        # AdderRTL/MulRTL: Fu(..., latency=1, vector_factor_power=0).
        # Load/store: one issue slot; SRAM timing is charged separately.
        return 1.0

