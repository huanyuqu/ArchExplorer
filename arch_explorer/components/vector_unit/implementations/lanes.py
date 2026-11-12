from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components import systolic_model
from arch_explorer.components.base import ComputeModel, Kernel



def _vector_instructions(kernel: Kernel):
    rows, cols = kernel.shape
    simt = rows * cols * kernel.simt + rows * kernel.row_simt
    sfu = rows * cols * kernel.sfu + rows * kernel.row_sfu
    return simt, sfu


class VectorUnitModel(ComputeModel):
    def kernel_time(self, kernel: Kernel) -> float:
        lanes, sfu_lanes = self.positive("lanes"), self.positive("sfu_lanes")
        if kernel.op in ("gemm", "mvm") and not self.param("matrix_kernels", True):
            raise NotImplementedError(f"{self.type_path}: matrix kernels disabled; use matrix hardware")
        if kernel.op == "gemm":
            self.precision(kernel)
            b, m, k, n = kernel.shape
            return b * m * k * n / lanes
        if kernel.op == "mvm":
            self.precision(kernel)
            b, m, k = kernel.shape
            return b * m * k / lanes
        if kernel.op == "vector":
            simt_instr, sfu_instr = _vector_instructions(kernel)
            return max(simt_instr / lanes, sfu_instr / sfu_lanes)
        raise NotImplementedError(f"{self.type_path}: vector_unit cannot run {kernel.op}")

