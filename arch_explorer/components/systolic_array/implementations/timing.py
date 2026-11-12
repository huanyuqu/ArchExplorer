from __future__ import annotations
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components import systolic_model
from arch_explorer.components.base import ComputeModel, Kernel



class SystolicArrayModel(ComputeModel):
    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.rows = self.positive("rows")
        self.cols = self.positive("cols")
        self.dataflow = self.param("dataflow")
        if self.dataflow not in systolic_model.DATAFLOWS:
            raise ValueError(f"{self.type_path}.dataflow must be one of {systolic_model.DATAFLOWS}")
        self.buffering = self.param("stationary_buffering", "single")
        if self.buffering not in systolic_model.BUFFERINGS:
            raise ValueError(f"{self.type_path}.stationary_buffering must be one of {systolic_model.BUFFERINGS}")
        if self.buffering == "double" and self.dataflow == "os":
            raise ValueError(f"{self.type_path}: stationary_buffering 'double' needs dataflow ws or is")
        self.model = context.options.get("systolic_model", "auto")

    def gemm(self, m, k, n):
        result = systolic_model.gemm(self.rows, self.cols, self.dataflow, m, k, n, model=self.model,
                                     buffering=self.buffering)
        source = ("SCALE-Sim v3 run at evaluation time (scalesim package)" if result.model.startswith("scalesim")
                  else "native closed form reproducing SCALE-Sim v3 dataflow timing")
        if self.buffering == "double":
            source += "; next stationary tile loaded during each fold (docs/components.md, systolic double buffering)"
        self.provenance.record(self.type_path, "gemm_model", result.model, "1", source)
        return result

    def kernel_time(self, kernel: Kernel) -> float:
        self.precision(kernel)
        if kernel.op == "gemm":
            b, m, k, n = kernel.shape
        elif kernel.op == "mvm":
            (b, m, k), n = kernel.shape, 1
        else:
            raise NotImplementedError(f"{self.type_path}: systolic_array runs gemm/mvm; map {kernel.op} to a vector_unit")
        result = self.gemm(m, k, n)
        if self.buffering == "single":
            return b * result.compute_cycles
        # The b GEMMs form one chain of folds: the first of each later GEMM also loads during its predecessor.
        interval = systolic_model.fold_interval(self.rows, self.cols, self.dataflow, m, k, n, "double")
        return result.compute_cycles + (b - 1) * result.row_folds * result.col_folds * interval

    def sram_accesses(self, kernel: Kernel):
        """Element reads/writes SCALE-Sim reports for the kernel (ifmap, filter, ofmap)."""
        b, m, k, n = kernel.shape if kernel.op == "gemm" else (*kernel.shape, 1)
        result = self.gemm(m, k, n)
        return (b * result.ifmap_sram_reads, b * result.filter_sram_reads, b * result.ofmap_sram_writes)

