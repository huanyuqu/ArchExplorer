import math
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import ComputeModel, Demand, NetworkModel



class TileComputeModel(ComputeModel):
    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.rate = self.positive("macs_per_cycle")
        self.vector_rate = self.positive("vector_ops_per_cycle")
        self.sync_cycles = self.param("sync_cycles", 0)
        if not isinstance(self.sync_cycles, (int, float)) or not math.isfinite(self.sync_cycles) or self.sync_cycles < 0:
            raise ValueError("sync_cycles must be finite and nonnegative")
        self.param("identity")
        accumulation = self.param("accumulation")
        output = self.param("output_precision")
        native = self.param("precision")
        expected = "fp16" if component.name == "ipu_tile" else "int8"
        if native != expected:
            raise ValueError(f"{component.name} supports only {expected} in this model")
        if (accumulation, output) != (("fp32", "fp16") if expected == "fp16" else ("int32", "int32")):
            raise ValueError("unsupported accumulation/output format for tile model")

    def kernel_time(self, kernel):
        if kernel.op == "fp32_add":
            if self.component.name != "ipu_tile" or kernel.precision != "fp32" or self.vector_rate is None:
                raise NotImplementedError("FP32 reduction requires IPU FP32 vector arithmetic")
            # Graphcore tile-level parallelism (2021): 2 FP32 operations/cycle,
            # one instruction issued across barrel-scheduled workers, not 6x2.
            return math.prod(kernel.shape) / self.vector_rate
        self.precision(kernel)
        if kernel.op == "gemm_accumulate":
            # GC200 HC2021 slide13: f16 multiply/f32 memory 125TF/s,
            # versus f16 memory 250TF/s. TianjicX already stores int32 sums.
            rate = self.rate / 2 if self.component.name == "ipu_tile" else self.rate
            return math.prod(kernel.shape) / rate
        if kernel.op in ("gemm", "mvm"):
            return math.prod(kernel.shape) / self.rate
        if kernel.op == "vector" and self.vector_rate is not None:
            rows, cols = kernel.shape
            if self.component.name == "ipu_tile" and (kernel.sfu != 0 or kernel.row_sfu != 0):
                raise NotImplementedError(
                    "IPU vector nonlinear work requires a sourced implementation/dtype-specific "
                    "exponential, reciprocal and conversion cost contract; SFU work cannot use "
                    "ordinary vector_ops_per_cycle or be omitted")
            return (rows * cols * kernel.simt + rows * kernel.row_simt) / self.vector_rate
        raise NotImplementedError(f"{self.component.name} does not time {kernel.op}")

