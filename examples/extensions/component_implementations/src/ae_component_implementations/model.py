"""Synthetic compute-only service; data transport belongs to the mapped memories."""
import math

from arch_explorer.components.api import ComputeModel


class SerialSMModel(ComputeModel):
    """FP32 GEMM/MVM occupy one service resource for startup + operations/rate ns.

    This does not claim instruction, cache, warp or memory-system simulation.
    The implementation owns compute service only. A mapping may separately
    schedule real SRAM/DRAM/network traffic without duplicating a local IO charge.
    """

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.rate = self.positive("ops_per_ns")
        self.startup = self.param("startup_ns")
        if (isinstance(self.startup, bool) or not isinstance(self.startup, (int, float))
                or not math.isfinite(self.startup) or self.startup < 0):
            raise ValueError("serial SM startup_ns must be finite and nonnegative")
        if self.param("precision") != "fp32":
            raise NotImplementedError("serial SM implementation supports FP32 only")

    def kernel_time(self, kernel):
        dimensions = {"gemm": 4, "mvm": 3}.get(kernel.op)
        if dimensions is None:
            raise NotImplementedError("serial SM accepts GEMM or MVM kernels only")
        self.precision(kernel)
        if (len(kernel.shape) != dimensions
                or any(type(size) is not int or size <= 0 for size in kernel.shape)):
            raise ValueError("serial SM kernel dimensions must be positive integers")
        operations = kernel.flops
        if not math.isfinite(operations) or operations <= 0:
            raise ValueError("serial SM requires finite positive operation count")
        duration = self.startup + operations / self.rate
        if not math.isfinite(duration):
            raise ValueError("serial SM service time overflow")
        return duration


def factory(resolved_component, materialized_point, context):
    """The same public factory contract used by built-in implementations."""
    return SerialSMModel(resolved_component, materialized_point, context)


class AssemblySerialSMModel(SerialSMModel):
    """The same matrix service plus explicitly sourced copy-issue occupancy.

    Only zero-arithmetic FP32 vector kernels represent assembly here. This is
    synthetic compute-side issue service, not another memory-bandwidth charge.
    """

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.assembly_rate = self.positive("assembly_elements_per_ns")
        self.assembly_startup = self.param("assembly_startup_ns")
        if (isinstance(self.assembly_startup, bool)
                or not isinstance(self.assembly_startup, (int, float))
                or not math.isfinite(self.assembly_startup) or self.assembly_startup < 0):
            raise ValueError("assembly startup must be finite and nonnegative")

    def kernel_time(self, kernel):
        if kernel.op != "vector":
            return super().kernel_time(kernel)
        self.precision(kernel)
        if len(kernel.shape) != 2 or any(type(size) is not int or size <= 0 for size in kernel.shape):
            raise ValueError("assembly vector dimensions must be two positive integers")
        counters = (kernel.simt, kernel.sfu, kernel.row_simt, kernel.row_sfu)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value != 0 for value in counters):
            raise NotImplementedError("assembly SM supports only vector kernels with zero arithmetic counters")
        elements = kernel.shape[0] * kernel.shape[1]
        try:
            duration = self.assembly_startup + elements / self.assembly_rate
        except OverflowError:
            raise ValueError("assembly service time overflow") from None
        if not math.isfinite(duration):
            raise ValueError("assembly service time overflow")
        return duration


def assembly_factory(resolved_component, materialized_point, context):
    return AssemblySerialSMModel(resolved_component, materialized_point, context)
