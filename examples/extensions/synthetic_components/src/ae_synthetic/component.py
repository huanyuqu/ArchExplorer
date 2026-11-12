"""A deliberately synthetic FP32 MAC component, with no device accuracy claim."""
import math

from arch_explorer.components.api import ComponentType, ComputeModel, Conventions, ParamSpec


class TutorialMacModel(ComputeModel):
    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.rate = self.positive("macs_per_ns")
        self.native_precision = self.param("precision")

    def kernel_time(self, kernel):
        if kernel.precision != self.native_precision or kernel.op not in ("gemm", "mvm"):
            raise NotImplementedError("tutorial_mac supports FP32 GEMM/MVM kernels only")
        # Kernel.flops counts one multiply-add as two operations.
        return kernel.flops / (2 * self.rate)


class TutorialMac(ComponentType):
    name = "tutorial_mac"
    role = "compute"
    params = {"macs_per_ns": ParamSpec("MAC/ns", extensive=True, domain=(1.0, 128.0)),
              "precision": ParamSpec("1", kind="str", domain=["fp32"])}
    conventions = Conventions(time_base="ns", arbitration="fifo")

    def validate(self, point):
        super().validate(point)
        rate = point.params["macs_per_ns"]
        if not math.isfinite(rate.value) or not 1 <= rate.value <= 128 or not rate.source.strip():
            raise ValueError("tutorial_mac rate must be in [1, 128] MAC/ns with a source")
        if point.params["precision"].value != "fp32":
            raise ValueError("tutorial_mac supports fp32 only")

    def build(self, point, context):
        self.validate(point)
        return TutorialMacModel(self, point, context)
