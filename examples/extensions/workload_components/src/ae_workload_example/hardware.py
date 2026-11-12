"""Synthetic scalar-lane arrays: cycles are not device measurements."""
import math

from arch_explorer.adf import Param, SpaceMatrix, SpacePoint
from arch_explorer.components.api import ComponentType, ComputeModel, Conventions, ParamSpec


class ArrayModel(ComputeModel):
    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.lanes = self.positive("lanes")
        self.launch = self.positive("launch_cycles")
        self.param("precision")

    def kernel_time(self, kernel):
        if (kernel.op != "vector" or kernel.precision != "fp32" or kernel.sfu or kernel.row_sfu):
            raise NotImplementedError("tutorial_array supports FP32 SIMD vector instructions without SFU only")
        rows, cols = kernel.shape
        instructions = rows * cols * kernel.simt + rows * kernel.row_simt
        return self.launch + math.ceil(instructions / self.lanes)


class TutorialArray(ComponentType):
    name = "tutorial_array"
    role = "compute"
    params = {"lanes": ParamSpec("count", kind="int", extensive=True, domain=[2, 4, 8, 16, 32]),
              "clock_ghz": ParamSpec("GHz", domain=(0.25, 4.0)),
              "launch_cycles": ParamSpec("cycle", kind="int", domain=(1, 64)),
              "precision": ParamSpec("1", kind="str", domain=["fp32"])}
    conventions = Conventions(time_base="cycle", arbitration="fifo")

    def validate(self, point):
        super().validate(point)
        value = {key: param.value for key, param in point.params.items()}
        if (value["lanes"] not in self.params["lanes"].domain
                or not math.isfinite(value["clock_ghz"]) or not 0.25 <= value["clock_ghz"] <= 4
                or not 1 <= value["launch_cycles"] <= 64 or value["precision"] != "fp32"):
            raise ValueError("tutorial_array parameter outside its declared domain")
        if any(not param.source.strip() for param in point.params.values()):
            raise ValueError("tutorial_array parameters require explicit sources")

    def build(self, point, context):
        self.validate(point)
        return ArrayModel(self, point, context)


def custom_adf(*, lanes=4, clock_ghz=1.0):
    """Explicit custom ADF; no device family or preset is selected."""
    source = "synthetic workload tutorial assumption; not measured hardware"
    def param(value, unit):
        return Param(value, unit, source)
    array = SpacePoint("array", "tutorial_array", {
        "lanes": param(lanes, "count"), "clock_ghz": param(clock_ghz, "GHz"),
        "launch_cycles": param(12, "cycle"), "precision": param("fp32", "1")}, repeat=2)
    TutorialArray().validate(array)
    memory = SpacePoint("shared", "sram", {"capacity": param(2**20, "B"),
                        "bandwidth": param(256.0, "B/ns"), "latency": param(1.0, "ns")})
    fabric = SpacePoint("fabric", "bus", {"bandwidth": param(256.0, "B/ns"), "latency": param(0.0, "ns")})
    return SpaceMatrix("custom", [array, memory], [fabric])
