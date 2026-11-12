"""One new implementation of the existing SM hardware type, not a new provider."""
import math
from dataclasses import replace

from arch_explorer.components.api import ComponentImplementation, Conventions, ParamSpec

from .model import factory, assembly_factory


def validate(point):
    if point.params["precision"].value != "fp32":
        raise NotImplementedError("example.sm.serial supports FP32 only")
    for name in ("ops_per_ns", "startup_ns"):
        parameter = point.params[name]
        value = parameter.value
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name == "ops_per_ns" and value == 0)):
            raise ValueError(f"example.sm.serial requires valid {name}")
        if not isinstance(parameter.source, str) or not parameter.source.strip():
            raise ValueError(f"example.sm.serial requires an explicit source for {name}")


DEFINITION = ComponentImplementation(
    component="sm", id="example.sm.serial", version="1",
    source={"basis": "synthetic external compute-only service, startup + operations/rate",
            "counting": "one MAC = two arithmetic operations; no memory traffic charge"},
    factory=factory, execution_domain="events", input_contract="events.kernel.v1",
    result_scope="Analytical FP32 GEMM/MVM compute service; no warp/cache or device accuracy claim",
    params={
        "ops_per_ns": ParamSpec("op/ns", domain=(1.0, 64.0), doc="Explicit synthetic arithmetic service rate"),
        "startup_ns": ParamSpec("ns", domain=(0.0, 32.0), doc="Startup per compute kernel; explicit synthetic assumption"),
    },
    hardware_parameters=("precision",),
    execution_modes=("demand",),
    conventions=Conventions(time_base="ns", arbitration="fifo", includes_startup=True),
    capabilities={"contained_components": [], "memory_traffic": "external_only",
                  "precisions": ["fp32"], "kernels": ["gemm", "mvm"]},
    validate=validate,
    adapters=("events",),
)


def validate_assembly(point):
    validate(point)
    for name in ("assembly_elements_per_ns", "assembly_startup_ns"):
        parameter = point.params[name]
        value = parameter.value
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name == "assembly_elements_per_ns" and value == 0)):
            raise ValueError(f"assembly SM requires valid {name}")
        if not isinstance(parameter.source, str) or not parameter.source.strip():
            raise ValueError(f"assembly SM requires an explicit source for {name}")


ASSEMBLY_DEFINITION = replace(
    DEFINITION, id="example.sm.serial_assembly", factory=assembly_factory,
    source={"basis": "explicit synthetic arithmetic service plus assembly issue occupancy",
            "assembly": "assembly_startup_ns + elements/assembly_elements_per_ns; zero arithmetic vector only",
            "memory": "external mapping owns all byte transfer and storage costs"},
    result_scope="Synthetic FP32 matrix and assembly-issue compute occupancy; no generic vector ISA or device accuracy claim",
    params={**DEFINITION.params,
            "assembly_elements_per_ns": ParamSpec("element/ns", domain=(1.0, 64.0),
                doc="Explicit synthetic assembly issue rate, not memory bandwidth"),
            "assembly_startup_ns": ParamSpec("ns", domain=(0.0, 32.0),
                doc="Explicit synthetic assembly command startup")},
    capabilities={"contained_components": [], "memory_traffic": "external_only",
                  "precisions": ["fp32"], "kernels": ["gemm", "mvm", "vector"],
                  "vector_contract": "positive 2D shape with simt/sfu/row_simt/row_sfu all zero; assembly issue only"},
    validate=validate_assembly,
    adapters=("example.serial.events",),
)
