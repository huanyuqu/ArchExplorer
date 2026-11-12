"""Compatibility seed exports and core-owned user demo scenes.

Concrete native configurations have one authority in components._support.accelsim.
"""
from arch_explorer.adf import Param, SpacePoint
from arch_explorer.components._support.accelsim.scenarios import (
    device_scenario, sm90_adf, sm120_mechanism_adf,
)


def program_inputs(model):
    """Core-owned complete tiny programs and mappings, independent of hardware."""
    from arch_explorer.adf import MappingSpec, WorkloadSpec
    if model not in ("vit", "clip"):
        raise ValueError("Demo model must be vit or clip; use --specs for another scene")
    workload = WorkloadSpec(model, {"profile": "synthetic", "dtype": "fp32",
                                  "config": {"layers": 1, "text_layers": 1}},
                            name="synthetic-" + model + "-forward")
    return {"workload": workload.to_dict(),
            "mappings": [MappingSpec("accelsim_program", {"threads_per_block": n}).to_dict()
                         for n in (32, 64)],
            "scope": "complete small forward; deterministic synthetic inputs/weights, "
                     "reference-generated SM90 scalar FP32 traces; no hardware precision claim"}


def program_demo_specs(model):
    """A core-owned tiny FP32 ViT/CLIP scene with an explicit provider choice."""
    from arch_explorer.adf import walk
    inputs = program_inputs(model)
    adf = sm90_adf()
    adf.name = "synthetic_fp32_gpu_demo"
    adf.child("cluster").repeat = 4
    adf.child("partition").repeat = 8
    adf.networks[0].params["booksim_k"] = Param(
        20, "count", "synthetic demo topology: four SM clusters and sixteen memory subpartitions")
    for _, node in walk(adf):
        if isinstance(node, SpacePoint):
            for key, param in list(node.params.items()):
                node.params[key] = Param(param.value, param.unit,
                    "synthetic reference-program configuration; inherited native parameter vocabulary; "
                    + param.source)
    return {"provider": "accelsim", "adf": adf.to_dict(), **inputs}
