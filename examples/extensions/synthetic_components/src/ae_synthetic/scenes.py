"""Standard objects used by the tutorial, separate from component mechanisms."""
from arch_explorer.adf import MappingSpec, Param, SpaceMatrix, SpacePoint, WorkloadSpec, walk


def mac_point():
    return SpacePoint("mac", "tutorial_mac", {
        "macs_per_ns": Param(8.0, "MAC/ns", "synthetic tutorial assumption; not a device measurement"),
        "precision": Param("fp32", "1", "synthetic tutorial: FP32 GEMM/MVM only")})


def workload():
    return WorkloadSpec("operator", {"op": "matmul", "m": 4, "k": 8, "n": 4, "dtype": "fp32"})


def provider_scene():
    return {"provider": "tutorial.serial_mac", "adf": SpaceMatrix("unit", [mac_point()]).to_dict(),
            "workload": workload().to_dict(),
            "mappings": [MappingSpec("tutorial_serial", {"tile_m": value}).to_dict() for value in (1, 2)],
            "options": {"startup_ns": 5}, "scope": "synthetic FP32 MAC tiles; compute only, no hardware claim"}


def event_objects():
    from arch_explorer.archs.adf_seeds import dmc_adf
    adf = dmc_adf(num_cores=1)
    adf.name = "tutorial_mac_with_memory"
    for _, node in walk(adf):
        if isinstance(node, SpacePoint) and node.component == "vector_unit":
            node.component = "tutorial_mac"
            node.params = mac_point().params
    mapping = MappingSpec("dmc_partition", {"split": "n", "weights": "local", "activations": "local"})
    return adf, workload(), mapping
