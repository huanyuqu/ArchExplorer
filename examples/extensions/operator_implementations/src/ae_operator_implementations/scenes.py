"""Standard user scenes; all planning is delegated to registered mappings."""
from arch_explorer.adf import MappingSpec, WorkloadSpec

from .events import SINGLE_ID, STEPS_ID
from .graphs import NODES


def event_mappings(*, memory="operator_board/shared", placement="round_robin"):
    return [MappingSpec("execution_graph", {
        "implementations": {node: implementation for node in NODES},
        "placement": placement, "serialize": False, "memory": memory,
    }) for implementation in (SINGLE_ID, STEPS_ID)]


def event_scene(adf, graph, mappings):
    return {"provider": "events", "adf": adf.to_dict(),
            "workload": WorkloadSpec("graph", graph.to_dict()).to_dict(),
            "mappings": [mapping.to_dict() for mapping in mappings],
            "options": {"trace": True, "check_capacity": True},
            "scope": "external FP32 shifted-square DAG on existing vector units; synthetic timing, not hardware accuracy"}


def gpu_case(adf, graph, engine):
    """Use the same ordinary graph on components and the independent oracle."""
    from pathlib import Path
    from .gpu import SINGLE_ID as GPU_SINGLE, STEPS_ID as GPU_STEPS
    mappings = [MappingSpec("accelsim_program", {
        "threads_per_block": threads, "implementations": {node: implementation for node in NODES},
    }).to_dict() for threads, implementation in ((32, GPU_SINGLE), (64, GPU_STEPS))]
    common = {"adf": adf.to_dict(), "workload": WorkloadSpec("graph", graph.to_dict()).to_dict(),
              "mappings": mappings, "options": {"engine": str(Path(engine).resolve()),
              "timeout_s": 600, "memory_limit_mb": 8192, "max_cycles": 10000000},
              "scope": "external FP32 DAG, single-kernel/32-thread and multi-kernel/64-thread mappings; synthetic reference trace"}
    # Reuse the installed core example's case format; no second execution runner.
    from arch_explorer.archs.gpu_instantiation import CASE_SCHEMA
    from copy import deepcopy
    return {"schema": CASE_SCHEMA, "scope": common["scope"],
            "candidate": {**deepcopy(common), "provider": "accelsim_components"},
            "oracle": {**deepcopy(common), "provider": "accelsim"}}
