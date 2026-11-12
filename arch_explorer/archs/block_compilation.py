"""User transport for the authoritative stream graph and physical block mapping."""


def block_scene(parameters, *, adf, compute, memories, external_memory=None):
    """Return (ADF, ordinary graph WorkloadSpec, producer-owned v4 mapping).

    parameters is the complete mla_stream keyword object, including real roots
    and any resource_budget. Ordered compute/memories lists and external memory
    are passed unchanged. No root partition, placement or capacity is inferred.
    The caller supplies hardware; a scene does not claim calibration or service.
    """
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import mla_stream, block_mapping
    if type(parameters) is not dict:
        raise ValueError('parameters must be the complete stream keyword object')
    graph = mla_stream(**parameters)
    mapping = block_mapping(graph, compute=compute, memories=memories,
                            external_memory=external_memory)
    return adf, WorkloadSpec('graph', graph.to_dict()), mapping


def block_hardware_scene(parameters, *, hardware):
    """Use explicit public factory options and its unmodified slot metadata.

    Returns (ADF, graph WorkloadSpec, MappingSpec, hardware metadata). The
    factory owns physical paths, capacities, candidate identity and timing
    declarations; block_scene still delegates the graph and mapping to SIM.
    """
    from arch_explorer.components.api import block_memory_hardware
    if type(hardware) is not dict:
        raise ValueError('hardware must be the explicit block_memory_hardware keyword object')
    adf, metadata = block_memory_hardware(**hardware)
    scene = block_scene(parameters, adf=adf, compute=metadata['compute'],
        memories=metadata['memories'], external_memory=metadata['external_memory'])
    return (*scene, metadata)
