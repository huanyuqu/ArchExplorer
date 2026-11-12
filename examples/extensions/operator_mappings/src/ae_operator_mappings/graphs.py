"""Compose the unknown square after the public attention graph."""


def scene(*, hardware=None, input_case="regular"):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.archs.attention import attention_scene
    from arch_explorer.workload.api import Workload
    from .semantics import KIND
    adf, workload, mapping = attention_scene(hardware=hardware, input_case=input_case)
    graph = Workload.from_dict(workload.params)
    source = graph.tensors["attention"]
    result = graph.tensor("attention_squared", source.shape)
    graph.add("postprocess", KIND, [source.name], [result], shape=list(source.shape))
    return adf, WorkloadSpec("graph", graph.to_dict()), mapping
