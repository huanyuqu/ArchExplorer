"""Explicit consumers of the registered LLM performance-graph contract."""
from copy import deepcopy
from dataclasses import asdict, fields

from arch_explorer.adf import MappingSpec, WorkloadSpec


def llm_workload(*, model, phase, batch, seq_len, layers, dtype, iterations):
    """Describe selected model layers, without weights or numerical activations.

    All experiment dimensions are required. Known model names resolve through
    the existing model registry; custom configurations must contain every
    ModelConfig field, including their source. Tensor parallelism is selected
    by the mapping request, never by a hidden workload parameter.
    """
    from arch_explorer.workload.api import workload_from_spec
    from arch_explorer.workload.llm import ModelConfig, model_config
    if isinstance(model, dict) and set(model) != {field.name for field in fields(ModelConfig)}:
        raise ValueError("Custom models require every ModelConfig field including source")
    config = model_config(model)
    if not isinstance(config.source, str) or not config.source.strip():
        raise ValueError("An explicit model configuration source is required")
    for name, value in (("batch", batch), ("seq_len", seq_len), ("layers", layers), ("iterations", iterations)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an explicit positive integer")
    if layers > config.layers:
        raise ValueError("Selected layers exceed the declared model layer count")
    spec = WorkloadSpec(kind="transformer", name=f"{config.name}-{phase}", params={
        "model": asdict(config), "phase": phase, "batch": batch, "seq_len": seq_len,
        "layers": layers, "dtype": dtype, "iterations": iterations})
    workload_from_spec(spec)  # The registered builder owns shapes and semantics.
    return spec


def llm_scene(*, adf, workload, mapping):
    """Return a checked ADF/spec/mapping triple without running a backend."""
    import arch_explorer.mapping  # Install the public built-in templates.
    from arch_explorer.adf import MAPPING_TEMPLATES
    from arch_explorer.agent_env.sampling import check_mapping
    from arch_explorer.structure import check
    if not isinstance(workload, WorkloadSpec) or workload.kind != "transformer":
        raise TypeError("An explicit registered transformer WorkloadSpec is required")
    if not isinstance(mapping, MappingSpec):
        raise TypeError("An explicit MappingSpec is required")
    check(adf)
    check_mapping(adf, workload, mapping, MAPPING_TEMPLATES)
    return deepcopy(adf), deepcopy(workload), deepcopy(mapping)


def llm_search_config(*, adf, workloads, backend, backend_options, mapping_template, mapping_requests,
                      parameter_domains, max_evaluations, parameter_samples,
                      mapping_samples, random_seed, search):
    """Use the existing Env search with explicit finite mapping requests.

    This is the ordinary performance-graph search contract. Its historical
    adf-result.v1 replay performs another evaluation. For read-only receipt
    validation prepare/evaluate an events request with the saved winning specs.
    """
    import arch_explorer.mapping
    from arch_explorer.adf import MAPPING_TEMPLATES
    from arch_explorer.agent_env.sampling import mapping_contract
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be explicitly named")
    if not workloads or any(not isinstance(w, WorkloadSpec) or w.kind != "transformer" for w in workloads):
        raise TypeError("workloads must contain explicit transformer WorkloadSpecs")
    if not isinstance(mapping_requests, list) or not mapping_requests:
        raise ValueError("mapping_requests must be a nonempty explicit list")
    for name, value in (("max_evaluations", max_evaluations), ("parameter_samples", parameter_samples),
                        ("mapping_samples", mapping_samples)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(random_seed) is not int:
        raise ValueError("random_seed must be an integer")
    template = MAPPING_TEMPLATES.get(mapping_template)
    for workload in workloads:
        mapping_contract(template, adf, workload, mapping_requests)
    return {"seed_adf": adf.to_dict(), "workloads": [w.to_dict() for w in workloads],
        "backend": backend, "backend_options": deepcopy(backend_options), "mapping_templates": [mapping_template],
        "mapping_selectors": {mapping_template: deepcopy(mapping_requests)},
        "parameter_domains": deepcopy(parameter_domains), "max_evaluations": max_evaluations,
        "parameter_samples": parameter_samples, "mapping_samples": mapping_samples,
        "random_seed": random_seed, "search": deepcopy(search)}
