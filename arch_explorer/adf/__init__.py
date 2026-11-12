"""Shared contract between the hardware builder, structure search and agents.

``model``: the recursive SpaceMatrix/SpacePoint architecture description.
``registry``: component types and mapping templates.
``evaluation``: workload/mapping specs, evaluation results, backend dispatch.
"""
from arch_explorer.adf.evaluation import Evaluation, MappingSpec, WorkloadSpec, evaluate, register_backend
from arch_explorer.adf.model import (Param, SpaceMatrix, SpacePoint, canonical_json, find, fingerprint,
                                     from_dict, instance_count, validate, walk)
from arch_explorer.adf.registry import (COMPONENTS, MAPPING_TEMPLATES, ComponentType, Conventions,
                                        MappingTemplate, ParamSpec, register_component, register_template)
from arch_explorer.adf.dynamic import Connection, instances, validate_connections

__all__ = [
    "COMPONENTS", "MAPPING_TEMPLATES", "ComponentType", "Conventions", "Evaluation", "MappingSpec",
    "MappingTemplate", "Param", "ParamSpec", "SpaceMatrix", "SpacePoint", "WorkloadSpec", "canonical_json",
    "evaluate", "find", "fingerprint", "from_dict", "instance_count", "register_backend",
    "register_component", "register_template", "validate", "walk",
    "Connection", "instances", "validate_connections",
]
