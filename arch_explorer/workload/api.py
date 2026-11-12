"""Public workload graph and semantic extension contract.

Custom builders accept ``params, *, tp=1``. Graphs are strict ordered DAGs;
``graph`` currently accepts only tp=1. Full primitives retain their distinct
type/lowering when serialized, rather than becoming compact vector operators.
"""
from .operators import Tensor, Operator, Workload
from .extensions import (OperatorDefinition, register_operator, register_workload,
                         get_operator, source_trace, source_objects, operator_identity)
from .execution import (ExecutionContext, ImplementationDefinition, register_implementation,
                        select_implementation, lower_implementation, ExecutionStep,
                        ExecutionProgram, as_program, implementation_identity,
                        implementation_source_objects)


def __getattr__(name):
    if name == 'transformer_preflight':
        from .llm import transformer_preflight
        return transformer_preflight
    if name in ('import_onnx', 'describe_onnx', 'onnx_preflight'):
        from . import onnx_import
        return getattr(onnx_import, name)
    if name in ('mla_budget','mla_preflight'):
        from importlib import import_module
        return getattr(import_module('.mla_budget',__package__),name)
    if name in ('mla_decode','mla_input_shapes'):
        from importlib import import_module
        return getattr(import_module('.mla_decode',__package__),name)
    if name in ('partition_input','restore_partition'):
        from importlib import import_module
        return getattr(import_module('.block_ops',__package__),name)
    if name in ('mla_stream','mla_stream_plan','block_mapping'):
        from importlib import import_module
        return getattr(import_module('.mla_stream',__package__),name)
    if name == 'normalize_budget':
        from .resource_budget import normalize_budget
        return normalize_budget
    if name in ('clip_input_shapes','clip_structure','clip_similarity'):
        from importlib import import_module
        return getattr(import_module('.clip_similarity',__package__),name)
    if name in ('vit_classifier','vit_classifier_structure','vit_classifier_input_shapes'):
        from importlib import import_module
        return getattr(import_module('.vit_classifier',__package__),name)
    if name in ('readonly_view','resolve_layout','element_offset','logical_accesses'):
        from . import storage
        return getattr(storage,name)
    if name == 'layout_candidates':
        from arch_explorer.mapping import view_rules
        return view_rules.layout_candidates
    if name == 'round_storage':
        from .precision import round_storage
        return round_storage
    if name in ('transformer_encoder_block','transformer_block_input_shapes','transformer_block_structure'):
        from . import transformer_block
        return getattr(transformer_block,name)
    if name in ('transformer_encoder','transformer_encoder_input_shapes'):
        from importlib import import_module
        return getattr(import_module('.transformer_encoder',__package__),name)
    if name in ('compose_workloads','qualified_name'):
        from . import composition
        return getattr(composition,name)
    if name == 'migrate_mapping_v2':
        from arch_explorer.mapping.compilation import migrate_mapping_v2
        return migrate_mapping_v2
    if name in ('register_compiler_target', 'compile_workload', 'validate_compilation',
                'compiler_targets', 'migrate_compilation_v1', 'migrate_compilation_v2'):
        from . import compilation
        return getattr(compilation, name)
    if name == 'CompilerTarget':
        from .compilation_v1 import CompilerTarget
        return CompilerTarget
    if name in ('MappingRuleBinding', 'MappingExpansion', 'OperatorMapping',
                'register_operator_mapping', 'operator_mappings'):
        from arch_explorer.mapping import rules
        return getattr(rules, name)
    if name in ('AdapterBinding','TaskAdapter','AdapterDomain','AdapterContext','AdaptedGroup',
                'TargetInput','register_task_adapter','register_adapter_domain','task_adapters'):
        from . import adapters
        return getattr(adapters,name)
    # Reference construction is owned by the target/reference child. Lazy
    # re-export avoids importing numerical machinery for semantic registration.
    if name in ('ReferenceDefinition', 'register_reference'):
        from . import reference_api
        return getattr(reference_api, name)
    if name == 'execute_reference':
        from .reference_execution import execute_reference
        return execute_reference
    raise AttributeError(name)


def workload_from_spec(spec, tp=1):
    """Discover and build a builtin or installed workload, then validate it."""
    from .extensions import _discover, _WORKLOADS, _record_source
    from .graph import finite_json, validate_graph
    if type(tp) is not int or tp < 1:
        raise ValueError("tp must be a positive integer")
    _discover()
    kind = getattr(spec, "kind", None)
    if kind not in _WORKLOADS:
        raise ValueError(f"unknown workload kind {kind!r}; registered: {sorted(_WORKLOADS)}")
    params = getattr(spec, "params", None)
    if type(params) is not dict:
        raise TypeError("workload params must be a JSON object")
    params = finite_json(params, tuples=True)
    builder = _WORKLOADS[kind]
    _record_source(builder)
    workload = builder(params, tp=tp)
    if type(workload) is not Workload:
        raise TypeError("workload builder must return Workload")
    return validate_graph(workload)


__all__ = ["Tensor", "Operator", "Workload", "OperatorDefinition", "register_operator",
           "readonly_view","resolve_layout","element_offset","logical_accesses","layout_candidates",
           "register_workload", "workload_from_spec", "get_operator", "source_trace",
           "source_objects", "operator_identity", "ExecutionContext", "ImplementationDefinition",
           "register_implementation", "select_implementation", "lower_implementation",
           "ExecutionStep", "ExecutionProgram", "as_program", "implementation_identity",
           "implementation_source_objects", "ReferenceDefinition", "register_reference", "execute_reference",
           "CompilerTarget", "TargetInput", "register_compiler_target", "compiler_targets", "compile_workload", "validate_compilation",
           "AdapterBinding", "TaskAdapter", "AdapterDomain", "AdapterContext", "AdaptedGroup", "register_task_adapter",
           "register_adapter_domain", "task_adapters", "migrate_compilation_v1", "migrate_compilation_v2",
           "MappingRuleBinding", "MappingExpansion", "OperatorMapping", "register_operator_mapping", "operator_mappings",
           "round_storage", "transformer_encoder_block", "transformer_block_input_shapes", "transformer_block_structure",
           "compose_workloads", "qualified_name", "transformer_encoder", "transformer_encoder_input_shapes", "migrate_mapping_v2",
           "vit_classifier","vit_classifier_structure","vit_classifier_input_shapes",
           "clip_input_shapes","clip_structure","clip_similarity","import_onnx","describe_onnx",
           "normalize_budget","onnx_preflight","mla_budget","mla_preflight","transformer_preflight",
           "partition_input","restore_partition","mla_stream","mla_stream_plan","block_mapping","mla_decode","mla_input_shapes"]
