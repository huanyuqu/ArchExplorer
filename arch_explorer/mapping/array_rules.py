"""Materialized mapping projections of the common typed array authority."""
from .rules import MappingExpansion, OperatorMapping, register_operator_mapping

_DEFINITIONS = None


def expand(op, tensors, options):
    from arch_explorer.workload.graph import validate_operator
    validate_operator(op, tensors)
    return MappingExpansion(operators=(op,))


def access(op, tensors, options):
    from arch_explorer.workload.array_ops import access_counts
    from arch_explorer.workload.extensions import _record_source
    _record_source(access_counts)
    return access_counts(op, tensors)


def inspect(op, tensors, options):
    from arch_explorer.workload.array_ops import validate
    from arch_explorer.workload.extensions import _record_source
    _record_source(validate)
    validate(op, tensors)
    axis = op.attrs.get('axis')
    return dict(materialized=True,
                input_shapes=[list(tensors[n].shape) for n in op.inputs],
                output_shape=list(tensors[op.outputs[0]].shape),
                axis=None if axis is None else axis % len(tensors[op.inputs[0]].shape),
                indices_scalar=op.attrs.get('indices_scalar', False),
                select_last_index=op.attrs.get('select_last_index', False),
                indexing='signed index domain [-dimension, dimension); negative indices normalized once'
                if op.op in ('gather', 'gather_elements') else 'declared logical order',
                numerical_domain='exact signed integers and direct storage comparisons; no float addressing',
                writes='independent output storage; input lifetimes include indices and conditions')


def install():
    from arch_explorer.workload.array_ops import KINDS
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS = tuple(OperatorMapping(k + '.materialized', '1', k,
                                            expand, access, inspect, default=True) for k in KINDS)
    for definition in _DEFINITIONS:
        register_operator_mapping(definition)
