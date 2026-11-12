"""Installed registration; no work generation or simulation at import."""


def register():
    from arch_explorer.workload.api import register_operator, register_operator_mapping
    from .semantics import SQUARE
    from .mapping import MATERIALIZED
    register_operator(SQUARE)
    register_operator_mapping(MATERIALIZED)
