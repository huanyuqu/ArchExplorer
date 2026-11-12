"""Installed registrations; no provider, simulation or device access at import."""
from arch_explorer.workload.api import OperatorDefinition

from .semantics import OPERATOR, reference, validate

SEMANTIC = OperatorDefinition(OPERATOR, validate, version="1", reference=reference)


def register():
    from arch_explorer.workload.api import register_implementation, register_operator
    from .events import IMPLEMENTATIONS as EVENTS
    from .gpu import IMPLEMENTATIONS as GPU
    register_operator(SEMANTIC)
    for implementation in (*EVENTS, *GPU):
        register_implementation(implementation)
