"""Component implementations and their public extension interface.

Ordinary builtin descriptors remain available on import. Optional engines and
external providers are discovered through registrars before execution.
"""
from .base import Demand, Kernel, Provenance, dtype_bytes
from .builtins import register_components

CATALOG = register_components()
__all__ = ['CATALOG', 'Demand', 'Kernel', 'Provenance', 'dtype_bytes']
