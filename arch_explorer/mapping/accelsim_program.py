"""Compatibility import; native ISA encoding is owned by the component."""
from importlib import import_module as _import_module

_impl = _import_module('arch_explorer.components.accelsim.lowering')
__all__ = [name for name in dir(_impl) if not name.startswith('_')]


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
