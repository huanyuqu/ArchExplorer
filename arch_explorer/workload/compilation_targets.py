"""Compatibility discovery shim; concrete adapters own their registrations."""
from .compilation import register_compiler_target, compiler_targets


def install():
    from .extensions import _discover
    _discover()
