"""Dynamic execution contracts.

``runtime.hpp`` is the generic single-threaded coordinator used by native
components and the independent synthetic component example. It is shipped as
package data; it contains no hardware-specific mechanism or pointer type.
"""
from importlib.resources import files


def runtime_header():
    """Return the installed, authoritative C++17 coordinator header path."""
    return files(__package__).joinpath('runtime.hpp')


__all__ = ['runtime_header']
