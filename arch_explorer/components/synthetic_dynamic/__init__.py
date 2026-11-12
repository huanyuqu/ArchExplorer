"""Independent CPU-only queue components exercising the generic coordinator."""
from importlib.resources import files


def source_path():
    return files(__package__).joinpath('probe.cpp')


__all__ = ['source_path']
