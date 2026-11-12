"""Installed unknown builder; no operator, component or timing extension."""
from arch_explorer.workload.api import register_workload
from .builders import build_scale

KIND='val57.external_scale'


def register():
    register_workload(KIND,build_scale)
