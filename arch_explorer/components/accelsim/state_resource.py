"""Compatibility module; one component-owned authority."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('arch_explorer.components._support.accelsim.state_resource')
