"""Compatibility import for the single adapter-owned program assembler."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('arch_explorer.workload.adapters.accelsim_program')
