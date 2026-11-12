"""Compatibility import; authority is source-specific shared support."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("arch_explorer.components._support.accelsim.standalone_engine")
