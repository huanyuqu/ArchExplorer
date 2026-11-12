"""Lazy compatibility package; native/source services have one authority."""
from importlib import import_module

__all__ = ['NativeComponent', 'AccelSimSm', 'AccelSimSm120Mechanism', 'AccelSimL1Shared',
           'AccelSimInterconnect', 'AccelSimL2', 'AccelSimDram', 'NATIVE_COMPONENTS', 'register']


def register():
    return import_module('arch_explorer.components._support.accelsim').register()


def __getattr__(name):
    return getattr(import_module('arch_explorer.components._support.accelsim'), name)
