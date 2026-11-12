"""Accel-Sim joint execution domain and native component descriptions.

Importing this package describes components only. Registration is explicit or
performed by the installed component entry point; engines are loaded only by
preflight/evaluation, never by discovery.
"""


def register():
    """Register this library using the same public contract as an extension."""
    from arch_explorer.components.accelsim.declarations import NATIVE_COMPONENTS
    from arch_explorer.adf.evaluation import register_backend
    from arch_explorer.adf.registry import register_template
    from arch_explorer.components.api import register_component, register_provider
    from arch_explorer.components.accelsim.provider import PROVIDER, evaluate_backend
    from arch_explorer.components.accelsim.component_provider import PROVIDER as COMPONENT_PROVIDER, evaluate_backend as component_backend
    from arch_explorer.components.accelsim.program_mapping import ProgramMapping
    register_template(ProgramMapping)
    for component in NATIVE_COMPONENTS:
        register_component(component)
    from arch_explorer.components.native_bindings import register as register_implementations
    register_implementations()
    from arch_explorer.workload.adapters.accelsim import install as install_adapter
    install_adapter()
    register_provider(PROVIDER)
    register_backend(PROVIDER.id, evaluate_backend)
    register_provider(COMPONENT_PROVIDER)
    register_backend(COMPONENT_PROVIDER.id, component_backend)
    return PROVIDER


def __getattr__(name):
    if name not in __all__ or name == 'register':
        raise AttributeError(name)
    from importlib import import_module
    return getattr(import_module('arch_explorer.components._support.accelsim.declarations'), name)


__all__ = [
    'NativeComponent', 'AccelSimSm', 'AccelSimSm120Mechanism', 'AccelSimL1Shared',
    'AccelSimInterconnect', 'AccelSimL2', 'AccelSimDram', 'NATIVE_COMPONENTS', 'register',
]
