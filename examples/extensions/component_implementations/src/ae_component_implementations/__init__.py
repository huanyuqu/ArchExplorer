"""Install after the main distribution; importing does not register or execute."""


def register():
    from arch_explorer.components.api import register_component_implementation
    from .implementation import DEFINITION, ASSEMBLY_DEFINITION
    register_component_implementation(DEFINITION)
    register_component_implementation(ASSEMBLY_DEFINITION)
    from arch_explorer.workload.api import register_task_adapter
    from .adapter import ADAPTER
    register_task_adapter(ADAPTER)


def scene(**kwargs):
    from .scenes import serial_scene
    return serial_scene(**kwargs)
