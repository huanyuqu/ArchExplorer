"""Builtin catalog registrar, using the same interface as installed extensions."""
from arch_explorer.adf.registry import register_component


def registrars():
    """The source checkout and wheel expose these identical entry-point targets."""
    return {'builtin': 'arch_explorer.components.builtins:register',
            'accelsim': 'arch_explorer.components.accelsim:register'}


def register_components():
    from .compute import SystolicArray, VectorUnit
    from .memory import Dram, Sram
    from .network import Bus, Crossbar, FifoLink, Mesh2d, P2pSwitch
    from .pim import PimUnit
    from .cgra import CgraPe
    from .manycore import IpuTile, TianjicCore, TileExchange
    from .service_ceiling import ServiceCeiling
    from .serial_service import SerialService
    from .multicast_tree import MulticastTree
    catalog = (SystolicArray, VectorUnit, PimUnit, CgraPe, Sram, Dram, Mesh2d, Bus, FifoLink,
               Crossbar, P2pSwitch, IpuTile, TianjicCore, TileExchange)
    for component in (*catalog, ServiceCeiling, SerialService, MulticastTree):
        register_component(component)
        from .implementations import register_legacy_implementation
        register_legacy_implementation(component)
    from .sm.definition import Sm
    from .sm.implementations.roofline import DEFINITION
    from .implementations import register_component_implementation
    register_component(Sm)
    register_component_implementation(DEFINITION)
    from .sm.implementations.events import DEFINITION as EVENTS_DEFINITION
    register_component_implementation(EVENTS_DEFINITION)
    from .sm.implementations.mac_events import DEFINITION as MAC_EVENTS_DEFINITION
    register_component_implementation(MAC_EVENTS_DEFINITION)
    return catalog


def register():
    """Register ordinary components and the generic event execution provider."""
    register_components()
    from .events_provider import register as register_events
    register_events()
    from arch_explorer.workload.adapters.events import install as install_adapters
    install_adapters()
    from arch_explorer.mapping.operator_rules import install as install_mappings
    install_mappings()
