"""Compatibility imports; production authority lives by component."""
from arch_explorer.components._support.network import FAIR,PortedModel
from arch_explorer.components.mesh2d.implementations.xy import MeshModel
from arch_explorer.components.mesh2d.definition import Mesh2d
from arch_explorer.components.bus.implementations.port import BusModel
from arch_explorer.components.bus.definition import Bus
from arch_explorer.components.fifo_link.definition import FifoLink
from arch_explorer.components.crossbar.implementations.ports import CrossbarModel
from arch_explorer.components.crossbar.definition import Crossbar
from arch_explorer.components.p2p_switch.implementations.ports import P2pModel
from arch_explorer.components.p2p_switch.definition import P2pSwitch
