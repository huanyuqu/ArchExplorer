"""Legacy configuration names; hardware-owned contracts remain unique."""
from .native_definition import NativeComponent, _params
from arch_explorer.components.sm.implementations.accelsim.definition import AccelSimSm
from arch_explorer.components.sm.implementations.accelsim.sm120_definition import AccelSimSm120Mechanism
from arch_explorer.components.l1_shared.implementations.accelsim.definition import AccelSimL1Shared
from arch_explorer.components.icnt.implementations.accelsim.definition import AccelSimInterconnect
from arch_explorer.components.l2.implementations.accelsim.definition import AccelSimL2
from arch_explorer.components.dram.implementations.accelsim.definition import AccelSimDram

NATIVE_COMPONENTS = (AccelSimSm, AccelSimSm120Mechanism, AccelSimL1Shared, AccelSimInterconnect, AccelSimL2, AccelSimDram)
