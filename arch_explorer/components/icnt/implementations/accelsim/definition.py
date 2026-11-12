"""Exact Accel-Sim configuration contract owned by this component."""
from arch_explorer.adf.registry import ParamSpec
from arch_explorer.components._support.accelsim.native_definition import NativeComponent, _params

class AccelSimInterconnect(NativeComponent):
    name, role = 'accelsim_icnt', 'network'
    params = _params('icnt')

