"""Exact Accel-Sim configuration contract owned by this component."""
from arch_explorer.adf.registry import ParamSpec
from arch_explorer.components._support.accelsim.native_definition import NativeComponent, _params

class AccelSimL1Shared(NativeComponent):
    name, role = 'accelsim_l1_shared', 'memory'
    params = _params('l1_shared')
    params['gpgpu_coalesce_arch'] = ParamSpec('1', 'int', domain=list(params['gpgpu_coalesce_arch'].domain)+[120],
        doc='120 only with SM120 mechanism; sector coalescing, not target timing')

