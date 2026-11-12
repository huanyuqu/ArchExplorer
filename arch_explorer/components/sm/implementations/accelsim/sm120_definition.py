"""Exact Accel-Sim configuration contract owned by this component."""
from arch_explorer.adf.registry import ParamSpec
from arch_explorer.components._support.accelsim.native_definition import NativeComponent, _params

class AccelSimSm120Mechanism(NativeComponent):
    """Logical SM120 subset and four synthetic register spaces, not a device profile."""
    name, role = 'accelsim_sm120_mechanism', 'compute'
    params = dict(_params('sm'),
        sim28_typed_regs=ParamSpec('1', 'int', domain=[1], doc='four synthetic typed register-bank spaces'))

