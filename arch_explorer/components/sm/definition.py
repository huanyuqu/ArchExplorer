"""SM hardware identity; no implicit timing implementation."""
from arch_explorer.adf.registry import ComponentType, ParamSpec


class Sm(ComponentType):
    implementation_policy = 'explicit'
    name, role = 'sm', 'compute'
    params = {
        'clock_mhz': ParamSpec('MHz', 'int', required=False, doc='Explicit core clock, when consumed'),
        'precision': ParamSpec('1', 'str', required=False, doc='Operand precision, when consumed'),
    }
    execution_modes = ()

    def build(self, point, context):
        raise ValueError('SM requires an explicit component implementation')
