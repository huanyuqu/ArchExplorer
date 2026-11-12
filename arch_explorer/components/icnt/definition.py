"""Hardware role, independent of implementation source."""
from arch_explorer.adf.registry import ComponentType

class Icnt(ComponentType):
    implementation_policy = 'explicit'
    name, role, execution_modes = 'icnt', 'network', ()
    def build(self, point, context):
        raise ValueError('ICNT requires an explicit component implementation')

