"""Hardware role, independent of implementation source."""
from arch_explorer.adf.registry import ComponentType

class L2(ComponentType):
    implementation_policy = 'explicit'
    name, role, execution_modes = 'l2', 'memory', ()
    def build(self, point, context):
        raise ValueError('L2 requires an explicit component implementation')

