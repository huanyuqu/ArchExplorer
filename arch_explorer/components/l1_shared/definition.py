"""Hardware role, independent of implementation source."""
from arch_explorer.adf.registry import ComponentType

class L1Shared(ComponentType):
    implementation_policy = 'explicit'
    name, role, execution_modes = 'l1_shared', 'memory', ()
    def build(self, point, context):
        raise ValueError('L1/shared is a contained SM configuration contributor')

