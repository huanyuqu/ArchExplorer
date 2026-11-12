"""Reusable optimistic service pools; deliberately not a full device model."""
from arch_explorer.adf.registry import ComponentType, Conventions, ParamSpec
from arch_explorer.components.base import PointModel, ComputeModel


class ServiceCeilingModel(ComputeModel):
    role = 'compute'

    def __init__(self, component, point, context):
        PointModel.__init__(self, component, point, context)
        if not point.params['rate'].source.strip():
            raise ValueError('service ceiling needs a source')
        self.unit = self.resource('service', self.positive('rate'), self.param('unit'))


class ServiceCeiling(ComponentType):
    name = 'service_ceiling'
    role = 'compute'
    params = {'rate': ParamSpec('work/ns'), 'unit': ParamSpec('1', kind='str'),
              'storage_capacity': ParamSpec('B', required=False),
              'cache_capacity': ParamSpec('B', required=False)}
    conventions = Conventions(arbitration='fair_share', includes_startup=False)

    def build(self, point, context):
        return ServiceCeilingModel(self, point, context)
