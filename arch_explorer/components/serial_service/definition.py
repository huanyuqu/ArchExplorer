"""Finite serial service with independently sourced positive duration per command."""
from arch_explorer.adf.registry import ComponentType,Conventions,ParamSpec
from arch_explorer.components.base import PointModel,ComputeModel


class SerialServiceModel(ComputeModel):
    role='compute'
    def __init__(self,component,point,context):
        PointModel.__init__(self,component,point,context)
        if not point.params['duration'].source.strip():raise ValueError('independent service source required')
        self.duration=self.positive('duration')
        self.server=self.resource('server',1.0,'ns')


class SerialService(ComponentType):
    name='serial_service'
    role='compute'
    params={'duration':ParamSpec('ns')}
    conventions=Conventions(arbitration='exclusive',includes_startup=True)
    def build(self,point,context):return SerialServiceModel(self,point,context)
