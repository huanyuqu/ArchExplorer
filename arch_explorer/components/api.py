"""Public component-author interface; all exports retain their original identity."""
from arch_explorer.adf.registry import (
    COMPONENTS, ComponentType, Conventions, ParamSpec, register_component,
)
from arch_explorer.components.base import (
    BuildContext, ComputeModel, Demand, Kernel, MemoryModel, NetworkModel,
    PointModel, Provenance, dtype_bytes,
)
from arch_explorer.simulator.events.engine import Resource, Task, TaskGraph, simulate
from arch_explorer.simulator.events.protocol import (
    DemandAdapter, ExecutionEvent, ExecutionModel, ExecutionRequest, valid_time,
)
from arch_explorer.simulator.events.builder import Simulator, build_simulator
from arch_explorer.simulator.plan import ExecutionPlan, MemoryGroup, PlanConstraints
from .extensions import (
    API_VERSION, ENTRY_POINT_GROUP, discover_components, get_provider,
    providers, register_provider,
)
from .implementations import (
    ComponentImplementation, ImplementationBinding, register_component_implementation,
    component_implementations, resolve_component, build_component, component_catalog, component_candidates,
)

__all__ = [
    'ComponentImplementation', 'ImplementationBinding', 'register_component_implementation',
    'component_implementations', 'resolve_component', 'build_component', 'component_catalog', 'component_candidates',
    'COMPONENTS', 'ComponentType', 'Conventions', 'ParamSpec', 'register_component',
    'BuildContext', 'ComputeModel', 'Demand', 'Kernel', 'MemoryModel', 'NetworkModel',
    'PointModel', 'Provenance', 'dtype_bytes', 'Resource', 'Task', 'TaskGraph', 'simulate',
    'ExecutionEvent', 'ExecutionModel', 'ExecutionRequest', 'DemandAdapter', 'valid_time',
    'Simulator', 'build_simulator',
    'ExecutionPlan', 'MemoryGroup', 'PlanConstraints',
    'API_VERSION', 'ENTRY_POINT_GROUP', 'discover_components', 'get_provider',
    'providers', 'register_provider',
    'block_memory_hardware',
]


def __getattr__(name):
    if name == 'block_memory_hardware':
        from .block_memory import block_memory_hardware
        return block_memory_hardware
    raise AttributeError(name)
