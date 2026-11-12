"""Finite native contract adapters; hardware identity stays source-independent."""
from copy import deepcopy
from dataclasses import dataclass

from arch_explorer.adf.model import SpacePoint, walk
from arch_explorer.adf.registry import ComponentType, ParamSpec, register_component
from .implementations import (
    ComponentImplementation, ImplementationBinding, _ALIASES, register_component_implementation,
    resolve_component,
)


from arch_explorer.components.l2.definition import L2


from arch_explorer.components.icnt.definition import Icnt


from arch_explorer.components.l1_shared.definition import L1Shared


@dataclass(frozen=True)
class NativeFactory:
    """C++ creation contract consumed by the native assembly/SDK, not a GPU host."""
    header: str
    symbol: str
    parameters: dict


def native_factory(component, point, context):
    capability = component.implementation.capabilities
    return NativeFactory(capability['factory_header'], capability['factory_symbol'],
                         {k: v.to_dict() for k,v in point.params.items()})


_DEFINITIONS = None


def register():
    global _DEFINITIONS
    from .accelsim.declarations import NATIVE_COMPONENTS
    from .sm.definition import Sm
    for cls in (Sm, L2, Icnt, L1Shared):
        register_component(cls)
    if _DEFINITIONS is None:
        definitions = []
        for legacy in NATIVE_COMPONENTS:
            canonical = legacy.name.removeprefix('accelsim_')
            if canonical == 'sm120_mechanism':
                canonical, ident = 'sm', 'sm.accelsim-sm120'
            else:
                ident = canonical+'.accelsim'
            shared = ('clock_mhz',) if canonical == 'sm' else ()
            params = {k: v for k,v in legacy.params.items() if k not in shared}
            header, symbol = ('shader_api.hpp', 'make_shader_cluster') if canonical == 'sm' else (
                ('icnt_api.hpp', 'make_icnt') if canonical == 'icnt' else (
                ('memory_api.hpp', 'make_'+canonical) if canonical in ('l2', 'dram') else
                ('shader_api.hpp', 'make_shader_cluster')))
            definitions.append(ComponentImplementation(
                canonical, ident, '1', {'upstream': 'Accel-Sim pinned typed-v1', 'legacy_contract': legacy.name},
                native_factory, 'accelsim', 'native.transaction.v1' if canonical != 'sm' else 'native.instruction-trace.v1',
                'Native software cycles and mechanism counters; no hardware calibration claim',
                params=params, execution_modes=('native',), hardware_parameters=shared,
                adapters=('accelsim',) if canonical == 'sm' else (),
                validate=legacy().validate,
                capabilities={'execution_type': legacy.name,
                    'factory_header': header, 'factory_symbol': symbol,
                    'provider_types': {p: legacy.name for p in ('accelsim', 'accelsim_components')},
                    'contained_components': ['l1', 'shared_memory'] if canonical == 'sm' else [],
                    'port_protocol': 'transaction.v1',
                    'operations': 'actual validated native ISA / memory transaction operations; not analytical F/B',
                    'precision': 'traced native register types; full reference frontend currently FP32',
                    'shape': 'native cluster/CTA/sector constraints; explicit configuration required',
                    'granularity': 'cluster' if canonical == 'sm' else canonical,
                    'independent_factory': canonical != 'l1_shared'}))
        _DEFINITIONS = tuple(definitions)
    for definition in _DEFINITIONS:
        register_component_implementation(definition)
        _ALIASES[definition.capabilities['execution_type']] = (definition.component, definition.id, definition.version)


def canonical_adf(adf):
    """Explicitly bind a legacy ADF without changing its original record."""
    result = deepcopy(adf)
    for _, node in walk(result):
        if isinstance(node, SpacePoint):
            resolved = resolve_component(node)
            node.component = resolved.name
            node.params = dict(resolved.point.params)
            node.implementation = deepcopy(resolved.binding)
    return result


def native_adf(adf):
    """Validate the selected native execution contracts and materialize their ABI."""
    result = deepcopy(adf)
    for _, node in walk(result):
        if not isinstance(node, SpacePoint):
            continue
        resolved = resolve_component(node, execution_domain='accelsim')
        if resolved.implementation.factory is not native_factory:
            raise ValueError('selected native factory has no adapter for this assembly; explicit integration required')
        native = resolved.implementation.capabilities.get('execution_type')
        if not native:
            raise ValueError('selected implementation has no compatible native assembly contract')
        node.component = native
        node.params = resolved.materialize().params
        node.implementation = None
    return result
