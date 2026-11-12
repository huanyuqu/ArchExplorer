"""Preserve generic independent-port/contention checks from retired GPU-local tests."""
import pytest
from arch_explorer.adf import Param, SpaceMatrix, SpacePoint
from arch_explorer.archs import dmc_adf
from arch_explorer.components.base import Demand
from arch_explorer.mapping.context import MappingContext
from arch_explorer.simulator.events.builder import build_simulator
from arch_explorer.simulator.events.engine import simulate


@pytest.mark.parametrize('shared', [False, True])
def test_separate_memory_ports_overlap_but_one_port_serializes(shared):
    sim = build_simulator(dmc_adf(2))
    ports = ['chip/core[0]/sram:port', 'chip/core[1]/sram:port']
    if shared:
        ports[1] = ports[0]
    context = MappingContext(sim)
    for index, resource in enumerate(ports):
        context.task(str(index), Demand({resource: 320}))
    assert simulate(context.graph).latency_ns == pytest.approx(10 if shared else 5)


def test_generic_serial_services_keep_explicit_dependency_duration():
    adf = SpaceMatrix('services', [SpacePoint(name, 'serial_service',
        {'duration': Param(duration, 'ns', 'independent synthetic service contract')})
        for name, duration in [('dispatch', 2), ('memory', 3)]])
    sim = build_simulator(adf)
    context = MappingContext(sim)
    first = sim.points['services/dispatch']
    second = sim.points['services/memory']
    a = context.task('first', Demand({first.server.name: first.duration}))
    context.task('second', Demand({second.server.name: second.duration}), [a])
    assert simulate(context.graph).latency_ns == 5
