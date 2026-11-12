"""Old non-GPU contracts that a component/implementation migration must retain.

No new selection API is guessed here. These are source-phase baseline controls;
they cannot establish that SM roofline/native or new registration works.
"""
from copy import deepcopy
import json

import pytest

from arch_explorer.adf import (
    Param, SpaceMatrix, SpacePoint, fingerprint, from_dict,
)
from arch_explorer.components.api import Kernel, build_simulator
from arch_explorer.mapping.context import MappingContext
from arch_explorer.simulator.events.engine import simulate


SOURCE = 'independent synthetic VAL52 legacy-control hardware'


def legacy_hardware():
    def point(name, component, values):
        return SpacePoint(name, component, {key: Param(value, unit, SOURCE)
                                           for key, (value, unit) in values.items()})
    return SpaceMatrix('chip', [
        point('v', 'vector_unit', {'lanes': (16, 'count'), 'sfu_lanes': (4, 'count'),
              'clock_ghz': (2.0, 'GHz'), 'precision': ('fp32', '1')}),
        point('ram', 'sram', {'capacity': (4096, 'B'), 'bandwidth': (64.0, 'B/ns'),
                            'latency': (0.0, 'ns')}),
    ])


def test_legacy_adf_roundtrip_keeps_parameters_sources_and_default_execution():
    adf = legacy_hardware()
    # This is the old request, before any implicit implementation normalization.
    old = json.loads(json.dumps(adf.to_dict()))
    assert all('implementation' not in point for point in old['elements'])
    again = from_dict(old)
    assert again.child('v').params == adf.child('v').params
    assert again.child('ram').params == adf.child('ram').params
    assert fingerprint(from_dict(again.to_dict())) == fingerprint(again)
    sim = build_simulator(again)
    unit = sim.points['chip/v']
    # 128 MAC / 16 lanes / 2 cycles-per-ns = 4ns; 256 FLOP, not 128 FLOP.
    kernel = Kernel('gemm', (1, 4, 8, 4), 'fp32')
    assert kernel.flops == 256
    assert unit.kernel_demand(kernel).resources == {unit.unit.name: 4.0}


def test_hardware_values_units_and_provenance_remain_in_identity():
    base = legacy_hardware()
    identities = {fingerprint(base)}
    for value, unit, source in [(1.0, 'GHz', SOURCE), (2.0, 'cycle', SOURCE),
                                (2.0, 'GHz', 'different parameter provenance')]:
        changed = deepcopy(base)
        changed.child('v').params['clock_ghz'] = Param(value, unit, source)
        identities.add(fingerprint(changed))
    assert len(identities) == 4


def test_component_rejects_unknown_parameter_wrong_unit_and_boolean_before_build():
    for value in [Param(2.0, 'cycle', SOURCE), Param(True, 'GHz', SOURCE)]:
        adf = legacy_hardware()
        adf.child('v').params['clock_ghz'] = value
        with pytest.raises(ValueError):
            build_simulator(adf)
    adf = legacy_hardware()
    adf.child('v').params['qa_undeclared'] = Param(1)
    with pytest.raises(ValueError):
        build_simulator(adf)


def test_legacy_vector_compute_does_not_recharge_explicit_memory_transfer():
    sim = build_simulator(legacy_hardware())
    unit, ram = sim.points['chip/v'], sim.points['chip/ram']
    context = MappingContext(sim)
    # Single explicit 1024B read at 64B/ns =16ns, then 128/16/2=4ns compute.
    read = context.task('read', ram.access(1024))
    demand = unit.kernel_demand(Kernel('vector', (2, 16), 'fp32', simt=4))
    assert demand.resources == {unit.unit.name: 4.0}
    context.task('compute', demand, [read])
    trace = simulate(context.graph)
    assert trace.latency_ns == 20.0
    assert [(record.start_ns, record.end_ns) for record in trace.records] == [(0.0, 16.0), (16.0, 20.0)]
