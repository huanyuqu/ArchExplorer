"""Actual compiler/engine consumption of the independent VAL54 IO oracle.

The component's declared adapter selects conversion; service/counting expected
values remain independent of the implementation and adapter descriptors.
"""
from copy import deepcopy

from adapter_boundary_expectations import region_case, serial_roofline_ns
from test_public_compilation_contract import prohibit_execution
from adapter_boundary_cases import adapter_api, roofline_scene


def test_component_io_service_is_paid_once_and_ignores_external_dram_bandwidth(
        adapter_api, monkeypatch, tmp_path):
    from arch_explorer.adf import Param

    w, s = adapter_api
    case = region_case()
    expected = float(serial_roofline_ns(case))
    adf, _, spec, mapping = roofline_scene(m=3, k=3, n=5,
                                         tiles={'m': 2, 'k': 2, 'n': 3})
    observed = []
    for bandwidth in (1, 1024):
        architecture = deepcopy(adf)
        architecture.elements[1].params['bandwidth'] = Param(
            bandwidth, 'B/ns', 'synthetic VAL54 isolation control; no calibration')
        with prohibit_execution(monkeypatch, s):
            compilation = w.compile_workload(architecture, spec, mapping)
            request = s.prepare_compiled(compilation, output_dir=tmp_path/str(bandwidth),
                                         options={'trace': True})
        accesses = [step['access'] for step in compilation['plan']['steps']]
        assert sum(a['operations'] for a in accesses) == case['operations']
        assert sum(a['read_bytes'] for a in accesses) == case['read_bytes']
        assert sum(a['write_bytes'] for a in accesses) == case['write_bytes']
        result = s.evaluate_request(request)
        observed.append(result['evaluation']['latency_ns'])
        assert observed[-1] == expected == 67.5
        assert s.evaluate_request(result) == result
    assert observed == [expected, expected]
