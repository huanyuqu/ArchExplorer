"""Public physical connections cover every endpoint and survive ADF JSON."""
from copy import deepcopy
from dataclasses import replace

import pytest

from arch_explorer.adf import from_dict
from arch_explorer.simulator import api  # Public registration precedes builder use.
from arch_explorer.components.accelsim.componentization import gpu_adf, compile_topology


@pytest.mark.parametrize('clusters', [2, 4])
def test_explicit_gpu_connection_coverage_and_roundtrip(clusters):
    adf = gpu_adf(sm_clusters=clusters, memory_partitions=8, l2_subpartitions=2,
                  clocks_mhz={'sm': 1300, 'icnt': 1400, 'l2': 2100, 'dram': 1600},
                  detailed_dram=True)
    assert from_dict(adf.to_dict()).to_dict() == adf.to_dict()
    lines = compile_topology(adf).splitlines()
    assert lines[0] == 'arch-explorer.accelsim-topology.v1'
    # One ICNT, one SM node per cluster, one DRAM per partition and two L2s per DRAM.
    assert sum(line.startswith('node\t') for line in lines) == 1 + clusters + 8 + 16
    # Request/reply cluster wires, ICNT/L2 wires, L2/DRAM wires.
    assert len(adf.connections) == 2 * clusters + 2 * 16 + 2 * 16
    assert len({(c.source, c.source_port, c.destination, c.destination_port)
                for c in adf.connections}) == len(adf.connections)
    assert all(c.protocol == 'transaction.v1' for c in adf.connections)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'endpoint', 'port', 'protocol'])
def test_invalid_gpu_connections_fail_before_any_native_execution(defect):
    adf = deepcopy(gpu_adf())
    if defect == 'missing':
        adf.connections.pop()
    elif defect == 'duplicate':
        adf.connections.append(adf.connections[0])
    else:
        field, value = {'endpoint': ('destination', 'unknown'),
                        'port': ('destination_port', 'unknown'),
                        'protocol': ('protocol', 'incompatible')}[defect]
        adf.connections[0] = replace(adf.connections[0], **{field: value})
    with pytest.raises((ValueError, TypeError, KeyError)):
        compile_topology(adf)
