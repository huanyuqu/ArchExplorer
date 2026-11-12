"""CPU-only VAL54 input builders for the frozen SIM43 public interface."""
from copy import deepcopy

import pytest

from test_public_compilation_contract import roofline_input, two_reader_graph_input


@pytest.fixture
def adapter_api():
    import numpy  # noqa: F401 -- declared numerical dependency before guards
    from arch_explorer.workload import api as w
    from arch_explorer.simulator import api as s
    for name in ('compile_workload', 'validate_compilation', 'task_adapters',
                 'AdapterBinding', 'TaskAdapter', 'AdapterDomain', 'AdaptedGroup',
                 'register_task_adapter', 'register_adapter_domain'):
        assert callable(getattr(w, name)), name
    return w, s


def roofline_scene(**kwargs):
    from arch_explorer.adf import MappingSpec
    adf, graph, spec, old = roofline_input(**kwargs)
    mapping = MappingSpec('compile_graph_v2', {k:v for k,v in old.params.items()
                                         if k != 'threads_per_block'})
    return adf, graph, spec, mapping


def events_scene():
    from arch_explorer.adf import Param
    from arch_explorer.components.api import ImplementationBinding
    adf, graph, spec, mapping = roofline_scene(m=3, k=3, n=5,
                                               tiles=dict(m=2, k=2, n=3))
    source = 'synthetic VAL54 lane resource; no calibration'
    adf.elements[0].implementation = ImplementationBinding('sm.events', '1', {
        'clock_ghz': Param(1, 'GHz', source), 'lanes': Param(8, 'count', source),
        'sfu_lanes': Param(1, 'count', source)})
    return adf, graph, spec, mapping


def mixed_group_scene():
    from arch_explorer.adf import MappingSpec
    adf, _, _, _ = roofline_scene()
    events, _, _, _ = events_scene()
    lane = deepcopy(events.elements[0])
    lane.name = 'lanes'
    adf.elements.insert(0, lane)
    graph, spec = two_reader_graph_input()
    return adf, graph, spec, MappingSpec('compile_graph_v2', {
        'compute': {'first': 'board/lanes', 'reader-one': 'board/lanes',
                    'reader-two': 'board/sm'},
        'memory': 'board/dram', 'rules': {'matmul': {'id':'matmul.tiled','version':'1',
            'options': {'tiles': {'m':2,'k':2,'n':2}}}}})


def native_scene():
    from arch_explorer.adf import MappingSpec
    from arch_explorer.components.assemblies.gpu import gpu_adf
    from test_public_compilation_contract import graph_input
    graph, spec = graph_input(m=3, k=3, n=5)
    return gpu_adf(sm_clusters=1), graph, spec, MappingSpec('compile_graph_v2', {
        'compute':'component_gpu/cluster/sm', 'memory':'component_gpu/partition/dram',
        'rules':{'matmul':{'id':'matmul.tiled','version':'1','options':{'tiles':{'m':2,'k':2,'n':3}}}}})
