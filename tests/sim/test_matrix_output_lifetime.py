"""Declared destination precision must not erase FP32 matrix temporaries."""
import pytest

from arch_explorer.adf import MappingSpec, Param, WorkloadSpec, walk
from arch_explorer.components.base import dtype_bytes
from arch_explorer.simulator.events.backend import run_mapping
from arch_explorer.workload.full_models import Graph
from storage_lifetime import assert_held_through_casts, storage_events


def probe(dtype, monkeypatch):
    import arch_explorer.mapping.full_models as mapper
    graph = Graph('matrix-conversion-probe', dtype)
    x = graph.w.tensor('x', (1, 2, 3))
    out = graph.mm('product', x, graph.weight('w', (3, 4)), 1, 2, 3, 4, True)
    graph.w.meta['full_model'] = {'outputs': {out: [1, 2, 4]}, 'parameter_bytes': 12 * dtype_bytes(dtype)}
    monkeypatch.setattr(mapper, 'workload_from_spec', lambda _: graph.w)
    return graph.w


@pytest.mark.parametrize('dtype', ['fp16', 'bf16', 'fp32'])
def test_matrix_conversion_storage_cache_bytes_and_last_consumer(dtype, monkeypatch):
    from arch_explorer.archs.adf_seeds import non_gpu_adf
    probe(dtype, monkeypatch)
    adf = non_gpu_adf(matrix_precision=dtype)
    from arch_explorer.archs.adf_seeds import memory
    adf.elements.append(memory('shared_cache', capacity=4096))
    for network in adf.networks:
        if network.component == 'mesh2d':
            network.params['cols'] = Param(network.value('cols') + 1, 'count', 'extra explicit shared-cache endpoint')
    sim, result = run_mapping(adf, WorkloadSpec('vit', {}), MappingSpec('gsm_tiled', {'tile_m': 64, 'tile_n': 64}))
    info = sim.mapping.details['full_model']
    row = info['operators'][0]
    assert row['operator'] == 'product'
    if dtype == 'fp32':
        assert row['fp32_matrix_scratch_bytes'] == row['explicit_output_casts'] == 0
        assert row['output_cast_read_bytes'] == row['output_cast_write_bytes'] == 0
        assert row['output_cast_tasks'] == []
        return
    # Eight FP32 accumulators exist in addition to the eight 16-bit outputs.
    assert row['fp32_matrix_scratch_bytes'] == row['matrix_output_bytes'] == 8 * 4
    assert row['write_bytes'] == row['output_cast_write_bytes'] == 8 * 2
    assert row['output_cast_read_bytes'] == 8 * 4
    assert row['explicit_output_casts'] == 8 and row['fused_output_casts'] == 0
    assert row['cache_reservation_bytes'] >= row['logical_input_bytes'] + 8 * 4
    scratch = row['fp32_matrix_scratch_task']
    casts = row['output_cast_tasks']
    assert scratch is not None and casts
    records, frees = storage_events(sim.mapping.graph, result)
    assert_held_through_casts(sim.mapping.graph, records, frees, scratch, casts)
    dram = next(path for path, node in walk(adf) if getattr(node, 'component', None) == 'dram')
    assert sum(sim.mapping.graph.tasks[task].demand.get(dram + ':port', 0) for task in casts) == 8 * (4 + 2)
    for task in casts:
        resources = sim.mapping.graph.tasks[task].demand
        assert any(path in resources and node.component == 'vector_unit' for path, node in walk(adf)
                   if getattr(node, 'component', None))
        assert not any(path in resources and node.component == 'systolic_array' for path, node in walk(adf)
                       if getattr(node, 'component', None))


@pytest.mark.parametrize('kind', ['vit', 'clip'])
def test_missing_conversion_capability_cannot_execute(kind):
    from arch_explorer.archs.adf_seeds import non_gpu_adf
    from arch_explorer.adf.registry import MAPPING_TEMPLATES
    adf = non_gpu_adf(matrix_precision='fp16')
    core = adf.child('core')
    core.elements = [node for node in core.elements if getattr(node, 'component', None) != 'vector_unit']
    workload = WorkloadSpec(kind, {'profile': 'synthetic', 'dtype': 'fp16'})
    with pytest.raises((ValueError, NotImplementedError)):
        run_mapping(adf, workload, MappingSpec('gsm_tiled', {}))
    assert not MAPPING_TEMPLATES.get('gsm_tiled').applicable(adf, workload)


@pytest.mark.parametrize('kind', ['vit', 'clip'])
@pytest.mark.parametrize('precision', ['fp16', 'fp32'])
def test_complete_non_gpu_capability_has_real_success_control(kind, precision):
    from arch_explorer.archs.adf_seeds import non_gpu_adf
    from arch_explorer.adf.registry import MAPPING_TEMPLATES
    adf = non_gpu_adf(matrix_precision=precision)
    workload = WorkloadSpec(kind, {'profile': 'synthetic', 'dtype': precision})
    assert MAPPING_TEMPLATES.get('gsm_tiled').applicable(adf, workload)
    sim, result = run_mapping(adf, workload, MappingSpec('gsm_tiled', {}))
    assert result.latency_ns > 0
    assert sim.mapping.details['full_model']['outputs']
