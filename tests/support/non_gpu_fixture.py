"""Synthetic shared-memory vector architecture for generic test contracts."""
from arch_explorer.adf import SpaceMatrix
from arch_explorer.archs.adf_seeds import memory, mesh, point


def shared_adf(num_cores=4, *, dram_capacity_bytes=2**30):
    compute = point('vector', 'vector_unit', {'lanes': (16, 'count'), 'sfu_lanes': (4, 'count'),
        'clock_ghz': (1.0, 'GHz'), 'precision': ('fp16', '1')}, source='non-GPU generic test fixture')
    core = SpaceMatrix('core', [compute], repeat=num_cores)
    return SpaceMatrix('chip', [core, memory('l2', capacity=2**24),
        memory('dram', technology='ddr', capacity=dram_capacity_bytes, bandwidth=64.0, latency=100.0)],
        [mesh(cols=num_cores + 2)])


def shared_configuration():
    return {'seed_adf': shared_adf().to_dict(), 'workloads': [{'kind': 'operator',
        'params': {'op': 'matmul', 'm': 16, 'k': 32, 'n': 32}}], 'backend': 'events',
        'mapping_templates': ['gsm_tiled'], 'max_evaluations': 8,
        'mapping_samples': 1, 'parameter_samples': 1, 'random_seed': 17}
