# Concrete Accel-Sim implementation; owned by the component provider.
"""Validated scalar and topology constraints of the pinned native implementation.

Ranges are implementation/input bounds, not a device or a performance model.
Compound options not implemented here retain their attributed fixed grammar.
"""

RANGES = {
    'clock_mhz': (1, 4294),  # native get_core_freq returns uint Hz; keep all domains safe
    'booksim_k': (2, 4096),  # bounded single-stage router; no padded/dummy endpoints
    'gpgpu_shader_registers': (1, 2**30),
    'gpgpu_registers_per_block': (1, 2**30),
    'gpgpu_shader_cta': (1, 32),  # MAX_CTA_PER_SHADER
    'gpgpu_shmem_size': (1, 2**30),
    'gpgpu_shmem_sizeDefault': (1, 2**30),
    'gpgpu_shmem_per_block': (1, 2**30),
    'gpgpu_n_chiplet_partition': (1, 4096),
    'gpgpu_compute_capability_major': (1, 12),
    'gpgpu_compute_capability_minor': (0, 9),
    'gpgpu_ptx_force_max_capability': (50, 120),
    'gpgpu_l1_latency': (1, 2**20),
    'gpgpu_smem_latency': (1, 2**20),
    'gpgpu_l2_rop_latency': (1, 2**20),
    'dram_latency': (1, 2**20),
}


def topology(clusters, channels, subpartitions, value):
    """Return checked native counts; no inference of unknown physical topology."""
    if clusters < 1 or channels < 1:
        raise ValueError('accelsim topology needs positive cluster/partition repeat')
    if subpartitions not in (1, 2, 4, 8, 16):
        raise ValueError('accelsim subpartitions must be a power of two in 1..16 DRAM banks')
    total = channels * subpartitions
    # Fixed indexing=6 is IPOLY-MODULO; hashing.cc implements degrees 16/32/64.
    if total & -total not in (16, 32, 64):
        raise ValueError('accelsim IPOLY-MODULO subpartitions require power-of-two factor 16/32/64; unsupported partition repeat')
    chiplets = value('dram', 'gpgpu_n_chiplet_partition')
    if clusters % chiplets or channels % chiplets:
        raise ValueError('accelsim SM clusters and memory channels must divide evenly across chiplet partitions')
    ports = clusters + total
    if ports > RANGES['booksim_k'][1] or value('icnt', 'booksim_k') != ports:
        raise ValueError('accelsim BookSim k ports must equal SM clusters + memory subpartitions, with at most 4096 ports')
    if value('sm', 'gpgpu_registers_per_block') > value('sm', 'gpgpu_shader_registers'):
        raise ValueError('accelsim registers_per_block exceeds per-SM register capacity')
    shared = value('l1_shared', 'gpgpu_shmem_size')
    if value('l1_shared', 'gpgpu_shmem_per_block') > shared:
        raise ValueError('accelsim shared_per_block exceeds per-SM shared capacity')
    if value('l1_shared', 'gpgpu_shmem_sizeDefault') != shared:
        raise ValueError('accelsim shared default and effective per-SM shared capacity conflict')
    if value('l1_shared', 'gpgpu_adaptive_cache_config'):
        choices = [int(x) * 1024 for x in value('l1_shared', 'gpgpu_shmem_option').split(',')]
        if shared != max(choices):
            raise ValueError('accelsim adaptive shared options must end at per-SM shared capacity')
    return total, ports
