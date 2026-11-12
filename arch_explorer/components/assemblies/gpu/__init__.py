"""GPU assembly of explicitly selected standalone components."""
from .component_topology import gpu_adf as _legacy_gpu_adf, connect_gpu, compile_topology

def gpu_adf(**options):
    from arch_explorer.components.native_bindings import canonical_adf
    return canonical_adf(_legacy_gpu_adf(**options))
