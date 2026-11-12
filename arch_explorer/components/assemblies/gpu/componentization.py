"""Public component-assembled GPU entry points, independent of user demos."""
from arch_explorer.components.accelsim.component_topology import connect_gpu, gpu_adf, compile_topology


def compare_results(component_result, oracle_result):
    """Read-only verify both bound results, then compare native integer facts."""
    from arch_explorer.components.accelsim.component_comparison import compare_results as compare
    return compare(component_result, oracle_result)


__all__ = ['gpu_adf', 'connect_gpu', 'compile_topology', 'compare_results']
