"""Matrix/vector dispatch is a hardware capability, not a timing correction."""
import pytest
from arch_explorer.adf import Param
from arch_explorer.adf.model import walk
from arch_explorer.components.base import Kernel
from arch_explorer.devices.pim import neupims
from arch_explorer.simulator.events import build_simulator


def models(array, enabled=False):
    adf = neupims(array, channels=1, banks=8)
    if enabled:
        for _, node in walk(adf):
            if getattr(node, 'component', None) == 'vector_unit':
                node.params.pop('matrix_kernels')
    sim = build_simulator(adf)
    return {name: next(p for p in sim.points.values() if p.component.name == name)
            for name in ('vector_unit', 'systolic_array')}


@pytest.mark.parametrize('array', [64, 128])
def test_matrix_dispatch_and_opt_out(array):
    current, old = models(array), models(array, True)
    for kernel in (Kernel('gemm', (1, 2, 128, 64), 'fp16'), Kernel('mvm', (1, 64, 128), 'fp16')):
        with pytest.raises(NotImplementedError):
            current['vector_unit'].kernel_time(kernel)
        assert old['vector_unit'].kernel_time(kernel) > 0
        assert current['systolic_array'].kernel_time(kernel) == old['systolic_array'].kernel_time(kernel)
    vector = Kernel('vector', (2, 128), 'fp16', simt=1)
    assert current['vector_unit'].kernel_time(vector) == old['vector_unit'].kernel_time(vector)


def test_systolic_fold_tail_is_unchanged():
    sa = models(64)['systolic_array']
    # A new output fold costs a 64-cycle double-buffered stationary interval;
    # its partial width does not remove the full array load.
    time = lambda n: sa.kernel_time(Kernel('gemm', (1, 2, 64, n), 'fp16'))
    # 2*64 + 64 + 2 - 3 = 191; second fold adds max(64,2).
    assert time(63) == time(64) == 191
    assert time(65) == 255
