"""SigLIP BF16 vs Gemma FP32 contract, from openpi 215abfb + Flax 0.10.2.

Primitive expected values were independently observed with JAX 0.5.3 CPU;
these checks do not equate host primitive arithmetic with GPU kernel accuracy.
"""
import math
import numpy as np
from arch_explorer.workload.full_primitives import FullOperator, reference
from arch_explorer.workload.pi05 import pi05_workload


def test_visual_and_language_attention_preserve_distinct_precision():
    w = pi05_workload({'num_steps': 1})
    qks = [o for o in w.operators if o.name.endswith('.attn.qk')]
    visual = [o for o in qks if '.vision' in o.name]
    language = [o for o in qks if '.vision' not in o.name]
    assert len(visual) == 3 * 27
    assert len(language) == 2 * 18
    for o in visual + language:
        dt = 'bf16' if o in visual else 'fp32'
        assert w.tensors[o.outputs[0]].dtype == dt
        softmax = next(x for x in w.operators if x.name == o.name[:-2] + 'softmax')
        assert w.tensors[softmax.outputs[0]].dtype == dt
        assert softmax.attrs['reduction_dtype'] == 'fp32'
        if dt == 'bf16':
            assert softmax.attrs['casts_per_element'] == 6
            assert softmax.attrs['extra_row_simt'] == 2
            assert softmax.attrs['write_bytes'] == math.prod(w.tensors[softmax.outputs[0]].shape) * 2
    assert w.tensors[w.meta['full_model']['outputs'][0]].dtype == 'fp32'
    assert all(w.tensors[t].dtype == 'bf16' for pair in w.meta['full_model']['prefix_cache_ids'][0] for t in pair)


def test_bf16_softmax_rounds_intermediates_not_only_final_storage():
    x = np.array([[.10009765625,.2001953125,.30078125,4.],[-4.,2.,0.,1.]],np.float32)
    expected = np.array([[.018798828125,.02099609375,.0230712890625,.93359375],
                         [.00164031982421875,.6640625,.08984375,.2431640625]],np.float32)
    op = FullOperator('softmax','softmax',('x',),('y',),dict(result_dtype='bf16',intermediate_dtype='bf16',reduction_dtype='fp32'))
    np.testing.assert_array_equal(reference(op,[x]),expected)
    old = FullOperator('softmax','softmax',('x',),('y',),dict(result_dtype='bf16'))
    assert not np.array_equal(reference(old,[x]),expected)


def test_visual_query_scaling_uses_bf16_sqrt_divisor():
    x=np.array([.10009765625,1.,3.140625,8.,-6.25,20.],np.float32)
    expected=np.array([.01177978515625,.11767578125,.369140625,.94140625,-.734375,2.359375],np.float32)
    source=FullOperator('scale','scale',('x',),('y',),dict(result_dtype='bf16',divisor=math.sqrt(72),divisor_dtype='bf16'))
    old=FullOperator('scale','scale',('x',),('y',),dict(result_dtype='bf16',factor=72**-.5))
    np.testing.assert_array_equal(reference(source,[x]),expected)
    assert not np.array_equal(reference(old,[x]),expected)
