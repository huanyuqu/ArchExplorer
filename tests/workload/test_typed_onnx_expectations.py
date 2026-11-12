"""Prepare dtype semantics independently before consuming the production tip."""
import math
import numpy as np
import pytest
from typed_onnx_expectations import cast,elementwise,matmul,layer_norm,gelu_erf_decomposition
from framework_numpy_interpreter import decode_bf16


@pytest.mark.parametrize('dtype,p,emin',[('fp32',24,-126),('fp16',11,-14),('bf16',8,-126)])
def test_quantizer_even_ties_subnormal_and_negative_zero(dtype,p,emin):
    step=2.0**(1-p)
    actual=cast([1+step/2,1+3*step/2,-1-step/2,2.0**(emin-p),-0.0],dtype)
    assert actual.tolist()[:4]==[1,1+2*step,-1,0]
    assert math.copysign(1,actual[-1])==-1


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_cast_against_independent_torch_storage(dtype):
    torch=pytest.importorskip('torch')
    # Distinct ties, subnormals and ordinary values; no pretrained data.
    x=np.array([-1.00390625,-.333251953125,-0.0,0.0,2**-24,2**-133,.333251953125,1.00390625,65504],dtype=np.float32)
    t=torch.tensor(x).to({'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[dtype]).float().numpy()
    np.testing.assert_array_equal(cast(x,dtype),t.astype(np.float64))


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_rectangular_batched_dot(dtype):
    a=np.arange(12).reshape(2,2,3)/16
    b=np.arange(12).reshape(1,3,4)/32
    expected=np.matmul(a,b) # Exactly representable small binary products/sums.
    np.testing.assert_array_equal(matmul(a,b,dtype=dtype),cast(expected,dtype))


def test_layernorm_float32_torch_nonlast_axis_three_outputs():
    torch=pytest.importorskip('torch')
    x=(np.arange(24).reshape(2,3,4)-11)/16
    scale=np.arange(12).reshape(3,4)/64+1;bias=np.arange(12).reshape(3,4)/128
    r=layer_norm(x,scale,bias,dtype='fp32',axis=1,epsilon=1e-4)
    actual=torch.nn.functional.layer_norm(torch.tensor(x,dtype=torch.float32),(3,4),
        torch.tensor(scale,dtype=torch.float32),torch.tensor(bias,dtype=torch.float32),eps=1e-4).numpy()
    np.testing.assert_allclose(r['Y'],actual,rtol=3e-6,atol=3e-6)
    assert r['Mean'].shape==r['InvStdDev'].shape==(2,1,1)


@pytest.mark.parametrize('dtype',['fp16','bf16'])
def test_layernorm_low_storage_boundaries_are_explicit(dtype):
    r=layer_norm([[.1,.2,.4,.8]], [1.1,.7,1.3,.9], [.03,.07,.11,.13],dtype=dtype)
    np.testing.assert_array_equal(r['Normalized'],cast(r['Normalized'],dtype))
    np.testing.assert_array_equal(r['NormalizedScaled'],cast(r['NormalizedScaled'],dtype))
    np.testing.assert_array_equal(r['Y'],cast(r['Y'],dtype))
    with pytest.raises(ValueError,match='stash_type=1'):layer_norm([[1,2]],[1,1],[0,0],dtype=dtype,stash_type=16)


def test_gelu_erf_path_preserves_negative_branch():
    x=np.array([-3,-.5,0,.5,3],dtype=np.float64)
    direct=x*.5*(1+np.vectorize(math.erf)(x/math.sqrt(2)))
    np.testing.assert_allclose(gelu_erf_decomposition(x,dtype='fp32'),direct,atol=3e-7,rtol=3e-6)
    assert gelu_erf_decomposition(x,dtype='fp16')[0]<0


def test_bfloat_scalar_decode_retains_shape_and_bit_width():
    for bits,value in [(0x3f80,1.),(0xbf80,-1.),(0x3f00,.5)]:
        got=decode_bf16(np.array(bits,dtype=np.uint16))
        assert got.shape==() and got.dtype==np.float32 and float(got)==value


def test_layernorm_formal_zero_epsilon_optional_bias_suffix():
    # Exact population mean=2, variance=1, inv_std=1; no fitted tolerance.
    r=layer_norm([[1,3]],[2,3],dtype='fp32',epsilon=0)
    np.testing.assert_array_equal(r['Y'],[[-2,3]])
    np.testing.assert_array_equal(r['Mean'],[[2]])
    np.testing.assert_array_equal(r['InvStdDev'],[[1]])
    with pytest.raises(ValueError,match='suffix shape'):
        layer_norm([[1,3]],[1],dtype='fp32')
