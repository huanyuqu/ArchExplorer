"""New logical controls, independently specified; no product imports."""
import numpy as np
import pytest
from onnx_index_expectations import (
    first_argmax, gather, gather_elements, shape_of, concat, slice_tensor,
    expand, unsqueeze, squeeze, integer_binary,
)
from onnx_conv_expectations import conv2d, conv_geometry


def official(op, feeds, attrs=None, outputs=('Y',)):
    onnx=pytest.importorskip('onnx')
    from onnx.reference import ReferenceEvaluator
    inputs=[onnx.helper.make_tensor_value_info(n,onnx.helper.np_dtype_to_tensor_dtype(v.dtype),list(v.shape))
            for n,v in feeds.items()]
    # Runtime reference infers result types; graph outputs deliberately untyped.
    result=[onnx.helper.make_empty_tensor_value_info(n) for n in outputs]
    node=onnx.helper.make_node(op,list(feeds),list(outputs),**(attrs or {}))
    model=onnx.helper.make_model(onnx.helper.make_graph([node],'qa-controls',inputs,result),
                                opset_imports=[onnx.helper.make_opsetid('',17)],ir_version=8)
    return ReferenceEvaluator(model).run(None,feeds)


def test_exact_int64_first_max_and_eot_lookup():
    # A float intermediate would turn the first row into a tie at index zero.
    tokens=np.array([[2**53,2**53+1,2**53+1],[9,11,11]],dtype=np.int64)
    selected=first_argmax(tokens,-1,0)
    np.testing.assert_array_equal(selected,[1,1])
    np.testing.assert_array_equal(first_argmax(tokens,-1,0,1),[2,2])
    embeddings=np.arange(12,dtype=np.float32).reshape(3,4)
    np.testing.assert_array_equal(gather(embeddings,selected),[[4,5,6,7],[4,5,6,7]])
    np.testing.assert_array_equal(official('ArgMax',{'X':tokens},{'axis':-1,'keepdims':0})[0],selected)


def test_gather_signed_bounds_and_elements():
    x=np.array([[10,20,30],[40,50,60]],dtype=np.int32)
    np.testing.assert_array_equal(gather(x,np.array([-3,-1,1]),1),[[10,30,20],[40,60,50]])
    indices=np.array([[2,1],[0,-1]],dtype=np.int64)
    expected=np.array([[30,20],[40,60]],dtype=np.int32)
    np.testing.assert_array_equal(gather_elements(x,indices,1),expected)
    np.testing.assert_array_equal(official('GatherElements',{'data':x,'indices':indices},{'axis':1})[0],expected)
    for bad in ([-4],[3],[1.0],[True]):
        with pytest.raises(ValueError):gather(x,bad,1)


def test_shape_slice_broadcast_rank_and_concat():
    x=np.arange(24,dtype=np.int64).reshape(2,3,4)
    np.testing.assert_array_equal(shape_of(x,-2,100),[3,4])
    starts=np.array([np.iinfo(np.int64).max],dtype=np.int64)
    ends=np.array([np.iinfo(np.int64).min],dtype=np.int64)
    axes=np.array([-1],dtype=np.int64);steps=np.array([-2],dtype=np.int64)
    expected=x[:,:,[3,1]]
    actual=slice_tensor(x,starts,ends,axes,steps)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(official('Slice',dict(data=x,starts=starts,ends=ends,axes=axes,steps=steps))[0],expected)
    np.testing.assert_array_equal(expand(np.array([[1],[2]],dtype=np.int64),[3]),[[1,1,1],[2,2,2]])
    expanded=unsqueeze(x,[0,-1]);assert expanded.shape==(1,2,3,4,1)
    np.testing.assert_array_equal(squeeze(expanded,[0,-1]),x)
    np.testing.assert_array_equal(concat([x[:1],x[1:]],0),x)
    with pytest.raises(ValueError):slice_tensor(x,[0,0],[1,1],[1,1])
    with pytest.raises(ValueError):slice_tensor(x,[0],[1],[0],[0])
    with pytest.raises(ValueError):unsqueeze(x,[0,-5])
    with pytest.raises(ValueError):squeeze(x,[0])
    with pytest.raises(ValueError):concat([x,x.astype(np.float64)])
    with pytest.raises(ValueError):expand(x,[2.0,3,4])


def test_integer_shape_arithmetic_never_float():
    a=np.array([2**53+1,-7,7],dtype=np.int64)
    np.testing.assert_array_equal(integer_binary('Add',a,np.array([2,0,0])),[2**53+3,-7,7])
    np.testing.assert_array_equal(integer_binary('Div',a,np.array([1,3,-3])),[2**53+1,-2,-2])
    with pytest.raises(ValueError):integer_binary('Add',np.array([2**63-1]),np.array([1]))
    with pytest.raises(ValueError):integer_binary('Div',np.array([1]),np.array([0]))


def test_conv_hand_arithmetic_and_low_precision_storage():
    x=np.arange(1,10,dtype=np.float32).reshape(1,1,3,3)
    w=np.array([1,0,0,-1],dtype=np.float32).reshape(1,1,2,2)
    b=np.array([.5],dtype=np.float32)
    expected=np.full((1,1,2,2),-3.5,dtype=np.float32)
    np.testing.assert_array_equal(conv2d(x,w,b),expected)
    np.testing.assert_array_equal(official('Conv',dict(X=x,W=w,B=b))[0],expected)
    for dtype,subnormal in [('fp16',2.**-24),('bf16',2.**-133)]:
        np.testing.assert_array_equal(conv2d([[[[subnormal]]]],[[[[1.]]]],dtype=dtype),[[[[subnormal]]]])
    # Halfway results round to even at the output storage boundary.
    assert conv2d([[[[1.]]]],[[[[1.]]]],np.array([2.**-8]),dtype='bf16').item()==1.
    with pytest.raises(ValueError,match='nonfinite'):
        conv2d([[[[65504.]]]],[[[[2.]]]],dtype='fp16')


def test_grouped_dilated_asymmetric_conv_official():
    x=np.arange(48,dtype=np.float32).reshape(1,2,4,6)
    w=np.array([1,-1,2,0, -2,1,0,3],dtype=np.float32).reshape(2,1,2,2)
    opts=dict(group=2,strides=[2,2],dilations=[2,1],pads=[1,0,0,1])
    np.testing.assert_array_equal(conv2d(x,w,**opts),official('Conv',dict(X=x,W=w),opts)[0])
    upper=conv_geometry([1,1,4,4],[1,1,3,3],strides=[2,2],auto_pad='SAME_UPPER')
    lower=conv_geometry([1,1,4,4],[1,1,3,3],strides=[2,2],auto_pad='SAME_LOWER')
    assert upper['output_shape']==lower['output_shape']==[1,1,2,2]
    assert upper['pads']==[0,0,1,1] and lower['pads']==[1,1,0,0]
    for auto in ('SAME_UPPER','SAME_LOWER'):
        opts=dict(strides=[2,2],auto_pad=auto)
        # ONNX 1.16.2 auto_pad reference uses X.shape[i] (N,C), not
        # X.shape[i+2] (H,W). Use independently derived explicit pads here.
        explicit=conv_geometry([1,1,4,4],[1,1,3,3],**opts)['pads']
        np.testing.assert_array_equal(conv2d(x[:,:1,:4,:4],np.ones((1,1,3,3)),**opts),
            official('Conv',dict(X=x[:,:1,:4,:4],W=np.ones((1,1,3,3),dtype=np.float32)),
                     dict(strides=[2,2],pads=explicit))[0])
    with pytest.raises(ValueError):conv_geometry([1,2,4,4],[2,2,3,3],group=2)
    with pytest.raises(ValueError):conv_geometry([1,1,4,4],[1,1,3,3],pads=[1,0,0,0],auto_pad='VALID')


def test_official_same_padding_discrepancy():
    onnx=pytest.importorskip('onnx')
    x=np.arange(16,dtype=np.float32).reshape(1,1,4,4)
    w=np.ones((1,1,3,3),dtype=np.float32)
    expected=np.array([[[[45,39],[66,50]]]],dtype=np.float32)
    np.testing.assert_array_equal(conv2d(x,w,strides=[2,2],auto_pad='SAME_UPPER'),expected)
    observed=official('Conv',dict(X=x,W=w),dict(strides=[2,2],auto_pad='SAME_UPPER'))[0]
    if onnx.__version__=='1.16.2' and not np.array_equal(observed,expected):
        pytest.xfail('ONNX1.16.2 op_conv auto_pad uses N,C in spatial padding calculation; not an oracle pass')
    np.testing.assert_array_equal(observed,expected)


def test_conv_shape_arithmetic_is_integer_exact_without_allocation():
    size=2**53+1
    geometry=conv_geometry([1,1,size,1],[1,1,1,1],strides=[2,1],auto_pad='SAME_UPPER')
    assert geometry['output_shape']==[1,1,2**52+1,1]
