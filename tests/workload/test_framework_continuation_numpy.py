"""Hand values discriminate explicit conversion/reduction and typed paths."""
import numpy as np
import pytest
from framework_continuation_numpy import explicit_cast,cast_input,execute,reduction,evaluate


def test_cast_and_reduction_boundaries():
    assert explicit_cast(np.array([2**53+1],dtype=np.int64),'fp32')[0]==2**53
    np.testing.assert_array_equal(explicit_cast(np.array([-1.9,2.9]),'int64'),[-1,2])
    with pytest.raises(ValueError):explicit_cast(np.array([2.**63]),'int64')
    with pytest.raises(ValueError):cast_input([1.0],'int64')
    with pytest.raises(ValueError):cast_input([True],'int64')
    with pytest.raises(ValueError):cast_input([float('inf')],'fp32')
    # FP32 sum 256+1-256 = 1; storing a partial sum as BF16 gives zero.
    x=np.array([[256.,1.,-256.]])
    np.testing.assert_array_equal(reduction('ReduceSum',x,[1],1,'bf16'),[[1.]])
    assert explicit_cast(np.array([1.+2.**-8]),'bf16')[0]==1.
    assert explicit_cast(np.array([2.**-133]),'bf16')[0]==2.**-133
    np.testing.assert_array_equal(reduction('ReduceL2',[[3.,4.]],[1],0,'fp32'),[5.])


def test_new_typed_dispatch():
    np.testing.assert_array_equal(execute('Range',[np.array(2**53),np.array(2**53+3),np.array(1)],{},'int64')[0],
                                  [2**53,2**53+1,2**53+2])
    np.testing.assert_array_equal(execute('Where',[np.array([True,False]),np.array([1,2]),np.array([3,4])],{},'int64')[0],[1,4])
    np.testing.assert_array_equal(execute('Reshape',[np.arange(6),np.array([2,-1])],{},'int64')[0],[[0,1,2],[3,4,5]])
    assert execute('Flatten',[np.zeros((2,3,4))],{'axis':2},'fp32')[0].shape==(6,4)
    assert execute('MatMul',[],{},'fp32') is None


def test_multiple_outputs_and_integer_feed_remains_exact():
    onnx=pytest.importorskip('onnx');h=onnx.helper;T=onnx.TensorProto
    nodes=[h.make_node('ArgMax',['tokens'],['which'],axis=1,keepdims=0),
           h.make_node('Gather',['table','which'],['selected'],axis=0)]
    graph=h.make_graph(nodes,'qa-control',[h.make_tensor_value_info('tokens',T.INT64,[2,3]),
        h.make_tensor_value_info('table',T.FLOAT,[3,2])],[h.make_tensor_value_info('which',T.INT64,[2]),
        h.make_tensor_value_info('selected',T.FLOAT,[2,2])])
    model=h.make_model(graph,ir_version=8,opset_imports=[h.make_opsetid('',17)])
    result=evaluate(model,dict(tokens=np.array([[2**53,2**53+1,0],[3,3,1]],dtype=np.int64),
                               table=np.array([[1,2],[3,4],[5,6]],dtype=np.float32)))
    np.testing.assert_array_equal(result['outputs']['which'],[1,0])
    np.testing.assert_array_equal(result['outputs']['selected'],[[3,4],[1,2]])
    assert result['stages']['which'].dtype==np.int64


def test_shared_default_remains_usable_without_extension():
    onnx=pytest.importorskip('onnx');h=onnx.helper;T=onnx.TensorProto
    from framework_numpy_interpreter import evaluate as default_evaluate
    graph=h.make_graph([h.make_node('MatMul',['A','B'],['Y'])],'default-control',
        [h.make_tensor_value_info('A',T.FLOAT,[1,2]),h.make_tensor_value_info('B',T.FLOAT,[2,1])],
        [h.make_tensor_value_info('Y',T.FLOAT,[1,1])])
    model=h.make_model(graph,ir_version=8,opset_imports=[h.make_opsetid('',17)])
    np.testing.assert_array_equal(default_evaluate(model,dict(A=[[1,2]],B=[[3],[4]]))['outputs']['Y'],[[11]])
