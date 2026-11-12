"""Deferred positive shape/attribute consumers of real standard ONNX files."""
import numpy as np
import onnx
import pytest
from onnx_import_cases import single_case,save_case
from onnx_numpy_interpreter import evaluate_model
from test_onnx_import_contract import onnx_api,compare_original_values
from test_onnx_independent_expectations import reference_values,close
from test_operator_mapping_attention_contract import no_backend

def edge_cases():
    values=np.arange(24,dtype=np.float32).reshape(2,3,4)/16-1
    result=[]
    for axis in (0,1,2,-1):result.append(single_case('Softmax',{'X':values},[2,3,4],attrs={'axis':axis}))
    result.append(single_case('Transpose',{'X':values},[4,3,2]))
    result.append(single_case('Transpose',{'X':values},[3,2,4],attrs={'perm':[1,0,2]}))
    result.append(single_case('Add',{'A':values,'B':np.asarray(.25,np.float32)},[2,3,4]))
    result.append(single_case('Mul',{'A':np.arange(6,dtype=np.float32).reshape(2,3,1),'B':np.arange(4,dtype=np.float32).reshape(1,1,4)},[2,3,4]))
    result.append(single_case('MatMul',{'A':values,'B':np.arange(20,dtype=np.float32).reshape(1,4,5)/16},[2,3,5]))
    for ta in (0,1):
        for tb in (0,1):
            a=np.arange(6,dtype=np.float32).reshape(2,3)/8;b=np.arange(12,dtype=np.float32).reshape(3,4)/8
            result.append(single_case('Gemm',{'A':a.T if ta else a,'B':b.T if tb else b,'C':np.asarray(.25,np.float32)},[2,4],attrs={'transA':ta,'transB':tb,'alpha':.5,'beta':-.5}))
    result.append(single_case('Identity',{'X':np.asarray(.125,np.float32)},[]))
    return result

@pytest.mark.parametrize('index',range(14))
def test_positive_nondefault_axes_scalars_broadcast_and_gemm(onnx_api,monkeypatch,tmp_path,index):
    module,w,_=onnx_api;case=edge_cases()[index];path=save_case(case,tmp_path/'edge')
    expected=evaluate_model(case.model,case.feeds)['stages'];official=reference_values(case.model,case.feeds)
    for name in expected:close(official[name],expected[name])
    with no_backend(monkeypatch):
        spec=module.import_onnx(path,inputs={k:v.tolist() for k,v in case.feeds.items()})
        description=module.describe_onnx(spec)
        assert compare_original_values(w.execute_reference(spec),description,expected)
