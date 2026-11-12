"""Release-pinned independent ONNX model/reference tests, no product calls."""
from copy import deepcopy
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import pytest
import onnx
from onnx.reference import ReferenceEvaluator
from onnx_import_cases import (mlp_case,attention_case,single_case,save_case,feed_values,_value)
from onnx_import_expectations import (ABS_TOL,REL_TOL,POLICY,attention,mlp,gemm,softmax,reshape_shape)

def close(actual,expected):
    actual=np.asarray(actual);expected=np.asarray(expected)
    assert actual.shape==expected.shape
    if expected.dtype==np.int64:np.testing.assert_array_equal(actual,expected)
    else:
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        np.testing.assert_allclose(actual,expected,atol=ABS_TOL,rtol=REL_TOL)

def reference_values(model,feeds):
    names=[name for node in model.graph.node for name in node.output]
    values=ReferenceEvaluator(model).run(names,feeds)
    return dict(zip(names,values))

@pytest.mark.parametrize('name,external',[('mlp',False),('attention',False),('mlp',True)])
def test_real_files_all_nodes_and_outputs_match_independent_closed_form(tmp_path,name,external):
    case={'mlp':mlp_case,'attention':attention_case}[name]()
    path=save_case(case,tmp_path/'model',external=external)
    model=onnx.load(str(path));manifest=json.loads(path.with_name('INPUTS.json').read_text())
    assert model.ir_version==8 and [(x.domain,x.version) for x in model.opset_import]==[('',17)]
    assert not manifest['contains_expected_answers']
    values=reference_values(model,feed_values(manifest))
    expected={'mlp':mlp,'attention':attention}[name](case.roots)
    assert set(values)==set(expected)
    for tensor in expected:close(values[tensor],expected[tensor])
    assert tuple(o.name for o in model.graph.output)==case.outputs
    names=[node.name for node in model.graph.node]
    assert '' in names and len(set(names))<len(names)
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest['model']['sha256']
    if external:
        for tensor in model.graph.initializer:
            close(onnx.numpy_helper.to_array(tensor),case.roots[tensor.name])
    record=dict(case=name,external=external,policy=POLICY,onnx_reference=True,
        independent_NumPy_closed_form=True,all_value_names=list(values),component_execution=False,
        model_sha256=hashlib.sha256(raw).hexdigest())
    (tmp_path/'REFERENCE-CHECK.json').write_text(json.dumps(record,indent=2)+'\n')

@pytest.mark.parametrize('factory,oracle',[(mlp_case,mlp),(attention_case,attention)])
def test_root_changes_are_traced_at_correct_intermediates(factory,oracle):
    case=factory();base=reference_values(case.model,case.feeds)
    for name,value in case.roots.items():
        roots=deepcopy(case.roots);roots[name].flat[0]+=.25
        model=onnx.ModelProto();model.CopyFrom(case.model);feeds=deepcopy(case.feeds)
        if name in feeds:feeds[name]=roots[name]
        else:
            tensor=next(t for t in model.graph.initializer if t.name==name)
            tensor.CopyFrom(onnx.numpy_helper.from_array(roots[name],name))
            assert model.SerializeToString()!=case.model.SerializeToString()
        actual=reference_values(model,feeds);expected=oracle(roots)
        for tensor in expected:close(actual[tensor],expected[tensor])
        assert any(not np.array_equal(actual[n],base[n]) for n in expected),name
        # Key bias can change every score equally; softmax invariance is valid.
        if name=='kb':assert not np.array_equal(actual['K'],base['K'])

def test_external_bytes_are_sliced_at_explicit_offset_and_portable(tmp_path):
    case=mlp_case();path=save_case(case,tmp_path/'first',external=True)
    raw=(path.parent/'weights/data.bin').read_bytes()
    assert len(raw)==16+case.roots['W1'].nbytes+3
    np.testing.assert_array_equal(np.frombuffer(raw[16:-3],dtype='<f4').reshape(4,3),case.roots['W1'])
    shutil.copytree(path.parent,tmp_path/'portable')
    moved=tmp_path/'portable'/'renamed.onnx'
    (tmp_path/'portable/model.onnx').rename(moved)
    for tensor,value in reference_values(onnx.load(str(moved)),case.feeds).items():
        close(value,mlp(case.roots)[tensor])
    (moved.parent/'weights/data.bin').unlink()
    with pytest.raises((OSError,onnx.checker.ValidationError)):
        onnx.checker.check_model(str(moved),full_check=True)

@pytest.mark.parametrize('transA,transB',[(0,0),(1,0),(0,1),(1,1)])
def test_hand_rectangular_gemm_alpha_beta_and_transpose(transA,transB,tmp_path):
    a=np.array([[1,2,-1],[.5,-.25,2]],np.float32)
    b=np.array([[1,0,.5,-1],[.25,2,-.5,.75],[-1,.5,1,.25]],np.float32)
    c=np.array([[.5,-.25,.125,1]],np.float32)
    aa=a.T.copy() if transA else a;bb=b.T.copy() if transB else b
    case=single_case('Gemm',{'A':aa,'B':bb,'C':c},(2,4),
        attrs=dict(alpha=.75,beta=-1.25,transA=transA,transB=transB))
    save_case(case,tmp_path/'gemm')
    wanted=np.array([[.75*sum(float(a[i,k])*float(b[k,j]) for k in range(3))-1.25*float(c[0,j])
                       for j in range(4)] for i in range(2)],np.float32)
    close(gemm(aa,bb,c,alpha=.75,beta=-1.25,transA=transA,transB=transB),wanted)
    close(ReferenceEvaluator(case.model).run(None,case.feeds)[0],wanted)

@pytest.mark.parametrize('shapeA,shapeB',[
    ((2,3,4),(4,2)),((2,1,3,4),(1,5,4,2)),((3,4),(2,4,5))])
def test_batched_matmul_broadcast_has_independent_scalar_dot(shapeA,shapeB):
    a=(np.arange(np.prod(shapeA),dtype=np.float32).reshape(shapeA)%11-5)/8
    b=(np.arange(np.prod(shapeB),dtype=np.float32).reshape(shapeB)%7-3)/4
    batch=np.broadcast_shapes(a.shape[:-2],b.shape[:-2]);m,k=a.shape[-2:];n=b.shape[-1]
    aa=np.broadcast_to(a,batch+(m,k));bb=np.broadcast_to(b,batch+(k,n))
    wanted=np.empty(batch+(m,n),np.float32)
    for leading in np.ndindex(batch):
        for i in range(m):
            for j in range(n):wanted[leading+(i,j)]=sum(float(aa[leading+(i,t)])*float(bb[leading+(t,j)]) for t in range(k))
    case=single_case('MatMul',{'A':a,'B':b},wanted.shape)
    close(ReferenceEvaluator(case.model).run(None,case.feeds)[0],wanted)

@pytest.mark.parametrize('axis',[0,1,2,-1,-2])
def test_softmax_axis_is_one_axis_not_legacy_flatten(axis):
    x=np.array([[[1,-2,3],[.25,.5,-.75]],[[2,1,0],[-1,-2,-3]]],np.float32)
    expected=softmax(x,axis)
    case=single_case('Softmax',{'X':x},x.shape,attrs=dict(axis=axis))
    close(ReferenceEvaluator(case.model).run(None,case.feeds)[0],expected)
    np.testing.assert_allclose(expected.sum(axis=axis),1,atol=2e-7,rtol=0)

def test_hand_uniform_softmax_and_scalar_broadcast():
    x=np.zeros((2,3,4),np.float32)
    np.testing.assert_array_equal(softmax(x,1),np.full(x.shape,np.float32(1/3)))
    case=single_case('Add',{'X':np.array([[1,-2],[.5,0]],np.float32),'scalar':np.array(.25,np.float32)},(2,2))
    close(ReferenceEvaluator(case.model).run(None,case.feeds)[0],[[1.25,-1.75],[.75,.25]])

@pytest.mark.parametrize('input_shape,target,allowzero',[
    ((2,3,4),(0,-1),0),((2,3,4),(0,0,-1),0),((2,3,4),(3,8),0),((2,0,3),(0,6),1)])
def test_reshape_zero_copy_inference_and_literal_zero(input_shape,target,allowzero):
    shape=reshape_shape(input_shape,list(target),allowzero=allowzero)
    x=np.arange(np.prod(input_shape),dtype=np.float32).reshape(input_shape)
    case=single_case('Reshape',{'X':x},shape,attrs=dict(allowzero=allowzero),
                     initializers={'shape':np.array(target,np.int64)})
    close(ReferenceEvaluator(case.model).run(None,case.feeds)[0],x.reshape(shape))

@pytest.mark.parametrize('shape,target,allowzero',[
    ((2,3),[-1,-1],0),((2,3),[4,-1],0),((2,3),[2,0,0],0),((2,3),[0,-1],1)])
def test_impossible_reshape_contract_is_rejected_by_independent_oracle(shape,target,allowzero):
    with pytest.raises(ValueError):reshape_shape(shape,target,allowzero=allowzero)

def test_onnx_initializer_graph_input_can_be_overridden_in_standard_reference():
    case=mlp_case();case.model.graph.input.append(_value('W2',case.roots['W2'].shape))
    onnx.checker.check_model(case.model,full_check=True)
    feeds=deepcopy(case.feeds);feeds['W2']=np.zeros_like(case.roots['W2'])
    actual=reference_values(case.model,feeds);roots=deepcopy(case.roots);roots['W2']=feeds['W2']
    for name,value in mlp(roots).items():close(actual[name],value)
    # Product must explicitly accept overrides or refuse them; policy is not guessed here.
