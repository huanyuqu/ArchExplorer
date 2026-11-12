"""Deferred product refusals bound to SIM51 REFUSALS.md; never broad Exception."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import onnx
import pytest
from onnx_import_cases import mlp_case,save_case,single_case
from onnx_import_negative_cases import model_mutation,external_mutation
from test_onnx_import_contract import onnx_api,compare_original_values
from onnx_numpy_interpreter import evaluate_model
from test_operator_mapping_attention_contract import no_backend

MODEL_ERRORS={
 'opset18':'ONNX supports exactly IR 8, standard empty domain, opset 17',
 'IR9':'ONNX supports exactly IR 8, standard empty domain, opset 17',
 'Sin':'ONNX unsupported node/domain:',
 'unknown_attribute':'ONNX unsupported/duplicate attributes for',
 'symbolic_batch':'ONNX unresolved root dimension; supply explicit input_shapes',
 'conflicting_shape':'ONNX (concrete root shape conflicts with annotation|checker/shape inference rejected model:|root X .*finite FP32 numbers)',
 'fp64':'ONNX tensor dtype must be FP32/FP16/BF16 or compile-time int64 shape',
 'missing_root':'ONNX inputs must exactly bind graph roots;',
 'nonfinite_root':'(ONNX root X .*finite FP32 numbers|workload graph requires finite JSON numbers)',
 'overridable_initializer':'ONNX duplicate root/initializer or initializer also listed in graph.input is unsupported'}

@pytest.mark.parametrize('condition',list(MODEL_ERRORS))
def test_model_refusals_before_backend_have_specific_diagnostics(onnx_api,monkeypatch,tmp_path,condition):
    module,w,_=onnx_api
    positive=mlp_case();path=save_case(positive,tmp_path/'positive')
    negative=model_mutation(tmp_path/'negative',condition)
    with no_backend(monkeypatch):
        assert w.workload_from_spec(module.import_onnx(path,inputs={k:v.tolist() for k,v in positive.feeds.items()}))
        with pytest.raises(ValueError,match=MODEL_ERRORS[condition]):
            module.import_onnx(negative.path,inputs={k:v.tolist() for k,v in negative.feeds.items()})

EXTERNAL_ERRORS={
 'missing':'ONNX external data unavailable:',
 'truncated':'ONNX external offset/length/dtype/shape byte range differs',
 'offset_beyond_file':'ONNX external offset/length/dtype/shape byte range differs',
 'length_short':'ONNX external offset/length/dtype/shape byte range differs',
 'negative_offset':'ONNX external offset must be a nonnegative decimal integer',
 'escape':'ONNX external location must be a normalized relative path without escape',
 'absolute_location':'ONNX external location must be a normalized relative path without escape',
 'symlink_escape':'ONNX external tensor escaped allowed root or is not a file'}

@pytest.mark.parametrize('condition',list(EXTERNAL_ERRORS))
def test_external_root_boundaries_preserve_explicit_byte_contract(onnx_api,monkeypatch,tmp_path,condition):
    module,w,_=onnx_api;case=mlp_case();positive=save_case(case,tmp_path/'positive',external=True)
    negative=external_mutation(tmp_path/'negative',condition)
    with no_backend(monkeypatch):
        assert w.workload_from_spec(module.import_onnx(positive,inputs={k:v.tolist() for k,v in case.feeds.items()},external_root=positive.parent))
        with pytest.raises(ValueError,match=EXTERNAL_ERRORS[condition]):
            module.import_onnx(negative.path,inputs={k:v.tolist() for k,v in negative.feeds.items()},external_root=negative.path.parent)

def test_explicit_symbolic_specialization_and_required_external_root(onnx_api,monkeypatch,tmp_path):
    module,w,_=onnx_api;row=model_mutation(tmp_path/'symbolic','symbolic_batch')
    feeds={k:v.tolist() for k,v in row.feeds.items()}
    expected=evaluate_model(onnx.load(str(row.path)),row.feeds)['stages']
    with no_backend(monkeypatch):
        spec=module.import_onnx(row.path,inputs=feeds,input_shapes={'X':[2,3]})
        assert compare_original_values(w.execute_reference(spec),module.describe_onnx(spec),expected)
        with pytest.raises(ValueError,match='ONNX concrete root shape conflicts with annotation'):
            module.import_onnx(row.path,inputs=feeds,input_shapes={'X':[2,4]})
        with pytest.raises(ValueError,match='ONNX root specialization rank conflicts with annotation'):
            module.import_onnx(row.path,inputs=feeds,input_shapes={'X':[6]})
        path=save_case(mlp_case(),tmp_path/'external',external=True)
        with pytest.raises(ValueError,match='ONNX external tensors require explicit external_root'):
            module.import_onnx(path,inputs=feeds)

@pytest.mark.parametrize('bad',[True,'1',float('inf'),1.e100])
def test_roots_never_silently_coerce_bool_string_nonfinite_or_overflow(onnx_api,monkeypatch,tmp_path,bad):
    module,_,_=onnx_api;case=mlp_case();path=save_case(case,tmp_path/'model')
    values={k:v.tolist() for k,v in case.feeds.items()};values['X'][0][0]=bad
    diagnostic=('(ONNX root X .*finite FP32 numbers|workload graph requires finite JSON numbers)'
                if isinstance(bad,float) and not np.isfinite(bad) else 'ONNX root X .*finite FP32 numbers')
    with no_backend(monkeypatch),pytest.raises(ValueError,match=diagnostic):
        module.import_onnx(path,inputs=values)

def test_external_location_is_model_relative_inside_larger_explicit_allowed_root(onnx_api,monkeypatch,tmp_path):
    module,w,_=onnx_api;case=mlp_case()
    allowed=tmp_path/'allowed';path=save_case(case,allowed/'nested/model',external=True)
    values={k:v.tolist() for k,v in case.feeds.items()}
    # Same named file at the allowed root deliberately has different bytes.
    # An implementation resolving location against external_root reads this
    # decoy and produces an independently detectable wrong intermediate.
    decoy=allowed/'weights/data.bin';decoy.parent.mkdir()
    raw=(path.parent/'weights/data.bin').read_bytes()
    decoy.write_bytes(raw[:16]+np.zeros((4,3),dtype='<f4').tobytes()+raw[-3:])
    expected=evaluate_model(onnx.load(str(path)),case.feeds)['stages']
    with no_backend(monkeypatch):
        narrow=module.import_onnx(path,inputs=values,external_root=path.parent)
        wide=module.import_onnx(path,inputs=values,external_root=allowed)
        assert narrow.to_dict()==wide.to_dict()
        assert compare_original_values(w.execute_reference(wide),module.describe_onnx(wide),expected)
        denied=allowed/'unrelated';denied.mkdir()
        with pytest.raises(ValueError,match='ONNX external tensor escaped allowed root or is not a file'):
            module.import_onnx(path,inputs=values,external_root=denied)

def test_broken_file_has_parser_error_and_missing_path_has_source_error(onnx_api,monkeypatch,tmp_path):
    module,_,_=onnx_api;bad=tmp_path/'broken.onnx';bad.write_bytes(b'not a ModelProto')
    with no_backend(monkeypatch):
        with pytest.raises(ValueError,match='ONNX ModelProto parse failed:'):
            module.import_onnx(bad,inputs={})
        with pytest.raises(ValueError,match='ONNX source model unavailable:'):
            module.import_onnx(tmp_path/'absent.onnx',inputs={})

def test_runtime_int64_shapes_and_data_are_not_silently_cast(onnx_api,monkeypatch,tmp_path):
    module,_,_=onnx_api
    runtime=single_case('Reshape',{'X':np.arange(6,dtype=np.float32).reshape(2,3),
        'shape':np.array([3,2],np.int64)},[3,2])
    path=save_case(runtime,tmp_path/'dynamic')
    integer=single_case('Identity',{'X':np.array([1,2],np.int64)},[2])
    # Output annotation must also have the integer type for a legal ONNX control.
    integer.model.graph.output[0].type.tensor_type.elem_type=onnx.TensorProto.INT64
    ipath=save_case(integer,tmp_path/'integer')
    with no_backend(monkeypatch):
        with pytest.raises(ValueError,match='ONNX (runtime roots require FP32|Reshape shape must be a literal)'):
            module.import_onnx(path,inputs={k:v.tolist() for k,v in runtime.feeds.items()})
        with pytest.raises(ValueError,match='ONNX (runtime roots require FP32|int64 values are only compile-time Reshape shapes)'):
            module.import_onnx(ipath,inputs={k:v.tolist() for k,v in integer.feeds.items()})

def test_dead_dynamic_node_is_preserved_in_graph_and_mapping(onnx_api,monkeypatch,tmp_path):
    module,w,_=onnx_api;case=mlp_case()
    case.model.graph.node.append(onnx.helper.make_node('Relu',['X'],['unused'],name=''))
    path=save_case(case,tmp_path/'dead')
    with no_backend(monkeypatch):
        spec=module.import_onnx(path,inputs={k:v.tolist() for k,v in case.feeds.items()})
        description=module.describe_onnx(spec);graph=w.workload_from_spec(spec)
        node=description['nodes'][-1]
        assert node['outputs']==['unused'] and not node['constant_folded']
        assert node['graph_operators'] and set(node['graph_operators'])<={op.name for op in graph.operators}
