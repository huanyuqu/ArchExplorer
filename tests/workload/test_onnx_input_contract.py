"""Frozen file provenance and valid-ONNX negative controls; not product refusals."""
import hashlib,json
from pathlib import Path
import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator
from onnx_import_cases import mlp_case,attention_case,feed_values
from onnx_import_expectations import mlp,attention
from test_onnx_independent_expectations import close,reference_values
from onnx_import_negative_cases import model_mutation

ROOT=Path(__file__).resolve().parents[1]/'fixtures/onnx_import'

@pytest.mark.parametrize('key',['mlp','attention','mlp_external'])
def test_frozen_real_model_and_external_bytes_have_complete_independent_inputs(key):
    top=json.loads((ROOT/'INPUTS.json').read_text());row=top['cases'][key]
    p=ROOT/row['model'];manifest=json.loads((ROOT/row['manifest']).read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['model_sha256']==manifest['model']['sha256']
    for name,record in manifest['external_data'].items():
        raw=(p.parent/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==record['sha256']
        assert record['offset']+record['length']<=len(raw)
    onnx.checker.check_model(str(p),full_check=True)
    case=attention_case() if key=='attention' else mlp_case()
    expected=(attention if key=='attention' else mlp)(case.roots)
    actual=reference_values(onnx.load(str(p)),feed_values(manifest))
    assert set(actual)==set(expected)
    for tensor in expected:close(actual[tensor],expected[tensor])
    assert not manifest['contains_expected_answers'] and not top['product_importer_used']

@pytest.mark.parametrize('condition',[
    'opset18','IR9','Sin','symbolic_batch','fp16','missing_root','nonfinite_root','overridable_initializer'])
def test_unsupported_or_policy_inputs_are_standard_models_not_malformed_protobuf(tmp_path,condition):
    item=model_mutation(tmp_path/condition,condition)
    assert item.standard_model_legal
    onnx.checker.check_model(str(item.path),full_check=True)
    # Acceptance/refusal by ArchExplorer is deliberately deferred to its API.

@pytest.mark.parametrize('condition',['unknown_attribute'])
def test_structurally_invalid_controls_have_specific_OFFICIAL_checker_errors(tmp_path,condition):
    item=model_mutation(tmp_path/condition,condition)
    with pytest.raises((onnx.checker.ValidationError,onnx.shape_inference.InferenceError)):
        onnx.checker.check_model(str(item.path),full_check=True)


def test_checker_alone_does_not_establish_root_or_contraction_consistency(tmp_path):
    item=model_mutation(tmp_path/'conflict','conflicting_shape')
    onnx.checker.check_model(str(item.path),full_check=True)
    model=onnx.load(str(item.path))
    declared=tuple(d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim)
    assert declared==(2,5) and item.feeds['X'].shape==(2,3)
    with pytest.raises(ValueError,match='shapes|matmul|mismatch'):
        ReferenceEvaluator(model).run(None,{'X':np.ones(declared,np.float32)})
