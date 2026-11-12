"""Three-way independent checks of owned and official frozen ModelProto files."""
import json,hashlib
from pathlib import Path
import numpy as np
import onnx
import pytest
from onnx_import_cases import feed_values
from onnx_import_expectations import mlp,attention
from onnx_numpy_interpreter import evaluate_model
from test_onnx_independent_expectations import reference_values,close
from onnx_import_cases import mlp_case,attention_case

ROOT=Path(__file__).resolve().parents[1]/'fixtures/onnx_import'

@pytest.mark.parametrize('key',['mlp','attention','mlp_external'])
def test_independent_interpreter_matches_separate_closed_form_and_OFFICIAL_reference(key):
    top=json.loads((ROOT/'INPUTS.json').read_text());row=top['cases'][key]
    path=ROOT/row['model'];inputs=json.loads((ROOT/row['manifest']).read_text())
    case=attention_case() if key=='attention' else mlp_case()
    form=(attention if key=='attention' else mlp)(case.roots)
    actual=evaluate_model(onnx.load(str(path)),feed_values(inputs))['stages']
    official=reference_values(onnx.load(str(path)),feed_values(inputs))
    for name in form:close(actual[name],form[name]);close(official[name],actual[name])

@pytest.mark.parametrize('key',['mlp','attention'])
def test_SIM51_frozen_ModelProto_inputs_have_independent_every_node_expectations(key):
    root=ROOT/'sim51';data=json.loads((root/'INPUTS.json').read_text());row=data['cases'][key]
    for name,record in data['files'].items():
        raw=(root/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==record['sha256']
    path=root/row['model'];onnx.checker.check_model(str(path),full_check=True)
    model=onnx.load(str(path));feeds={name:np.asarray(value,np.float32) for name,value in row['inputs'].items()}
    actual=evaluate_model(model,feeds);official=reference_values(model,feeds)
    assert set(actual['stages'])==set(official)
    for name in official:close(official[name],actual['stages'][name])
    assert set(actual['outputs'])=={v.name for v in model.graph.output}
    assert all(np.isfinite(v).all() for v in actual['outputs'].values())
