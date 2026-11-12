"""Validate positive edge ModelProto inputs before any product consumption."""
import onnx
import pytest
from onnx_numpy_interpreter import evaluate_model
from test_onnx_import_edges import edge_cases
from test_onnx_independent_expectations import reference_values,close

@pytest.mark.parametrize('index',range(14))
def test_owned_edge_files_check_and_independent_values_match_official(index):
    case=edge_cases()[index];onnx.checker.check_model(case.model,full_check=True)
    independent=evaluate_model(case.model,case.feeds)['stages'];official=reference_values(case.model,case.feeds)
    assert independent.keys()==official.keys()
    for name in independent:close(official[name],independent[name])
