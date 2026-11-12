"""Source-only fixture checks; no claim about the new composition API."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from test_operator_mapping_attention_contract import no_backend


@pytest.fixture
def builder(monkeypatch):
    import numpy
    source=Path(__file__).resolve().parents[1]/'fixtures/composition_builder_extension/src'
    monkeypatch.syspath_prepend(str(source))
    import val57_composition_extension as extension
    extension.register()
    return extension


def test_fixture_has_real_registered_builder_source_and_closed_form_reference(builder,monkeypatch):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload import api
    from arch_explorer.workload.extensions import callable_identity
    source=Path(__file__).resolve().parents[1]/'fixtures/composition_builder_extension/src/val57_composition_extension/builders.py'
    inputs={'x':[[1.,2.,-1.,.5]]};before=deepcopy(inputs)
    spec=WorkloadSpec(builder.KIND,dict(shape=[1,4],factor=2.,inputs=inputs))
    with no_backend(monkeypatch),api.source_trace() as used:
        graph=api.workload_from_spec(spec)
        reference=api.execute_reference(spec)
    assert builder.build_scale in used
    assert reference['outputs']['y'].tolist()==[[2.,4.,-2.,1.]]
    assert inputs==before
    inputs['x'][0][0]=99.
    assert graph.meta['inputs']==before
    identity=callable_identity(builder.build_scale)
    assert identity['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert identity['module']=='val57_composition_extension.builders'


def test_fixture_supports_structural_unbound_input_without_fictitious_answer(builder,monkeypatch):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload import api
    spec=WorkloadSpec(builder.KIND,dict(shape=[1,4],factor=3.))
    with no_backend(monkeypatch):
        graph=api.workload_from_spec(spec)
    assert graph.meta['inputs']=={}
    assert [(op.inputs,op.outputs) for op in graph.operators]==[(('x',),('y',))]
