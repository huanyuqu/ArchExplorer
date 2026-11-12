"""Reject fabricated cross-layer answers at the real public builder boundary."""
from copy import deepcopy

import pytest

from multilayer_contract_cases import REJECT,composition_api,scene
from test_operator_mapping_attention_contract import no_backend


@pytest.mark.parametrize('mutation',['later-X','precomputed-activation','missing-layer-parameter'])
def test_encoder_builder_rejects_non_parameter_layer_inputs(composition_api,monkeypatch,mutation):
    from arch_explorer.adf import WorkloadSpec
    w,_=composition_api;_,valid,_=scene('events-2-fp32-regular')
    with no_backend(monkeypatch):assert w.workload_from_spec(valid).operators
    document=deepcopy(valid.to_dict());values=document['params']['inputs']
    if mutation=='later-X':values['layers'][1]['X']=deepcopy(values['X'])
    elif mutation=='precomputed-activation':values['hidden_answer']=deepcopy(values['X'])
    else:del values['layers'][1]['Wq']
    before=deepcopy(document)
    # Reject while building the public graph, not because an unrelated backend
    # or later compile step happens to fail.
    with no_backend(monkeypatch),pytest.raises(REJECT):
        w.workload_from_spec(WorkloadSpec(**document))
    assert document==before
