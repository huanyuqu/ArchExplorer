"""Consumers of frozen CORE28 data-only API; collection is not acceptance.

Run this file only against the lead's installed candidate once it exists.
It deliberately does not guess future composition or selector schemas.
"""
from copy import deepcopy
import json
import math

import pytest

from multilayer_expectations import PARAMETER_NAMES,reference_for
from test_operator_mapping_attention_contract import no_backend


SHAPES={**{key:(8,8) for key in ('Wq','Wk','Wv','Wo')},
        **{key:(8,) for key in ('bq','bk','bv','bo','norm1_gamma','norm1_beta','norm2_gamma','norm2_beta','b2')},
        'W1':(8,16),'b1':(16,),'W2':(16,8)}


def exact_shape(value,shape):
    if not shape:
        assert type(value) in (float,int) and math.isfinite(value)
        return
    assert type(value) is list and len(value)==shape[0]
    for child in value:exact_shape(child,shape[1:])


@pytest.fixture
def encoder_data():
    import numpy
    from arch_explorer.archs.multilayer import encoder_inputs
    return encoder_inputs


@pytest.mark.parametrize('layers',[2,4])
@pytest.mark.parametrize('case',['regular','boundary','constant'])
def test_installed_encoder_inputs_have_one_X_distinct_parameters_and_no_intermediate_answers(
        encoder_data,monkeypatch,layers,case):
    with no_backend(monkeypatch):data=encoder_data(layers=layers,input_case=case)
    assert set(data)=={'X','layers'}
    exact_shape(data['X'],(2,3,8));assert len(data['layers'])==layers
    for parameters in data['layers']:
        assert set(parameters)==PARAMETER_NAMES
        for key,value in parameters.items():exact_shape(value,SHAPES[key])
    encoded=[json.dumps(parameters,sort_keys=True,allow_nan=False) for parameters in data['layers']]
    assert len(set(encoded))==layers
    before=deepcopy(data)
    # Independent expected chain only. This does not validate new production
    # composition/reference execution, which requires the SIM46 public API.
    for dtype in ('fp32','fp16','bf16'):
        expected=reference_for(dict(config=dict(batch=2,sequence=3,hidden=8,heads=2,mlp_hidden=16,epsilon=1e-5),
            dtype=dtype,X=data['X'],instances=[dict(id='data-'+str(i),parameters=p)
                                             for i,p in enumerate(data['layers'])]))
        exact_shape(expected['output'],(2,3,8))
    assert data==before
    with no_backend(monkeypatch):assert encoder_data(layers=layers,input_case=case)==before
    data['layers'][0]['b2'][0]+=1
    assert data['layers'][1]==before['layers'][1]
    with no_backend(monkeypatch):assert encoder_data(layers=layers,input_case=case)==before


def test_installed_data_cases_change_only_X(encoder_data,monkeypatch):
    with no_backend(monkeypatch):
        regular=encoder_data(layers=4,input_case='regular')
        boundary=encoder_data(layers=4,input_case='boundary')
        constant=encoder_data(layers=4,input_case='constant')
    assert regular['layers']==boundary['layers']==constant['layers']
    assert regular['X']!=boundary['X'] and regular['X']!=constant['X']


@pytest.mark.parametrize('changes',[{'layers':1},{'layers':5},{'layers':True},
    {'layers':2.5},{'input_case':'unknown'}])
def test_installed_data_refusals_follow_normal_control(encoder_data,monkeypatch,changes):
    with no_backend(monkeypatch):
        assert len(encoder_data(layers=2)['layers'])==2
        with pytest.raises((ValueError,TypeError)):
            encoder_data(**changes)
