"""Installed unknown nested builder and real source-byte invalidation."""
from copy import deepcopy
import importlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from multilayer_contract_cases import composition_api,fork_spec,basic_mapping,scene
from multilayer_contract_cases import REJECT
from test_operator_attention_search_contract import command
from test_operator_mapping_attention_contract import no_backend,files
import pytest


def test_installed_registered_non_model_builder_reuses_existing_components(
        composition_api,monkeypatch,tmp_path):
    w,s=composition_api
    extension=importlib.import_module('val57_composition_extension')
    assert 'site-packages' in extension.__file__
    adf,_,_=scene('generic-fork');spec=fork_spec(w,registered=True)
    with no_backend(monkeypatch):
        raw=w.execute_reference(spec)
        compilation=w.compile_workload(adf,spec,basic_mapping())
        request=s.prepare_compiled(compilation,output_dir=tmp_path/'unknown',options={'trace':True})
    assert raw['outputs']['Y'].tolist()==[[4.,8.,-4.,2.]]
    for row in raw['graph'].meta['composition']['instances']:
        if row['id'] in ('make','left','right'):
            assert row['workload']['kind']==extension.KIND
            assert row['sources']
    result=s.evaluate_request(request)
    before=files(tmp_path/'unknown')
    with no_backend(monkeypatch):assert s.evaluate_request(result)==result
    assert files(tmp_path/'unknown')==before


def test_mutating_nested_builder_source_rejects_old_compilation_and_request(
        composition_api,tmp_path):
    module=importlib.import_module('val57_composition_extension')
    assert 'site-packages' in module.__file__
    copied=tmp_path/'copy/val57_composition_extension'
    shutil.copytree(Path(module.__file__).parent,copied)
    adf,_,_=scene('generic-fork')
    data=tmp_path/'ADF.json';data.write_text(json.dumps(adf.to_dict()))
    tests=Path(__file__).resolve().parent
    code=r'''
import importlib,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import numpy,pytest
from arch_explorer.adf import from_dict
from arch_explorer.workload import api as w
from arch_explorer.simulator import api as s
from multilayer_contract_cases import fork_spec,basic_mapping,REJECT
from test_operator_mapping_attention_contract import no_backend
extension=importlib.import_module('val57_composition_extension')
assert str(Path(extension.__file__).resolve()).startswith(str(Path(sys.argv[2]).resolve()))
adf=from_dict(json.loads(Path(sys.argv[3]).read_text()))
with pytest.MonkeyPatch.context() as patch,no_backend(patch):
    compilation=w.compile_workload(adf,fork_spec(w,registered=True),basic_mapping())
    request=s.prepare_compiled(compilation,output_dir=Path(sys.argv[4])/'prepared')
    assert w.validate_compilation(compilation)==compilation
    assert s.assess_request(request)['supported']
    target=Path(extension.__file__).with_name('builders.py');original=target.read_bytes()
    Path(sys.argv[4],'BUILDER-before.bin').write_bytes(original)
    try:
        target.write_bytes(original+b'\n# deliberate QA mutation of actual nested builder source\n')
        Path(sys.argv[4],'BUILDER-mutated.bin').write_bytes(target.read_bytes())
        with pytest.raises(REJECT):w.validate_compilation(compilation)
        with pytest.raises(REJECT):s.evaluate_request(request)
    finally:target.write_bytes(original)
    assert w.validate_compilation(compilation)==compilation
print('nested-builder-source-invalidation-without-execution')
'''
    result=command([sys.executable,'-c',code,str(tests),str(copied.parent),str(data),str(tmp_path)],tmp_path)
    assert result.stdout.strip()=='nested-builder-source-invalidation-without-execution'


def test_original_single_block_uses_the_same_authoritative_structure(composition_api,monkeypatch):
    from arch_explorer.adf import WorkloadSpec
    from test_precision_transformer_contract import scene as single_scene
    w,_=composition_api;_,spec,_=single_scene('bf16')
    before=deepcopy(spec.params)
    kwargs={key:value for key,value in spec.params.items() if key!='inputs'}
    with no_backend(monkeypatch):
        structure=w.workload_from_spec(w.transformer_block_structure(**kwargs))
        original=w.workload_from_spec(w.transformer_encoder_block(**spec.params))
    assert structure.to_dict()['operators']==original.to_dict()['operators']
    assert structure.to_dict()['tensors']==original.to_dict()['tensors']
    assert not structure.meta.get('inputs')
    assert original.meta['inputs']==before['inputs'] and spec.params==before


def test_exact_historical_single_block_result_is_rejected_after_builder_source_change(
        composition_api,monkeypatch):
    # The acceptance runner supplies a genuinely captured, sealed VAL56 result;
    # private operational data are not embedded in the public source fixture.
    path=Path(os.environ['VAL57_HISTORICAL_RESULT'])
    expected=os.environ['VAL57_HISTORICAL_RESULT_SHA256']
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
    payload=json.loads(raw)
    assert payload['schema']=='arch-explorer.compiled-result.v3'
    assert payload['request']['compilation']['source']['workload']['kind']=='transformer_block'
    _,s=composition_api
    with no_backend(monkeypatch),pytest.raises(REJECT):s.evaluate_request(payload)
    assert path.read_bytes()==raw
