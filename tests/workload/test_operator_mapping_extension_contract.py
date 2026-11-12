"""Real installed unknown mapping, source binding and transaction contracts."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import pytest

from test_operator_mapping_attention_contract import (
    api,check_values,no_backend,node_spec,primitive_scene,REJECT)


def unknown_scene(w,mode='events',kind='val55.square',rule='val55.square.materialized'):
    from arch_explorer.adf import MappingSpec
    spec = node_spec(kind,[(2,3)],(2,3),{},[[[1,-2,.5],[0,.25,-.5]]])
    adf,spec,mapping = primitive_scene(spec,mode)
    return adf,spec,MappingSpec(mapping.template,dict(mapping.params,
        rules={kind:w.MappingRuleBinding(rule,'1').to_dict()}))


def reseal(document):
    result = deepcopy(document)
    result.pop('binding',None)
    encoded = json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    result['binding'] = dict(sha256=hashlib.sha256(encoded).hexdigest())
    return result


@pytest.mark.parametrize('mode',['events','roofline'])
def test_installed_unknown_semantic_rule_reuses_existing_adapter_and_backend(api,monkeypatch,tmp_path,mode):
    from arch_explorer.adf import WorkloadSpec
    import val55_mapping_extension as extension
    w,s = api
    catalog = w.operator_mappings(extension.KIND)
    assert any(row['id']==extension.RULE_ID for row in catalog)
    adf,spec,mapping = unknown_scene(w,mode)
    before = deepcopy(spec.to_dict())
    with no_backend(monkeypatch):
        raw = w.execute_reference(spec)
        compilation = w.compile_workload(adf,spec,mapping)
        mapped = w.execute_reference(WorkloadSpec('graph',compilation['program']),inputs=spec.params['meta']['inputs'])
        request = s.prepare_compiled(compilation,output_dir=tmp_path/mode,options={'trace':True})
    assert spec.to_dict()==before
    expected = [[1,4,.25],[0,.0625,.25]]
    check_values(raw['outputs']['y'],expected)
    check_values(mapped['outputs']['y'],expected)
    assert {op['op'] for op in compilation['program']['operators']} == {'mul'}
    result = s.evaluate_request(request)
    assert result['evaluation']['latency_ns']>0
    with no_backend(monkeypatch):
        assert s.evaluate_request(result)==result


def test_two_mapping_rules_are_ambiguous_until_explicit_binding(api,monkeypatch):
    from arch_explorer.adf import MappingSpec
    import val55_mapping_extension as extension
    w,_ = api
    kind = 'val55.ambiguity.'+uuid4().hex
    w.register_operator(w.OperatorDefinition(kind,extension.validate,version='1',reference=extension.reference))
    first = w.OperatorMapping(kind+'.a','1',kind,extension.expand,extension.access)
    second = w.OperatorMapping(kind+'.b','1',kind,extension.expand,extension.access)
    w.register_operator_mapping(first)
    w.register_operator_mapping(second)
    adf,spec,explicit = unknown_scene(w,kind=kind,rule=first.id)
    automatic = MappingSpec(explicit.template,dict(explicit.params,rules={}))
    with no_backend(monkeypatch):
        with pytest.raises(REJECT,match='(?i)ambig|explicit|multiple'):
            w.compile_workload(adf,spec,automatic)
        assert w.compile_workload(adf,spec,explicit)['schema']=='arch-explorer.compilation.v3'
        bad = MappingSpec(explicit.template,dict(explicit.params,
            rules={kind:dict(id=first.id,version='999',options={})}))
        with pytest.raises(REJECT):
            w.compile_workload(adf,spec,bad)


def test_mapping_definition_mutation_and_duplicate_identity_refused(api,monkeypatch):
    import val55_mapping_extension as extension
    w,_ = api
    kind = 'val55.identity.'+uuid4().hex
    definition = w.OperatorMapping(kind,'1',extension.KIND,extension.expand,extension.access)
    w.register_operator_mapping(definition)
    with no_backend(monkeypatch):
        with pytest.raises(ValueError):
            w.register_operator_mapping(w.OperatorMapping(kind,'1',extension.KIND,extension.expand,extension.access))
        try:
            object.__setattr__(definition,'version','999')
            with pytest.raises(ValueError):
                w.operator_mappings(extension.KIND)
        finally:
            object.__setattr__(definition,'version','1')
        assert any(row['id']==kind for row in w.operator_mappings(extension.KIND))


def test_unselected_additional_rule_does_not_change_saved_selected_identity(api,monkeypatch,tmp_path):
    import val55_mapping_extension as extension
    w,s = api
    adf,spec,mapping = unknown_scene(w)
    with no_backend(monkeypatch):
        original = w.compile_workload(adf,spec,mapping)
        request = s.prepare_compiled(original,output_dir=tmp_path/'selected')
        extra = w.OperatorMapping('val55.unselected.'+uuid4().hex,'1',extension.KIND,
                                  extension.expand,extension.access)
        w.register_operator_mapping(extra)
        assert w.validate_compilation(original)==original
        assert w.compile_workload(adf,spec,mapping)==original
        assert s.assess_request(request)['supported']


def test_actual_delegated_mapping_source_byte_edit_rejects_rehashed_documents(api,monkeypatch,tmp_path):
    # Import using importlib: the package also publishes an access callback.
    import importlib
    helper = importlib.import_module('val55_mapping_extension.access')
    w,s = api
    adf,spec,mapping = unknown_scene(w)
    with no_backend(monkeypatch):
        compilation = w.compile_workload(adf,spec,mapping)
        assert w.validate_compilation(compilation)==compilation
        request = s.prepare_compiled(compilation,output_dir=tmp_path/'bound')
    path = Path(helper.__file__)
    before,stat = path.read_bytes(),path.stat()
    try:
        path.write_bytes(before+b'\n# VAL55 transitive mapping source mutation\n')
        with no_backend(monkeypatch):
            with pytest.raises(REJECT):
                w.validate_compilation(reseal(compilation))
            assert not s.assess_request(reseal(request))['supported']
            with pytest.raises(REJECT):
                s.evaluate_request(reseal(request))
    finally:
        path.write_bytes(before)
        os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert path.read_bytes()==before and path.stat().st_mtime_ns==stat.st_mtime_ns


def test_failing_installed_registrar_rolls_back_semantics_and_new_mapping_registry(api):
    code = r'''
from importlib import metadata
original = metadata.entry_points
failure = metadata.EntryPoint(name='val55-failing',value='val55_mapping_extension:register_then_fail',group='arch_explorer.components')
def entries(*args,**kwargs):
    if kwargs.get('group')=='arch_explorer.components': return [failure]
    return original(*args,**kwargs)
metadata.entry_points = entries
from arch_explorer.components.extensions import discover_components
from arch_explorer.workload.api import operator_mappings
from arch_explorer.workload.extensions import get_operator
state = discover_components()
assert any('VAL55 intentional' in text for text in state['failures'].values()),state
assert 'val55.orphan.mapping' not in {row['id'] for row in operator_mappings()}
try: get_operator('val55.orphan.semantic')
except ValueError: pass
else: raise AssertionError('semantic registration leaked from failed transaction')
print('val55-semantic-mapping-rollback')
'''
    env = {k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    completed = subprocess.run([sys.executable,'-c',code],cwd=os.getcwd(),env=env,
                               capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    assert completed.stdout.strip()=='val55-semantic-mapping-rollback'
