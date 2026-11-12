"""Consumers of frozen SIM43 binding API; execute on the unique candidate.

Collection remains possible on the baseline. Missing production API is an
execution failure, never a skip/stub/fallback. No native execution is used.
"""
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from adapter_boundary_cases import (
    adapter_api, roofline_scene, events_scene, mixed_group_scene, native_scene,
)
from compilation_expectations import reference_matmul
from test_compilation_execution_contract import independently_sweep_storage
from test_public_compilation_contract import prohibit_execution


REJECT = (ValueError, TypeError, NotImplementedError)


def rebind(document):
    result = deepcopy(document)
    body = {k:v for k,v in result.items() if k != 'binding'}
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('ascii')
    result['binding'] = {'sha256':hashlib.sha256(encoded).hexdigest()}
    return result


@pytest.fixture(scope='module')
def extension(tmp_path_factory):
    # Test-generated copy, never modify a checkout or shared installation when
    # checking actual source-file mutation. The same fixture also has a real
    # distribution entry point for later outside-repository wheel acceptance.
    source = Path(__file__).resolve().parents[1]/'fixtures/adapter_boundary_extension/src'
    copied = tmp_path_factory.mktemp('val54-unknown-extension')/'src'
    shutil.copytree(source, copied)
    sys.path.insert(0, str(copied))
    try:
        from arch_explorer.workload.api import task_adapters
        task_adapters()  # ordinary installed discovery before direct fixture use
        module = importlib.import_module('val54_adapter_extension')
        module.register()
        yield module
    finally:
        sys.path.remove(str(copied))


def external_scene(implementation='val54.sm'):
    from arch_explorer.components.api import ImplementationBinding
    adf, graph, spec, mapping = events_scene()
    binding = adf.elements[0].implementation
    adf.elements[0].implementation = ImplementationBinding(implementation, '1', binding.params)
    return adf, graph, spec, mapping


@pytest.mark.parametrize('scene,adapter', [(events_scene, 'events'), (roofline_scene, 'roofline')])
def test_component_declared_unique_adapter_and_explicit_binding_match(
        adapter_api, monkeypatch, tmp_path, scene, adapter):
    w, s = adapter_api
    assert 'target' not in inspect.signature(w.compile_workload).parameters
    adf, _, spec, mapping = scene()
    assert set(mapping.params) == {'compute', 'memory', 'tiles'}
    path = mapping.params['compute']
    with prohibit_execution(monkeypatch, s):
        discovered = w.task_adapters()
        assert adapter in {d['id'] for d in discovered}
        auto = w.compile_workload(adf, spec, mapping)
        explicit = w.compile_workload(adf, spec, mapping,
                                      bindings={path:w.AdapterBinding(adapter, '1', {})})
        assert explicit == auto == w.validate_compilation(auto)
        assert auto['schema'] == 'arch-explorer.compilation.v3'
        assert auto['bindings'][path] == {'id':adapter, 'version':'1', 'options':{}}
        request = s.prepare_compiled(auto, output_dir=tmp_path/'prepared')
        assert request['schema'] == 'arch-explorer.compiled-request.v3'
        assert s.assess_request(request)['supported']


@pytest.mark.parametrize('bad', ['unknown', 'version', 'option', 'domain', 'undeclared'])
def test_bad_binding_is_rejected_before_output_or_execution(adapter_api, monkeypatch, tmp_path, bad):
    w, s = adapter_api
    adf, _, spec, mapping = events_scene()
    identifier, version, options = 'events', '1', {}
    if bad == 'unknown':
        identifier = 'val54.unknown.not.registered'
    elif bad == 'version':
        version = '999'
    elif bad == 'option':
        options = {'threads_per_block':32}
    elif bad == 'domain':
        identifier, options = 'accelsim', {'threads_per_block':32}
    else:
        identifier = 'roofline'  # same domain but not declared for sm.events
    out = tmp_path/'must-not-exist'
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(REJECT):
            c = w.compile_workload(adf, spec, mapping,
                                   bindings={'board/sm':w.AdapterBinding(identifier, version, options)})
            s.prepare_compiled(c, output_dir=out)
    assert not out.exists()


@pytest.mark.parametrize('operation', ['compile', 'validate', 'prepare'])
def test_custom_registry_rejection_stays_before_output_on_new_binding_path(
        adapter_api, monkeypatch, tmp_path, operation):
    from arch_explorer.adf import COMPONENTS
    w, s = adapter_api
    adf, _, spec, mapping = events_scene()
    c = w.compile_workload(adf, spec, mapping, registry=COMPONENTS)
    private = deepcopy(COMPONENTS)
    out = tmp_path/'private-registry'
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(NotImplementedError, match='(?i)registry'):
            if operation == 'compile':
                w.compile_workload(adf, spec, mapping, registry=private)
            elif operation == 'validate':
                w.validate_compilation(c, registry=private)
            else:
                s.prepare_compiled(c, output_dir=out, registry=private)
    assert not out.exists()


def test_unknown_external_adapter_uses_existing_provider_and_exact_semantics(
        adapter_api, extension, monkeypatch, tmp_path):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import execute_reference
    w, s = adapter_api
    adf, graph, spec, mapping = external_scene()
    with prohibit_execution(monkeypatch, s):
        c = w.compile_workload(adf, spec, mapping)
        assert c['bindings']['board/sm']['id'] == extension.FIRST.id
        assert c['lowered']['provider'] == 'events'
        assert c['lowered']['workload'] == WorkloadSpec('graph', c['program']).to_dict()
        request = s.prepare_compiled(c, output_dir=tmp_path/'run', options={'trace':True})
    # This is numerical execution of the actual public mapped graph, not a
    # private regenerated graph or comparison of FLOPs/bytes only.
    values = graph.meta['inputs']
    expected = reference_matmul(values['A'][0], values['B'][0])
    numerical = execute_reference(WorkloadSpec('graph', c['program']))
    assert numerical['outputs']['C'].tolist() == [expected]
    result = s.evaluate_request(request)
    assert result['schema'] == 'arch-explorer.compiled-result.v2'
    assert s.evaluate_request(result) == result


@pytest.mark.parametrize('implementation', ['val54.ambiguous', 'val54.absent'])
def test_ambiguity_or_missing_component_adapter_never_uses_first_registration(
        adapter_api, extension, monkeypatch, implementation):
    w, s = adapter_api
    adf, _, spec, mapping = external_scene(implementation)
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(REJECT):
            w.compile_workload(adf, spec, mapping)
        if implementation == 'val54.ambiguous':
            a = w.compile_workload(adf, spec, mapping,
                bindings={'board/sm':w.AdapterBinding(extension.FIRST.id, '1')})
            b = w.compile_workload(adf, spec, mapping,
                bindings={'board/sm':w.AdapterBinding(extension.SECOND.id, '1')})
            assert a['program'] == b['program']
            assert a['binding'] != b['binding']


def test_two_adapter_groups_run_in_one_graph_with_real_last_reader_storage(
        adapter_api, tmp_path):
    from arch_explorer.adf import WorkloadSpec
    w, s = adapter_api
    adf, graph, spec, mapping = mixed_group_scene()
    c = w.compile_workload(adf, spec, mapping)
    groups = c['groups']
    assert len(groups) == 2
    assert {g['compute'] for g in groups} == {'board/lanes', 'board/sm'}
    assert {g['adapter']['id'] for g in groups} == {'events', 'roofline'}
    covered = [name for g in groups for name in g['tasks']]
    assert len(covered) == len(set(covered))
    assert set(covered) == {op['name'] for op in c['program']['operators']}
    assert c['lowered']['provider'] == 'events'
    assert c['lowered']['mapping']['template'] == 'adapted_events'
    assert c['lowered']['workload'] == WorkloadSpec('graph', c['program']).to_dict()
    task_group = {name:g for g in groups for name in g['tasks']}
    producers = {tensor:op['name'] for op in c['program']['operators'] for tensor in op['outputs']}
    inputs = {op['name']:op['inputs'] for op in c['program']['operators']}
    cross_group_edges = []
    for step in c['plan']['steps']:
        assert task_group[step['name']]['compute'] == mapping.params['compute'][step['origin']]
        assert set(step['dependencies']) == {producers[t] for t in inputs[step['name']] if t in producers}
        cross_group_edges.extend((parent, step['name']) for parent in step['dependencies']
                                 if task_group[parent]['id'] != task_group[step['name']]['id'])
    assert cross_group_edges
    values = graph.meta['inputs']
    intermediate = reference_matmul(values['A'][0], values['B'][0])
    expected = {out:[reference_matmul(intermediate, values[weight][0])]
                for out,weight in [('Y','D'), ('Z','E')]}
    numerical = w.execute_reference(WorkloadSpec('graph', c['program']))
    assert {key:numerical['outputs'][key].tolist() for key in expected} == expected
    result = s.evaluate_request(s.prepare_compiled(c, output_dir=tmp_path/'grouped',
                                                  options={'trace':True}))
    peak, intervals = independently_sweep_storage(c, result)
    assert peak == result['evaluation']['details']['stats']['memory_peak']['board/dram']
    shared = next(t for t in intervals if t['tensor'] == 'T')
    assert len(shared['readers']) >= 2
    assert shared['last_reader'] > shared['allocation']
    assert s.evaluate_request(result) == result


def test_no_bridge_rejects_cross_domain_groups_before_composing(
        adapter_api, extension, monkeypatch, tmp_path):
    from arch_explorer.components.api import ImplementationBinding
    w, s = adapter_api
    adf, _, spec, mapping = mixed_group_scene()
    current = adf.elements[0].implementation
    adf.elements[0].implementation = ImplementationBinding('val54.foreign', '1', current.params)
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(REJECT):
            c = w.compile_workload(adf, spec, mapping)
            s.prepare_compiled(c, output_dir=tmp_path/'cross-domain')
    assert not (tmp_path/'cross-domain').exists()


@pytest.mark.parametrize('options', [None, {}, {'threads_per_block':33},
                                   {'threads_per_block':True},
                                   {'threads_per_block':32, 'invented':1}])
def test_native_requires_its_own_explicit_valid_thread_options(
        adapter_api, monkeypatch, options):
    w, s = adapter_api
    adf, _, spec, mapping = native_scene()
    bindings = None if options is None else {
        mapping.params['compute']:w.AdapterBinding('accelsim', '1', options)}
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(REJECT):
            w.compile_workload(adf, spec, mapping, bindings=bindings)


def test_native_valid_option_edit_cannot_reuse_old_converted_product(adapter_api, monkeypatch):
    w, s = adapter_api
    adf, _, spec, mapping = native_scene()
    path = mapping.params['compute']
    with prohibit_execution(monkeypatch, s):
        c = w.compile_workload(adf, spec, mapping,
                              bindings={path:w.AdapterBinding('accelsim', '1', {'threads_per_block':32})})
        assert c['bindings'][path]['options'] == {'threads_per_block':32}
        changed = deepcopy(c)
        changed['bindings'][path]['options']['threads_per_block'] = 64
        with pytest.raises(REJECT):
            w.validate_compilation(rebind(changed))


def test_deprecated_target_registration_rejects_without_calling_a_second_pipeline(
        adapter_api, monkeypatch):
    w, s = adapter_api
    def unused(*args):
        raise AssertionError('deprecated compiler callback must not execute')
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(REJECT, match='(?i)adapter|deprecated|legacy|removed'):
            w.register_compiler_target(w.CompilerTarget('val54.old-compiler', '1', unused))


def test_real_v1_document_needs_explicit_migration_to_new_bound_compilation(
        adapter_api, monkeypatch, tmp_path):
    w, s = adapter_api
    path = Path(__file__).resolve().parents[1]/'fixtures/adapter_boundary_extension/V1-COMPILATION.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '5f73c606b0cbda7342771b78a4dcb89aef96e48f9606f72b4ef6b29bc785b666'
    old = json.loads(path.read_text())
    assert old['schema'] == 'arch-explorer.compilation.v1'
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(REJECT):
            w.validate_compilation(old)
        with pytest.raises(REJECT):
            s.prepare_compiled(old, output_dir=tmp_path/'implicit-upgrade')
        new = w.migrate_compilation_v1(deepcopy(old))
        assert new['schema'] == 'arch-explorer.compilation.v3'
        assert new['source']['adf'] == old['source']['adf']
        assert new['source']['workload'] == old['source']['workload']
        legacy = old['source']['mapping']['params']
        assert new['source']['mapping']['template']=='compile_graph_v2'
        assert new['source']['mapping']['params'] == {
            'compute':legacy['compute'],'memory':legacy['memory'],
            'rules':{'matmul':{'id':'matmul.tiled','version':'1','options':{'tiles':legacy['tiles']}}}}
        assert new['binding'] != old['binding']
        assert w.validate_compilation(new) == new
    assert not (tmp_path/'implicit-upgrade').exists()
    # Frozen old bytes retain their actual producer and never become a v3 result.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '5f73c606b0cbda7342771b78a4dcb89aef96e48f9606f72b4ef6b29bc785b666'


@pytest.mark.parametrize('field', ['options', 'adapter-version', 'group-product'])
def test_rehashed_adapter_or_product_edits_cannot_prepare_old_compilation(
        adapter_api, extension, monkeypatch, tmp_path, field):
    w, s = adapter_api
    adf, _, spec, mapping = external_scene()
    c = w.compile_workload(adf, spec, mapping)
    changed = deepcopy(c)
    if field == 'options':
        changed['bindings']['board/sm']['options']['invented'] = 1
    elif field == 'adapter-version':
        changed['bindings']['board/sm']['version'] = '2'
    else:
        tasks = changed['groups'][0]['payload']['tasks']
        tasks[next(iter(tasks))]['work']['shape'][1] += 1
    changed = rebind(changed)
    with prohibit_execution(monkeypatch, s):
        with pytest.raises(REJECT):
            w.validate_compilation(changed)
        with pytest.raises(REJECT):
            s.prepare_compiled(changed, output_dir=tmp_path/'edited')
    assert not (tmp_path/'edited').exists()


def test_actual_external_source_byte_edit_invalidates_bound_compilation(
        adapter_api, extension, monkeypatch):
    w, s = adapter_api
    adf, _, spec, mapping = external_scene()
    c = w.compile_workload(adf, spec, mapping)
    source = Path(extension.__file__)
    before, stat = source.read_bytes(), source.stat()
    try:
        source.write_bytes(before+b'\n# VAL54 source mutation sensitivity probe\n')
        with prohibit_execution(monkeypatch, s):
            with pytest.raises(REJECT):
                w.validate_compilation(c)
    finally:
        source.write_bytes(before)
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert source.read_bytes() == before
    assert w.validate_compilation(c) == c


def test_failed_real_registrar_discovery_rolls_back_component_and_adapter(
        adapter_api, extension):
    # A cold process exercises actual entry-point load and registrar transaction.
    # The entry-point object is injected; real installed metadata is tested only
    # after building/installing the fixture with the lead's unique main wheel.
    code = '''
from importlib import metadata
original = metadata.entry_points
failing = metadata.EntryPoint(name='val54-failure', value='val54_adapter_extension:register_then_fail', group='arch_explorer.components')
def entries(*, group=None, **kwargs):
    values = original(group=group, **kwargs)
    return (*values, failing) if group == 'arch_explorer.components' else values
metadata.entry_points = entries
from arch_explorer.components.api import discover_components, component_implementations
from arch_explorer.workload.api import task_adapters
state = discover_components()
assert any('VAL54 intentional registrar failure' in text for text in state['failures'].values()), state
assert 'val54.orphan' not in {v['id'] for v in task_adapters()}
assert 'val54.orphan' not in {v['id'] for v in component_implementations()}
print('component-and-adapter-rollback-verified')
'''
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(Path(extension.__file__).parents[1]),
                                      env.get('PYTHONPATH', '')])
    run = subprocess.run([sys.executable, '-c', code], env=env,
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == 'component-and-adapter-rollback-verified'
