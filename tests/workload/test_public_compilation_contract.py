"""Public compilation consumers on SIM44 v3 operator-rule bindings.

Collection does not require the new API. Execution DOES: no stub, fallback,
skip or xfail makes the old baseline look like a working compiler. Historical
Historical documents retain their producer; normal paths require v3 rules.
"""
from contextlib import contextmanager
from copy import deepcopy
import ctypes
import importlib
import json
import math
import os
import re
import subprocess
import sys
from uuid import uuid4

import pytest


REJECT = (ValueError, TypeError, KeyError, NotImplementedError)
SOURCE = 'explicit synthetic VAL53 input assumption; no hardware calibration'


def test_public_compiler_entrypoints_import_in_a_fresh_process():
    # Do not let the component initialization in other fixtures hide a cold
    # public-import cycle. The new API must work in the documented order.
    code = ('from arch_explorer.workload.api import compile_workload, validate_compilation, '
            'TaskAdapter, AdapterBinding, TargetInput, register_task_adapter; '
            'from arch_explorer.simulator.api import prepare_compiled, assess_request, evaluate_request; '
            'print("public-compiler-imports-ready")')
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    completed = subprocess.run([sys.executable, '-c', code], env=env,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == 'public-compiler-imports-ready'


@pytest.fixture
def compiler_api():
    # Ordinary declared numerical dependencies may initialize before the guard;
    # no engine/library or target execution is warmed here.
    import numpy  # noqa: F401
    from arch_explorer.components import Kernel  # noqa: F401
    w = importlib.import_module('arch_explorer.workload.api')
    s = importlib.import_module('arch_explorer.simulator.api')
    for name in ('compile_workload', 'validate_compilation', 'TaskAdapter',
                 'AdapterBinding', 'TargetInput', 'register_task_adapter', 'task_adapters'):
        assert callable(getattr(w, name)), name
    assert callable(s.prepare_compiled)
    return w, s


def graph_input(*, m=3, k=2, n=5, dtype='fp32', iterations=1):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import workload_from_spec
    graph = workload_from_spec(WorkloadSpec('operator', dict(op='matmul', m=m, k=k, n=n,
                                                          dtype=dtype, iterations=iterations)))
    produced = {name for op in graph.operators for name in op.outputs}
    def values(shape, offset):
        if len(shape) == 1:
            return [(offset+i+1)/8 for i in range(shape[0])]
        stride = 1
        for x in shape[1:]:
            stride *= x
        return [values(shape[1:], offset+i*stride) for i in range(shape[0])]
    graph.meta['inputs'] = {name: values(t.shape, index*11)
                            for index, (name, t) in enumerate(graph.tensors.items())
                            if name not in produced}
    return graph, WorkloadSpec('graph', graph.to_dict())


def roofline_input(*, tiles=None, capacity=2**20, **graph_options):
    from arch_explorer.adf import Param, SpaceMatrix, SpacePoint, MappingSpec
    from arch_explorer.components.api import ImplementationBinding
    from arch_explorer.adf import COMPONENTS
    sm = SpacePoint('sm', 'sm', {'precision': Param('fp32', '1', SOURCE)},
                    implementation=ImplementationBinding('sm.roofline', '1', {
                        'peak_ops_per_ns': Param(16, 'op/ns', SOURCE),
                        'bandwidth_bytes_per_ns': Param(8, 'B/ns', SOURCE),
                        'boundary': Param('component_io', '1', SOURCE)}))
    technology = COMPONENTS.get('dram').params['technology'].domain[0]
    memory = SpacePoint('dram', 'dram', {
        'technology': Param(technology, '1', SOURCE),
        'capacity': Param(capacity, 'B', SOURCE),
        'bandwidth': Param(8, 'B/ns', SOURCE), 'latency': Param(2, 'ns', SOURCE)})
    adf = SpaceMatrix('board', [sm, memory])
    graph, spec = graph_input(**graph_options)
    mapping = MappingSpec('compile_graph_v2', {'compute': 'board/sm', 'memory': 'board/dram',
        'rules': {'matmul': {'id':'matmul.tiled','version':'1',
            'options': {'tiles': tiles or {'m':2,'k':2,'n':2}}}}})
    return adf, graph, spec, mapping


def two_reader_graph_input():
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import Workload
    graph = Workload('qa-two-reader-matmul', dtype='fp32')
    for name, shape in [('A', (1, 3, 2)), ('B', (1, 2, 3)), ('T', (1, 3, 3)),
                        ('D', (1, 3, 2)), ('E', (1, 3, 4)),
                        ('Y', (1, 3, 2)), ('Z', (1, 3, 4))]:
        graph.tensor(name, shape)
    graph.add('first', 'matmul', ('A', 'B'), ('T',), batch=1, m=3, k=2, n=3)
    graph.add('reader-one', 'matmul', ('T', 'D'), ('Y',), batch=1, m=3, k=3, n=2)
    graph.add('reader-two', 'matmul', ('T', 'E'), ('Z',), batch=1, m=3, k=3, n=4)
    graph.meta['inputs'] = {name: [[[(1+offset+i*cols+j)/8 for j in range(cols)]
                                   for i in range(rows)]]
                            for offset, (name, rows, cols) in enumerate([
                                ('A', 3, 2), ('B', 2, 3), ('D', 3, 2), ('E', 3, 4)])}
    graph.meta['outputs'] = ['Y', 'Z']
    graph.validate()
    return graph, WorkloadSpec('graph', graph.to_dict())


@contextmanager
def prohibit_execution(monkeypatch, simulator_api):
    def forbidden(*args, **kwargs):
        raise AssertionError('compile/prepare attempted simulation, native load, subprocess or search')
    with monkeypatch.context() as patch:
        for name in ('Popen', 'run', 'check_call', 'check_output'):
            patch.setattr(subprocess, name, forbidden)
        patch.setattr(os, 'system', forbidden)
        if hasattr(os, 'posix_spawn'):
            patch.setattr(os, 'posix_spawn', forbidden)
        patch.setattr(ctypes, 'CDLL', forbidden)
        patch.setattr(ctypes, 'PyDLL', forbidden)
        patch.setattr(simulator_api, 'evaluate_request', forbidden)
        engine = importlib.import_module('arch_explorer.simulator.events.engine')
        patch.setattr(engine, 'simulate', forbidden)
        patch.setattr(engine.ConsistentScheduler, 'run', forbidden)
        backend = importlib.import_module('arch_explorer.simulator.events.backend')
        if hasattr(backend, 'simulate'):
            patch.setattr(backend, 'simulate', forbidden)
        yield


def replace_json_subtree(value, original, replacement):
    """Locate exact saved public input values, without guessing artifact keys."""
    count = 0
    def visit(node):
        nonlocal count
        if node == original:
            count += 1
            return deepcopy(replacement)
        if isinstance(node, dict):
            return {key: visit(item) for key, item in node.items()}
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node
    result = visit(value)
    return result, count


def compiled_roofline(api, **options):
    w, _ = api
    adf, graph, spec, mapping = roofline_input(**options)
    return w.compile_workload(adf, spec, mapping), (adf, graph, spec, mapping)


def test_public_compilation_is_deterministic_JSON_and_validates_from_saved_sources(compiler_api):
    w, _ = compiler_api
    first, (adf, _, spec, mapping) = compiled_roofline(compiler_api)
    assert first['schema'] == 'arch-explorer.compilation.v3'
    restored = json.loads(json.dumps(first, sort_keys=True, allow_nan=False))
    assert restored == first == w.compile_workload(adf, spec, mapping)
    assert isinstance(w.validate_compilation(restored), dict)
    assert restored == first


@pytest.mark.parametrize('tiles', [{'m': 2, 'k': 2, 'n': 2}, {'m': 4, 'k': 2, 'n': 4}])
def test_compile_and_prepare_are_legal_under_execution_guard(compiler_api, monkeypatch, tmp_path, tiles):
    w, s = compiler_api
    with prohibit_execution(monkeypatch, s):
        compiled, _ = compiled_roofline(compiler_api, tiles=tiles)
        assert isinstance(w.validate_compilation(compiled), dict)
        request = s.prepare_compiled(compiled, output_dir=tmp_path/'prepared', options={})
        assert request['schema'] == 'arch-explorer.compiled-request.v3'
        assert s.assess_request(request)['supported']


@pytest.mark.parametrize('mutation', ['input', 'mapping', 'adapter', 'digest'])
def test_saved_compilation_tampering_refuses_before_output_creation(compiler_api, tmp_path, mutation):
    w, s = compiler_api
    compiled, (_, graph, _, mapping) = compiled_roofline(compiler_api)
    if mutation == 'input':
        original = graph.to_dict()
        edited = deepcopy(original)
        inputs = edited['meta']['inputs']
        name = next(iter(inputs))
        def change_first(x):
            if isinstance(x[0], list):
                change_first(x[0])
            else:
                x[0] += 1
        change_first(inputs[name])
    elif mutation == 'mapping':
        original, edited = mapping.params, deepcopy(mapping.params)
        edited['rules']['matmul']['options']['tiles']['m'] += 1
    elif mutation == 'adapter':
        original, edited = 'roofline', 'qa.unregistered.task-adapter'
    else:
        def digests(x):
            if isinstance(x, dict):
                for v in x.values():
                    yield from digests(v)
            elif isinstance(x, list):
                for v in x:
                    yield from digests(v)
            elif isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x):
                yield x
        original = next(digests(compiled))
        edited = 'f'*64 if original != 'f'*64 else 'e'*64
    altered, matches = replace_json_subtree(compiled, original, edited)
    assert matches > 0, 'published saved source/adapter/binding absent'
    with pytest.raises(REJECT):
        w.validate_compilation(altered)
    output = tmp_path/'must-not-exist'
    with pytest.raises(REJECT):
        s.prepare_compiled(altered, output_dir=output, options={})
    assert not output.exists()


@pytest.mark.parametrize('field,value', [('m', 0), ('k', True), ('n', -1), ('n', '2')])
def test_illegal_tiles_reject_without_execution(compiler_api, monkeypatch, field, value):
    w, s = compiler_api
    adf, _, spec, mapping = roofline_input()
    mapping.params['rules']['matmul']['options']['tiles'][field] = value
    with prohibit_execution(monkeypatch, s), pytest.raises(REJECT):
        w.compile_workload(adf, spec, mapping)


@pytest.mark.parametrize('options', [{'dtype': 'fp16'}, {'dtype':'bf16'}, {'iterations': 2}])
def test_extended_storage_precision_and_unsupported_iterations_before_execution(compiler_api, monkeypatch, options):
    w, s = compiler_api
    adf, _, spec, mapping = roofline_input(**options)
    with prohibit_execution(monkeypatch, s):
        if 'dtype' in options:
            compilation=w.compile_workload(adf,spec,mapping)
            assert compilation['program']['dtype']==options['dtype']
            assert all(t['dtype']==options['dtype'] for t in compilation['program']['tensors'])
        else:
            with pytest.raises(REJECT):w.compile_workload(adf,spec,mapping)


def test_unknown_adapter_never_silently_selects_another_backend(compiler_api):
    w, _ = compiler_api
    adf, _, spec, mapping = roofline_input()
    with pytest.raises(REJECT):
        w.compile_workload(adf, spec, mapping,
                           bindings={'board/sm':w.AdapterBinding('qa.never-registered-adapter', '1')})


@pytest.mark.parametrize('old_target', ['roofline', 'qa.never-registered-target'])
def test_retired_target_keyword_is_explicitly_rejected(compiler_api, old_target):
    w, _ = compiler_api
    adf, _, spec, mapping = roofline_input()
    with pytest.raises(REJECT):
        w.compile_workload(adf, spec, mapping, target=old_target)


def test_task_adapter_duplicate_identity_is_rejected(compiler_api):
    w, _ = compiler_api
    def first(context):
        raise AssertionError('registration must not execute adapter')
    def second(context):
        raise AssertionError('registration must not execute adapter')
    identity = 'qa.duplicate.'+uuid4().hex
    semantics = ('compiled.matmul', 'compiled.assemble')
    definition = w.TaskAdapter(identity, '1', 'events', semantics, first)
    w.register_task_adapter(definition)
    with pytest.raises(REJECT):
        w.register_task_adapter(w.TaskAdapter(identity, '1', 'events', semantics, second))


def bind_inspection_adapter(w, adf, lower, identity):
    """Bind a new conversion to a real compatible component, without new costs."""
    from dataclasses import replace
    from arch_explorer.components.api import ImplementationBinding, register_component_implementation
    from arch_explorer.components.sm.implementations.roofline import DEFINITION
    w.register_task_adapter(w.TaskAdapter(identity, '1', 'events',
                            ('compiled.matmul', 'compiled.assemble'), lower))
    component_id = identity+'.component'
    register_component_implementation(replace(DEFINITION, id=component_id, adapters=(identity,)))
    point = adf.elements[0]
    point.implementation = ImplementationBinding(component_id, '1', point.implementation.params)


def test_invalid_adapter_product_rejects_instead_of_running_provider(compiler_api, monkeypatch):
    w, s = compiler_api
    called = []
    def wrong(context):
        called.append(context.program)
        return None
    identity = 'qa.wrong-product.'+uuid4().hex
    adf, _, spec, mapping = roofline_input()
    bind_inspection_adapter(w, adf, wrong, identity)
    with prohibit_execution(monkeypatch, s), pytest.raises(REJECT):
        w.compile_workload(adf, spec, mapping)
    assert len(called) == 1


@pytest.mark.parametrize('tiles', [{'m': 2, 'k': 2, 'n': 2}, {'m': 4, 'k': 2, 'n': 4}])
def test_external_adapter_observes_real_common_regions_and_nondivisible_K_semantics(compiler_api, tiles):
    """Inspect real compiler output before deliberately rejecting a bad product.

    This proves graph construction/reference consumption, not successful
    external adapter execution. Real installed group consumption remains
    a separate candidate-gated acceptance requirement.
    """
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import execute_reference
    from compilation_expectations import reference_matmul
    w, _ = compiler_api
    adf, graph, spec, mapping = roofline_input(tiles=tiles, k=3)
    source_op = graph.operators[0]
    a, b = [graph.meta['inputs'][name] for name in source_op.inputs]
    expected = [reference_matmul(a[0], b[0])]
    observed = []
    def inspect_then_invalid(context):
        program = context.program
        assert set(context.tasks) == {op.name for op in program.operators}
        regions = [op for op in program.operators if op.op == 'compiled.matmul']
        assemblies = [op for op in program.operators if op.op == 'compiled.assemble']
        assert len(regions) > 1 and assemblies
        # Region output extents cover logical M*N, not nominal padded tiles.
        assert sum(math.prod(program.tensors[name].shape)
                   for op in regions for name in op.outputs) == 3*5
        result = execute_reference(WorkloadSpec('graph', program.to_dict()),
                                   inputs=graph.meta['inputs'])
        assert result['outputs'][source_op.outputs[0]].tolist() == expected
        assert result['reference']['complete']
        observed.append(program.to_dict())
        return None  # explicitly invalid AdaptedGroup: provider must never run
    identity = 'qa.graph-inspection.'+uuid4().hex
    bind_inspection_adapter(w, adf, inspect_then_invalid, identity)
    with pytest.raises(REJECT):
        w.compile_workload(adf, spec, mapping)
    assert len(observed) == 1


def test_common_graph_keeps_temporary_producer_and_both_actual_readers(compiler_api):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import execute_reference
    from compilation_expectations import reference_matmul
    w, _ = compiler_api
    adf, _, _, mapping = roofline_input()
    graph, spec = two_reader_graph_input()
    values = graph.meta['inputs']
    temporary = reference_matmul(values['A'][0], values['B'][0])
    expected = {'Y': [reference_matmul(temporary, values['D'][0])],
                'Z': [reference_matmul(temporary, values['E'][0])]}
    observed = []
    def inspect_then_invalid(context):
        program = context.program
        assert program.producers()['T'].op == 'compiled.assemble'
        assert len([op for op in program.operators if 'T' in op.inputs]) >= 2
        result = execute_reference(WorkloadSpec('graph', program.to_dict()), inputs=values)
        assert {n: result['outputs'][n].tolist() for n in expected} == expected
        observed.append(program.to_dict())
        return None
    identity = 'qa.two-readers.'+uuid4().hex
    bind_inspection_adapter(w, adf, inspect_then_invalid, identity)
    with pytest.raises(REJECT):
        w.compile_workload(adf, spec, mapping)
    assert len(observed) == 1
