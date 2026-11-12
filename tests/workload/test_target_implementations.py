"""VAL50 public target contracts, independent of author example identities.

No target/native execution is accepted merely because these CPU tests pass.
"""
from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from arch_explorer.adf import MappingSpec, SpaceMatrix, WorkloadSpec
from arch_explorer.archs.adf_seeds import memory, mesh, point
from arch_explorer.components.api import Kernel
from arch_explorer.simulator.api import assess_request, evaluate_request, prepare_request
from arch_explorer.workload import api
from open_execution_cases import SHIFT_SQUARE


# Keep the shared join on its existing built-in Kernel compatibility adapter
# while comparing the two external branch products. Public discovery also
# installs compiled.events.add, so this control needs an explicit selection.
JOIN_IMPLEMENTATION = 'legacy:add'


def validate(op, tensors):
    if set(op.attrs) != {'bias'} or type(op.attrs['bias']) not in (int, float):
        raise ValueError('explicit numeric bias required')
    if not np.isfinite(op.attrs['bias']):
        raise ValueError('finite bias required')
    if len(op.inputs) != 1 or len(op.outputs) != 1:
        raise ValueError('one input and output required')
    assert tensors[op.inputs[0]].shape == tensors[op.outputs[0]].shape


def hardware(capacity=2**20):
    vector = point('vector', 'vector_unit', {
        'lanes': (4, 'count'), 'sfu_lanes': (1, 'count'),
        'clock_ghz': (1.0, 'GHz'), 'precision': ('fp32', '1')}, source='synthetic QA')
    return SpaceMatrix('chip', [SpaceMatrix('core', [vector], repeat=2),
        memory('l2', capacity=2**20), memory('dram', technology='ddr', capacity=capacity,
                                            bandwidth=16.0, latency=2.0)], [mesh(cols=4)])


@pytest.fixture
def external():
    name = 'qa50.shift_square.' + uuid4().hex
    seen = {'lower': 0, 'reference': 0}

    def reference(op, inputs):
        seen['reference'] += 1
        assert not inputs[0].flags.writeable
        return (inputs[0] + op.attrs['bias']) ** 2

    def atomic(op, context):
        seen['lower'] += 1
        return Kernel('vector', context.tensors[op.outputs[0]].shape, context.dtype, simt=2)

    def steps(op, context):
        seen['lower'] += 1
        shape = context.tensors[op.outputs[0]].shape
        kernel = Kernel('vector', shape, context.dtype, simt=1)
        return api.ExecutionProgram((api.Tensor('tmp', shape, context.dtype),), (
            api.ExecutionStep('shift', kernel, op.inputs, ('tmp',)),
            api.ExecutionStep('square', kernel, ('tmp',), op.outputs, ('shift',))))

    semantic = api.OperatorDefinition(name, validate, version='1', reference=reference)
    api.register_operator(semantic)
    first = api.ImplementationDefinition(name+'.atomic', '1', name, 'events', atomic,
                                         requirements=('kernel:vector',), dtypes=('fp32',))
    second = api.ImplementationDefinition(name+'.steps', '1', name, 'events', steps,
                                          requirements=('kernel:vector',), dtypes=('fp32',))
    api.register_implementation(second)  # Reverse installation order cannot resolve ambiguity.
    api.register_implementation(first)
    graph = api.Workload('arbitrary graph with no model identity', dtype='fp32')
    for tensor in ('x', 'left', 'right', 'out'):
        graph.tensor(tensor, (2, 4))
    for side in ('left', 'right'):
        graph.add(side, name, ['x'], [side], bias=SHIFT_SQUARE['bias'][side])
    graph.add('join', 'add', ['left', 'right'], ['out'], rows=2, cols=4)
    graph.meta['inputs'] = {'x': deepcopy(SHIFT_SQUARE['input'])}
    return graph, first, second, seen


def select(graph, implementation=None, **changes):
    return api.select_implementation(graph.operators[0], 'events', dtype=changes.get('dtype', 'fp32'),
        tensors=graph.tensors, capabilities=changes.get('capabilities', ('kernel:vector',)),
        implementation=implementation)


def test_selection_is_explicit_ambiguous_without_id_and_never_lowers(external):
    graph, first, second, seen = external
    with pytest.raises(ValueError, match='ambiguous'):
        select(graph)
    assert select(graph, first.id) is first and select(graph, second.id) is second
    assert seen == {'lower': 0, 'reference': 0}


def test_builtin_join_also_requires_explicit_selection(external):
    graph, _, _, seen = external
    join = graph.operators[-1]
    options = dict(dtype='fp32', tensors=graph.tensors, capabilities=('kernel:vector',))
    with pytest.raises(ValueError, match="node='join'.*ambiguous"):
        api.select_implementation(join, 'events', **options)
    compatibility = api.select_implementation(join, 'events',
        implementation=JOIN_IMPLEMENTATION, **options)
    compiled = api.select_implementation(join, 'events',
        implementation='compiled.events.add', **options)
    assert compatibility.id == JOIN_IMPLEMENTATION
    assert compatibility.operator == compiled.operator == 'add'
    assert compatibility.target == compiled.target == 'events'
    assert compatibility is not compiled
    assert seen == {'lower': 0, 'reference': 0}


@pytest.mark.parametrize('bad', ['absent', 'capability', 'dtype'])
def test_explicit_selection_rejects_missing_or_incompatible_without_lowering(external, bad):
    graph, first, _, seen = external
    with pytest.raises(ValueError, match='node=.*implementation=.*target='):
        select(graph, 'qa50.no_such_implementation' if bad == 'absent' else first.id,
               **({'capabilities': ()} if bad == 'capability' else {'dtype': 'bf16'} if bad == 'dtype' else {}))
    assert seen == {'lower': 0, 'reference': 0}


def test_same_id_cannot_replace_implementation_or_version(external):
    _, first, _, _ = external
    api.register_implementation(first)
    with pytest.raises(ValueError, match='conflict'):
        api.register_implementation(replace(first, version='2'))


@pytest.mark.parametrize('bad', ['missing-producer-dependency', 'unproduced-input',
                                'missing-output', 'colliding-temp'])
def test_invalid_multi_step_product_is_not_an_executable_plan(external, bad):
    graph, _, second, _ = external
    op = graph.operators[0]
    context = api.ExecutionContext('events', 'fp32', graph.tensors,
                                   capabilities=('kernel:vector',))
    product = api.lower_implementation(second, op, context)
    if bad == 'missing-producer-dependency':
        product = replace(product, steps=(product.steps[0], replace(product.steps[1], deps=())))
    elif bad == 'unproduced-input':
        product = replace(product, tensors=(*product.tensors, api.Tensor('future', (2, 4), 'fp32')),
                          steps=(replace(product.steps[0], inputs=('future',)), product.steps[1]))
    elif bad == 'missing-output':
        product = replace(product, steps=product.steps[:1])
    else:
        product = replace(product, tensors=(api.Tensor('x', (2, 4), 'fp32'),))
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        api.as_program(product, op, context)


def test_reference_matches_frozen_independent_dyadic_answers(external):
    graph, _, _, seen = external
    result = api.execute_reference(WorkloadSpec('graph', graph.to_dict()))
    for name, expected in [('left', 'expected_left'), ('right', 'expected_right'), ('out', 'expected_join')]:
        assert np.array_equal(result['tensors'][name], np.asarray(SHIFT_SQUARE[expected], dtype=np.float32))
    assert seen == {'lower': 0, 'reference': 2}


def test_events_consume_both_products_and_replay_does_not_execute(external, tmp_path, monkeypatch):
    graph, first, second, _ = external
    outcomes = []
    for implementation in (first, second):
        mapping = MappingSpec('execution_graph', {'implementations': {
            'left': implementation.id, 'right': implementation.id,
            'join': JOIN_IMPLEMENTATION}, 'placement': 'round_robin'})
        request = prepare_request(hardware(), WorkloadSpec('graph', graph.to_dict()), mapping,
            provider='events', output_dir=tmp_path/implementation.id, options={'trace': True})
        assert assess_request(request)['supported']
        result = evaluate_request(request)
        assert result['evaluation']['latency_ns'] > 0
        before = {str(p):p.read_bytes() for p in (tmp_path/implementation.id).rglob('*') if p.is_file()}
        def forbidden(*args, **kwargs):
            raise AssertionError('replay invoked execution')
        with monkeypatch.context() as guard:
            guard.setattr('arch_explorer.components.events_provider.EventsProvider._execute', forbidden)
            guard.setattr('arch_explorer.simulator.events.engine.simulate', forbidden)
            assert evaluate_request(result) == result
        assert before == {str(p):p.read_bytes() for p in (tmp_path/implementation.id).rglob('*') if p.is_file()}
        outcomes.append(result['evaluation'])
    # This is a real target accounting distinction, separate from numerical reference validation.
    assert outcomes[0]['latency_ns'] != outcomes[1]['latency_ns']


@pytest.mark.parametrize('bad', ['ambiguous', 'unknown-node', 'impossible-capacity'])
def test_preparation_refuses_before_creating_output_directory(external, tmp_path, bad):
    graph, _, second, _ = external
    choices = {} if bad == 'ambiguous' else {
        'left': second.id, 'right': second.id, 'join': JOIN_IMPLEMENTATION}
    if bad == 'unknown-node':
        choices['not-a-node'] = second.id
    output = tmp_path/'must-not-exist'
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        prepare_request(hardware(1 if bad == 'impossible-capacity' else 2**20),
            WorkloadSpec('graph', graph.to_dict()), MappingSpec('execution_graph', {'implementations':choices}),
            provider='events', output_dir=output, options={'check_capacity':False})
    assert not output.exists()


def test_semantic_without_any_target_implementation_is_not_runnable(external):
    graph, _, _, _ = external
    name = 'qa50.no_implementation.' + uuid4().hex
    api.register_operator(api.OperatorDefinition(name, validate))
    op = replace(graph.operators[0], op=name)
    with pytest.raises(ValueError, match='no compatible implementation'):
        api.select_implementation(op, 'events', dtype='fp32', tensors=graph.tensors,
                                  capabilities=('kernel:vector',))


def test_declared_shape_layout_constraint_is_checked(external):
    graph, first, _, _ = external

    def contiguous_two_rows(op, context):
        tensor = context.tensors[op.inputs[0]]
        return tensor.shape[0] == 2 and tensor.strides is None

    implementation = replace(first, id=first.id+'.layout', supports=contiguous_two_rows)
    api.register_implementation(implementation)
    assert select(graph, implementation.id) is implementation
    altered = deepcopy(graph)
    altered.tensors['qa50.alias.storage'] = api.Tensor('qa50.alias.storage', (4, 2), 'fp32')
    altered.tensors['x'] = replace(altered.tensors['x'], alias_of='qa50.alias.storage', strides=(1, 2))
    with pytest.raises(ValueError, match='shape/layout'):
        select(altered, implementation.id)


@pytest.mark.parametrize('failure_stage', ['import', 'register'])
def test_failed_discovery_rolls_back_all_four_public_registrations(monkeypatch, failure_stage):
    from types import SimpleNamespace
    from arch_explorer.components import extensions
    extensions.discover_components()
    identity = 'qa50.failed.' + uuid4().hex

    def lower(op, context):
        return Kernel('vector', (2, 4), 'fp32', simt=2)

    def initialize(spec, graph, external_names):
        return {'inputs': {}}

    def builder(params, *, tp=1):
        return api.Workload('unused')

    def register_then_fail():
        api.register_operator(api.OperatorDefinition(identity, validate))
        api.register_implementation(api.ImplementationDefinition(identity+'.impl', '1', identity, 'events', lower))
        api.register_workload(identity, builder)
        api.register_reference(api.ReferenceDefinition(identity, '1', initialize))
        raise RuntimeError('QA controlled '+failure_stage+' failure')

    def load():
        if failure_stage == 'import':
            register_then_fail()
        return register_then_fail

    entry = SimpleNamespace(name=identity, value=identity+':register', load=load,
                            dist=SimpleNamespace(name=identity))
    monkeypatch.setattr(extensions, '_DISCOVERED', False)
    monkeypatch.setattr(extensions, '_LOADED', set(extensions._LOADED))
    monkeypatch.setattr(extensions, '_FAILURES', dict(extensions._FAILURES))
    monkeypatch.setattr(extensions.metadata, 'entry_points', lambda **kwargs: [entry])
    result = extensions.discover_components()
    assert any('QA controlled' in error for error in result['failures'].values())
    with pytest.raises(ValueError, match='unknown operator'):
        api.get_operator(identity)
    # New objects with the same stable IDs must be accepted after rollback.
    api.register_operator(api.OperatorDefinition(identity, validate))
    api.register_implementation(api.ImplementationDefinition(identity+'.impl', '1', identity, 'events', lower))
    api.register_workload(identity, lambda params, *, tp=1: api.Workload('replacement'))
    api.register_reference(api.ReferenceDefinition(identity, '1', lambda spec, graph, names: {'inputs':{}}))
    old = api.workload_from_spec(WorkloadSpec('operator', {'op':'softmax','rows':1,'cols':8,'dtype':'fp32'}))
    assert old.operators[0].kernel('fp32').sfu == 1
