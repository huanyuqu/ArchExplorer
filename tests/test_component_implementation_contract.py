"""SIM41 public-contract checks, to execute only on the lead's VAL52 candidate.

Collection on the old baseline is possible, execution must fail if the actual
new API is missing. There are no skip/xfail markers or compatibility substitutes.
Real external distribution installation and native consumption remain separate
candidate-gated checks; entry-point test doubles below test rollback only.
"""
from copy import deepcopy
import importlib
import json
import subprocess
from types import SimpleNamespace
from uuid import uuid4

import pytest

from arch_explorer.adf import MappingSpec, Param, SpaceMatrix, SpacePoint, WorkloadSpec, fingerprint, from_dict
from arch_explorer.components.api import BuildContext, ComputeModel, Conventions, Kernel, ParamSpec, Provenance
from component_implementation_oracles import ROOFLINE_CASES, check_roofline_duration


REJECT = (ValueError, TypeError, KeyError, NotImplementedError)
SOURCE = 'independent VAL52 synthetic hand-input assumption, not device calibration'


@pytest.fixture
def api():
    module = importlib.import_module('arch_explorer.components.api')
    for name in ('ComponentImplementation', 'ImplementationBinding', 'component_catalog',
                 'component_implementations', 'register_component_implementation',
                 'resolve_component', 'build_component', 'component_candidates'):
        assert callable(getattr(module, name)), name
    return module


def roofline_point(api, *, rate=128, bandwidth=64, boundary='component_io', precision='fp32'):
    return SpacePoint('sm', 'sm', {'precision': Param(precision, '1', SOURCE)},
        implementation=api.ImplementationBinding('sm.roofline', '1', {
            'peak_ops_per_ns': Param(rate, 'op/ns', SOURCE),
            'bandwidth_bytes_per_ns': Param(bandwidth, 'B/ns', SOURCE),
            'boundary': Param(boundary, '1', SOURCE)}))


def context():
    return BuildContext('chip/sm', 'chip/sm', (0,), Provenance())


def sm_catalog(api):
    rows = api.component_catalog()['components']
    return next(row for row in rows if row['name'] == 'sm')


def test_one_sm_identity_describes_two_distinct_inputs_without_running_engines(api, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('descriptor discovery attempted external execution')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    data = api.component_catalog()
    assert data['schema'] == 'arch-explorer.component-catalog.v1'
    row = sm_catalog(api)
    implementations = {item['id']: item for item in row['implementations']}
    analytic, native = implementations['sm.roofline'], implementations['sm.accelsim']
    assert analytic['component'] == native['component'] == row['name'] == 'sm'
    assert analytic['version'] == native['version'] == '1'
    assert analytic['input_contract'] != native['input_contract']
    assert analytic['execution_domain'] != native['execution_domain']
    assert {'peak_ops_per_ns', 'bandwidth_bytes_per_ns', 'boundary'} <= analytic['parameters'].keys()
    assert analytic['source'] and native['source'] and analytic['result_scope'] and native['result_scope']
    # Returned JSON must not be a mutable alias into the registry authority.
    saved = deepcopy(row)
    row['implementations'].clear()
    assert sm_catalog(api) == saved


@pytest.mark.parametrize('case', [case for case in ROOFLINE_CASES if case.name != 'zero_work'],
                         ids=lambda case: case.name)
def test_actual_roofline_service_matches_independent_hand_answer(api, case):
    work_type = importlib.import_module(
        'arch_explorer.components.sm.implementations.roofline').RooflineWork
    point = roofline_point(api, rate=case.peak_flops_per_ns, bandwidth=case.bandwidth_bytes_per_ns)
    model = api.build_component(point, context(), execution_domain='events')
    work = work_type(case.flops, case.boundary_bytes, 'component_io')
    demand = model.kernel_demand(work)
    assert set(demand.resources) == {model.unit.name}
    assert demand.latency_ns == 0
    check_roofline_duration(case, demand.resources[model.unit.name])


@pytest.mark.parametrize('field', ['peak_ops_per_ns', 'bandwidth_bytes_per_ns', 'boundary'])
def test_missing_roofline_configuration_is_not_inferred(api, field):
    point = roofline_point(api)
    raw = point.to_dict()
    del raw['implementation']['params'][field]
    with pytest.raises(REJECT):
        api.build_component(from_dict(raw), context(), execution_domain='events')


@pytest.mark.parametrize('field', ['rate', 'bandwidth'])
@pytest.mark.parametrize('value', [0, -1, True, '128', float('nan'), float('inf')])
def test_invalid_roofline_rates_fail_instead_of_silent_defaults(api, field, value):
    with pytest.raises(REJECT):
        api.build_component(roofline_point(api, **{field: value}), context(), execution_domain='events')


@pytest.mark.parametrize('operations,byte_count,boundary', [
    (0, 0, 'component_io'), (-1, 64, 'component_io'), (64, -1, 'component_io'),
    (True, 64, 'component_io'), (64, True, 'component_io'),
    (float('nan'), 64, 'component_io'), (64, float('inf'), 'component_io'),
    (64, 64, 'dram'),
])
def test_invalid_or_wrong_boundary_work_cannot_create_demand(api, operations, byte_count, boundary):
    work_type = importlib.import_module(
        'arch_explorer.components.sm.implementations.roofline').RooflineWork
    model = api.build_component(roofline_point(api), context(), execution_domain='events')
    with pytest.raises(REJECT):
        model.kernel_demand(work_type(operations, byte_count, boundary))


def test_ordinary_kernel_is_not_used_to_infer_roofline_boundary_bytes(api):
    model = api.build_component(roofline_point(api), context(), execution_domain='events')
    with pytest.raises(REJECT):
        model.kernel_demand(Kernel('gemm', (1, 4, 8, 16), 'fp32'))


def fork_join_graph():
    from arch_explorer.workload.api import Workload
    graph = Workload('qa_roofline_fork_join', dtype='fp32')
    for name in ('x', 'left', 'right', 'out'):
        graph.tensor(name, (2, 8))
    graph.add('left', 'softmax', ['x'], ['left'], rows=2, cols=8)
    graph.add('right', 'gelu', ['x'], ['right'], rows=2, cols=8)
    graph.add('join', 'add', ['left', 'right'], ['out'], rows=2, cols=8)
    # Caller-declared synthetic F/B, not operator-name inference or device fit.
    work = {'left': {'operations': 4096, 'boundary_bytes': 128, 'boundary': 'component_io'},
            'right': {'operations': 256, 'boundary_bytes': 8192, 'boundary': 'component_io'},
            'join': {'operations': 1024, 'boundary_bytes': 512, 'boundary': 'component_io'}}
    return WorkloadSpec('graph', graph.to_dict()), work


def test_roofline_graph_serializes_one_sm_without_extra_external_traffic_and_replays_readonly(
        api, tmp_path, monkeypatch):
    from arch_explorer.simulator.api import assess_request, evaluate_request, prepare_request
    spec, work = fork_join_graph()
    request = prepare_request(SpaceMatrix('chip', [roofline_point(api)]), spec,
        MappingSpec('roofline_graph', {'compute': 'chip/sm', 'work': work}),
        provider='events', output_dir=tmp_path/'run', options={'trace': True})
    assert assess_request(request)['supported']
    result = evaluate_request(request)
    # Parallel DAG branches still contend for one exclusive SM:32+128+8=168ns.
    assert result['evaluation']['latency_ns'] == 168
    trace = result['evaluation']['details']['trace']
    compute = [item for item in trace if item['kind'] == 'compute']
    assert {item['operator'] for item in compute} == {'left', 'right', 'join'}
    assert not any(item['kind'] == 'comm' for item in trace)
    raw = tmp_path/'run/RAW.json'
    original = raw.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError('public replay tried to execute')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr('arch_explorer.simulator.events.engine.simulate', forbidden)
    monkeypatch.setattr('arch_explorer.simulator.events.engine.ConsistentScheduler.run', forbidden)
    assert evaluate_request(result) == result
    assert raw.read_bytes() == original


@pytest.mark.parametrize('bad', ['missing_work', 'extra_work', 'unknown_point', 'native_implementation'])
def test_wrong_roofline_graph_contract_rejects_before_output_publication(api, tmp_path, bad):
    from arch_explorer.simulator.api import prepare_request
    spec, work = fork_join_graph()
    point = roofline_point(api)
    compute = 'chip/sm'
    if bad == 'missing_work':
        del work['right']
    elif bad == 'extra_work':
        work['ghost'] = dict(work['left'])
    elif bad == 'unknown_point':
        compute = 'chip/unknown'
    elif bad == 'native_implementation':
        raw = point.to_dict()
        raw['implementation'] = {'id': 'sm.accelsim', 'version': '1', 'params': {}}
        point = from_dict(raw)
    directory = tmp_path/'must_not_publish'
    with pytest.raises(REJECT):
        prepare_request(SpaceMatrix('chip', [point]), spec,
            MappingSpec('roofline_graph', {'compute': compute, 'work': work}),
            provider='events', output_dir=directory)
    assert not directory.exists()


def test_binding_is_exact_in_json_and_candidate_identity(api):
    point = roofline_point(api)
    raw = json.loads(json.dumps(point.to_dict(), allow_nan=False))
    assert raw['implementation'] == {
        'id': 'sm.roofline', 'version': '1', 'params': {
            key: value.to_dict() for key, value in point.implementation.params.items()}}
    again = from_dict(raw)
    assert again.to_dict() == raw and fingerprint(again) == fingerprint(point)
    identities = {fingerprint(point)}
    for target, name, value in [
        ('binding', 'id', 'qa.other'), ('binding', 'version', '2'),
        ('parameter', 'peak_ops_per_ns', Param(256, 'op/ns', SOURCE).to_dict()),
        ('parameter', 'bandwidth_bytes_per_ns', Param(32, 'B/ns', SOURCE).to_dict()),
        ('parameter', 'boundary', Param('other', '1', SOURCE).to_dict()),
    ]:
        modified = deepcopy(raw)
        item = modified['implementation'] if target == 'binding' else modified['implementation']['params']
        item[name] = value
        identities.add(fingerprint(from_dict(modified)))
    assert len(identities) == 6
    other = roofline_point(api, rate=256)
    candidates = api.component_candidates(point, [point.implementation, other.implementation])
    assert {fingerprint(item) for item in candidates} == {fingerprint(point), fingerprint(other)}
    with pytest.raises(REJECT):
        api.component_candidates(point, [point.implementation, point.implementation])


@pytest.mark.parametrize('mutation', ['missing', 'unknown_id', 'unknown_version', 'unknown_field',
                                    'overrides_hardware', 'wrong_unit', 'wrong_domain'])
def test_binding_ambiguity_and_configuration_are_rejected_before_execution(api, mutation):
    raw = roofline_point(api).to_dict()
    if mutation == 'missing':
        del raw['implementation']
    elif mutation == 'unknown_id':
        raw['implementation']['id'] = 'qa.nonexistent'
    elif mutation == 'unknown_version':
        raw['implementation']['version'] = '999'
    elif mutation == 'unknown_field':
        raw['implementation']['qa_silent_field'] = 1
    elif mutation == 'overrides_hardware':
        raw['implementation']['params']['precision'] = Param('fp16').to_dict()
    elif mutation == 'wrong_unit':
        raw['implementation']['params']['peak_ops_per_ns']['unit'] = 'op/cycle'
    with pytest.raises(REJECT):
        api.resolve_component(from_dict(raw), execution_domain='dynamic' if mutation == 'wrong_domain' else 'events')


class ExternalSerialSm(ComputeModel):
    def kernel_time(self, kernel):
        self.precision(kernel)
        return kernel.flops / self.positive('qa_ops_per_ns')


def external_factory(component, point, build_context):
    return ExternalSerialSm(component, point, build_context)


def external_definition(api, identity):
    return api.ComponentImplementation(
        component='sm', id=identity, version='1', source={'basis': SOURCE},
        factory=external_factory, execution_domain='events', input_contract='events.kernel.v1',
        result_scope='Synthetic serial compute only, external memory costs excluded',
        params={'qa_ops_per_ns': ParamSpec('op/ns')}, hardware_parameters=('precision',),
        conventions=Conventions(time_base='ns', arbitration='fifo', includes_startup=False),
        capabilities={'contained_components': [], 'memory_traffic': 'external_only'})


def test_new_implementation_of_existing_sm_really_builds_and_duplicate_conflict_rejects(api):
    identity = 'qa.sm.serial.' + uuid4().hex
    definition = external_definition(api, identity)
    api.register_component_implementation(definition)
    api.register_component_implementation(definition)  # same unchanged object is idempotent
    with pytest.raises(REJECT):
        api.register_component_implementation(external_definition(api, identity))
    identities = set()
    for rate, expected_ns in [(32, 32), (64, 16)]:
        point = SpacePoint('sm', 'sm', {'precision': Param('fp32')},
            implementation=api.ImplementationBinding(identity, '1', {
                'qa_ops_per_ns': Param(rate, 'op/ns', SOURCE)}))
        model = api.build_component(point, context(), execution_domain='events')
        demand = model.kernel_demand(Kernel('gemm', (1, 4, 8, 16), 'fp32'))
        assert demand.resources == {model.unit.name: expected_ns}
        identities.add(fingerprint(point))
    assert len(identities) == 2


@pytest.mark.parametrize('stage', ['load', 'register'])
def test_failed_discovery_rolls_back_component_implementations_without_harming_existing(api, monkeypatch, stage):
    from arch_explorer.components import extensions
    extensions.discover_components()
    before = deepcopy(api.component_catalog())
    identity = 'qa.sm.rollback.' + uuid4().hex

    def register_then_fail():
        api.register_component_implementation(external_definition(api, identity))
        assert identity in {item['id'] for item in sm_catalog(api)['implementations']}
        raise RuntimeError('VAL52 broken ' + stage)

    def load():
        if stage == 'load':
            register_then_fail()
        return register_then_fail

    entry = SimpleNamespace(name='val52_broken', value='qa_component_package:register', load=load,
                            dist=SimpleNamespace(name='qa_component_distribution'))
    monkeypatch.setattr(extensions, '_DISCOVERED', False)
    monkeypatch.setattr(extensions, '_LOADED', set(extensions._LOADED))
    monkeypatch.setattr(extensions, '_FAILURES', dict(extensions._FAILURES))
    monkeypatch.setattr(extensions.metadata, 'entry_points', lambda **kwargs: [entry])
    result = extensions.discover_components()
    assert any('VAL52 broken ' + stage in message for message in result['failures'].values())
    assert api.component_catalog() == before
    assert identity not in {item['id'] for item in sm_catalog(api)['implementations']}
