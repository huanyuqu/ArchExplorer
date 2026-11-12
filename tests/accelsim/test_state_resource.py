"""Pure parser fixtures: no test record confers actual resource authority."""
import copy
import hashlib
import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from arch_explorer.simulator.accelsim import measurement_state as state
from arch_explorer.simulator.accelsim import state_resource as resource
from arch_explorer.simulator.accelsim.microcontract import CAPS


NS = 10**9
UUID = '00112233445566778899aabbccddeeff'
GPU_UUID = 'GPU-00112233-4455-6677-8899-aabbccddeeff'
EMPTY = {'sm_mhz': None, 'memory_mhz': None, 'pstate': None}


def original(tmp_path, name, value):
    path = tmp_path/name
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def bundle(n=2, preparation=2):
    cases = {f'case-{i}': {'id': f'case-{i}', 'family': 'dep' if i % 2 == 0 else 'chase',
                         'function': f'candidate_{i}',
                         'partition': 'fit' if i < n-1 else 'heldout'} for i in range(n)}
    plan = {'schema': state.PLAN, 'lifecycle': dict(state.LIFECYCLE),
            'cases': [{'case': cid, 'diagnostic': 1, 'preparation': preparation,
                       'formal': 3} for cid in cases]}
    limits = dict(CAPS, cases=n, host_processes=n+1, capture_launches=n, timing_launches=3*n,
                  diagnostic_launches=n, preparation_launches=preparation*n,
                  uninstrumented_launches=(4+preparation)*n,
                  application_launches=(5+preparation)*n)
    return SimpleNamespace(document={'schema': state.REGISTRY, 'work_item': 'WP-SIM-34',
                                     'measurement_state': plan, 'budget': limits},
                           sha256='a'*64, cases=cases)


def fixture(tmp_path, *, synthetic=False, n=2):
    b = bundle(n)
    design = {'run_id': 'run-34', 'created_realtime_ns': 1000*NS,
              'design_sha256': 'b'*64, 'plan_sha256': state.sha_plan(b),
              'budget': copy.deepcopy(b.document['budget'])}
    evidence = original(tmp_path, 'owner.json', {
        'kind': 'synthetic' if synthetic else 'hardware', 'run_id': design['run_id'],
        'no_resource_authority': synthetic, 'resource_released': True,
        'note': 'CPU parser fixture only; external authenticity is not supplied by this record',
    })
    handoff = {
        'schema': resource.SYNTHETIC_HANDOFF if synthetic else resource.HANDOFF,
        'work_item': 'WP-SIM-34', 'run_id': design['run_id'], 'resource_released': True,
        'external_compute_empty': True,
        'coordinator': 'synthetic-coordinator' if synthetic else 'resident_2e1684fdfc1b',
        'resource_owner': 'synthetic-no-device-owner' if synthetic else 'owner-format-fixture',
        'coordination_evidence': [evidence], 'issued_realtime_ns': 1001*NS,
        'lock_file': str(tmp_path/'never-opened.lock'),
        'absolute_deadline_realtime_ns': 2801*NS, 'close_reserve_s': 60,
        'registry_sha256': b.sha256, 'plan_sha256': state.sha_plan(b),
        'budget': copy.deepcopy(design['budget']), 'design_sha256': design['design_sha256'],
    }
    if synthetic:
        handoff['no_resource_authority'] = True
    return b, design, handoff


def snapshot(tmp_path, *, synthetic=False, position='before'):
    b, frozen, handoff = fixture(tmp_path, synthetic=synthetic)
    resource.validate_handoff(b, frozen, handoff, synthetic=synthetic)
    view = dict(frozen, absolute_deadline_realtime_ns=handoff['absolute_deadline_realtime_ns'],
                close_reserve_s=60)
    start = 1002*NS if position == 'before' else 1011*NS
    stdout = original(tmp_path, 'query.stdout', (GPU_UUID+', 1980, 12000, P0\n').encode())
    stderr = original(tmp_path, 'query.stderr', b'')
    data = {
        'schema': 'arch-explorer.measurement-state-snapshot.v1',
        'kind': 'synthetic' if synthetic else 'hardware', 'no_resource_authority': synthetic,
        'run_id': frozen['run_id'], 'case': 'case-0', 'position': position, 'uuid_hex': UUID,
        'argv': resource.snapshot_policy()['query'], 'status': 'returned', 'exit_code': 0,
        'started_realtime_ns': start, 'ended_realtime_ns': start+NS//10,
        'started_monotonic_ns': 40*NS, 'ended_monotonic_ns': 40*NS+NS//10,
        'stdout': stdout, 'stderr': stderr,
        'values': {'sm_mhz': 1980, 'memory_mhz': 12000, 'pstate': 'P0'}, 'reason': None,
    }
    process = {'run_id': frozen['run_id'], 'case': 'case-0',
               'started_realtime_ns': 1003*NS, 'ended_realtime_ns': 1010*NS}
    return data, view, process


@pytest.mark.parametrize('synthetic', [False, True])
def test_handoff_binds_complete_plan_originals_and_mode(tmp_path, synthetic):
    b, d, h = fixture(tmp_path, synthetic=synthetic, n=8)
    before = copy.deepcopy((d, h))
    assert resource.validate_handoff(b, d, h, synthetic=synthetic) is h
    assert (d, h) == before
    assert not (tmp_path/'never-opened.lock').exists()


@pytest.mark.parametrize('field', sorted(resource._HANDOFF_FIELDS))
def test_handoff_missing_identity_or_role_field_is_refused(tmp_path, field):
    b, d, h = fixture(tmp_path)
    del h[field]
    with pytest.raises(ValueError, match='exactly fields'):
        resource.validate_handoff(b, d, h)


@pytest.mark.parametrize('edit', [
    {'schema': 'arch-explorer.measurement-resource-handoff.v2'},
    {'run_id': 'foreign-run'}, {'work_item': 'WP-SIM-32'}, {'registry_sha256': 'c'*64},
    {'plan_sha256': 'c'*64}, {'design_sha256': 'c'*64},
    {'resource_released': False}, {'external_compute_empty': False},
    {'resource_released': 1}, {'coordinator': 'other'}, {'resource_owner': ''},
    {'resource_owner': 'synthetic-no-device-owner'}, {'lock_file': 'relative.lock'},
    {'issued_realtime_ns': 1000*NS}, {'issued_realtime_ns': True},
    {'absolute_deadline_realtime_ns': 1001*NS},
    {'absolute_deadline_realtime_ns': 2801*NS+1}, {'close_reserve_s': 59},
    {'close_reserve_s': True}, {'coordination_evidence': []},
    {'budget': {'cases': 8}}, {'no_resource_authority': False},
])
def test_handoff_cannot_borrow_or_extend_authority(tmp_path, edit):
    b, d, h = fixture(tmp_path)
    h.update(edit)
    with pytest.raises(ValueError):
        resource.validate_handoff(b, d, h)


@pytest.mark.parametrize('field', ['run_id', 'design_sha256', 'created_realtime_ns', 'plan_sha256', 'budget'])
def test_frozen_design_identity_is_required(tmp_path, field):
    b, d, h = fixture(tmp_path)
    del d[field]
    with pytest.raises(ValueError):
        resource.validate_handoff(b, d, h)


@pytest.mark.parametrize('edit', ['too-many', 'legacy-registry', 'less-preparation', 'more-preparation',
                                 'no-diagnostic', 'missing-formal', 'reordered', 'boolean-count',
                                 'hidden-total', 'hidden-host', 'larger-window'])
def test_reclosed_registry_cannot_authorize_a_different_role_budget(tmp_path, edit):
    b, d, h = fixture(tmp_path, n=9 if edit == 'too-many' else 2)
    if edit == 'legacy-registry':
        b.document['schema'] = 'arch-explorer.measurement-registry.v2'
    elif edit in ('less-preparation', 'more-preparation'):
        n = len(b.cases)
        p = 0 if edit == 'less-preparation' else 3
        for row in b.document['measurement_state']['cases']:
            row['preparation'] = p
        b.document['budget'].update(preparation_launches=n*p,
                                    uninstrumented_launches=n*(4+p), application_launches=n*(5+p))
    elif edit == 'no-diagnostic':
        b.document['measurement_state']['cases'][0]['diagnostic'] = 0
    elif edit == 'missing-formal':
        b.document['measurement_state']['cases'][0]['formal'] = 2
    elif edit == 'reordered':
        b.document['measurement_state']['cases'].reverse()
    elif edit == 'boolean-count':
        b.document['measurement_state']['cases'][0]['diagnostic'] = True
    elif edit == 'hidden-total':
        b.document['budget']['application_launches'] += 1
    elif edit == 'hidden-host':
        b.document['budget']['host_processes'] += 1
    elif edit == 'larger-window':
        b.document['budget']['total_window_s'] += 1
    d.update(plan_sha256=state.sha_plan(b), budget=copy.deepcopy(b.document['budget']))
    h.update(plan_sha256=d['plan_sha256'], budget=copy.deepcopy(d['budget']))
    with pytest.raises(ValueError):
        resource.validate_handoff(b, d, h)


def test_synthetic_evidence_cannot_be_promoted_by_handoff_relabel(tmp_path):
    b, d, h = fixture(tmp_path, synthetic=True)
    h.pop('no_resource_authority')
    h.update(schema=resource.HANDOFF, coordinator='resident_2e1684fdfc1b',
             resource_owner='owner-format-fixture')
    with pytest.raises(ValueError, match='synthetic coordination bytes'):
        resource.validate_handoff(b, d, h)


@pytest.mark.parametrize('record', [
    {'kind': 'synthetic', 'no_resource_authority': True, 'run_id': 'foreign'},
    {'kind': 'synthetic', 'run_id': 'run-34'},
    {'kind': 'hardware', 'no_resource_authority': False, 'run_id': 'run-34'},
])
def test_synthetic_handoff_requires_marked_current_run_originals(tmp_path, record):
    b, d, h = fixture(tmp_path, synthetic=True)
    h['coordination_evidence'] = [original(tmp_path, 'changed-evidence.json', record)]
    with pytest.raises(ValueError, match='explicitly marked current-run'):
        resource.validate_handoff(b, d, h, synthetic=True)


def test_hardware_rejects_nested_fixture_marker_and_foreign_run(tmp_path):
    b, d, h = fixture(tmp_path)
    for i, record in enumerate([{'owner': {'no_resource_authority': True}}, {'run_id': 'foreign'}]):
        h['coordination_evidence'] = [original(tmp_path, f'evidence-{i}.json', record)]
        with pytest.raises(ValueError):
            resource.validate_handoff(b, d, h)


def test_coordination_uses_exact_original_bytes_and_allows_plain_hardware_messages(tmp_path):
    b, d, h = fixture(tmp_path)
    r = original(tmp_path, 'owner-message.txt', b'CPU format fixture of an owner release message\n')
    h['coordination_evidence'] = [r]
    resource.validate_handoff(b, d, h)
    (tmp_path/'owner-message.txt').write_bytes(b'different bytes\n')
    with pytest.raises(ValueError, match='changed/missing'):
        resource.validate_handoff(b, d, h)
    r['pid'] = 99
    with pytest.raises(ValueError, match='exactly fields'):
        resource.validate_handoff(b, d, h)


def test_policy_is_fixed_and_returns_fresh_containers():
    expected = {'schema': 'arch-explorer.measurement-state-snapshot-policy.v1',
                'query': ['/usr/bin/nvidia-smi', '--id=0',
                          '--query-gpu=uuid,clocks.current.sm,clocks.current.memory,pstate',
                          '--format=csv,noheader,nounits'],
                'timeout_s': 5, 'per_host': 2, 'positions': ['before', 'after'],
                'clock_interpretation': 'instantaneous; not average or steady/cache evidence'}
    assert resource.snapshot_policy() == expected
    bad = resource.snapshot_policy()
    bad['query'].append('--loop=1')
    bad['positions'].clear()
    assert resource.snapshot_policy() == expected


@pytest.mark.parametrize('synthetic', [False, True])
@pytest.mark.parametrize('position', ['before', 'after'])
def test_snapshot_raw_values_identity_and_position(tmp_path, synthetic, position):
    data, view, process = snapshot(tmp_path, synthetic=synthetic, position=position)
    before = copy.deepcopy((data, view, process))
    assert resource.validate_snapshot(data, view, 'case-0', position, UUID, process,
                                      synthetic=synthetic) == data['values']
    assert (data, view, process) == before


@pytest.mark.parametrize('edit', [
    {'schema': 'older'}, {'kind': 'synthetic'}, {'no_resource_authority': True},
    {'run_id': 'wrong'}, {'case': 'case-1'}, {'position': 'after'}, {'uuid_hex': 'f'*32},
    {'status': 'passed'}, {'exit_code': True}, {'exit_code': 2}, {'reason': 'replacement'},
    {'values': {'sm_mhz': 1981, 'memory_mhz': 12000, 'pstate': 'P0'}},
    {'values': {'sm_mhz': True, 'memory_mhz': 12000, 'pstate': 'P0'}},
    {'values': {'sm_mhz': 1980.0, 'memory_mhz': 12000, 'pstate': 'P0'}},
    {'values': {'sm_mhz': 1980, 'memory_mhz': 12000, 'pstate': 'P9'}},
    {'values': {'sm_mhz': 0, 'memory_mhz': 12000, 'pstate': 'P0'}},
    {'values': {'sm_mhz': 1980, 'memory_mhz': 100001, 'pstate': 'P0'}},
    {'values': {'sm_mhz': 1980, 'memory_mhz': 12000, 'pstate': 'N/A'}},
    {'values': {'sm_mhz': 1980, 'memory_mhz': 12000}},
    {'physical_core_mhz': 1980}, {'stdout': None}, {'stderr': None},
    {'argv': ['/usr/bin/nvidia-smi', '--query-gpu=clocks.sm']},
])
def test_snapshot_cannot_relabel_or_invent_raw_values(tmp_path, edit):
    data, view, process = snapshot(tmp_path)
    data.update(edit)
    with pytest.raises(ValueError):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process)


@pytest.mark.parametrize('field', sorted(resource._SNAPSHOT_FIELDS))
def test_snapshot_missing_field_is_refused(tmp_path, field):
    data, view, process = snapshot(tmp_path)
    del data[field]
    with pytest.raises(ValueError, match='exactly fields'):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process)


@pytest.mark.parametrize('edit', [
    {'started_realtime_ns': 1000*NS}, {'ended_realtime_ns': 1001*NS},
    {'ended_realtime_ns': 2741*NS}, {'ended_realtime_ns': 2742*NS},
    {'started_monotonic_ns': True}, {'ended_monotonic_ns': 39*NS},
    {'ended_monotonic_ns': 46*NS+1},
])
def test_snapshot_cannot_cross_deadline_or_query_duration(tmp_path, edit):
    data, view, process = snapshot(tmp_path)
    data.update(edit)
    with pytest.raises(ValueError):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID)


def test_window_view_and_preflight_uuid_are_required(tmp_path):
    data, view, process = snapshot(tmp_path)
    for key in ('absolute_deadline_realtime_ns', 'close_reserve_s', 'created_realtime_ns', 'run_id'):
        bad = dict(view)
        bad.pop(key)
        with pytest.raises(ValueError):
            resource.validate_snapshot(data, bad, 'case-0', 'before', UUID)
    for bad_uuid in (None, '', GPU_UUID, 'z'*32):
        with pytest.raises(ValueError):
            resource.validate_snapshot(data, view, 'case-0', 'before', bad_uuid)


@pytest.mark.parametrize('edit', [{'run_id': 'foreign'}, {'case': 'case-1'},
                                 {'started_realtime_ns': 1002*NS},
                                 {'ended_realtime_ns': 1001*NS}])
def test_before_snapshot_cannot_borrow_process_scope_or_overlap(tmp_path, edit):
    data, view, process = snapshot(tmp_path)
    process.update(edit)
    with pytest.raises(ValueError):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process)


def test_after_snapshot_cannot_overlap_the_measured_process(tmp_path):
    data, view, process = snapshot(tmp_path, position='after')
    process['ended_realtime_ns'] = data['started_realtime_ns']+1
    with pytest.raises(ValueError, match='overlaps'):
        resource.validate_snapshot(data, view, 'case-0', 'after', UUID, process)


@pytest.mark.parametrize('raw', [
    b'', (GPU_UUID+', N/A, 12000, P0\n').encode(),
    (GPU_UUID+', 0, 12000, P0\n').encode(), (GPU_UUID+', 100001, 12000, P0\n').encode(),
    (GPU_UUID+', 1.5, 12000, P0\n').encode(), (GPU_UUID+', +1980, 12000, P0\n').encode(),
    (GPU_UUID+', 1980 MHz, 12000, P0\n').encode(), (GPU_UUID+', 1980, 12000\n').encode(),
    (GPU_UUID+', 1980, 12000, N/A\n').encode(), (GPU_UUID+', 1980, 12000, P0\n\n').encode(),
    ((GPU_UUID+', 1980, 12000, P0\n')*2).encode(), b'\xff\xfe',
])
def test_unavailable_query_stays_unknown_without_clock_values(tmp_path, raw):
    data, view, process = snapshot(tmp_path)
    data.update(status='unknown', values=dict(EMPTY), reason='query data unavailable',
                stdout=original(tmp_path, 'unavailable.stdout', raw))
    assert resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process) == EMPTY
    data.update(status='returned', reason=None)
    with pytest.raises(ValueError, match='complete successful original CSV'):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process)


@pytest.mark.parametrize('code', [None, -9, 1])
def test_failed_or_interrupted_query_retains_bytes_but_no_known_values(tmp_path, code):
    data, view, process = snapshot(tmp_path)
    data.update(status='unknown', exit_code=code, values=dict(EMPTY), reason='bounded query failed')
    assert resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process) == EMPTY
    data['values']['sm_mhz'] = 1980
    with pytest.raises(ValueError, match='unknown snapshot'):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process)


def test_successful_complete_csv_cannot_be_hidden_as_unknown(tmp_path):
    data, view, process = snapshot(tmp_path)
    data.update(status='unknown', values=dict(EMPTY), reason='pretend unsupported')
    with pytest.raises(ValueError, match='hide a complete successful query'):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process)


@pytest.mark.parametrize('side', ['stdout', 'stderr'])
def test_unknown_with_foreign_uuid_is_refused_even_on_error(tmp_path, side):
    data, view, process = snapshot(tmp_path)
    raw = b'query failed for GPU-ffeeddcc-bbaa-9988-7766-554433221100\n'
    data.update(status='unknown', exit_code=1, values=dict(EMPTY), reason='failure')
    data[side] = original(tmp_path, 'foreign.'+side, raw)
    with pytest.raises(ValueError, match='foreign UUID'):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process)


def test_returned_csv_normalizes_uuid_without_clock_assumption(tmp_path):
    data, view, process = snapshot(tmp_path)
    data['stdout'] = original(tmp_path, 'limits.stdout',
                              (GPU_UUID.upper()+', 1, 100000, P12\r\n').encode())
    data['values'] = {'sm_mhz': 1, 'memory_mhz': 100000, 'pstate': 'P12'}
    assert resource.validate_snapshot(data, view, 'case-0', 'before', UUID) == data['values']


def test_decimal_leading_zeroes_do_not_change_original_value(tmp_path):
    data, view, process = snapshot(tmp_path)
    data['stdout'] = original(tmp_path, 'zero-padded.stdout',
                              (GPU_UUID+', 000001980, 0000012000, P0\n').encode())
    assert resource.validate_snapshot(data, view, 'case-0', 'before', UUID) == data['values']


def test_not_started_preserves_absent_query_and_all_null_values(tmp_path):
    data, view, process = snapshot(tmp_path)
    data.update(status='not_started', exit_code=None, stdout=None, stderr=None,
                values=dict(EMPTY), reason='earlier execution failure')
    assert resource.validate_snapshot(data, view, 'case-0', 'before', UUID) == EMPTY
    for edit in ({'exit_code': 0}, {'stdout': original(tmp_path, 'unexpected', b'')},
                 {'reason': None}, {'values': {'sm_mhz': 1980, 'memory_mhz': None, 'pstate': None}}):
        with pytest.raises(ValueError, match='not-started'):
            resource.validate_snapshot(dict(data, **edit), view, 'case-0', 'before', UUID)


def test_query_original_hash_mismatch_is_refused(tmp_path):
    data, view, process = snapshot(tmp_path)
    (tmp_path/'query.stdout').write_bytes(b'replaced bytes')
    with pytest.raises(ValueError, match='changed/missing'):
        resource.validate_snapshot(data, view, 'case-0', 'before', UUID)


def test_contracts_do_not_call_external_execution_or_resource_apis(tmp_path, monkeypatch):
    b, design, h = fixture(tmp_path, synthetic=True)
    data, view, process = snapshot(tmp_path, synthetic=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('external/resource operation in a pure validator')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    import ctypes
    import fcntl
    import time
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(fcntl, 'flock', forbidden)
    monkeypatch.setattr(time, 'time_ns', forbidden)
    monkeypatch.setattr(time, 'monotonic_ns', forbidden)
    resource.validate_handoff(b, design, h, synthetic=True)
    resource.validate_snapshot(data, view, 'case-0', 'before', UUID, process, synthetic=True)
