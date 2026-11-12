# Concrete Accel-Sim implementation; owned by the component provider.
"""Read-only resource and snapshot contracts for finite state measurements.

These validators check supplied records and their original bytes. They neither
acquire resources nor establish the authenticity of an external resource owner.
The controller must validate the frozen design, installation and actual handoff
before constructing the snapshot window/process views accepted here.
"""
import csv
import json
import re
from pathlib import Path

from arch_explorer.components.accelsim.measurement_state import sha_plan, stateful, validate_budget, validate_plan
from arch_explorer.components.accelsim.microcontract import budget, work_item
from arch_explorer.components.accelsim.microrecords import original
from arch_explorer.components.accelsim.trace import exact, load_json, same_json


HANDOFF = 'arch-explorer.measurement-resource-handoff.v3'
SYNTHETIC_HANDOFF = 'arch-explorer.measurement-state-synthetic-handoff.v1'
_SNAPSHOT = 'arch-explorer.measurement-state-snapshot.v1'
_NS = 10**9
_UINT64 = 2**64 - 1
_REFERENCE_BYTES = 2**20
_HANDOFF_FIELDS = {
    'schema', 'work_item', 'run_id', 'resource_released', 'external_compute_empty',
    'coordinator', 'resource_owner', 'coordination_evidence', 'issued_realtime_ns',
    'lock_file', 'absolute_deadline_realtime_ns', 'close_reserve_s',
    'registry_sha256', 'plan_sha256', 'budget', 'design_sha256',
}
_SNAPSHOT_FIELDS = {
    'schema', 'kind', 'no_resource_authority', 'run_id', 'case', 'position',
    'uuid_hex', 'argv', 'status', 'exit_code', 'started_realtime_ns',
    'ended_realtime_ns', 'started_monotonic_ns', 'ended_monotonic_ns',
    'stdout', 'stderr', 'values', 'reason',
}
_UUID = re.compile(r'[0-9a-f]{32}')
_RAW_UUID = re.compile(
    r'(?<![0-9a-z])(?:GPU-)?(?:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}'
    r'|[0-9a-f]{32})(?![0-9a-z])', re.IGNORECASE)


def _require(condition, message):
    if not condition:
        raise ValueError('measurement state resource: ' + message)


def _integer(value, low, high, name):
    _require(type(value) is int and low <= value <= high,
             'bounded integer required: ' + name)
    return value


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _synthetic_marker(value):
    if isinstance(value, dict):
        return (value.get('kind') == 'synthetic'
                or value.get('no_resource_authority') is True
                or value.get('real_world_resource_authority') is False
                or any(_synthetic_marker(v) for v in value.values()))
    if isinstance(value, list):
        return any(_synthetic_marker(v) for v in value)
    return False


def _coordination(references, run_id, synthetic):
    _require(isinstance(references, list) and bool(references),
             'coordination originals required')
    for reference in references:
        path = original(reference, maximum=_REFERENCE_BYTES)
        try:
            record, _ = load_json(path)
        except (json.JSONDecodeError, UnicodeError):
            record = None
        if synthetic:
            _require(isinstance(record, dict) and record.get('kind') == 'synthetic'
                     and record.get('no_resource_authority') is True
                     and record.get('run_id') == run_id,
                     'synthetic coordination must be an explicitly marked current-run record')
        else:
            _require(not _synthetic_marker(record),
                     'synthetic coordination bytes cannot authorize hardware')
            if isinstance(record, dict) and 'run_id' in record:
                _require(record['run_id'] == run_id, 'foreign coordination run')


def _role_budget(bundle):
    _require(stateful(bundle), 'state registry v3 required')
    plan = bundle.document['measurement_state']
    validate_plan(plan, bundle.cases)
    values = validate_budget(bundle)
    n = len(bundle.cases)
    _require(1 <= n <= 8 and values['host_processes'] <= 9,
             'at most eight cases and nine measurement hosts')
    for row in plan['cases']:
        _require((row['diagnostic'], row['preparation'], row['formal']) == (1, 2, 3),
                 'each case requires diagnostic 1, preparation 2, formal 3')
    # validate_budget already binds every total to the unique ordered plan.
    _require(values['uninstrumented_launches'] == 6*n <= 48
             and values['application_launches'] == 7*n <= 56,
             'full finite role/structure launch denominator differs')
    return budget(bundle)


def validate_handoff(bundle, design, data, *, synthetic=False):
    """Validate an original handoff without clocks, locks or device operations.

    Hardware ownership/authenticity and design seals belong to the controller.
    Synthetic coordination has a separate schema and requires marked originals;
    changing only a handoff schema/label cannot promote those bytes to hardware.
    """
    _require(type(synthetic) is bool, 'explicit boundary mode required')
    from arch_explorer.components.accelsim.state_idle import ADMISSION, validate_admission
    if data.get('schema') == ADMISSION:
        _require(not synthetic, 'idle admission cannot promote a synthetic boundary')
        return validate_admission(bundle, design, data)
    exact(data, _HANDOFF_FIELDS | ({'no_resource_authority'} if synthetic else set()),
          'role-bound resource handoff')
    values = _role_budget(bundle)
    run_id = design.get('run_id')
    _require(_text(run_id) and _sha(bundle.sha256)
             and _sha(design.get('design_sha256')),
             'frozen run, registry and design identities required')
    _require(design.get('plan_sha256') == sha_plan(bundle)
             and same_json(design.get('budget'), values),
             'frozen design plan/budget differs from registry')
    expected_schema = SYNTHETIC_HANDOFF if synthetic else HANDOFF
    coordinator = 'synthetic-coordinator' if synthetic else 'resident_2e1684fdfc1b'
    _require(data['schema'] == expected_schema
             and data['work_item'] == work_item(bundle)
             and data['run_id'] == run_id
             and data['registry_sha256'] == bundle.sha256
             and data['plan_sha256'] == sha_plan(bundle)
             and data['design_sha256'] == design['design_sha256']
             and same_json(data['budget'], values),
             'handoff run/registry/plan/budget/design differs')
    _require(data['resource_released'] is True and data['external_compute_empty'] is True
             and data['coordinator'] == coordinator and _text(data['resource_owner']),
             'actual release, compute-empty and coordinator/owner required')
    if synthetic:
        _require(data['no_resource_authority'] is True
                 and data['resource_owner'] == 'synthetic-no-device-owner',
                 'synthetic handoff grants no resource authority')
    else:
        _require(data['resource_owner'] not in ('synthetic-no-device-owner', 'synthetic-coordinator'),
                 'synthetic owner cannot authorize hardware')
    _require(isinstance(data['lock_file'], str) and Path(data['lock_file']).is_absolute(),
             'absolute common lock path required')
    created = _integer(design.get('created_realtime_ns'), 1, _UINT64, 'design creation')
    issued = _integer(data['issued_realtime_ns'], 1, _UINT64, 'owner issue')
    end = _integer(data['absolute_deadline_realtime_ns'], 1, _UINT64, 'absolute deadline')
    _require(type(data['close_reserve_s']) is int and data['close_reserve_s'] == 60,
             'fixed sixty-second cleanup reserve required')
    _require(created < issued < end <= issued + values['total_window_s']*_NS,
             'absolute handoff time cannot extend the frozen window')
    _coordination(data['coordination_evidence'], run_id, synthetic)
    return data


def handoff_evidence(data):
    """Return admission originals without inventing an owner/coordinator."""
    from arch_explorer.components.accelsim.state_idle import ADMISSION
    if data.get('schema') == ADMISSION:
        return [data['request'], data['observation'], data['claim']]
    return data['coordination_evidence']


def snapshot_policy():
    """Return a fresh immutable-in-meaning policy; never execute the query."""
    return {
        'schema': 'arch-explorer.measurement-state-snapshot-policy.v1',
        'query': ['/usr/bin/nvidia-smi', '--id=0',
                  '--query-gpu=uuid,clocks.current.sm,clocks.current.memory,pstate',
                  '--format=csv,noheader,nounits'],
        'timeout_s': 5,
        'per_host': 2,
        'positions': ['before', 'after'],
        'clock_interpretation': 'instantaneous; not average or steady/cache evidence',
    }


def _uuid(value):
    text = value.strip()
    if text[:4].lower() == 'gpu-':
        text = text[4:]
    text = text.replace('-', '').lower()
    return text if _UUID.fullmatch(text) else None


def _raw_values(stdout, stderr, uuid_hex):
    # Refuse a recognizable foreign UUID even on a failed/unsupported query.
    for raw in (stdout, stderr):
        for match in _RAW_UUID.finditer(raw.decode('utf-8', errors='replace')):
            _require(_uuid(match.group()) == uuid_hex, 'raw query names a foreign UUID')
    try:
        lines = stdout.decode('utf-8').splitlines()
        if len(lines) != 1:
            return None
        rows = list(csv.reader(lines, strict=True, skipinitialspace=True))
    except (UnicodeError, csv.Error):
        return None
    if len(rows) != 1 or len(rows[0]) != 4:
        return None
    raw_uuid, sm, memory, pstate = (x.strip() for x in rows[0])
    observed_uuid = _uuid(raw_uuid)
    if observed_uuid is not None:
        _require(observed_uuid == uuid_hex, 'raw query names a foreign UUID')
    if (observed_uuid is None or not re.fullmatch(r'[0-9]+', sm)
            or not re.fullmatch(r'[0-9]+', memory)
            or not re.fullmatch(r'P(?:0|[1-9][0-9]*)', pstate)):
        return None
    # Bound significant digits before int conversion, retaining decimal leading
    # zero support without assuming a physical clock or parsing giant integers.
    sm, memory = sm.lstrip('0'), memory.lstrip('0')
    if (not sm or not memory or len(sm) > 6 or len(memory) > 6
            or not 1 <= int(sm) <= 100000 or not 1 <= int(memory) <= 100000):
        return None
    return {'sm_mhz': int(sm), 'memory_mhz': int(memory), 'pstate': pstate}


def validate_snapshot(data, design, case_id, position, uuid_hex, process=None, *, synthetic=False):
    """Read original query outputs against a validated window/process view.

    ``design`` adds the validated handoff's absolute_deadline_realtime_ns and
    close_reserve_s to the frozen design without resealing or modifying it.
    ``process`` is an optional normalized run_id/case/start/end receipt view.
    """
    _require(type(synthetic) is bool, 'explicit boundary mode required')
    exact(data, _SNAPSHOT_FIELDS, 'bounded device state snapshot')
    exact(data['values'], {'sm_mhz', 'memory_mhz', 'pstate'}, 'snapshot values')
    _require(_text(case_id) and position in ('before', 'after')
             and isinstance(uuid_hex, str) and _UUID.fullmatch(uuid_hex) is not None,
             'registered case/position and preflight UUID required')
    _require(_text(design.get('run_id')) and data['schema'] == _SNAPSHOT
             and data['kind'] == ('synthetic' if synthetic else 'hardware')
             and data['no_resource_authority'] is synthetic
             and data['run_id'] == design['run_id'] and data['case'] == case_id
             and data['position'] == position and data['uuid_hex'] == uuid_hex,
             'snapshot boundary/run/case/position/UUID differs')
    policy = snapshot_policy()
    _require(data['argv'] == policy['query'], 'snapshot query differs from frozen policy')
    _require(data['status'] in ('returned', 'unknown', 'not_started'), 'snapshot status differs')
    created = _integer(design.get('created_realtime_ns'), 1, _UINT64, 'design creation')
    end = _integer(design.get('absolute_deadline_realtime_ns'), 1, _UINT64, 'window deadline')
    _require(type(design.get('close_reserve_s')) is int and design['close_reserve_s'] == 60,
             'validated fixed cleanup reserve required')
    start_rt = _integer(data['started_realtime_ns'], 1, _UINT64, 'query realtime start')
    end_rt = _integer(data['ended_realtime_ns'], 1, _UINT64, 'query realtime end')
    start_mt = _integer(data['started_monotonic_ns'], 1, _UINT64, 'query monotonic start')
    end_mt = _integer(data['ended_monotonic_ns'], 1, _UINT64, 'query monotonic end')
    _require(created < start_rt <= end_rt < end - 60*_NS,
             'snapshot outside preregistered work interval')
    _require(0 <= end_mt - start_mt <= (policy['timeout_s'] + 1)*_NS,
             'snapshot exceeded query/cleanup time bound')
    if process is not None:
        _require(isinstance(process, dict) and process.get('run_id') == design['run_id']
                 and process.get('case') == case_id, 'foreign snapshot process scope')
        begin = _integer(process.get('started_realtime_ns'), 1, _UINT64, 'process start')
        finish = _integer(process.get('ended_realtime_ns'), 1, _UINT64, 'process end')
        _require(begin <= finish, 'process time reversed')
        _require(end_rt <= begin if position == 'before' else start_rt >= finish,
                 'snapshot overlaps the measured process')
    values = data['values']
    _require(all(type(values[k]) is int and 1 <= values[k] <= 100000
                 or values[k] is None for k in ('sm_mhz', 'memory_mhz'))
             and (values['pstate'] is None or isinstance(values['pstate'], str)
                  and re.fullmatch(r'P(?:0|[1-9][0-9]*)', values['pstate'])),
             'snapshot values have invalid types or ranges')
    if data['status'] == 'not_started':
        _require(data['exit_code'] is None and data['stdout'] is None and data['stderr'] is None
                 and all(x is None for x in values.values()) and _text(data['reason']),
                 'not-started snapshot cannot contain query results')
        return dict(values)
    if data['exit_code'] is not None:
        _integer(data['exit_code'], -2**31, 2**31-1, 'query exit code')
    stdout = original(data['stdout'], maximum=_REFERENCE_BYTES).read_bytes()
    stderr = original(data['stderr'], maximum=_REFERENCE_BYTES).read_bytes()
    parsed = _raw_values(stdout, stderr, uuid_hex)
    if data['status'] == 'returned':
        _require(type(data['exit_code']) is int and data['exit_code'] == 0
                 and parsed is not None and same_json(values, parsed) and data['reason'] is None,
                 'returned snapshot differs from complete successful original CSV')
    else:
        _require(all(x is None for x in values.values()) and _text(data['reason'])
                 and not (data['exit_code'] == 0 and parsed is not None),
                 'unknown snapshot cannot invent values or hide a complete successful query')
    return dict(values)
