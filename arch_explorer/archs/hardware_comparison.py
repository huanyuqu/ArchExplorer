"""Strict paired timing comparison, independent of simulator and measurement tools.

Implements arch-explorer.hardware-pairs.v1. Source admission remains a separate
review of original artifacts; a supplied digest/review flag is not authentication.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import statistics


def _mean(values):
    # Average can remain finite when the unscaled intermediate sum overflows.
    values = list(values)
    return math.fsum(x / len(values) for x in values)


def _fields(value, names, path):
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError(f'{path}: exact fields required: {names}')


def _text(value, path):
    if (not isinstance(value, str) or not value.strip()
            or value.strip().lower() in ('unknown', 'unspecified', 'todo')):
        raise ValueError(f'{path}: a known nonempty identity is required')


def _number(value, path, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{path}: a finite {"positive" if positive else "nonnegative"} number is required')


def _known(value, path):
    if value is None:
        raise ValueError(f'{path}: missing identity')
    if isinstance(value, str):
        _text(value, path)
    elif type(value) is dict:
        if not value and not path.endswith('.attributes'):
            raise ValueError(f'{path}: empty identity')
        for key, item in value.items():
            _text(key, path + '.key')
            _known(item, path + '.' + key)
    elif type(value) is list:
        if not value and not path.endswith('.shape'):
            raise ValueError(f'{path}: scalar shapes require an explicit shape: [] object')
        if path.endswith('.shape') and any(type(x) is not int or x <= 0 for x in value):
            raise ValueError(f'{path}: known positive dimensions required')
        for i, item in enumerate(value):
            _known(item, f'{path}[{i}]')
    elif type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f'{path}: non-finite identity')
    elif type(value) is not bool:
        raise ValueError(f'{path}: JSON identity required')


def _conditions(value, path):
    _fields(value, 'device software clock operation precision mapping timing', path)
    for name, keys in [('device', 'vendor sku architecture'), ('software', 'sdk driver kernel'),
                       ('clock', 'mode core_hz memory_hz'),
                       ('operation', 'op inputs outputs attributes'),
                       ('precision', 'inputs outputs accumulate rounding'),
                       ('timing', 'start end synchronization warmup statistic')]:
        _fields(value[name], keys, path + '.' + name)
    _known(value, path)
    for name in ('core_hz', 'memory_hz'):
        _number(value['clock'][name], path + '.clock.' + name, positive=True)
    for name in ('vendor', 'sku', 'architecture'):
        _text(value['device'][name], path + '.device.' + name)
    for name in ('sdk', 'driver', 'kernel'):
        _text(value['software'][name], path + '.software.' + name)
    _text(value['clock']['mode'], path + '.clock.mode')
    _text(value['operation']['op'], path + '.operation.op')
    for name in ('inputs', 'outputs', 'attributes'):
        if type(value['operation'][name]) is not dict or (name != 'attributes' and not value['operation'][name]):
            raise ValueError(f'{path}.operation.{name}: explicit object required')
    for name in ('inputs', 'outputs'):
        for tensor in value['operation'][name].values():
            shape = tensor.get('shape') if type(tensor) is dict else tensor
            if (type(shape) is not list or (not shape and type(tensor) is not dict)
                    or any(type(d) is not int or d <= 0 for d in shape)):
                raise ValueError(f'{path}.operation.{name}: complete positive shapes or explicit scalar shape objects required')
    for name in ('inputs', 'outputs'):
        if type(value['precision'][name]) is not list or not value['precision'][name]:
            raise ValueError(f'{path}.precision.{name}: nonempty dtype array required')
        for dtype in value['precision'][name]:
            _text(dtype, path + '.precision.' + name)
    for name in ('accumulate', 'rounding'):
        _text(value['precision'][name], path + '.precision.' + name)
    if type(value['mapping']) is not dict or not value['mapping']:
        raise ValueError(f'{path}.mapping: complete effective mapping object required')
    timing = value['timing']
    for name in ('start', 'end', 'synchronization'):
        _text(timing[name], path + '.timing.' + name)
    if type(timing['warmup']) is not int or timing['warmup'] < 0:
        raise ValueError(f'{path}.timing.warmup: nonnegative integer required')
    if timing['statistic'] not in ('mean', 'median', 'min', 'max'):
        raise ValueError(f'{path}.timing.statistic: unsupported statistic')


def _observation(value, role):
    _fields(value, 'kind source conditions unit value repetitions raw_values', role)
    if value['kind'] != role:
        raise ValueError(f'{role}.kind differs from its slot')
    source = value['source']
    _fields(source, 'locator sha256 license redistribution evidence_class audit', role + '.source')
    for name in ('locator', 'license', 'redistribution'):
        _text(source[name], role + '.source.' + name)
    if not isinstance(source['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', source['sha256']):
        raise ValueError(f'{role}.source.sha256: lowercase SHA256 required')
    if source['evidence_class'] not in ('raw-device-timing', 'paper-digitization', 'simulator', 'synthetic-control'):
        raise ValueError(f'{role}.source.evidence_class: unsupported class')
    if source['audit'] not in ('reviewed', 'unreviewed'):
        raise ValueError(f'{role}.source.audit: reviewed/unreviewed required')
    _conditions(value['conditions'], role + '.conditions')
    factors = {'s': 1e9, 'ms': 1e6, 'us': 1e3, 'ns': 1}
    if value['unit'] not in factors:
        raise ValueError(f'{role}.unit: s/ms/us/ns required')
    _number(value['value'], role + '.value')
    raw, repetitions = value['raw_values'], value['repetitions']
    if type(repetitions) is not int or repetitions < 1 or type(raw) is not list or len(raw) != repetitions:
        raise ValueError(f'{role}: raw_values must match positive repetitions')
    for item in raw:
        _number(item, role + '.raw_values')
    operation = {'mean': _mean, 'median': statistics.median, 'min': min, 'max': max}
    expected = operation[value['conditions']['timing']['statistic']](raw)
    if not math.isclose(value['value'], expected, rel_tol=1e-12, abs_tol=0):
        raise ValueError(f'{role}.value differs from declared raw statistic')
    result = value['value'] * factors[value['unit']]
    _number(result, role + '.value_ns')
    return result


def compare_hardware(document):
    """Return accepted errors, explicit refusals and split/class/audit aggregates.

    No fitting, inference of missing measurement identities or file I/O occurs.
    Zero measured values have null relative errors and are excluded from MAPE.
    All duplicated IDs are refused, including their first occurrence.
    """
    _fields(document, 'schema samples', 'input')
    if document['schema'] != 'arch-explorer.hardware-pairs.v1' or type(document['samples']) is not list:
        raise ValueError('arch-explorer.hardware-pairs.v1 samples array required')
    rows = document['samples']
    ids = Counter(r.get('sample_id') for r in rows if type(r) is dict and isinstance(r.get('sample_id'), str))
    accepted, rejected, groups = [], [], defaultdict(list)
    for index, row in enumerate(rows):
        try:
            _fields(row, 'sample_id split measured predicted', f'samples[{index}]')
            sid = row['sample_id']
            _text(sid, 'sample_id')
            if ids[sid] != 1:
                raise ValueError('duplicate sample_id')
            if row['split'] not in ('calibration', 'development', 'holdout'):
                raise ValueError('split: calibration/development/holdout required')
            measured, predicted = _observation(row['measured'], 'measured'), _observation(row['predicted'], 'predicted')
            m, p = row['measured'], row['predicted']
            canonical = lambda x: json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)
            if canonical(m['conditions']) != canonical(p['conditions']):
                raise ValueError('conditions mismatch: device/software/clock/shape/precision/mapping/timing must all agree')
            classes = (m['source']['evidence_class'], p['source']['evidence_class'])
            if classes == ('synthetic-control', 'synthetic-control'):
                evidence = 'synthetic-control'
            elif classes == ('raw-device-timing', 'simulator'):
                evidence = 'hardware-pair'
            else:
                raise ValueError('evidence_class mismatch: raw-device-timing/simulator or two synthetic controls required')
            audit = m['source']['audit'] + '/' + p['source']['audit']
            signed = predicted - measured
            relative = signed / abs(measured) if measured else None
            if not math.isfinite(signed) or (relative is not None and not math.isfinite(relative)):
                raise ValueError('non-finite error after unit conversion')
            record = {'sample_id': sid, 'split': row['split'], 'evidence_class': evidence, 'audit': audit,
                'measured_ns': measured, 'predicted_ns': predicted, 'signed_error_ns': signed,
                'absolute_error_ns': abs(signed), 'signed_relative_error': relative,
                'absolute_relative_error': abs(relative) if relative is not None else None,
                'zero_reference': measured == 0,
                'conditions_sha256': hashlib.sha256(canonical(m['conditions']).encode()).hexdigest(),
                'sources': {'measured': dict(m['source']), 'predicted': dict(p['source'])}}
            accepted.append(record)
            groups[(row['split'], evidence, audit)].append(record)
        except (ValueError, TypeError, OverflowError) as exc:
            rejected.append({'index': index, 'sample_id': row.get('sample_id') if type(row) is dict else None,
                             'reason': str(exc)})
    summaries = []
    for (split, evidence, audit), records in sorted(groups.items()):
        relative = [r for r in records if not r['zero_reference']]
        worst_abs = max(records, key=lambda r: r['absolute_error_ns'])
        worst_rel = max(relative, key=lambda r: r['absolute_relative_error']) if relative else None
        summaries.append({'split': split, 'evidence_class': evidence, 'audit': audit,
            'count': len(records), 'zero_reference_count': len(records) - len(relative),
            'relative_denominator_count': len(relative),
            'mean_signed_error_ns': _mean(r['signed_error_ns'] for r in records),
            'mean_absolute_error_ns': _mean(r['absolute_error_ns'] for r in records),
            'mape': _mean(r['absolute_relative_error'] for r in relative) if relative else None,
            'worst_absolute_sample_id': worst_abs['sample_id'],
            'worst_relative_sample_id': worst_rel['sample_id'] if worst_rel else None})
    return {'schema': 'arch-explorer.hardware-comparison.v1', 'accepted': accepted,
        'rejected': rejected, 'groups': summaries, 'all_samples_accepted': not rejected,
        'source_admission': 'Source digests and audit flags are supplied metadata; verify original artifacts separately.',
        'fitting_performed': False}
