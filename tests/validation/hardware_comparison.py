"""Independent strict pairing and error arithmetic; no simulator imports.

The caller must audit provenance. A self-declared SHA or evidence kind is not
an authentication, device measurement, or permission to redistribute data.
"""
import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

SCHEMA = 'arch-explorer.hardware-pairs.v1'
UNITS = {'s': 1e9, 'ms': 1e6, 'us': 1e3, 'ns': 1.0}


def exact_keys(value, keys, where):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(where + ': expected fields ' + ','.join(sorted(keys)))


def known(value, where):
    if value == [] and where.endswith('.shape'):
        return  # Explicit scalar shape; an unknown shape remains null.
    if value is None or value == '' or value == [] or value == {}:
        raise ValueError(where + ': missing identity')
    if isinstance(value, str) and value.lower() in {'unknown', 'unspecified', 'todo'}:
        raise ValueError(where + ': unknown identity')
    if isinstance(value, dict):
        for k, v in value.items():
            known(v, where + '.' + str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            known(v, where + '.' + str(i))
    elif not isinstance(value, (str, bool, int, float)):
        raise ValueError(where + ': unsupported identity value')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(where + ': nonfinite identity')


def number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(where + ': expected finite nonnegative number')
    return value


def observation(record, kind):
    exact_keys(record, {'kind', 'source', 'conditions', 'unit', 'value', 'repetitions', 'raw_values'}, kind)
    if record['kind'] != kind:
        raise ValueError(kind + ': kind mismatch')
    source = record['source']
    exact_keys(source, {'locator', 'sha256', 'license', 'redistribution', 'evidence_class', 'audit'}, kind + '.source')
    known(source, kind + '.source')
    if not isinstance(source['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', source['sha256']):
        raise ValueError(kind + ': invalid source SHA256')
    if source['evidence_class'] not in {'raw-device-timing', 'paper-digitization', 'simulator', 'synthetic-control'}:
        raise ValueError(kind + ': invalid evidence class')
    if source['audit'] not in {'reviewed', 'unreviewed'}:
        raise ValueError(kind + ': invalid audit status')
    c = record['conditions']
    exact_keys(c, {'device', 'software', 'clock', 'operation', 'precision', 'mapping', 'timing'}, kind + '.conditions')
    exact_keys(c['device'], {'vendor', 'sku', 'architecture'}, 'device')
    exact_keys(c['software'], {'sdk', 'driver', 'kernel'}, 'software')
    exact_keys(c['clock'], {'mode', 'core_hz', 'memory_hz'}, 'clock')
    exact_keys(c['operation'], {'op', 'inputs', 'outputs', 'attributes'}, 'operation')
    exact_keys(c['precision'], {'inputs', 'outputs', 'accumulate', 'rounding'}, 'precision')
    exact_keys(c['timing'], {'start', 'end', 'synchronization', 'warmup', 'statistic'}, 'timing')
    if type(c['timing']['warmup']) is not int or c['timing']['warmup'] < 0:
        raise ValueError('timing.warmup: expected nonnegative integer')
    # An empty attributes object means a genuinely attribute-free operation.
    attrs = c['operation']['attributes']
    if not isinstance(attrs, dict):
        raise ValueError('operation.attributes: expected object')
    for key in c:
        if key == 'operation':
            known({k: v for k, v in c[key].items() if k != 'attributes'}, 'operation')
            for k, v in attrs.items(): known(v, 'operation.attributes.' + k)
        else:
            known(c[key], key)
    for k in ('core_hz', 'memory_hz'):
        if number(c['clock'][k], 'clock.' + k) == 0:
            raise ValueError('clock.' + k + ': clock must be positive')
    if record['unit'] not in UNITS:
        raise ValueError(kind + ': unsupported/missing time unit')
    number(record['value'], kind + '.value')
    count = record['repetitions']
    if type(count) is not int or count < 1:
        raise ValueError(kind + ': repetitions must be positive integer')
    raw = record['raw_values']
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError(kind + ': raw_values/repetitions mismatch')
    for v in raw: number(v, kind + '.raw_values')
    funcs = {'mean': statistics.mean, 'median': statistics.median, 'min': min, 'max': max}
    stat = c['timing']['statistic']
    if stat not in funcs:
        raise ValueError(kind + ': unsupported timing statistic')
    expected = funcs[stat](raw)
    if not math.isclose(expected, record['value'], rel_tol=1e-12, abs_tol=0):
        raise ValueError(kind + ': declared statistic differs from raw_values')
    return record['value'] * UNITS[record['unit']]


def compare_pair(pair):
    exact_keys(pair, {'sample_id', 'split', 'measured', 'predicted'}, 'sample')
    known(pair['sample_id'], 'sample_id')
    if pair['split'] not in {'calibration', 'development', 'holdout'}:
        raise ValueError('sample: invalid split')
    m, p = pair['measured'], pair['predicted']
    measured = observation(m, 'measured')
    predicted = observation(p, 'predicted')
    # JSON encoding avoids Python's True == 1 matching different identities.
    if json.dumps(m['conditions'], sort_keys=True) != json.dumps(p['conditions'], sort_keys=True):
        raise ValueError('sample: prediction/measurement conditions differ')
    mc, pc = m['source']['evidence_class'], p['source']['evidence_class']
    if mc not in {'raw-device-timing', 'synthetic-control'} or pc not in {'simulator', 'synthetic-control'}:
        raise ValueError('sample: evidence classes cannot establish this pair')
    if (mc == 'synthetic-control') != (pc == 'synthetic-control'):
        raise ValueError('sample: synthetic evidence must remain separate')
    delta = predicted - measured
    return dict(sample_id=pair['sample_id'], split=pair['split'], measured_ns=measured,
                predicted_ns=predicted, signed_error_ns=delta, absolute_error_ns=abs(delta),
                signed_relative_error=None if measured == 0 else delta / abs(measured),
                absolute_relative_error=None if measured == 0 else abs(delta) / abs(measured),
                zero_reference=measured == 0,
                evidence_class='synthetic-control' if mc == 'synthetic-control' else 'hardware',
                reviewed=m['source']['audit'] == p['source']['audit'] == 'reviewed')


def compare(document):
    exact_keys(document, {'schema', 'samples'}, 'document')
    if document['schema'] != SCHEMA or not isinstance(document['samples'], list):
        raise ValueError('document: unsupported schema/samples')
    rows, rejected = [], []
    counts=Counter(p.get('sample_id') for p in document['samples']
                   if isinstance(p,dict) and isinstance(p.get('sample_id'),str))
    for pair in document['samples']:
        sid = pair.get('sample_id') if isinstance(pair, dict) else None
        try:
            if not isinstance(sid, str) or counts[sid]>1:
                raise ValueError('sample: missing/duplicate sample_id')
            rows.append(compare_pair(pair))
        except ValueError as e:
            rejected.append(dict(sample_id=sid, reason=str(e)))
    groups = []
    for key in sorted({(r['split'], r['evidence_class'], r['reviewed']) for r in rows}):
        group = [r for r in rows if (r['split'], r['evidence_class'], r['reviewed']) == key]
        nonzero = [r for r in group if not r['zero_reference']]
        worst_abs = max(group, key=lambda r: r['absolute_error_ns'])
        worst_rel = max(nonzero, key=lambda r: r['absolute_relative_error']) if nonzero else None
        groups.append(dict(split=key[0], evidence_class=key[1], reviewed=key[2], count=len(group),
                           zero_reference_count=len(group)-len(nonzero), relative_error_count=len(nonzero),
                           mape_pct=None if not nonzero else 100*statistics.mean(r['absolute_relative_error'] for r in nonzero),
                           worst_absolute_sample_id=worst_abs['sample_id'],
                           worst_relative_sample_id=None if worst_rel is None else worst_rel['sample_id']))
    return dict(schema=SCHEMA + '.result', samples=rows, rejected=rejected, groups=groups,
                provenance_authenticated=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output')
    a = p.parse_args()
    result = compare(json.loads(Path(a.input).read_text()))
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return bool(result['rejected'])


if __name__ == '__main__':
    raise SystemExit(main())
