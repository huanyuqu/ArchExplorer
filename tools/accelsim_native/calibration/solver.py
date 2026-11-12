"""Exact CPU service-observable fit, conditional bounds, and held-out check.

No profile mutation or hardware invocation. Unknown clocks and mixed memory
paths stay unknown, and synthetic samples can never become physical evidence.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from common import read_json, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(plan):
    return digest({k: v for k, v in plan.items() if k != 'design_sha256'})


def number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('numeric observation required')
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError('finite observation required') from None


def affine(points):
    """Closed normal equations for an intercept and one slope, rank checked."""
    n = len(points)
    sx = sum(x for x, y in points); sy = sum(y for x, y in points)
    xx = sum(x * x for x, y in points); xy = sum(x * y for x, y in points)
    det = n * xx - sx * sx
    if n < 2 or not det:
        raise ValueError('rank-deficient steps; cannot separate intercept from slope')
    slope = (n * xy - sx * sy) / det
    intercept = (sy - slope * sx) / n
    if slope <= 0 or intercept < 0:
        raise ValueError('nonpositive service slope or negative fixed overhead')
    return intercept, slope


def fraction(value):
    return str(value.numerator) + '/' + str(value.denominator)


def max_bounds(dependent, quadruple):
    """Solve d=max(L,I), q=max(L,4I) on positive intervals.

    This is a conditional scheduling hypothesis, never an independently proved
    physical decomposition. In ns it also cannot identify native cycle values.
    """
    if not 0 < dependent <= quadruple <= 4 * dependent:
        return {'status': 'refused', 'reason': 'observables contradict max(L,I), max(L,4I)'}
    if dependent == quadruple:
        return {'status': 'rank_deficient', 'L_ns': {'point': fraction(dependent)},
                'I_ns': {'lower_open': '0/1', 'upper': fraction(dependent / 4)}}
    if quadruple == 4 * dependent:
        return {'status': 'rank_deficient', 'I_ns': {'point': fraction(dependent)},
                'L_ns': {'lower_open': '0/1', 'upper': fraction(dependent)}}
    return {'status': 'unique_under_hypothesis', 'L_ns': {'point': fraction(dependent)},
            'I_ns': {'point': fraction(quadruple / 4)}}


def solve(plan, observation):
    if plan.get('schema') != 'arch-explorer.calibration-plan.v1' or seal(plan) != plan.get('design_sha256'):
        raise ValueError('plan identity changed after design freeze')
    if observation.get('schema') != 'arch-explorer.calibration-observations.v1':
        raise ValueError('observation schema')
    if observation.get('design_sha256') != plan['design_sha256']:
        raise ValueError('observation/design identity differs')
    kind = observation.get('kind')
    if kind not in ('synthetic', 'hardware') or observation.get('unit') != 'ns':
        raise ValueError('explicit synthetic/hardware class and elapsed ns required')
    if not observation.get('samples'):
        return {'status': 'blocked', 'reason': 'no observations', 'physical_parameters': None,
                'hardware_accuracy': 'blocked', 'kind': kind}
    if kind == 'hardware':
        raise ValueError('SIM29 protocol is not authorized/executed; hardware samples require a future reviewed executor/raw-evidence validator')
    cases = {row['id']: row for row in plan['cases']}
    samples = {}
    required = ('case_id', 'cubin_sha256', 'text_sha256', 'input_sha256', 'auxiliary_sha256',
                'output_sha256', 'sample_index', 'elapsed_ns', 'correctness', 'raw_ref')
    for row in observation['samples']:
        if set(row) != set(required) or row['case_id'] not in cases:
            raise ValueError('complete registered sample fields required')
        c = cases[row['case_id']]
        for name in ('cubin', 'text', 'input', 'auxiliary', 'output'):
            if row[name + '_sha256'] != c[name + '_sha256']:
                raise ValueError('sample compilation/input/output identity conflict')
        if row['correctness'] is not True or row['raw_ref'] != 'synthetic:not-hardware':
            raise ValueError('synthetic result must carry successful oracle and explicit raw marker')
        idx = row['sample_index']
        if type(idx) is not int or not 0 <= idx < plan['budget']['samples_per_case']:
            raise ValueError('sample budget/index')
        key = (row['case_id'], idx)
        if key in samples:
            raise ValueError('duplicate sample; no replacement or best-of selection')
        y = number(row['elapsed_ns'])
        if y <= 0:
            raise ValueError('nonpositive elapsed observation')
        samples[key] = y
    count = plan['budget']['samples_per_case']
    if set(samples) != {(c, i) for c in cases for i in range(count)}:
        raise ValueError('all fit/holdout cases and failed/unstarted denominators must be retained')
    means = {}
    for cid in cases:
        values = [samples[(cid, i)] for i in range(count)]
        mean = sum(values) / count
        if (max(values) - min(values)) / mean > number(plan['quality']['max_sample_range_over_mean']):
            raise ValueError('sample dispersion exceeds preregistered quality; no selective replacement')
        means[cid] = mean
    fit = {}
    for family in ('dep', 'quad', 'chase'):
        rows = [c for c in cases.values() if c['family'] == family and c['partition'] == 'fit']
        a, b = affine([(Fraction(c['steps']), means[c['id']]) for c in rows])
        residuals = {c['id']: abs(means[c['id']] - a - b * c['steps']) / means[c['id']] for c in rows}
        if max(residuals.values()) > number(plan['quality']['max_fit_relative_residual']):
            raise ValueError('service model inconsistent on fit observations')
        fit[family] = {'fixed_overhead_ns': fraction(a), 'effective_ns_per_round': fraction(b),
                       'fit_rank': 2, 'fit_residuals': {c: fraction(e) for c, e in residuals.items()}}
    # Predictions are functions only of FIT rows, hashed before holdout values
    # are inspected. A hardware executor must persist the same separate freeze
    # before opening any target window; no such executor is provided here.
    predictions = {c['id']: fraction(Fraction(fit[c['family']]['fixed_overhead_ns']) +
                                   Fraction(fit[c['family']]['effective_ns_per_round']) * c['steps'])
                   for c in cases.values() if c['partition'] == 'holdout'}
    holdout_errors = {cid: abs(means[cid] - Fraction(prediction)) / means[cid] for cid, prediction in predictions.items()}
    passed = all(e <= number(plan['quality']['max_holdout_relative_error']) for e in holdout_errors.values())
    return {'status': 'passed' if passed else 'refused', 'kind': 'synthetic', 'fit': fit,
            'heldout_predictions_ns': predictions, 'fit_prediction_sha256': digest({'fit': fit, 'predictions': predictions}),
            'holdout_errors': {c: fraction(e) for c, e in holdout_errors.items()},
            'conditional_integer_max_hypothesis': max_bounds(Fraction(fit['dep']['effective_ns_per_round']),
                                                           Fraction(fit['quad']['effective_ns_per_round'])),
            'hypothesis_requires': ['single-warp steady-state issue/release evidence',
                'unchanged opcode/path/register allocation and no additional overlapping bottleneck',
                'independently established core clock to convert ns into native cycles'],
            'memory_identification': {'observed': 'one effective chase interval including address calculation, dependency, cache/memory path and queues',
                'equation': 's_chase = s_address + s_dependency + s_cache_or_memory + s_queue',
                'matrix': [[1, 1, 1, 1]], 'rank': 1, 'unknowns': 4,
                'separate_native_values': None},
            'physical_parameters': None, 'hardware_accuracy': 'blocked', 'device_observations': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path); parser.add_argument('observations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new output required; retain original failures')
    try:
        result = solve(read_json(args.plan), read_json(args.observations))
    except ValueError as error:
        result = {'status': 'refused', 'reason': str(error), 'hardware_accuracy': 'blocked'}
    write_json(args.output, result)
    print(json.dumps(result))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
