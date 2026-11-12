"""CPU contract checks; no CUDA imports, GPU queries or native execution.

Usage: PYTHONPATH=. python tools/accelsim_native/calibration/test_cpu.py --plan PATH
"""
import argparse
import copy
from fractions import Fraction as Q
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from common import read_json, write_json
from dryrun import synthetic, validate_bundle
from replay import replay
from solver import affine, max_bounds, solve
from static_check import assess, instruction


class CPUChecks(unittest.TestCase):
    plan = None

    def positive(self):
        return synthetic(self.plan)

    def test_empty_means_blocked(self):
        obs = self.positive()
        obs.update(kind='hardware', samples=[])
        result = solve(self.plan, obs)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['physical_parameters'])
        self.assertEqual(result['hardware_accuracy'], 'blocked')

    def test_exact_fit_and_conditional_unique_solution(self):
        result = solve(self.plan, self.positive())
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['fit']['dep']['effective_ns_per_round'], '6/1')
        self.assertEqual(result['conditional_integer_max_hypothesis']['I_ns']['point'], '3/1')
        self.assertEqual(result['memory_identification']['rank'], 1)
        self.assertIsNone(result['physical_parameters'])
        self.assertEqual(result['device_observations'], 0)

    def test_piecewise_bounds_and_contradiction(self):
        self.assertEqual(max_bounds(Q(4), Q(4))['I_ns']['upper'], '1/1')
        self.assertEqual(max_bounds(Q(4), Q(16))['L_ns']['upper'], '4/1')
        self.assertEqual(max_bounds(Q(4), Q(8))['status'], 'unique_under_hypothesis')
        self.assertEqual(max_bounds(Q(4), Q(17))['status'], 'refused')
        self.assertEqual(max_bounds(Q(4), Q(3))['status'], 'refused')

    def test_affine_rank_and_nonphysical_fit(self):
        with self.assertRaisesRegex(ValueError, 'rank-deficient'):
            affine([(Q(8), Q(20)), (Q(8), Q(30))])
        with self.assertRaises(ValueError):
            affine([(Q(8), Q(20)), (Q(16), Q(10))])
        with self.assertRaises(ValueError):
            affine([(Q(8), Q(2)), (Q(16), Q(20))])

    def test_holdout_cannot_change_fit_or_frozen_predictions(self):
        obs = self.positive()
        before = solve(self.plan, obs)
        for row in obs['samples']:
            if row['case_id'] == 'dep-24':
                row['elapsed_ns'] *= 2
        after = solve(self.plan, obs)
        self.assertEqual(before['fit_prediction_sha256'], after['fit_prediction_sha256'])
        self.assertEqual(before['fit'], after['fit'])
        self.assertEqual(after['status'], 'refused')

    def test_no_missing_duplicate_or_incorrect_samples(self):
        for change in ('missing', 'duplicate', 'identity', 'incorrect', 'hardware', 'nan', 'infinite', 'negative'):
            with self.subTest(change=change):
                obs = self.positive()
                if change == 'missing': obs['samples'].pop()
                elif change == 'duplicate': obs['samples'].append(copy.deepcopy(obs['samples'][0]))
                elif change == 'identity': obs['samples'][0]['text_sha256'] = '0' * 64
                elif change == 'incorrect': obs['samples'][0]['correctness'] = False
                elif change == 'hardware': obs['kind'] = 'hardware'
                elif change == 'nan': obs['samples'][0]['elapsed_ns'] = float('nan')
                elif change == 'infinite': obs['samples'][0]['elapsed_ns'] = float('inf')
                else: obs['samples'][0]['elapsed_ns'] = -1
                with self.assertRaises(ValueError): solve(self.plan, obs)

    def test_quality_failure_keeps_all_denominators(self):
        obs = self.positive()
        obs['samples'][0]['elapsed_ns'] *= 2
        with self.assertRaisesRegex(ValueError, 'dispersion'):
            solve(self.plan, obs)

    def test_sealed_input_and_plan_identity(self):
        obs = self.positive()
        obs['design_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'design identity'):
            solve(self.plan, obs)
        p = copy.deepcopy(self.plan)
        p['cases'][0]['steps'] += 1
        with self.assertRaisesRegex(ValueError, 'design freeze'):
            solve(p, self.positive())

    def test_real_cubin_sass_bytes_and_negative_whole_functions(self):
        root = Path(self.plan['build']['path']).parent
        b, cases = validate_bundle(root)
        self.assertEqual(len(cases), 12)
        n = b['negative_fixtures']
        facts = assess((root / n['cubin']['path']).read_bytes(), (root / n['disassembly']['path']).read_text())
        for name, fragment in [('clock', 'clock'), ('barrier', 'BAR.'), ('branch', 'branch')]:
            row = facts['sim29_negative_' + name]
            self.assertEqual(row['static_native_subset'], 'blocked')
            self.assertTrue(any(fragment in r['reason'] for r in row['blockers']))
        row = b['candidates'][0]
        cubin = (root / row['cubin']['path']).read_bytes()
        assembly = (root / row['disassembly']['path']).read_text()
        # Alter a real encoded instruction in its .text, while keeping the
        # recorded disassembler bytes. Every emitted instruction is checked.
        from arch_explorer.components.accelsim.capture_input import elf_texts
        body = elf_texts(cubin)[row['function']]
        offset = cubin.index(body)
        altered = bytearray(cubin)
        altered[offset + 1] ^= 1
        with self.assertRaisesRegex(ValueError, 'byte mismatch'):
            assess(bytes(altered), assembly)

    def test_typed_descriptor_is_not_constant_bank_and_unsupported_stays_blocked(self):
        self.assertEqual(instruction('LDG.E R2, desc[UR4][R2.64]', 12)[0], 'LDG.E')
        for text in ('LDG.E.STRONG.SM R2, desc[UR4][R2.64]', 'CS2R.32 R2, SR_CLOCKLO',
                     'LDC R2, c[0x3][0x10]', 'S2R R2, SR_SMID', 'MOV R15, R0'):
            with self.subTest(text=text), self.assertRaises(ValueError): instruction(text, 12)

    def test_file_only_full_replay_starts_no_process(self):
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('dry-run must never start a process')):
            result = replay(self.plan, self.positive())
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(self.plan['current_binding']['status'], 'blocked')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new private result directory')
    args = parser.parse_args()
    CPUChecks.plan = read_json(args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'checks.log').open('w') as stream:
        result = unittest.TextTestRunner(verbosity=2, stream=stream).run(unittest.defaultTestLoader.loadTestsFromTestCase(CPUChecks))
    report = {'status': 'passed' if result.wasSuccessful() else 'failed',
        'failures': [str(t) for t, _ in result.failures], 'errors': [str(t) for t, _ in result.errors],
        'gpu_queries': 0, 'gpu_executions': 0, 'native_runs': 0, 'hardware_accuracy': 'blocked'}
    write_json(args.output / 'RESULT.json', report)
    print(report)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
