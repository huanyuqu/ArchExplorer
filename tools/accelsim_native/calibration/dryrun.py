"""File-only calibration preregistration and replay. Never starts a subprocess.

The sole executor in this directory is build.py's offline compiler/CPU oracle.
This module reads artifacts, runs Python arithmetic, and writes a new plan.
"""
import argparse
import json
from pathlib import Path
import re
import struct

from arch_explorer.components.accelsim.capabilities import preflight_cubin
from common import checked, read_json, ref, sha, write_json
from inputs import ADDEND, FAMILIES, MULTIPLIER, STEPS, expected
from solver import seal, solve
from static_check import assess

HERE = Path(__file__).resolve().parent
BASE = 'c2d4bdc735d622a26586656e129a30765dfafe85'


def external(row):
    """Verify an explicitly listed external artifact, not a recursive scan."""
    path = Path(row['path']).resolve()
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('external identity changed: ' + str(path))
    return path


def resource_log(text):
    """Independent ptxas declaration cross-check against cubin metadata."""
    result, name = {}, None
    for line in text.splitlines():
        match = re.search(r"Compiling entry function '([^']+)' for 'sm_120a'", line)
        if match:
            name = match[1]
            result[name] = {}
        if name:
            match = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
            if match:
                result[name].update(zip(('stack_frame_bytes', 'spill_store_bytes', 'spill_load_bytes'), map(int, match.groups())))
            match = re.search(r'Used (\d+) registers, used (\d+) barriers', line)
            if match:
                result[name].update(registers_per_thread=int(match[1]), barriers=int(match[2]))
    if not result or any(set(row) != {'stack_frame_bytes', 'spill_store_bytes', 'spill_load_bytes', 'registers_per_thread', 'barriers'} for row in result.values()):
        raise ValueError('complete ptxas function resource declarations required')
    return result


def validate_bundle(root):
    root = Path(root).resolve()
    b = read_json(root / 'BUILD.json')
    if (b.get('schema') != 'arch-explorer.calibration-build.v1' or b.get('status') != 'built'
            or b.get('compiler_only') is not True or any(b.get(k) != 0 for k in ('gpu_queries', 'gpu_executions', 'native_executions'))):
        raise ValueError('successful offline-only build required')
    if len(b['subprocesses']) != 24 or b['budgets']['subprocesses_max'] != 24:
        raise ValueError('finite offline compiler/oracle process closure differs')
    processes = {r['label']: r for r in b['subprocesses']}
    labels = {'nvcc-version', 'oracle-compile', 'nvcc-negative', 'nvdisasm-negative'}
    labels |= {prefix + str(s) for s in STEPS for prefix in ('nvcc-', 'nvdisasm-')}
    labels |= {'oracle-' + f + '-' + str(s) for s in STEPS for f in FAMILIES}
    if set(processes) != labels or len(processes) != len(b['subprocesses']):
        raise ValueError('missing/extra compiler/CPU oracle command')
    for p in processes.values():
        if p['exit_code'] != 0 or p['timeout_s'] != 120:
            raise ValueError('failed/unbounded offline child')
        checked(root, p['log'])
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > 64 * 1024**2:
        raise ValueError('offline artifact memory/storage budget exceeded')
    for row in b['toolchain'].values():
        external(row)
    critical = ('kernels.cu', 'negative_kernels.cu', 'oracle.cpp', 'SOURCES.json')
    source_rows = {Path(r['path']).name: r for r in b['sources']}
    for name in critical:
        original = checked(root, source_rows[name])
        if sha(HERE / name) != sha(original):
            raise ValueError('compilation source changed after offline build: ' + name)
    origin = read_json(HERE / 'SOURCES.json')
    if b['upstream']['commit'] != origin['commit'] or len(b['upstream']['files']) != len(origin['files']):
        raise ValueError('public source pin differs')
    for original, wanted in zip(b['upstream']['files'], origin['files']):
        checked(root, original)
        if original['sha256'] != wanted['sha256'] or original['origin_path'] != wanted['path']:
            raise ValueError('pinned public source bytes differ')
    values = {name: list(struct.unpack('<' + 'I' * (row['bytes'] // 4), checked(root, row).read_bytes()))
              for name, row in b['inputs'].items()}
    registered = {'dep': [(0x9e3779b9 * (i + 1)) & 0xffffffff for i in range(32)],
                  'quad': [(0x85ebca6b * (i + 7)) & 0xffffffff for i in range(128)],
                  'constants': [MULTIPLIER, ADDEND], 'next': [(17 * i + 11) % 4096 for i in range(4096)],
                  'start': [(127 * i + 3) % 4096 for i in range(32)]}
    if values != registered:
        raise ValueError('concrete input specification differs')
    if {r['id'] for r in b['candidates']} != {f + '-' + str(s) for f in FAMILIES for s in STEPS} or len(b['candidates']) != 12:
        raise ValueError('all registered candidates required')
    compiled, cases = {}, []
    for row in b['candidates']:
        family, steps = row['family'], row['steps']
        if (row['id'] != family + '-' + str(steps) or row['function'] != 'sim29_' + family
                or row['partition'] != ('holdout' if steps == 24 else 'fit') or row['grid'] != [1, 1, 1] or row['block'] != [32, 1, 1]):
            raise ValueError('registered function/design/partition/geometry conflict')
        for field in ('cubin', 'disassembly', 'expected', 'input', 'auxiliary'):
            checked(root, row[field])
        if row['cubin']['sha256'] not in compiled:
            facts = assess((root / row['cubin']['path']).read_bytes(), (root / row['disassembly']['path']).read_text())
            resources = resource_log((root / processes['nvcc-' + str(steps)]['log']['path']).read_text())
            if set(facts) != {'sim29_' + f for f in FAMILIES} or set(resources) != set(facts):
                raise ValueError('all actual functions require static/resource preflight')
            for name, f in facts.items():
                if f['static_native_subset'] != 'eligible':
                    raise ValueError('whole compiled candidate outside the native/capture static subset')
                if any(resources[name][k] != f['resources'][k] for k in ('registers_per_thread', 'stack_frame_bytes')):
                    raise ValueError('ptxas/cubin actual resource conflict')
                if any(resources[name][k] for k in ('stack_frame_bytes', 'spill_store_bytes', 'spill_load_bytes', 'barriers')):
                    raise ValueError('unexpected spill/stack/barrier in candidate')
            argv = processes['nvcc-' + str(steps)]['argv']
            if argv[1:7] != ['-std=c++17', '-arch=sm_120a', '-O3', '--cubin', '-Xptxas=-v', '-DCAL_STEPS=' + str(steps)] or Path(argv[7]).name != 'kernels.cu':
                raise ValueError('complete offline compilation flags differ')
            compiled[row['cubin']['sha256']] = facts
        facts = compiled[row['cubin']['sha256']][row['function']]
        if facts != row['facts']:
            raise ValueError('static disassembly/resources changed after build')
        if (root / row['expected']['path']).read_bytes() != expected(family, steps, values):
            raise ValueError('independent CPU output/analytic oracle differs')
        input_name, aux_name = ('next', 'start') if family == 'chase' else (family, 'constants')
        if row['input'] != b['inputs'][input_name] or row['auxiliary'] != b['inputs'][aux_name]:
            raise ValueError('function/input binding differs')
        cases.append({k: row[k] for k in ('id', 'family', 'steps', 'partition', 'function', 'grid', 'block')} |
            {'cubin_sha256': row['cubin']['sha256'], 'text_sha256': facts['text_sha256'],
             'input_sha256': row['input']['sha256'], 'auxiliary_sha256': row['auxiliary']['sha256'],
             'output_sha256': row['expected']['sha256'], 'resources': facts['resources'],
             'flow_opcodes': facts['flow_opcode_histogram'], 'static_native_subset': 'eligible',
             'complete_text_bytes': facts['text_bytes']})
    negative = b['negative_fixtures']
    for name in ('cubin', 'disassembly'):
        checked(root, negative[name])
    neg = assess((root / negative['cubin']['path']).read_bytes(), (root / negative['disassembly']['path']).read_text())
    if neg != negative['facts'] or set(neg) != {'sim29_negative_' + f for f in ('clock', 'branch', 'barrier')}:
        raise ValueError('whole unsupported instruction controls differ')
    if any(f['static_native_subset'] != 'blocked' for f in neg.values()):
        raise ValueError('clock/taken branch/barrier must be accurately blocked')
    return b, cases


def history_check(cases, history_path):
    path = Path(history_path).resolve()
    h = read_json(path)
    loaded = {}
    for key in ('history_reference', 'static_reference'):
        row = h[key]
        target = Path(row['path'])
        if sha(target) != row['sha256']:
            raise ValueError('closed history reference differs')
        loaded[key] = read_json(target)
    bins = set(loaded['history_reference']['closed_binary_sha256']) | {h['new_binary']}
    texts = set(h['new_texts'].values())
    for row in loaded['static_reference']:
        texts.update(t['sha256'] for t in row['text_sections'])
    overlaps = [c['id'] for c in cases if c['cubin_sha256'] in bins or c['text_sha256'] in texts]
    if overlaps:
        raise ValueError('known old binary/execution bytes overlap: ' + ','.join(overlaps))
    return {'history_check': ref(path), 'closed_history': h['history_reference'], 'static_history': h['static_reference'],
            'known_binary_sha_count': len(bins), 'enumerated_old_text_sha_count': len(texts), 'overlap': overlaps,
            'unresolved_text_registry': 'SIM28 summary reports 328 old text hashes but does not enumerate them; only frozen enumerated digests compared',
            'holdout_meaning': 'preregistered step-24 timing partition, no observations yet; novelty conditional on completing prior-text registry before hardware phase',
            'old_SIM28_cases': 'all observed already; never count as new timing holdout', 'old_windows_and_leases': 'closed, never reusable'}


def make_plan(build_dir, engine, history_path):
    root = Path(build_dir).resolve()
    b, cases = validate_bundle(root)
    # Reuse the installed capability parser for the exact fixed engine and the
    # common SM120 ELF target. Every separate function/text was decoded above.
    example = root / b['candidates'][0]['cubin']['path']
    native = preflight_cubin(example, engine=engine)
    if native['isa'] != 'sm120' or any(r['kind'] != 'dynamic_trace_missing' for r in native['blockers']):
        raise ValueError('actual engine does not implement this SM120 static ISA class')
    cuda = Path(b['toolchain']['nvcc']['path']).parent.parent
    dependencies = ('bin/ptxas', 'nvvm/bin/cicc', 'bin/cudafe++',
                    'include/cuda_runtime.h', 'nvvm/libdevice/libdevice.10.bc')
    missing = [name for name in dependencies if not (cuda / name).is_file()]
    if missing:
        raise ValueError('explicit CUDA offline compilation dependency missing: ' + ','.join(missing))
    plan = {'schema': 'arch-explorer.calibration-plan.v1', 'baseline_commit': BASE,
        'status': 'offline_design_only', 'source_kind': 'new_design_from_pinned_public_microbenchmarks',
        'source_files': [ref(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.suffix in ('.py', '.cu', '.cpp', '.json', '.md')],
        'build': ref(root / 'BUILD.json'), 'cpu_oracle_binary': ref(root / 'cpu-oracle'),
        'offline_toolchain_dependencies': [ref(cuda / name) for name in dependencies],
        'engine': ref(Path(engine)), 'public_static_preflight': native,
        'cases': sorted(cases, key=lambda c: c['id']), 'history': history_check(cases, history_path),
        'native_execution': 'none; full dynamic records/masks/addresses missing',
        'current_binding': {'status': 'blocked', 'object': 'accelsim.capture_input.read_records / derive hardware binding',
            'reason': 'current capture ABI requires grid4/block128 and SIM28 cases; proposed grid1/block32 requires a separately registered capture/geometry contract',
            'no_workaround': 'retain entire SM120 instructions and true future addresses; do not relabel ISA, truncate or synthesize hardware records'},
        'budget': {'samples_per_case': 3, 'future_registered_cases_max': 12,
            'future_host_processes_max': 13, 'future_kernel_launches_max': 48, 'future_capture_launches_per_case_max': 1,
            'future_timing_launches_per_case_max': 3, 'process_timeout_s': 120, 'whole_window_timeout_s': 1800,
            'aggregate_device_buffers_bytes_max': 1048576, 'aggregate_capture_bytes_max': 16777216,
            'raw_host_memory_bytes_max': 67108864, 'on_failure': 'record and stop; no retry/replacement/new case'},
        'quality': {'max_sample_range_over_mean': '0.03', 'max_fit_relative_residual': '0.02', 'max_holdout_relative_error': '0.10',
            'meaning': 'new draft service-consistency checks; no change to historical MAPE/worst-case acceptance lines'},
        'future_protocol': [
            'New coordinator-approved window and new live lease/common flock; old SIM28 reservation/RefUnit/terminal evidence is closed.',
            'Before opening: complete history digest registry; reviewed generic capture binding for exact single-warp geometry; exact tracer/NVBit/driver/tool bytes and full static subset proof.',
            'All 12 actual function handles remain alive together. Query each function actual registers/static shared/local/barriers/maximum block and occupancy against the same live device capacities/UUID/CC. Reconcile every offline resource declaration before first allocation or launch.',
            'Design freeze persists complete sources, compiler/ELF/text/SASS/resources/input/oracle, exact engine/profile/ADF, fit/holdout identity, raw schema, sampling/order/failure budgets and metric.',
            'Capture complete fit streams once, verify masks/control/typed operands/addresses/outputs against exact static bytes; instrumented capture is never a timing observation. No deletes or generated addresses.',
            'A separately reviewed future runner may time uninstrumented fit cases in fixed registration order, with no warmup or repeated best-of. Clock/cache state must be observed or remain a confounder; always preserve failed denominators.',
            'Fit only the registered effective service observables. Persist fit raw identity, complete native scenario, direct/public predictions and a second prediction freeze before any step-24 hardware launch or observation.',
            'Open each held-out case once for capture and three fixed timing samples; compare only after the second freeze. On refusal retain all records; no residual/shape coefficients, cache reclassification, or target-based search.',
            'Close and reap own processes, release new lock/lease, notify the new coordinator; never reuse old training terminal success.'
        ],
        'raw_record_schema': ref(HERE / 'RAW-RECORD-SCHEMA.json'),
        'compiler_confounders': {family: {
            'registers_by_steps': {str(c['steps']): c['resources']['registers_per_thread'] for c in cases if c['family'] == family},
            'interpretation': 'full compiled functions vary with unroll factor; no assertion that a common physical latency/issue bottleneck is isolated'} for family in FAMILIES},
        'native_configuration_constraints': {
            'detailed_DRAM_identification_requires': {'gpgpu_simple_dram_model': 0},
            'SIM28_seed_simple_dram_model': '1; simplified fixed service bypasses detailed dram_t command counters',
            'latency_timestamps': 'dram_latency and gpgpu_l2_rop_latency are CORE ready-cycle timestamps polled by DRAM/L2 paths',
            'DRAM_MHz': 'controller clock; dram_data_command_freq_ratio is independent; no assumed DDR doubling',
            'current_parent_parameter_domain': 'L1/shared/ROP/dram_latency positive cycle integers 1..2^20; no zero unknown sentinel',
            'physical_values': None, 'profile_status': 'unknown; parent CPU synthetic parameter-consumption examples are not hardware calibration'},
        'parameter_targets': {
            'integer': {'native_key': '-trace_opcode_latency_initiation_int', 'native_unit': 'core cycles per latency/initiation interval',
                'observable_unit': 'ns per repeated round', 'equations': ['d=max(L,I)', 'q=max(L,4I)'],
                'condition': 'steady-state single-warp dependence/issue model must be separately demonstrated; intercept includes full load/store/prologue work',
                'cycle_conversion': 'cycles = ns * independently_observed_core_MHz / 1000; unknown frequency means unknown cycles'},
            'memory': {'native_keys': ['-gpgpu_l1_latency', '-gpgpu_l2_rop_latency', '-dram_latency'],
                'observable_unit': 'ns per dependent load/address round',
                'native_units': ['core-domain L1 latency queue ticks', 'core-cycle-tagged L2 ROP ingress delay', 'core-cycle-tagged DRAM scheduling delay'],
                'equation': 'chase_slope = address + dependency + cache_or_memory + queue', 'matrix': [[1, 1, 1, 1]], 'rank': 1,
                'next_discriminator': 'independently establish path hits/misses, address-only dependent baseline and queue-free state; otherwise only report combined slope'}},
        'measurements': [], 'physical_parameters': None, 'hardware_accuracy': 'blocked'}
    plan['design_sha256'] = seal(plan)
    return plan


def synthetic(plan):
    """Exactly identified synthetic exercise, not simulated GPU observations."""
    rows = []
    for c in plan['cases']:
        slope = {'dep': 6, 'quad': 12, 'chase': 20}[c['family']]
        for i in range(plan['budget']['samples_per_case']):
            rows.append({'case_id': c['id'], **{k + '_sha256': c[k + '_sha256'] for k in ('cubin', 'text', 'input', 'auxiliary', 'output')},
                'sample_index': i, 'elapsed_ns': 100 + slope * c['steps'], 'correctness': True,
                'raw_ref': 'synthetic:not-hardware'})
    return {'schema': 'arch-explorer.calibration-observations.v1', 'design_sha256': plan['design_sha256'],
            'kind': 'synthetic', 'unit': 'ns', 'samples': rows}


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = make_plan(args.build, args.engine, args.history)
        write_json(out / 'PLAN.json', plan)
        empty = {'schema': 'arch-explorer.calibration-observations.v1', 'design_sha256': plan['design_sha256'],
                 'kind': 'hardware', 'unit': 'ns', 'samples': []}
        write_json(out / 'OBSERVATIONS-empty.json', empty)
        write_json(out / 'RESULT-empty.json', solve(plan, empty))
        sample = synthetic(plan)
        write_json(out / 'SYNTHETIC-observations.json', sample)
        write_json(out / 'SYNTHETIC-result.json', solve(plan, sample))
        write_json(out / 'DRYRUN.json', {'status': 'passed', 'design_sha256': plan['design_sha256'],
            'static_candidates': len(plan['cases']), 'subprocesses_started': 0, 'native_runs': 0, 'gpu_queries': 0,
            'gpu_executions': 0, 'actual_observations': 0, 'hardware_accuracy': 'blocked'})
    except Exception as error:
        write_json(out / 'REFUSAL.json', {'status': 'refused', 'error': type(error).__name__ + ': ' + str(error),
                                         'hardware_accuracy': 'blocked', 'gpu_executions': 0})
        raise
    print(json.dumps(read_json(out / 'DRYRUN.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('build', 'engine', 'history', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
