"""Offline compiler only. Compiles cubins and a CPU-only oracle, never a CUDA host.

All subprocesses and fetched originals are recorded. Existing output directories
are refused so failed builds cannot be overwritten by a later successful attempt.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import urllib.request

from common import ref, sha, read_json, write_json
from inputs import STEPS, FAMILIES, expected, make_inputs, input_rows
from static_check import assess

HERE = Path(__file__).resolve().parent


def child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def build(args):
    out = args.output.resolve()
    previous = read_json(out / 'BUILD.json') if args.resume else None
    if previous:
        if previous['status'] != 'incomplete':
            raise ValueError('only an incomplete CPU build may resume; never a GPU attempt')
        suffix = len(list(out.glob('BUILD.failed-*.json')))
        write_json(out / ('BUILD.failed-' + str(suffix) + '.json'), previous)
        for name in ('kernels.cu', 'negative_kernels.cu', 'oracle.cpp', 'SOURCES.json'):
            if (out / 'sources' / name).read_bytes() != (HERE / name).read_bytes():
                raise ValueError('cannot reuse compiler output after compilation source changed')
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'logs').mkdir()
        (out / 'sources').mkdir()
        for path in sorted(HERE.iterdir()):
            if path.is_file():
                (out / 'sources' / path.name).write_bytes(path.read_bytes())
    record = previous or {'schema': 'arch-explorer.calibration-build.v1', 'status': 'incomplete',
              'gpu_queries': 0, 'gpu_executions': 0, 'native_executions': 0,
              'compiler_only': True, 'subprocesses': [], 'candidates': [], 'negative_fixtures': [],
              'toolchain': {name: ref(path) for name, path in [('nvcc', args.nvcc), ('nvdisasm', args.nvdisasm), ('cxx', args.cxx)]},
              'sources': [ref(path, out) for path in sorted((out / 'sources').iterdir())],
              'budgets': {'subprocesses_max': 24, 'subprocess_timeout_s': 120, 'subprocess_address_space_bytes': 2 * 1024**3,
                          'network_files': 3, 'network_file_bytes_max': 65536, 'no_gpu_host_built': True}}

    def run(label, command):
        argv = list(map(str, command))
        old = next((r for r in record['subprocesses'] if r['label'] == label), None)
        if old:
            if old['argv'] != argv or old.get('exit_code') != 0:
                raise ValueError('only identical successful CPU compiler/oracle commands may be reused')
            log = out / old['log']['path']
            if sha(log) != old['log']['sha256']:
                raise ValueError('previous command log changed')
            old['reused_in_resume'] = True
            return log.read_text(errors='replace')
        if len(record['subprocesses']) >= record['budgets']['subprocesses_max']:
            raise ValueError('offline process budget exceeded')
        row = {'label': label, 'argv': argv, 'timeout_s': 120}
        record['subprocesses'].append(row)
        log = out / 'logs' / (label + '.log')
        try:
            with log.open('wb') as stream:
                done = subprocess.run(row['argv'], stdout=stream, stderr=subprocess.STDOUT,
                                      env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'LC_ALL': 'C'},
                                      timeout=120, preexec_fn=child_limits)
            row.update(exit_code=done.returncode, log=ref(log, out))
            if done.returncode:
                raise ValueError('offline subprocess failed: ' + label)
        except subprocess.TimeoutExpired:
            row.update(exit_code=None, timeout=True, log=ref(log, out))
            raise
        return log.read_text(errors='replace')

    try:
        origins = read_json(HERE / 'SOURCES.json')
        upstream = out / 'upstream'
        upstream.mkdir(exist_ok=True)
        record['upstream'] = {'commit': origins['commit'], 'files': []}
        for i, row in enumerate(origins['files']):
            url = origins['repository'].replace('github.com', 'raw.githubusercontent.com') + '/' + origins['commit'] + '/' + row['path']
            path = upstream / (str(i) + '-' + Path(row['path']).name)
            if path.exists():
                raw = path.read_bytes()
            else:
                with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ArchExplorer-offline-calibration'}), timeout=30) as stream:
                    raw = stream.read(65537)
            path.write_bytes(raw)
            if len(raw) > 65536 or sha(path) != row['sha256']:
                raise ValueError('pinned public source differs: ' + row['path'])
            record['upstream']['files'].append({'url': url, 'origin_path': row['path'], **ref(path, out)})
        run('nvcc-version', [args.nvcc, '--version'])
        run('oracle-compile', [args.cxx, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', HERE / 'oracle.cpp', '-o', out / 'cpu-oracle'])
        values = make_inputs(out / 'inputs')
        record['inputs'] = input_rows(out / 'inputs')
        for steps in STEPS:
            cubin = out / ('steps-' + str(steps) + '.cubin')
            run('nvcc-' + str(steps), [args.nvcc, '-std=c++17', '-arch=sm_120a', '-O3', '--cubin', '-Xptxas=-v',
                                      '-DCAL_STEPS=' + str(steps), HERE / 'kernels.cu', '-o', cubin])
            assembly = run('nvdisasm-' + str(steps), [args.nvdisasm, '-hex', cubin])
            facts = assess(cubin.read_bytes(), assembly)
            if set(facts) != {'sim29_' + name for name in FAMILIES}:
                raise ValueError('unexpected/missing full compiled function')
            for family in FAMILIES:
                oracle = out / (family + '-' + str(steps) + '.expected.bin')
                inp = out / 'inputs' / (('next' if family == 'chase' else family) + '.bin')
                aux = out / 'inputs' / (('start' if family == 'chase' else 'constants') + '.bin')
                run('oracle-' + family + '-' + str(steps), [out / 'cpu-oracle', family, steps, inp, aux, oracle])
                if oracle.read_bytes() != expected(family, steps, values):
                    raise ValueError('independent C++ / analytic Python oracle disagreement')
                record['candidates'].append({'id': family + '-' + str(steps), 'family': family,
                    'function': 'sim29_' + family, 'steps': steps, 'partition': 'holdout' if steps == 24 else 'fit',
                    'cubin': ref(cubin, out), 'disassembly': ref(out / 'logs' / ('nvdisasm-' + str(steps) + '.log'), out),
                    'expected': ref(oracle, out), 'input': ref(inp, out), 'auxiliary': ref(aux, out),
                    'grid': [1, 1, 1], 'block': [32, 1, 1], 'facts': facts['sim29_' + family],
                    'oracle': 'independent C++ table/update loop equals analytic Python affine composition'})
        negative = out / 'negative.cubin'
        run('nvcc-negative', [args.nvcc, '-std=c++17', '-arch=sm_120a', '-O3', '--cubin', '-Xptxas=-v', HERE / 'negative_kernels.cu', '-o', negative])
        assembly = run('nvdisasm-negative', [args.nvdisasm, '-hex', negative])
        record['negative_fixtures'] = {'cubin': ref(negative, out),
            'disassembly': ref(out / 'logs' / 'nvdisasm-negative.log', out), 'facts': assess(negative.read_bytes(), assembly),
            'purpose': 'static refusal controls only; not candidates and never allowed to execute'}
        record['status'] = 'built'
        record.pop('error', None)
    except Exception as error:
        record['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        write_json(out / 'BUILD.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('nvcc', 'nvdisasm', 'cxx', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='reuse identical successful CPU commands of an incomplete build; retain its failure manifest')
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({'status': result['status'], 'candidates': len(result['candidates']),
                      'subprocesses': len(result['subprocesses']), 'gpu_executions': 0,
                      'static': {r['id']: r['facts']['static_native_subset'] for r in result['candidates']}}))


if __name__ == '__main__':
    main()
