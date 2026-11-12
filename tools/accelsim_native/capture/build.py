"""Offline-only build of SIM28 CUDA input, host, and NVBit tool.

Never invokes the resulting host, CUDA driver, device queries, or GPU kernels.
Only the explicit compiler/disassembler/nm commands below are allowed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
NVBIT_ASSET_SHA = '9c383591bc62be02dd0bcba51f98320525817de67d30d997ec3d618adcd3d618'


def digest(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--nvbit', type=Path, required=True)
    parser.add_argument('--nvbit-archive', type=Path, required=True)
    parser.add_argument('--nvdisasm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', choices=('tool-link','host','nvdisasm'),
        help='Resume a failed stage after changing only that stage/its successors; retained earlier commands must match')
    args = parser.parse_args()
    cuda, nvbit, out = (p.resolve() for p in (args.cuda, args.nvbit, args.output))
    assert digest(args.nvbit_archive) == NVBIT_ASSET_SHA, 'wrong NVBit release bytes'
    assert '#define NVBIT_VERSION "1.8.1"' in (nvbit/'core/nvbit.h').read_text()
    assert 'URZ_BLACKWELL_PLUS = 255' in (nvbit/'core/instr_types.h').read_text()
    out.mkdir(parents=True, exist_ok=True)
    logs = out/'logs'; logs.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    env.pop('LD_PRELOAD', None)
    env.pop('CUDA_INJECTION64_PATH', None)
    env['PATH'] = str(cuda/'bin') + ':/usr/bin:/bin'
    commands = []
    prior = [json.loads(line) for line in (out/'BUILD-COMMANDS.jsonl').read_text().splitlines()] if (out/'BUILD-COMMANDS.jsonl').exists() else []
    skipping = bool(args.resume_from)

    def run(label, argv):
        nonlocal skipping
        argv = list(map(str, argv))
        if label == args.resume_from:
            skipping = False
        if skipping:
            old = next((r for r in reversed(prior) if r['label']==label and r['argv']==argv and r['exit_code']==0), None)
            assert old and digest(Path(old['log']))==old['log_sha256'], 'cannot reuse missing/changed successful stage '+label
            # This option is for an interrupted build, not source changes to
            # an earlier stage. Exact commands/log identities stay in the receipt.
            commands.append(dict(old, reused=True))
            return Path(old['log'])
        suffix = len(list(logs.glob(label+'.*.log')))
        file = logs/(label+'.'+str(suffix)+'.log')
        with file.open('xb') as log:
            result = subprocess.run(argv, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'label': label, 'argv': argv, 'exit_code': result.returncode, 'log': str(file), 'log_sha256': digest(file)}
        commands.append(row)
        with (out/'BUILD-COMMANDS.jsonl').open('a') as receipt:
            receipt.write(json.dumps(row)+'\n')
        if result.returncode:
            raise SystemExit('offline build failed; retain '+str(file))
        return file

    nvcc = cuda/'bin/nvcc'
    version = run('nvcc-version', [nvcc, '--version'])
    assert 'V12.8.93' in version.read_text(), 'unreviewed CUDA compiler'
    inc = ['-I'+str(nvbit/'core'), '-I'+str(HERE)]
    common = [nvcc, '-std=c++17', '-arch=sm_120a']
    run('kernels', [*common, '-O3', '--cubin', HERE/'kernels.cu', '-o', out/'kernels.cubin'])
    run('inject', [*common, *inc, '-O3', '-D_FORCE_INLINES', '-Xptxas', '-astoolspatch',
        '--keep-device-functions', '-Xcompiler', '-Wall,-fPIC', '-c', HERE/'inject_funcs.cu', '-o', out/'inject_funcs.o'])
    run('tracer', [*common, *inc, '-O3', '-D_FORCE_INLINES', '-Xptxas', '-cloning=no',
        '-Xcompiler', '-Wall,-fPIC', '-dc', HERE/'tracer.cu', '-o', out/'tracer.o'])
    run('tool-link', [*common, '-shared', '-Xcompiler', '-fPIC', out/'tracer.o', out/'inject_funcs.o',
        nvbit/'core/libnvbit.a', '-L'+str(cuda/'lib/stubs'), '-L'+str(cuda/'lib'), '-lcuda', '-lcudart_static', '-ldl', '-lrt',
        '-Xcompiler', '-pthread', '-o', out/'sim28_capture.so'])
    run('host', ['/usr/bin/g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-I'+str(cuda/'include'),
        HERE/'host.cpp', '-L'+str(cuda/'lib/stubs'), '-Wl,--no-as-needed', '-lcuda', '-ldl', '-o', out/'host'])
    disasm = run('nvdisasm', [args.nvdisasm.resolve(), '-hex', out/'kernels.cubin'])
    run('nm-tool', ['/usr/bin/nm', '-D', '--defined-only', out/'sim28_capture.so'])
    run('nm-host', ['/usr/bin/nm', '-D', '--undefined-only', out/'host'])
    files = [p for p in HERE.iterdir() if p.is_file()]
    record = {'schema': 'arch-explorer.sim28-capture-build.v1', 'commands_this_run': commands,
        'sources': {p.name:digest(p) for p in sorted(files)},
        'nvbit': {'asset_sha256': NVBIT_ASSET_SHA, 'core_library_sha256': digest(nvbit/'core/libnvbit.a'),
            'headers': {n:digest(nvbit/'core'/n) for n in ('nvbit.h','instr_types.h','nvbit_tool.h')},
            'license_files': {n:digest(nvbit/n) for n in ('LICENSE','EULA.txt') if (nvbit/n).is_file()}},
        'tools': {str(p):digest(p) for p in (nvcc, args.nvdisasm.resolve(), Path('/usr/bin/g++'))},
        'artifacts': {n:digest(out/n) for n in ('kernels.cubin','sim28_capture.so','host')},
        'disassembly': str(disasm), 'gpu_queries': 0, 'gpu_executions': 0, 'timing_or_calibration': False,
        'scope': 'offline build only; no claim of successful NVBit/device capture'}
    (out/'BUILD.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'artifacts': record['artifacts'], 'gpu_queries': 0, 'gpu_executions': 0}))


if __name__ == '__main__':
    main()
