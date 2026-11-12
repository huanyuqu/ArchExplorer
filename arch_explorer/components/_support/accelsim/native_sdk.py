"""Installed native factory headers and explicit, bounded CPU caller builds.

Importing or reading this SDK never builds, loads a library, or executes a
simulation. The caller explicitly selects a source-bound standalone archive.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from arch_explorer.components.accelsim.component_execution import check_ref, digest, file_ref, json_read, write
from arch_explorer.components._support.accelsim.native_resources import (
    native_files, native_include_directories,
)

SDK_SCHEMA = 'arch-explorer.accelsim-standalone-sdk.v1'


def include_directories():
    from arch_explorer.simulator.dynamic import runtime_header
    return (*native_include_directories(), runtime_header().parent)


def headers():
    from arch_explorer.simulator.dynamic import runtime_header
    return {name: digest(path.read_bytes()) for name, path in native_files().items()
            if path.suffix in {'.hpp', '.h'}} | {
        'runtime.hpp': digest(runtime_header().read_bytes())}


def load_sdk(path):
    document = json_read(path)
    if (not isinstance(document, dict) or set(document) != {'schema', 'headers', 'archives', 'build', 'dependencies'}
            or document['schema'] != SDK_SCHEMA or document['headers'] != headers()
            or not isinstance(document['archives'], dict) or set(document['archives']) != {'icnt', 'memory', 'gpu'}):
        raise ValueError('exact installed-header standalone SDK manifest required')
    for ref in document['archives'].values():
        check_ref(ref)
    build = document['build']
    if (not isinstance(build, dict) or set(build) != set(document['archives'])
            or not isinstance(document['dependencies'], dict) or set(document['dependencies']) != set(build)):
        raise ValueError('each standalone archive requires its independent build receipt')
    for target, ref in build.items():
        raw = check_ref(ref)
        receipt = json.loads(raw)
        if (receipt.get('status') != 'built' or receipt.get('target') != target
                or receipt.get('binary') != document['archives'][target]
                or receipt.get('GPU_host_translation_units') != [] or receipt.get('jobs') != 1):
            raise ValueError('standalone archive receipt differs from explicit source closure')
        for source in receipt['sources']:
            check_ref(source)
        if not isinstance(receipt.get('compiler_dependencies'),list) or not receipt['compiler_dependencies']:
            raise ValueError('actual compiler header dependency closure required')
        for dependency in receipt['compiler_dependencies']:
            check_ref(dependency)
        if not isinstance(document['dependencies'][target], list):
            raise ValueError('explicit native link dependencies required')
        for dependency in document['dependencies'][target]:
            check_ref(dependency)
    return document


def _run(argv, directory, timeout_s=600, memory_limit_mb=8192):
    if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 600:
        raise ValueError('CPU command timeout must be in (0,600]')
    if type(memory_limit_mb) is not int or not 1 <= memory_limit_mb <= 8192:
        raise ValueError('CPU command memory limit must be in [1,8192] MiB')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    record = {'argv': list(map(str, argv)), 'cwd': str(directory), 'timeout_s': timeout_s,
              'memory_limit_mb': memory_limit_mb, 'started_realtime_ns': time.time_ns()}
    env = dict(PATH='/usr/bin:/bin', LC_ALL='C', CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
    record['environment'] = env

    def bound():
        limit = memory_limit_mb*2**20
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    with (directory/'stdout.log').open('xb') as stdout, (directory/'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(record['argv'], cwd=directory, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True, preexec_fn=bound)
        record['pid'] = process.pid
        try:
            process.wait(timeout=timeout_s)
            record['timed_out'] = False
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            record['timed_out'] = True
            raise
        finally:
            record.update(exit_code=process.returncode, wait_completed=True, ended_realtime_ns=time.time_ns())
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                record['group_closed'] = True
            else:
                os.killpg(process.pid, signal.SIGKILL)
                record['group_closed'] = False
            record['stdout'] = file_ref(str(directory/'stdout.log'))
            record['stderr'] = file_ref(str(directory/'stderr.log'))
            write(directory/'PROCESS.json', record)
    if record['exit_code'] or not record['group_closed']:
        raise RuntimeError('bounded standalone CPU command failed; see '+str(directory/'PROCESS.json'))
    return record


def compile_consumer(source, sdk, output_dir, *, targets=('memory', 'icnt'), native_types=False, timeout_s=600):
    """Compile an external C++ caller against factory archives; no GPU link."""
    document = load_sdk(sdk)
    if not targets or len(set(targets)) != len(targets) or set(targets)-set(document['archives']):
        raise ValueError('select unique standalone archive targets')
    source = Path(source).resolve(strict=True)
    if source.suffix not in {'.cc', '.cpp', '.cxx'}:
        raise ValueError('external C++ caller source required')
    directory = Path(output_dir).resolve()
    binary = directory/'consumer'
    flags=[]
    if type(native_types) is not bool:
        raise ValueError('native_types must be explicit boolean')
    if native_types:
        for target in targets:
            receipt=json.loads(check_ref(document['build'][target]))
            for flag in receipt['compiler_flags']:
                if flag.startswith('-I'):
                    path=Path(flag[2:])
                    if not path.is_absolute() or not path.is_dir():
                        raise ValueError('native type include closure is unavailable')
                    flags.append(flag)
                elif flag.startswith('-DCUDART_VERSION=') and flag.split('=',1)[1].isdigit():
                    flags.append(flag)
    dependencies=list(dict.fromkeys(ref['path'] for target in targets for ref in document['dependencies'][target]))
    command = ['g++', '-std=c++17', '-O1', '-g', *('-I'+str(p) for p in include_directories()),*dict.fromkeys(flags),
               str(source), *(document['archives'][target]['path'] for target in targets),
               *dependencies,'-Wl,--gc-sections', '-lz', '-lm','-pthread', '-o', str(binary)]
    receipt = _run(command, directory, timeout_s)
    from arch_explorer.components.accelsim.standalone_engine import elf_symbols
    import re
    names=elf_symbols(binary.read_bytes())
    forbidden=[name for name in names if re.search(r'_ZNK?(?:9gpgpu_sim|15trace_gpgpu_sim|14exec_gpgpu_sim)',name)]
    if forbidden:
        raise ValueError('external caller linked a GPU host: '+', '.join(sorted(forbidden)[:8]))
    result = {'schema': 'arch-explorer.accelsim-standalone-consumer.v1',
              'sdk': file_ref(str(Path(sdk).resolve())), 'source': file_ref(str(source)),
              'binary': file_ref(str(binary)), 'targets': list(targets),'native_types':native_types,
              'constructor_audit':{'ELF_symbols':len(names),'GPU_host_symbols':forbidden},'build': receipt}
    write(directory/'CONSUMER.json', result)
    return result


def run_consumer(consumer, output_dir, *, arguments=(), timeout_s=600, memory_limit_mb=8192):
    """Explicitly execute the compiled external caller in a fresh CPU group."""
    if not isinstance(consumer, dict) or consumer.get('schema') != 'arch-explorer.accelsim-standalone-consumer.v1':
        raise ValueError('bound standalone caller receipt required')
    check_ref(consumer['binary'])
    check_ref(consumer['source'])
    load_sdk(consumer['sdk']['path'])
    check_ref(consumer['sdk'])
    if not isinstance(arguments, (tuple, list)) or any(not isinstance(x, str) for x in arguments):
        raise ValueError('caller arguments must be strings')
    return _run([consumer['binary']['path'], *arguments], output_dir, timeout_s, memory_limit_mb)
