"""Read-only validation of independently linked component and oracle images."""
from pathlib import Path
import hashlib
import json
import os
import re
import struct
from arch_explorer.components._support.accelsim.native_resources import native_files

SCHEMA = 'arch-explorer.accelsim-component-engine.v2'
ORACLE_BINARY = '05d466b007d4f7c3288d02c33f19b276e4c263411fdf6737e83e2dd0c03b1035'
ORACLE_CPU_LIBRARY = 'dd0753eb03604ffb23029d2b5ea1781d7ed90a8562abec2696f2865bd12bc1c8'
PATCHES = {'accelsim': 'f90e0be472a1e657e02b3a3de5c2f7217d8be8dcb169bf2eb46e41eefa37b477',
           'gpgpusim': 'b250da0dfd4d69154f7280e4311cf4641ba14e969738568d839f64038b389891'}
_LEGACY_BASELINE = '3a6e614ad619b51270ffeaaf941cdb22259a0bda'
_LEGACY_TABLE_SHA256 = '295bb6e8242ea21082c7d637abe6411342cfcb6b2da08ec53c1cafc038f3d42d'


def _legacy_native_inputs():
    from arch_explorer.components._support.accelsim import native_resources
    path = Path(native_resources.__file__).with_name('native-baseline-3a6e614.json')
    with path.open('rb') as file:
        raw = file.read(32*2**10 + 1)
    if len(raw) > 32*2**10 or hashlib.sha256(raw).hexdigest() != _LEGACY_TABLE_SHA256:
        raise ValueError('frozen native baseline table differs from its pinned source')
    document = json.loads(raw)
    if (set(document) != {'schema', 'source_commit', 'inputs'}
            or document['schema'] != 'arch-explorer.accelsim-native-baseline.v1'
            or document['source_commit'] != _LEGACY_BASELINE):
        raise ValueError('exact native baseline source identity required')
    return document['inputs']


def _check_native_inputs(inputs, *, read_only=False):
    from arch_explorer.components.accelsim.component_execution import digest
    if type(read_only) is not bool:
        raise ValueError('read_only must be an explicit boolean')
    current = {'component_native/'+name: digest(source.read_bytes())
               for name, source in native_files().items()}
    bound = {key: ref.get('sha256') for key, ref in inputs.items()
             if key.startswith('component_native/') and Path(key).suffix in {'.hpp', '.h', '.cc', '.cpp'}}
    if bound == current:
        return 'current'
    if read_only and bound == _legacy_native_inputs():
        return _LEGACY_BASELINE
    raise ValueError('standalone native source/member vector differs from the installed package'
                     + (' or the exact read-only baseline' if read_only else ''))


def elf_symbols(raw):
    """Inspect the real ELF symbol tables without launching nm or the image."""
    if len(raw) < 64 or raw[:6] != b'\x7fELF\x02\x01':
        raise ValueError('standalone image requires an unstripped ELF64 little-endian symbol table')
    offset = struct.unpack_from('<Q', raw, 40)[0]
    width, count = struct.unpack_from('<HH', raw, 58)
    if width != 64 or not 0 < count < 65535 or offset+width*count > len(raw):
        raise ValueError('invalid ELF section directory')
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, offset+width*i) for i in range(count)]
    symbols, full = set(), False
    for entry in sections:
        if entry[1] not in (2, 11):
            continue
        full |= entry[1] == 2
        begin, size, strings_index, stride = entry[4], entry[5], entry[6], entry[9]
        if stride != 24 or size % stride or begin+size > len(raw) or strings_index >= count:
            raise ValueError('invalid ELF symbol directory')
        strings = sections[strings_index]
        start, length = strings[4], strings[5]
        if start+length > len(raw):
            raise ValueError('invalid ELF symbol strings')
        names = raw[start:start+length]
        for pos in range(begin, begin+size, stride):
            name = struct.unpack_from('<I', raw, pos)[0]
            if name >= len(names):
                raise ValueError('invalid ELF symbol name')
            end = names.find(b'\0', name)
            if end < 0:
                raise ValueError('unterminated ELF symbol name')
            symbols.add(names[name:end].decode('ascii', errors='replace'))
    if not full:
        raise ValueError('retain the full native symbol table for independent construction audit')
    return symbols


def load(path, *, read_only=False):
    """Validate an engine; historical native sources require explicit replay use.

    The read-only exception matches one complete frozen baseline vector. It
    never changes the engine document or bypasses any other receipt check.
    """
    if type(read_only) is not bool:
        raise ValueError('read_only must be an explicit boolean')
    from arch_explorer.components.accelsim.component_execution import check_ref, digest, file_ref, json_read
    from arch_explorer.components.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA
    from arch_explorer.simulator.dynamic import runtime_header
    engine = json_read(path, 4*2**20)
    fields = {'schema', 'sources', 'binary', 'libraries', 'variant', 'build', 'oracle', 'sdk'}
    if (not isinstance(engine, dict) or set(engine) != fields or engine['schema'] != SCHEMA
            or engine['sources'] != {'accelsim': ACCELSIM_SHA, 'gpgpusim': GPGPUSIM_SHA}
            or engine['variant'] != {'id': 'standalone-factories-v1', 'modes': ['components', 'oracle'],
                                    'typed_patches': PATCHES, 'profile_status': 'experimental-unvalidated'}):
        raise ValueError('exact independently linked standalone component engine contract required')
    raw = check_ref(engine['binary'])
    if not os.access(engine['binary']['path'], os.X_OK):
        raise ValueError('standalone image is not executable')
    names = elf_symbols(raw)
    forbidden = [name for name in names if
                 re.search(r'(?:_ZNK?(?:9gpgpu_sim|15trace_gpgpu_sim|14exec_gpgpu_sim)|_ZTV(?:9gpgpu_sim|15trace_gpgpu_sim|14exec_gpgpu_sim))', name)
                 or name in {'cuInit', 'cuDeviceGet', 'cudaGetDevice', 'cudaMalloc', 'cudaFree', 'cudaLaunchKernel'}]
    if forbidden:
        raise ValueError('standalone image links a forbidden GPU host/device service: '+', '.join(sorted(forbidden)[:8]))
    build = engine['build']
    if (not isinstance(build, dict) or set(build) != {'status', 'jobs', 'source_trees', 'inputs', 'link'}
            or build['status'] != 'built' or build['jobs'] != 1
            or not isinstance(build['source_trees'], dict) or set(build['source_trees']) != {'accelsim', 'gpgpusim'}
            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{40}', v) for v in build['source_trees'].values())):
        raise ValueError('complete single-job native source/build identity required')
    inputs = build['inputs']
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('standalone build inputs absent')
    for item in inputs.values():
        check_ref(item)
    if (inputs.get('runtime.hpp', {}).get('sha256') != digest(runtime_header().read_bytes())
            or inputs.get('accelsim-typed-v1.patch', {}).get('sha256') != PATCHES['accelsim']
            or inputs.get('gpgpusim-typed-v1.patch', {}).get('sha256') != PATCHES['gpgpusim']):
        raise ValueError('standalone build does not match installed Runtime or fixed typed patches')
    _check_native_inputs(inputs, read_only=read_only)
    link = json_read(build['link']['path'])
    check_ref(build['link'])
    if (link.get('binary') != engine['binary'] or link.get('status') != 'built'
            or link.get('target') != 'gpu' or link.get('jobs') != 1
            or link.get('GPU_host_translation_units') != []):
        raise ValueError('standalone link receipt differs')
    for source in link['sources']:
        check_ref(source)
        if Path(source['path']).name in {'gpu-sim.cc', 'gpgpusim_entrypoint.cc', 'accel-sim.cc', 'main.cc', 'cuda_runtime_api.cc'}:
            raise ValueError('standalone source closure includes a GPU host translation unit')
    if not isinstance(link.get('compiler_dependencies'),list) or not link['compiler_dependencies']:
        raise ValueError('standalone compiler header closure absent')
    for dependency in link['compiler_dependencies']:
        check_ref(dependency)
    oracle = json_read(engine['oracle']['path'], 4*2**20)
    check_ref(engine['oracle'])
    if (oracle.get('schema') != 'arch-explorer.accelsim-component-engine.v1'
            or oracle.get('sources') != engine['sources']
            or oracle.get('binary', {}).get('sha256') != ORACLE_BINARY
            or oracle.get('variant', {}).get('typed_patches') != PATCHES
            or len(oracle.get('libraries', [])) != 1
            or oracle['libraries'][0].get('sha256') != ORACLE_CPU_LIBRARY):
        raise ValueError('independent unchanged original-cycle oracle identity differs')
    check_ref(oracle['binary'])
    for value in oracle['libraries']:
        check_ref(value)
    for value in oracle['build']['inputs'].values():
        check_ref(value)
    if engine['libraries'] != oracle['libraries']:
        raise ValueError('standalone/oracle runtime library directory binding differs')
    from arch_explorer.components.accelsim.native_sdk import load_sdk
    check_ref(engine['sdk'])
    load_sdk(engine['sdk']['path'])
    return engine, file_ref(str(Path(path).resolve()))


def selected_binary(engine, mode):
    if mode == 'oracle':
        from arch_explorer.components.accelsim.component_execution import json_read
        return json_read(engine['oracle']['path'])['binary']['path']
    return engine['binary']['path']
