# Concrete Accel-Sim implementation; owned by the component provider.
"""Hash-bound trace request, restricted to one legacy SM90 kernel and stream."""
import hashlib
import json
from pathlib import Path
import re

from arch_explorer.components.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA, PROFILE
from arch_explorer.components.accelsim.native_isa import require_adapter_isa

TRACE_SCHEMA = 'arch-explorer.accelsim-trace.v1'
WORKLOAD_FIELDS = frozenset({'op', 'kernel', 'grid', 'block'})
MAPPING_FIELDS = frozenset({'trace_id', 'compiler', 'grid', 'block'})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def exact(obj, keys, where):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError(f'{where}: expected exactly fields {sorted(keys)}')


def load_json(path):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f'duplicate JSON key {key}')
            obj[key] = value
        return obj
    raw = Path(path).read_bytes()
    def constant(value):
        raise ValueError(f'non-finite JSON constant {value}')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant), raw


def identity(workload):
    return {'kind': workload.kind, 'params': {k: v for k, v in workload.params.items() if k != 'accelsim'}}


def same_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _text(value, where):
    if type(value) is not str or not value.strip():
        raise ValueError(f'{where}: requires a nonempty string')


def _name(value, where):
    _text(value, where)
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', value):
        raise ValueError(f'{where}: requires a kernel identifier')


def _integer(value, where, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{where}: requires an integer >= {minimum}; bool/float are not integers')


def _geometry(value, where):
    if type(value) is not list or len(value) != 3 or any(type(v) is not int or v <= 0 for v in value):
        raise ValueError(f'{where}: requires exactly three positive integers in an array; bool/float/string are unsupported')


def _sha256(value, where):
    if type(value) is not str or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError(f'{where}: requires a lowercase SHA-256 digest')


def _workload_params(params, where, *, reference=False):
    exact(params, WORKLOAD_FIELDS | ({'accelsim'} if reference else set()), where)
    _text(params['op'], f'{where}.op')
    if params['op'] != 'trace_kernel':
        raise ValueError(f'{where}.op: unsupported operator; accelsim v1 requires trace_kernel')
    _name(params['kernel'], f'{where}.kernel')
    for key in ['grid', 'block']:
        _geometry(params[key], f'{where}.{key}')
    if reference:
        ref = params['accelsim']
        exact(ref, {'manifest', 'sha256'}, f'{where}.accelsim')
        _text(ref['manifest'], f'{where}.accelsim.manifest')
        _sha256(ref['sha256'], f'{where}.accelsim.sha256')


def _mapping_spec(spec, where):
    exact(spec, {'template', 'params'}, where)
    _text(spec['template'], f'{where}.template')
    if spec['template'] != 'accelsim_trace':
        raise ValueError('accelsim requires mapping template accelsim_trace')
    params = spec['params']
    exact(params, MAPPING_FIELDS, f'{where}.params')
    for key in ['trace_id', 'compiler']:
        _text(params[key], f'{where}.params.{key}')
    for key in ['grid', 'block']:
        _geometry(params[key], f'{where}.params.{key}')


def _bind_facts(params, fields, actual, where):
    for key, fact in fields.items():
        if params[key] != actual[fact]:
            raise ValueError(f'{where}.{key} differs from actual trace {fact}: {params[key]!r} != {actual[fact]!r}')


def _trace_info(raw):
    """Validate the deliberately small v1 input grammar before native startup.

    Only one 32-thread warp/CTA, no control flow except final EXIT, no copies,
    barriers, clusters or tracez. Upstream supports more; v1 does not certify it.
    """
    lines = [s.strip() for s in raw.decode('ascii').splitlines() if s.strip()]
    marker = lines.index('#BEGIN_TB') if '#BEGIN_TB' in lines else -1
    if marker < 1 or lines[-1] != '#END_TB' or lines.count('#BEGIN_TB') != 1 or lines.count('#END_TB') != 1:
        raise ValueError('trace: exactly one complete thread block is supported')
    headers = {}
    for line in lines[:marker]:
        if line.startswith('#'):
            continue
        if not line.startswith('-') or ' = ' not in line:
            raise ValueError('trace: invalid header')
        k, v = line[1:].split(' = ', 1)
        if k in headers:
            raise ValueError(f'trace: duplicate header {k}')
        headers[k] = v
    required = {'kernel name', 'kernel id', 'grid dim', 'block dim', 'shmem', 'nregs', 'binary version',
                'cuda stream id', 'enable lineinfo', 'nvbit version', 'accelsim tracer version',
                'shmem base_addr', 'local mem base_addr'}
    exact(headers, required, 'trace header')
    require_adapter_isa('sm'+headers['binary version'])
    fixed = {'kernel id': '1', 'shmem': '0', 'binary version': '90', 'cuda stream id': '0', 'enable lineinfo': '0',
             'accelsim tracer version': '3', 'shmem base_addr': '0x0', 'local mem base_addr': '0x0'}
    if any(headers[k] != v for k, v in fixed.items()) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', headers['kernel name']):
        raise ValueError('trace header: requires SM90 v3 single 32-thread CTA/stream=0 without shared/local memory')
    if not headers['nregs'].isdigit() or not 1 <= int(headers['nregs']) <= 255:
        raise ValueError('trace header: nregs must be in 1..255')
    dimensions = {}
    for key in ['grid', 'block']:
        match = re.fullmatch(r'\(([0-9]+),([0-9]+),([0-9]+)\)', headers[f'{key} dim'])
        if match is None:
            raise ValueError(f'trace header: invalid {key} dim')
        dimensions[key] = [int(v) for v in match.groups()]
        _geometry(dimensions[key], f'trace header.{key}')
    if dimensions['grid'] != [1, 1, 1] or dimensions['block'] != [32, 1, 1]:
        raise ValueError('trace header: requires single 32-thread CTA; grid=[1,1,1], block=[32,1,1]')
    body = lines[marker + 1:-1]
    if len(body) < 4 or body[:2] != ['thread block = 0,0,0', 'warp = 0'] or not body[2].startswith('insts = '):
        raise ValueError('trace: one thread block and warp=0 required')
    instructions = body[3:]
    if not body[2][8:].isdigit() or int(body[2][8:]) != len(instructions):
        raise ValueError('trace: instruction count differs from complete warp trace')
    pc_set, thread_instructions = set(), 0
    for i, line in enumerate(instructions):
        tokens = line.split()
        try:
            pc, mask, ndst = int(tokens[0], 16), int(tokens[1], 16), int(tokens[2])
            if pc in pc_set or mask != 0xffffffff or not 0 <= ndst <= 1:
                raise ValueError()
            pc_set.add(pc)
            thread_instructions += mask.bit_count()
            for reg in tokens[3:3 + ndst]:
                if not re.fullmatch(r'R\d+', reg) or int(reg[1:]) >= int(headers['nregs']):
                    raise ValueError()
            op = tokens[3 + ndst]
            nsrc = int(tokens[4 + ndst])
            if not 0 <= nsrc <= 3:
                raise ValueError()
            for reg in tokens[5 + ndst:5 + ndst + nsrc]:
                if not re.fullmatch(r'R\d+', reg) or int(reg[1:]) >= int(headers['nregs']):
                    raise ValueError()
            tail = tokens[5 + ndst + nsrc:]
            if op in {'IADD3', 'EXIT'}:
                if tail != ['0', '0'] or ndst != (1 if op == 'IADD3' else 0):
                    raise ValueError()
            elif op in {'LDG.E', 'STG.E'}:
                if len(tail) != 5 or tail[:2] != ['4', '1'] or tail[3:] != ['4', '0'] or ndst != (1 if op == 'LDG.E' else 0):
                    raise ValueError()
                int(tail[2], 16)
            else:
                raise ValueError()
            if (op == 'EXIT') != (i == len(instructions) - 1):
                raise ValueError()
        except (ValueError, IndexError):
            raise ValueError(f'trace: unsupported v1 instruction at line {i + 1}: {line}') from None
    return {'name': headers['kernel name'], **dimensions, 'stream': int(headers['cuda stream id']),
            'thread_instructions': thread_instructions,
            'ctas': dimensions['grid'][0] * dimensions['grid'][1] * dimensions['grid'][2]}


def load_trace(workload, mapping):
    if workload.kind != 'operator' or not isinstance(workload.params, dict):
        raise ValueError('accelsim v1 supports a manifest-bound operator only; full ViT/CLIP/model traces are unsupported')
    exact(workload.params, WORKLOAD_FIELDS | {'accelsim'}, 'workload.params')
    ref = workload.params['accelsim']
    exact(ref, {'manifest', 'sha256'}, 'workload.params.accelsim')
    _text(ref['manifest'], 'workload.params.accelsim.manifest')
    _sha256(ref['sha256'], 'workload.params.accelsim.sha256')
    path = Path(ref['manifest']).resolve()
    manifest, raw = load_json(path)
    if digest(raw) != ref['sha256']:
        raise ValueError('trace manifest SHA-256 mismatch')
    if manifest.get('schema') == 'arch-explorer.accelsim-trace.v3':
        from arch_explorer.components.accelsim.typed_trace import load_v3
        return load_v3(workload, mapping, path, manifest, raw)
    if manifest.get('schema') == 'arch-explorer.accelsim-trace.v2':
        from arch_explorer.components.accelsim.trace_v2 import load_v2
        return load_v2(workload, mapping, path, manifest, raw)
    _workload_params(workload.params, 'workload.params', reference=True)
    _mapping_spec(mapping.to_dict(), 'mapping')
    exact(manifest, {'schema', 'format', 'isa', 'tracer_version', 'source', 'engine',
                     'workload', 'mapping', 'commands', 'kernels', 'scope'}, 'trace manifest')
    require_adapter_isa(manifest['isa'])
    if manifest['schema'] != TRACE_SCHEMA or manifest['format'] != 'traceg' or type(manifest['tracer_version']) is not int or manifest['tracer_version'] != 3:
        raise ValueError('unsupported trace format/ISA/version: requires traceg v3 sm90; Blackwell unsupported')
    if manifest['engine'] != {'accelsim': ACCELSIM_SHA, 'gpgpusim': GPGPUSIM_SHA, 'profile': PROFILE}:
        raise ValueError('trace manifest engine/config identity mismatch')
    exact(manifest['workload'], {'kind', 'params'}, 'manifest.workload')
    if manifest['workload']['kind'] != 'operator':
        raise ValueError('manifest.workload.kind: accelsim v1 requires operator')
    _workload_params(manifest['workload']['params'], 'manifest.workload.params')
    _mapping_spec(manifest['mapping'], 'manifest.mapping')
    if manifest['scope'] != 'single-kernel-device-cycles' or not same_json(manifest['workload'], identity(workload)) or not same_json(manifest['mapping'], mapping.to_dict()):
        raise ValueError('trace manifest workload/mapping identity or scope mismatch; changed shapes/compiler/mapping require a new trace')
    exact(manifest['source'], {'kind', 'gpu', 'toolchain', 'description'}, 'trace source')
    for key, value in manifest['source'].items():
        _text(value, f'trace source.{key}')
    if manifest['source']['kind'] not in {'synthetic', 'hardware'}:
        raise ValueError('trace source must identify kind, GPU, toolchain and origin')
    source = manifest['source']
    if (source['kind'] == 'synthetic' and source['gpu'] != 'none') or (source['kind'] == 'hardware' and source['gpu'] != 'NVIDIA H100 (SM90)'):
        raise ValueError('trace source GPU disagrees with supported ISA/origin: synthetic requires none; hardware requires NVIDIA H100 (SM90)')
    kernels = manifest['kernels']
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise ValueError('accelsim v1 requires exactly one kernel; multi-kernel/concurrency/copies unsupported')
    kernel = kernels[0]
    exact(kernel, {'file', 'sha256', 'original_sha256', 'name', 'grid', 'block', 'stream', 'thread_instructions', 'ctas'}, 'kernel')
    _name(kernel['name'], 'manifest.kernels[0].name')
    for key in ['grid', 'block']:
        _geometry(kernel[key], f'manifest.kernels[0].{key}')
    for key in ['stream', 'thread_instructions', 'ctas']:
        _integer(kernel[key], f'manifest.kernels[0].{key}', minimum=0 if key == 'stream' else 1)
    for key in ['sha256', 'original_sha256']:
        _sha256(kernel[key], f'manifest.kernels[0].{key}')
    if not isinstance(kernel['file'], str) or not re.fullmatch(r'kernel-[A-Za-z0-9_-]+\.traceg', kernel['file']):
        raise ValueError('trace filename must be a plain kernel-*.traceg basename')
    trace = (path.parent / kernel['file']).read_bytes()
    if digest(trace) != kernel['sha256'] or kernel['original_sha256'] != kernel['sha256']:
        raise ValueError('trace file hash/original mismatch; v1 accepts unmodified original text traces')
    expected = _trace_info(trace)
    if any(type(kernel[k]) is not type(v) or kernel[k] != v for k, v in expected.items()):
        raise ValueError('trace header/count differs from manifest expected completion')
    workload_facts = {'kernel': 'name', 'grid': 'grid', 'block': 'block'}
    mapping_facts = {'grid': 'grid', 'block': 'block'}
    _bind_facts(workload.params, workload_facts, expected, 'workload.params')
    _bind_facts(manifest['workload']['params'], workload_facts, expected, 'manifest.workload.params')
    _bind_facts(mapping.params, mapping_facts, expected, 'mapping.params')
    _bind_facts(manifest['mapping']['params'], mapping_facts, expected, 'manifest.mapping.params')
    exact(manifest['commands'], {'file', 'sha256'}, 'commands')
    _sha256(manifest['commands']['sha256'], 'commands.sha256')
    if manifest['commands']['file'] != 'kernelslist.g':
        raise ValueError('commands file must be kernelslist.g')
    commands = (path.parent / 'kernelslist.g').read_bytes()
    if digest(commands) != manifest['commands']['sha256'] or commands != (kernel['file'] + '\n').encode():
        raise ValueError('commands hash/content mismatch; exactly one kernel, no memcpy commands')
    return manifest, trace, digest(raw), expected
