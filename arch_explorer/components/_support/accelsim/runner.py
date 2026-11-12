# Concrete Accel-Sim implementation; owned by the component provider.
"""Pinned executable lifecycle and complete retained run evidence."""
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile

from arch_explorer.components.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA
from arch_explorer.components.accelsim.trace import digest, exact, load_json

ENGINE_SCHEMA = 'arch-explorer.accelsim-engine.v1'


def _checked_file(record, at):
    exact(record, {'path', 'sha256'}, at)
    path = Path(record['path']).resolve()
    if not path.is_file():
        raise ImportError(f'accelsim optional native engine file missing: {path}; build the pinned v2.0.1 engine (docs/accelsim.md)')
    if digest(path.read_bytes()) != record['sha256']:
        raise ValueError(f'accelsim {at} SHA-256 mismatch: {path}')
    return path


def load_engine(path):
    if path is None or not Path(path).is_file():
        raise ImportError('accelsim requires the optional native engine: set engine to a pinned engine manifest; see docs/accelsim.md')
    manifest, raw = load_json(path)
    from arch_explorer.components.accelsim.variant import SCHEMA as VARIANT_SCHEMA
    is_variant=manifest.get('schema')==VARIANT_SCHEMA
    exact(manifest, {'schema', 'sources', 'binary', 'libraries', 'build'} | ({'variant'} if is_variant else set()), 'engine manifest')
    if manifest['schema'] not in (ENGINE_SCHEMA,VARIANT_SCHEMA) or manifest['sources'] != {'accelsim': ACCELSIM_SHA, 'gpgpusim': GPGPUSIM_SHA}:
        raise ValueError('accelsim engine must use the pinned v2.0.1 source identities')
    if not isinstance(manifest['build'], dict) or not manifest['build']:
        raise ValueError('accelsim engine build provenance is required')
    binary = _checked_file(manifest['binary'], 'binary')
    if not os.access(binary, os.X_OK):
        raise ImportError(f'accelsim engine is not executable: {binary}')
    if not isinstance(manifest['libraries'], list) or not manifest['libraries']:
        raise ValueError('accelsim engine manifest requires the simulator shared library hashes')
    libraries = [_checked_file(row, 'library') for row in manifest['libraries']]
    if not any(path.name == 'libcudart.so' for path in libraries):
        raise ValueError('accelsim engine requires its GPGPU-Sim libcudart.so, not the CUDA runtime')
    if is_variant:
        from arch_explorer.components.accelsim.variant import validate
        validate(manifest)
    return manifest, binary, libraries, digest(raw)


def run_native(config, trace, engine, *, output_dir=None, timeout_s=120, max_cycles=1_000_000, memory_limit_mb=4096):
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 3600:
        raise ValueError('accelsim timeout_s must be finite in (0, 3600]')
    if type(max_cycles) is not int or not 1 <= max_cycles <= 1_000_000_000:
        raise ValueError('accelsim max_cycles must be an integer in 1..1000000000')
    if type(memory_limit_mb) is not int or not 512 <= memory_limit_mb <= 16384:
        raise ValueError('accelsim memory_limit_mb must be an integer in 512..16384')
    manifest, binary, libraries, engine_hash = load_engine(engine)
    parent = Path(output_dir).resolve() if output_dir is not None else Path(tempfile.gettempdir())
    # Upstream's legacy PipeReader invokes `cat <filename>` through popen.
    # Stage only fixed basenames inside a shell-safe absolute path.
    if not re.fullmatch(r'[A-Za-z0-9_/.-]+', str(parent)):
        raise ValueError('accelsim run directory requires a shell-safe absolute path for upstream PipeReader')
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='accelsim-', dir=parent))
    for name, content in config.artifacts.items():
        (directory / name).write_text(content)
    (directory / 'limits.config').write_text(f'-gpgpu_max_cycle {max_cycles}\n')
    if isinstance(trace,bytes):
        name='kernel-1.traceg';(directory/name).write_bytes(trace)
    else:
        name=trace.stage(directory)
    names = [name] if isinstance(name, str) else name
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(n, str) or re.fullmatch(r'kernel-[1-9][0-9]*\.(?:traceg|tracez)(?:\.xz)?', n) is None
                   or not (directory/n).is_file() for n in names)):
        raise ValueError('accelsim staged kernel list must contain unique canonical trace files')
    (directory / 'kernelslist.g').write_text(''.join(n+'\n' for n in names))
    args = [str(binary), '-config', 'gpgpusim.config', '-config', 'limits.config', '-trace', './kernelslist.g']
    env = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'CUDA_VISIBLE_DEVICES': '',
           'LD_LIBRARY_PATH': ':'.join(dict.fromkeys(str(p.parent) for p in libraries))}
    record = dict(schema='arch-explorer.accelsim-run.v1', argv=args, cwd=str(directory), environment=env,
                  engine=manifest, engine_manifest_sha256=engine_hash,
                  config_sha256=config.sha256, adf_sha256=config.adf_sha256,
                  limits={'timeout_s': timeout_s, 'max_cycles': max_cycles, 'memory_limit_mb': memory_limit_mb},
                  files={p.name: digest(p.read_bytes()) for p in directory.iterdir()}, status='starting')
    from arch_explorer.components.accelsim.source_compat import adapter_sources
    record['adapter_sources_sha256'] = adapter_sources()
    def save():
        (directory / 'run.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_mb * 2**20,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(timeout_s) + 1,) * 2)
    save()
    try:
        with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            process = subprocess.Popen(args, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, start_new_session=True, preexec_fn=limits)
            record['pid'] = process.pid
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The owned process group exited at the timeout boundary.
                process.wait()
                record['status'] = 'timeout'
            record['exit_code'] = process.returncode
    except OSError as error:
        record.update(status='spawn_failed', error=str(error))
        save()
        raise ValueError(f'accelsim native spawn failed; evidence: {directory}') from error
    record['stdout_sha256'] = digest((directory / 'stdout.log').read_bytes())
    record['stderr_sha256'] = digest((directory / 'stderr.log').read_bytes())
    if record['status'] == 'timeout' or record['exit_code'] != 0:
        record['status'] = 'timeout' if record['status'] == 'timeout' else 'native_failed'
        save()
        raise ValueError(f'accelsim native {record["status"]}, exit={record["exit_code"]}; evidence: {directory}')
    record['status'] = 'executed'
    save()
    return directory, record
