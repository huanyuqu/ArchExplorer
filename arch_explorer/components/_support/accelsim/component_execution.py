"""Bounded CPU process adapter and strict raw-evidence reader for component GPUs.

The timing mechanism is in the native components. This module stages the same
complete program for either assembly or the untouched original-cycle oracle;
neither invocation consumes the other invocation's output.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile
import time

from arch_explorer.components.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA

ENGINE_SCHEMA = 'arch-explorer.accelsim-component-engine.v1'
METRICS_SCHEMA = 'arch-explorer.accelsim-component-metrics.v1'
RUN_SCHEMA = 'arch-explorer.accelsim-component-run.v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON field: '+key)
        value[key] = item
    return value


def read(path, maximum=64*2**20):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('missing/oversized component artifact: '+str(path))
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise ValueError('component artifact grew past its read bound')
    return raw


def json_read(path, maximum=64*2**20):
    value = json.loads(read(path, maximum), object_pairs_hook=_pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    canonical(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value)+b'\n'
    with path.open('xb') as stream:
        stream.write(raw)
    return {'path': str(path.resolve()), 'sha256': digest(raw)}


def file_ref(path):
    path = Path(path)
    if not path.is_absolute() or str(path.resolve()) != str(path):
        raise ValueError('absolute canonical component artifact path required')
    return {'path': str(path), 'sha256': digest(read(path, 512*2**20))}


def check_ref(ref, maximum=512*2**20):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('exact path/SHA component artifact reference required')
    if not isinstance(ref['path'], str) or not re.fullmatch('[0-9a-f]{64}', str(ref['sha256'])):
        raise ValueError('invalid component artifact identity')
    path = Path(ref['path'])
    if not path.is_absolute() or str(path.resolve()) != str(path):
        raise ValueError('component artifact path must be canonical absolute')
    raw = read(path, maximum)
    if digest(raw) != ref['sha256']:
        raise ValueError('component artifact bytes differ: '+str(path))
    return raw


def is_component_engine(path):
    return path is not None and json_read(path, 4*2**20).get('schema') in {
        ENGINE_SCHEMA, 'arch-explorer.accelsim-component-engine.v2'}


def load_engine(path, *, read_only=False):
    if not isinstance(path, str) or not path:
        raise ValueError('explicit component engine manifest required')
    ref = file_ref(path)
    engine = json_read(path, 4*2**20)
    if engine.get('schema') == 'arch-explorer.accelsim-component-engine.v2':
        from arch_explorer.components.accelsim.standalone_engine import load
        return load(path, read_only=read_only)
    if engine.get('schema') != ENGINE_SCHEMA:
        raise ValueError('independent component-engine manifest required')
    if engine.get('sources') != {'accelsim': ACCELSIM_SHA, 'gpgpusim': GPGPUSIM_SHA}:
        raise ValueError('component engine upstream source versions differ')
    build = engine.get('build')
    if (not isinstance(build, dict) or build.get('status') != 'built'
            or type(build.get('jobs')) is not int or not 1 <= build['jobs'] <= 2):
        raise ValueError('complete bounded native build receipt required')
    trees = build.get('source_trees')
    if (not isinstance(trees, dict) or set(trees) != {'accelsim', 'gpgpusim'}
            or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value)
                   for value in trees.values())):
        raise ValueError('both exact native source trees must be retained')
    check_ref(engine.get('binary'))
    binary = Path(engine['binary']['path'])
    if not os.access(binary, os.X_OK):
        raise ValueError('component engine binary is not executable')
    libraries = engine.get('libraries')
    if not isinstance(libraries, list) or not libraries:
        raise ValueError('component engine simulator libraries are required')
    for library in libraries:
        check_ref(library)
    if not any(Path(row['path']).name == 'libcudart.so' for row in libraries):
        raise ValueError('explicit CPU simulator libcudart library required')
    inputs = engine.get('build', {}).get('inputs')
    if not isinstance(inputs, dict) or not inputs or 'runtime.hpp' not in inputs:
        raise ValueError('component engine complete build inputs/runtime binding required')
    for value in inputs.values():
        check_ref(value)
    from arch_explorer.components.accelsim.variant import APPROVED
    inherited = {'accelsim': inputs.get('accelsim-typed-v1.patch', {}).get('sha256'),
                 'gpgpusim': inputs.get('gpgpusim-typed-v1.patch', {}).get('sha256')}
    if not any(row['patches'] == inherited for row in APPROVED.values()):
        raise ValueError('component extraction does not inherit an approved exact native patch pair')
    variant = engine.get('variant')
    if (not isinstance(variant, dict) or set(variant) != {'id', 'modes', 'typed_patches', 'profile_status'}
            or variant['id'] != 'component-extraction-v1'
            or variant['profile_status'] != 'experimental-unvalidated'
            or variant['typed_patches'] != inherited
            or not isinstance(variant['modes'], list) or len(variant['modes']) != 2
            or sorted(variant['modes']) != ['components', 'oracle']):
        raise ValueError('unsupported or falsely physical native component variant contract')
    from arch_explorer.simulator.dynamic import runtime_header
    if inputs['runtime.hpp']['sha256'] != digest(runtime_header().read_bytes()):
        raise ValueError('native engine uses a different installed generic runtime')
    from arch_explorer.components._support.accelsim.native_resources import native_files
    packaged = list(native_files().values())
    values = {entry['sha256'] for entry in inputs.values()}
    if not packaged or any(digest(p.read_bytes()) not in values for p in packaged):
        raise ValueError('native engine adapter inputs differ from installed component sources')
    return engine, ref


def options(value):
    if not isinstance(value, dict) or set(value)-{'engine', 'timeout_s', 'max_cycles', 'memory_limit_mb'}:
        raise ValueError('component options require engine and bounded runtime limits only')
    result = json.loads(canonical(value))
    if not isinstance(result.get('engine'), str) or not result['engine']:
        raise ValueError('component engine must be an explicit manifest path')
    timeout = result.get('timeout_s', 600)
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError('component timeout must be in (0,600] seconds')
    for name, default, minimum, maximum in (
            ('max_cycles', 10000000, 1, 1000000000),
            ('memory_limit_mb', 8192, 512, 8192)):
        number = result.get(name, default)
        if type(number) is not int or not minimum <= number <= maximum:
            raise ValueError('component '+name+' outside bounded execution contract')
    return result


def limits(value):
    return {key: value.get(key, default) for key, default in
            (('timeout_s', 600), ('max_cycles', 10000000), ('memory_limit_mb', 8192))}


def _argv(engine, directory, mode):
    binary = engine['binary']['path']
    if engine['schema'] == 'arch-explorer.accelsim-component-engine.v2':
        from arch_explorer.components.accelsim.standalone_engine import selected_binary
        binary = selected_binary(engine, mode)
    args = [binary, '-ae_mode', mode,
            '-ae_metrics', str(directory/'metrics.json')]
    if mode == 'components':
        args += ['-ae_topology', str(directory/'topology.tsv')]
    return args+['-config', 'gpgpusim.config', '-config', 'limits.config', '-trace', './kernelslist.g']


def _environment(engine):
    return {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'CUDA_VISIBLE_DEVICES': '',
            'NVIDIA_VISIBLE_DEVICES': 'none',
            'LD_LIBRARY_PATH': ':'.join(sorted({str(Path(x['path']).parent) for x in engine['libraries']}))}


def run(config, document, program_root, topology, engine_path, output_dir, runtime_limits, mode):
    if mode not in ('components', 'oracle'):
        raise ValueError('invalid native component mode')
    engine, engine_ref = load_engine(engine_path)
    root = Path(output_dir).resolve()
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', str(root)):
        raise ValueError('native trace reader needs a shell-safe absolute evidence path')
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=mode+'-', dir=root))
    from arch_explorer.components.accelsim.program import _ProgramFiles
    names = _ProgramFiles(document, program_root).stage(directory)
    inputs = {**config.artifacts,
              'limits.config': '-gpgpu_max_cycle '+str(runtime_limits['max_cycles'])+'\n',
              'topology.tsv': topology, 'kernelslist.g': ''.join(n+'\n' for n in names)}
    for name, text in inputs.items():
        with (directory/name).open('xb') as stream:
            stream.write(text.encode())
    files = {name: digest(read(directory/name)) for name in [*inputs, *names]}
    argv, env = _argv(engine, directory, mode), _environment(engine)
    receipt = {'schema': RUN_SCHEMA, 'mode': mode, 'status': 'starting',
               'engine': engine_ref, 'engine_document': engine, 'cwd': str(directory),
               'argv': argv, 'environment': env, 'limits': runtime_limits, 'files': files,
               'started_realtime_ns': time.time_ns(), 'started_monotonic_ns': time.monotonic_ns(),
               'pid': None, 'exit_code': None, 'wait_completed': False, 'group_closed': False}

    def bound():
        memory = runtime_limits['memory_limit_mb']*2**20
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu = int(math.ceil(runtime_limits['timeout_s']))+5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    proc = None
    with (directory/'stdout.log').open('xb') as stdout, (directory/'stderr.log').open('xb') as stderr:
        try:
            proc = subprocess.Popen(argv, cwd=directory, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True, preexec_fn=bound)
            receipt['pid'] = proc.pid
            try:
                receipt['exit_code'] = proc.wait(timeout=runtime_limits['timeout_s'])
                receipt['status'] = 'exited' if proc.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                receipt['exit_code'] = proc.wait()
                receipt['status'] = 'timeout'
            receipt['wait_completed'] = True
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                receipt['group_closed'] = True
            else:
                os.killpg(proc.pid, signal.SIGKILL)
                receipt['status'] = 'residual_process_group'
        except (OSError, ValueError) as error:
            receipt['status'] = 'launch_error'
            receipt['error'] = str(error)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                receipt['exit_code'] = proc.wait()
                receipt['wait_completed'] = True
    receipt.update(ended_realtime_ns=time.time_ns(), ended_monotonic_ns=time.monotonic_ns(),
                   stdout_sha256=digest(read(directory/'stdout.log', 256*2**20)),
                   stderr_sha256=digest(read(directory/'stderr.log', 16*2**20)))
    if (directory/'metrics.json').is_file():
        receipt['metrics_sha256'] = digest(read(directory/'metrics.json'))
    write(directory/'run.json', receipt)
    return directory, receipt


def consume(request, document, config, program_root, directory, receipt, *, read_only=False):
    """Reconstruct facts from raw staged input, stdout and native metrics."""
    directory = Path(directory)
    engine, engine_ref = load_engine(request['options']['engine'], read_only=read_only)
    mode = request['mode']
    expected_limits = limits(request['options'])
    if (receipt.get('schema') != RUN_SCHEMA or receipt.get('mode') != mode
            or receipt.get('engine') != engine_ref or receipt.get('engine_document') != engine
            or receipt.get('cwd') != str(directory) or directory.parent != Path(request['output_dir'])/'native'
            or receipt.get('argv') != _argv(engine, directory, mode)
            or receipt.get('environment') != _environment(engine)
            or receipt.get('limits') != expected_limits or receipt.get('exit_code') != 0
            or receipt.get('wait_completed') is not True or receipt.get('group_closed') is not True
            or receipt.get('status') != 'exited' or type(receipt.get('pid')) is not int or receipt['pid'] <= 0):
        raise ValueError('component native process/engine/scope/closure differs')
    for clock in ('realtime', 'monotonic'):
        start, end = receipt.get('started_'+clock+'_ns'), receipt.get('ended_'+clock+'_ns')
        if type(start) is not int or type(end) is not int or not 0 < start <= end:
            raise ValueError('component native clock scope differs')
    names = [item['file'] for item in document['kernels']]
    texts = {**config.artifacts, 'topology.tsv': check_ref(request['topology']).decode(),
             'limits.config': '-gpgpu_max_cycle '+str(expected_limits['max_cycles'])+'\n',
             'kernelslist.g': ''.join(name+'\n' for name in names)}
    expected_files = {key: digest(value.encode()) for key, value in texts.items()}
    expected_files.update({row['file']: row['sha256'] for row in document['kernels']})
    if receipt.get('files') != expected_files:
        raise ValueError('component staged input set differs from the complete bound program')
    for name, sha in expected_files.items():
        if digest(read(directory/name)) != sha:
            raise ValueError('component staged configuration/program changed: '+name)
    stdout, stderr = read(directory/'stdout.log', 256*2**20), read(directory/'stderr.log', 16*2**20)
    metrics_raw = read(directory/'metrics.json')
    if (digest(stdout) != receipt.get('stdout_sha256') or digest(stderr) != receipt.get('stderr_sha256')
            or digest(metrics_raw) != receipt.get('metrics_sha256')):
        raise ValueError('component raw output binding differs')
    if ACCELSIM_SHA[:7].encode() not in stdout or GPGPUSIM_SHA[:7].encode() not in stdout:
        raise ValueError('native upstream build identity absent from raw output')
    from arch_explorer.components.accelsim.program_stats import parse_program_stats
    core_mhz = next(row['value'] for row in config.component_map
                    if row['param'] == 'clock_mhz' and row['clock_domain'] == 'sm')
    stats = parse_program_stats(stdout.decode(), stderr.decode(),
                                [row['facts'] for row in document['kernels']], core_mhz)
    from fractions import Fraction
    exact_ns = Fraction(stats['gpu_tot_sim_cycle']*1000, core_mhz)
    stats['latency_ns_exact'] = {'numerator': exact_ns.numerator, 'denominator': exact_ns.denominator, 'unit': 'ns'}
    metrics = json_read(directory/'metrics.json')
    standalone = receipt['engine_document']['schema'] == 'arch-explorer.accelsim-component-engine.v2'
    expected_metrics = ('arch-explorer.accelsim-component-metrics.v2'
                        if standalone and mode == 'components' else METRICS_SCHEMA)
    if metrics.get('schema') != expected_metrics:
        raise ValueError('native metrics schema differs from its bound assembly image')
    validate_metrics(metrics, mode, stats, config, check_ref(request['topology']).decode())
    if metrics['final']['total_cycles'] > expected_limits['max_cycles']:
        raise ValueError('physical system drain exceeds the complete native cycle bound')
    physical_ns = Fraction(metrics['final']['total_cycles']*1000, core_mhz)
    stats.update(system_drained_cycles=metrics['final']['total_cycles'],
                 system_drained_latency_ns=float(physical_ns),
                 system_drained_latency_ns_exact={'numerator': physical_ns.numerator,
                                                 'denominator': physical_ns.denominator, 'unit': 'ns'},
                 drain_ticks=metrics['drain_ticks'], drain_core_cycles=metrics['drain_core_cycles'])
    return stats, metrics


def validate_metrics(metrics, mode, stats, config, topology):
    """Require explicit mode, complete kernel observation and physical drain."""
    fields = {'schema', 'mode', 'original_cycle_calls', 'component_ticks', 'clock_periods',
              'phase_calls', 'routing', 'ports', 'lifecycle', 'source', 'kernels',
              'foreground_final', 'drain_ticks', 'drain_core_cycles', 'final'}
    standalone = isinstance(metrics,dict) and metrics.get('schema') == 'arch-explorer.accelsim-component-metrics.v2'
    if standalone:
        fields |= {'allocation_domain','oracle_observations'}
        if mode != 'components':
            raise ValueError('standalone metrics cannot describe the original-cycle image')
    if (not isinstance(metrics, dict) or set(metrics) != fields
            or metrics.get('schema') not in {METRICS_SCHEMA, 'arch-explorer.accelsim-component-metrics.v2'} or metrics.get('mode') != mode):
        raise ValueError('native component metrics version/mode differs')
    if type(metrics.get('original_cycle_calls')) is not int or type(metrics.get('component_ticks')) is not int:
        raise ValueError('native invocation counters must be explicit integers')
    if mode == 'components':
        if metrics['original_cycle_calls'] != 0 or metrics['component_ticks'] <= 0:
            raise ValueError('component mode used whole-engine cycle or did not advance components')
    elif metrics['component_ticks'] != 0 or metrics['original_cycle_calls'] <= 0:
        raise ValueError('oracle must use original cycle without component execution')
    nodes, edges = {}, set()
    for line in topology.splitlines()[1:]:
        row = line.split('\t')
        if row[0] == 'node':
            nodes[row[1]] = row[2]
        elif row[0] == 'connect':
            edges.add(tuple(row[1:]))
    periods = metrics['clock_periods']
    expected_clocks = {('core' if row['clock_domain'] == 'sm' else row['clock_domain']):
                       1/(row['value']*1e6) for row in config.component_map if row['param'] == 'clock_mhz'}
    if not isinstance(periods, dict) or set(periods) != set(expected_clocks):
        raise ValueError('all four native clock periods are required')
    if any(type(periods[key]) not in (int, float) or periods[key] != value
           for key, value in expected_clocks.items()):
        raise ValueError('native clock periods differ from compiled component clocks')

    def integers(value, names):
        if (not isinstance(value, dict) or set(value) != set(names)
                or any(type(number) is not int or number < 0 for number in value.values())):
            raise ValueError('missing/invalid native integer counter set: '+str(names))

    scalar_names = {'cycles', 'total_cycles', 'instructions', 'total_instructions', 'total_cta',
                    'stall_core_to_icnt', 'stall_icnt_to_l2', 'stall_l2_to_icnt', 'stall_icnt_to_core',
                    'icnt_injected', 'icnt_ejected', 'icnt_packet_bytes', 'native_busy', 'pending'}
    l2_names = {'index', 'accesses', 'hits', 'misses', 'pending_hits', 'reservation_fails',
                'port_available_cycles', 'data_port_busy_cycles', 'fill_port_busy_cycles'}
    dram_names = {'index', 'commands', 'activity', 'nop', 'activate', 'precharge',
                  'read', 'write', 'writeback', 'requests'}

    def snapshot(value):
        if (not isinstance(value, dict) or set(value) != scalar_names | {'l2', 'dram', 'system_drained'}
                or type(value['system_drained']) is not bool):
            raise ValueError('incomplete native mechanism snapshot')
        integers({key: value[key] for key in scalar_names}, scalar_names)
        if value['system_drained'] and (value['native_busy'] or value['pending']):
            raise ValueError('native physical drain contradicts pending state')
        for kind, names in (('l2', l2_names), ('dram', dram_names)):
            rows = value[kind]
            if not isinstance(rows, list) or len(rows) != sum(v == kind for v in nodes.values()):
                raise ValueError('native mechanism instance coverage differs')
            for index, row in enumerate(rows):
                integers(row, names)
                if row['index'] != index:
                    raise ValueError('native mechanism instance order/identity differs')
                if kind == 'l2' and row['accesses'] != row['hits']+row['misses']+row['pending_hits']:
                    raise ValueError('native L2 HIT/MISS/HIT_RESERVED accounting differs')

    kernels = metrics.get('kernels')
    if not isinstance(kernels, list) or len(kernels) != len(stats['kernels']):
        raise ValueError('native component metrics kernel coverage is incomplete')
    final = metrics.get('final')
    if not isinstance(final, dict) or final.get('system_drained') is not True:
        raise ValueError('native component system did not physically drain')
    cumulative_instructions, cumulative_cta = 0, 0
    for kernel, expected in zip(kernels, stats['kernels']):
        if (not isinstance(kernel, dict) or set(kernel) != {'uid', 'metrics'}
                or type(kernel['uid']) is not int or kernel['uid'] != expected['uid']):
            raise ValueError('native mechanism kernel order/UID differs')
        snap = kernel['metrics']
        snapshot(snap)
        cumulative_instructions += expected['thread_instructions']
        cumulative_cta += expected['ctas']
        for key, value in (('cycles', expected['cycles']), ('total_cycles', expected['cumulative_cycles']),
                           ('instructions', expected['thread_instructions']),
                           ('total_instructions', cumulative_instructions), ('total_cta', cumulative_cta)):
            if snap[key] != value:
                raise ValueError('raw metrics and stdout kernel counters differ: '+key)
    snapshot(final)
    foreground = metrics['foreground_final']
    snapshot(foreground)
    if (foreground['total_cycles'] != stats['gpu_tot_sim_cycle']
            or foreground['total_instructions'] != stats['gpu_tot_sim_insn']
            or foreground['total_cta'] != stats['gpu_tot_issued_cta']):
        raise ValueError('original foreground mechanism counters do not cover the complete program')
    integers({key: metrics[key] for key in ('drain_ticks', 'drain_core_cycles')},
             {'drain_ticks', 'drain_core_cycles'})
    if (metrics['drain_core_cycles'] > metrics['drain_ticks']
            or final['total_cycles'] != foreground['total_cycles']+metrics['drain_core_cycles']
            or final['total_instructions'] != foreground['total_instructions']
            or final['total_cta'] != foreground['total_cta']):
        raise ValueError('physical drain changed retirement/CTA or has inconsistent elapsed cycles')
    total_ticks = metrics['component_ticks'] if mode == 'components' else metrics['original_cycle_calls']
    if metrics['drain_ticks'] >= total_ticks:
        raise ValueError('native clock accounting leaves no foreground execution')
    if foreground['native_busy'] and not metrics['drain_ticks']:
        raise ValueError('busy foreground cannot become drained without actual clock advancement')
    if foreground['system_drained'] and (metrics['drain_ticks'] or metrics['drain_core_cycles']
                                         or final != foreground):
        raise ValueError('already drained foreground must not receive synthetic tail cycles')
    ports = metrics['ports']
    if not isinstance(ports, list) or len(ports) != len(edges):
        raise ValueError('complete explicit native port directory required')
    observed = set()
    port_total = 0
    injected, ejected, packet_bytes = 0, 0, 0
    for port in ports:
        endpoints = ('source', 'source_port', 'destination', 'destination_port')
        counters = ('accepted', 'logical_bytes', 'transfer_bytes')
        if not isinstance(port, dict) or set(port) != set(endpoints+counters):
            raise ValueError('invalid native port counter fields')
        identity = tuple(port[key] for key in endpoints)
        if any(not isinstance(token, str) or not token for token in identity):
            raise ValueError('native port endpoint tokens must be strings')
        if identity not in edges or identity in observed:
            raise ValueError('native port counters refer to duplicate/unknown wire')
        observed.add(identity)
        integers({key: port[key] for key in counters}, counters)
        port_total += port['accepted']
        if port['destination'] == 'icnt':
            injected += port['accepted']
            packet_bytes += port['transfer_bytes']
        if port['source'] == 'icnt':
            ejected += port['accepted']
    if (not port_total or injected != final['icnt_injected'] or ejected != final['icnt_ejected']
            or packet_bytes != final['icnt_packet_bytes']):
        raise ValueError('explicit wire accounting differs from actual native ICNT traffic')
    if standalone:
        allocation=metrics['allocation_domain']
        integers(allocation,{'allocated','destroyed','live'})
        if allocation['allocated'] != allocation['destroyed'] or allocation['live']:
            raise ValueError('the actual native allocation domain did not drain')
        legacy=metrics['oracle_observations']
        if not isinstance(legacy,dict) or set(legacy)!={'schema','ports'} or legacy['schema']!=METRICS_SCHEMA:
            raise ValueError('explicit frozen-oracle observation contract required')
        expected=[]
        for row in ports:
            item=dict(row)
            if row['source']=='icnt' and row['source_port'].startswith('request_mem'):
                # The fixed native protocol has 8B request/reply control
                # packets. SIM38's egress observer used reply length at both
                # ejection directions. Preserve that observation separately
                # while checking the actual packet lengths above.
                item['transfer_bytes']=16*row['accepted']+row['logical_bytes']-row['transfer_bytes']
                if item['transfer_bytes']<0:
                    raise ValueError('native wire lengths cannot produce the pinned oracle observation')
            expected.append(item)
        if canonical(legacy['ports'])!=canonical(expected):
            raise ValueError('frozen-oracle port observations contradict actual transactions')
    lifecycle = metrics['lifecycle']
    integers(lifecycle, {'native_requests', 'native_destroyed', 'consumer_ready',
                         'physical_complete', 'resource_ready', 'pending'})
    if lifecycle['pending'] or lifecycle['native_requests'] != lifecycle['native_destroyed']:
        raise ValueError('native request lifetime did not close')
    integers(metrics['source'], {'requests', 'accepted', 'completed', 'rejected'})
    if any(metrics['source'].values()):
        raise ValueError('complete shader program unexpectedly used an independent source')
    if mode == 'components':
        required_phases = ({'core.retire_dispatch', 'icnt.advance'} if standalone else
                           {'l2.reply_begin', 'l2.reply_end', 'l2.cache_begin', 'l2.cache_end',
                            'core.begin', 'core.retire_dispatch', 'credits.notify', 'icnt.advance'})
        for name, kind in nodes.items():
            required_phases.update(name+'.'+phase for phase in
                (('receive', 'advance') if kind == 'sm' else
                 (('reply', 'advance') if standalone else ('chiplet_receive', 'reply', 'advance')) if kind == 'l2' else
                 ('advance',) if kind == 'dram' else ()))
        integers(metrics['phase_calls'], required_phases)
        if not all(metrics['phase_calls'].values()):
            raise ValueError('declared native component/phase never advanced')
        audit = metrics['routing']
        integers(audit, {'ticks', 'phase_calls', 'accepted', 'rejected', 'wakeups', 'consumer_ready',
                         'physical_complete', 'resource_ready', 'completed'})
        if (audit['ticks'] != metrics['component_ticks'] or audit['accepted'] != port_total
                or audit['phase_calls'] != sum(metrics['phase_calls'].values())
                or not audit['completed'] or audit['completed'] > lifecycle['native_destroyed']
                or any(audit[key] != audit['completed'] for key in
                       ('consumer_ready', 'physical_complete', 'resource_ready'))):
            raise ValueError('dynamic route/phase/lifecycle accounting differs')
    elif metrics['routing'] != {} or metrics['phase_calls'] != {}:
        raise ValueError('original-cycle oracle must not assemble the component coordinator')
