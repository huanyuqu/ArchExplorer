"""Bounded Linux process-group supervision; no simulation semantics.

An isolated subreaper inherits the stage lock and owns the command group until
its members stop and are reaped. A leader exit never implies a group exit.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import threading
import time


_POLL_SECONDS = 0.02
_CLEANUP_REPORT_SECONDS = 2.0


def scale_lock_path():
    """The fixed three-party lock; independent of TMPDIR and hostname."""
    return '/tmp/archexplorer-paper-scale.lock'


def _write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _group(pgid):
    """Return Linux group membership and explicit observation errors."""
    members, errors = [], []
    try:
        paths = list(Path('/proc').iterdir())
    except OSError as exc:
        return members, [{'type': type(exc).__name__, 'message': str(exc)}]
    for path in paths:
        if not path.name.isdecimal():
            continue
        pid = int(path.name)
        try:
            if os.getpgid(pid) != pgid:
                continue
            value = (path / 'stat').read_text()
            fields = value[value.rindex(')') + 2:].split()
            if int(fields[2]) != pgid:
                continue
            members.append({'pid': pid, 'ppid': int(fields[1]), 'state': fields[0],
                'start_ticks': int(fields[19]), 'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
                'cpu_seconds': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')})
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, ValueError, IndexError) as exc:
            errors.append({'pid': pid, 'type': type(exc).__name__, 'message': str(exc)})
    return sorted(members, key=lambda item: item['pid']), errors


def _signal_group(pgid, number):
    try:
        os.killpg(pgid, number)
        return True
    except ProcessLookupError:
        return False


def _emit(value):
    # Atomic, bounded control events; the command uses separate stdout/stderr.
    try:
        value = (json.dumps(value, separators=(',', ':')) + '\n').encode()
        return os.write(sys.stdout.fileno(), value) == len(value)
    except OSError:
        return False


def _subreaper():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _exec_command(config_path, gate):
    config = json.loads(Path(config_path).read_text())
    if os.read(gate, 1) != b'1':
        raise RuntimeError('Stage launch was not acknowledged')
    os.close(gate)
    os.execvpe(config['argv'][0], config['argv'], os.environ)


def _supervise(config_path, lock_fd):
    """Fresh process holds the lock, adopts orphans and reaps the managed group."""
    config = json.loads(Path(config_path).read_text())
    record = config['record']
    directory = Path(config_path).parent
    started, process = time.monotonic(), None
    deadline = started + record['limits']['wall_seconds']
    cancelled, reaped, cpu_seen = [], [], {}
    record.update(supervisor_pid=os.getpid(), supervisor_subreaper=False,
        group_peak_rss_bytes=0, group_observed_cpu_seconds=0.0,
        cleanup={'status': 'not-started', 'confirmed_no_running': False,
                 'remaining_pids': [], 'signals': [], 'lock_retained': True})
    def persist():
        try:
            _write(directory / 'SUPERVISION.json', record)
        except OSError as exc:
            record['journal_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    def cancelled_by_signal(number, frame):
        cancelled.append(number)
    def sample():
        members, errors = _group(process.pid)
        live = [p for p in members if p['state'] not in ('Z', 'X')]
        record['group_peak_rss_bytes'] = max(record['group_peak_rss_bytes'], sum(p['rss_bytes'] for p in live))
        for item in members:
            key = (item['pid'], item['start_ticks'])
            cpu_seen[key] = max(cpu_seen.get(key, 0.0), item['cpu_seconds'])
        record['group_observed_cpu_seconds'] = max(record['group_observed_cpu_seconds'], sum(cpu_seen.values()))
        record['cleanup'].update(remaining_pids=[p['pid'] for p in members], observation_errors=errors)
        return members, live, errors
    def reap(pid):
        try:
            found, status, usage = os.wait4(pid, os.WNOHANG)
        except ChildProcessError:
            return False
        if not found:
            return False
        code = os.waitstatus_to_exitcode(status)
        reaped.append({'pid': pid, 'returncode': code, 'user_seconds': usage.ru_utime,
                       'system_seconds': usage.ru_stime, 'max_rss_bytes': usage.ru_maxrss * 1024})
        if pid == process.pid:
            process.returncode = code
            record.update(returncode=code, max_rss_bytes=usage.ru_maxrss * 1024,
                          child_user_seconds=usage.ru_utime, child_system_seconds=usage.ru_stime)
        return True
    def cleanup():
        if process is None:
            record['cleanup'].update(status='not-started', confirmed_no_running=True)
            return
        clean_start = time.monotonic()
        kill_at = min(deadline, clean_start + 1.0)
        if record['status'] == 'resource-exceeded':
            kill_at = clean_start
        sent_term, sent_kill, reported = False, False, False
        record['cleanup']['status'] = 'checking'
        while True:
            members, live, errors = sample()
            descendants = [p for p in members if p['pid'] != process.pid]
            if live:
                if record.get('reason') == 'leader-exit' and any(p['pid'] != process.pid for p in live):
                    record['orphaned_running_descendants'] = True
                    if record['status'] == 'success':
                        record['status'] = 'failed'
                number = signal.SIGKILL if time.monotonic() >= kill_at else signal.SIGTERM
                if (number == signal.SIGTERM and not sent_term) or (number == signal.SIGKILL and not sent_kill):
                    try:
                        _signal_group(process.pid, number)
                    except OSError as exc:
                        record['cleanup']['signal_error'] = {'type': type(exc).__name__, 'message': str(exc)}
                        persist()
                        time.sleep(_POLL_SECONDS)
                        continue
                    record['cleanup']['signals'].append({'signal': number, 'at_seconds': time.monotonic() - started})
                    sent_term |= number == signal.SIGTERM
                    sent_kill |= number == signal.SIGKILL
            for item in descendants:
                if item['state'] in ('Z', 'X'):
                    reap(item['pid'])
            # The leader remains a WNOWAIT zombie anchoring this PGID until
            # every descendant has exited and been reaped.
            if not live and not errors and not descendants and reap(process.pid):
                record['cleanup'].update(status='confirmed', confirmed_no_running=True, remaining_pids=[],
                    observation_errors=[], duration_seconds=time.monotonic() - clean_start,
                    confirmation='No group descendants in /proc; adopted children and leader reaped by wait4.')
                record['reaped_processes'] = reaped
                record['group_cpu_seconds'] = sum(r['user_seconds'] + r['system_seconds'] for r in reaped)
                if record['group_cpu_seconds'] > record['limits']['cpu_seconds'] and record['status'] == 'success':
                    record.update(status='resource-exceeded', reason='group-cpu-budget')
                return
            if time.monotonic() - clean_start >= _CLEANUP_REPORT_SECONDS and not reported:
                record['cleanup'].update(status='incomplete', confirmed_no_running=False, lock_retained=True)
                persist()
                _emit({'event': 'cleanup-incomplete', 'pgid': process.pid})
                reported = True
            # No unlock on an uninterruptible or unobservable member. The
            # supervisor stays alive as lock keeper and keeps checking.
            persist()
            time.sleep(0.1 if reported else _POLL_SECONDS)
    try:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, cancelled_by_signal)
        signal.signal(signal.SIGINT, cancelled_by_signal)
        _subreaper()
        record['supervisor_subreaper'] = True
        persist()
        if os.getppid() != config['controller_pid']:
            record.update(status='cancelled', reason='controller-exited-before-launch')
            return
        gate_read, gate_write = os.pipe()
        try:
            def limits():
                resource.setrlimit(resource.RLIMIT_AS, (record['limits']['address_space_bytes'],) * 2)
                resource.setrlimit(resource.RLIMIT_CPU, (record['limits']['cpu_seconds'],) * 2)
            with (directory / 'stdout.txt').open('wb') as stdout, (directory / 'stderr.txt').open('wb') as stderr:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run-command',
                    str(config_path), str(gate_read)], cwd=record['cwd'], pass_fds=(gate_read,),
                    stdout=stdout, stderr=stderr, start_new_session=True, preexec_fn=limits)
                record['pid'] = record['process_group_id'] = process.pid
                persist()
                if _emit({'event': 'started', 'pgid': process.pid}):
                    os.write(gate_write, b'1')
                else:
                    cancelled.append(signal.SIGTERM)
                os.close(gate_write)
                gate_write = None
                while True:
                    members, live, errors = sample()
                    for item in members:
                        if item['pid'] != process.pid and item['state'] == 'Z':
                            reap(item['pid'])
                    if cancelled or os.getppid() != config['controller_pid']:
                        record.update(status='cancelled', reason='cancelled-or-controller-exited')
                        break
                    if time.monotonic() >= deadline:
                        record.update(status='timed-out', reason='group-wall-budget')
                        break
                    if errors:
                        record.update(status='failed', reason='group-observation-failed')
                        break
                    if sum(p['rss_bytes'] for p in live) > record['limits']['address_space_bytes']:
                        record.update(status='resource-exceeded', reason='group-rss-budget')
                        break
                    if record['group_observed_cpu_seconds'] > record['limits']['cpu_seconds']:
                        record.update(status='resource-exceeded', reason='group-cpu-budget')
                        break
                    pending = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    if pending is not None:
                        code = pending.si_status if pending.si_code == os.CLD_EXITED else -pending.si_status
                        record.update(status='success' if code == 0 else 'failed', reason='leader-exit')
                        break
                    time.sleep(_POLL_SECONDS)
        finally:
            os.close(gate_read)
            if gate_write is not None:
                os.close(gate_write)
    except BaseException as exc:
        record.update(status='failed', reason='supervisor-exception',
                      error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        cleanup()
        record['supervision_seconds'] = time.monotonic() - started
        persist()
        _emit({'event': 'complete', 'pgid': record.get('pid'),
               'confirmed': record['cleanup']['confirmed_no_running'],
               'returncode': record['returncode'], 'status': record['status']})
        os.close(lock_fd)


def _emergency_cleanup(pgid, record, journal):
    """Caller retains its lock when the isolated supervisor died unexpectedly."""
    record.update(status='failed', reason='supervisor-exited-without-cleanup-confirmation')
    record['cleanup'] = {'status': 'checking', 'confirmed_no_running': False, 'lock_retained': True}
    started = time.monotonic()
    while True:
        members, errors = _group(pgid)
        live = [p for p in members if p['state'] not in ('Z', 'X')]
        record['cleanup'].update(remaining_pids=[p['pid'] for p in members], observation_errors=errors)
        if not live and not errors:
            record['cleanup'].update(status='confirmed-no-running', confirmed_no_running=True,
                remaining_zombie_pids=[p['pid'] for p in members],
                confirmation='Emergency /proc check; no running members. Zombies explicitly reported.')
            return
        if time.monotonic() - started >= _CLEANUP_REPORT_SECONDS:
            record['cleanup']['status'] = 'incomplete'
        try:
            _write(journal, record)
        except OSError as exc:
            record['journal_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        try:
            _signal_group(pgid, signal.SIGKILL)
        except OSError as exc:
            record['cleanup']['signal_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        time.sleep(_POLL_SECONDS)


def run_scale_stage(argv, *, cwd, output_dir, wall_seconds, memory_bytes,
                    stage, dimensions, estimate):
    """Run a stage with an inherited lock and a descendant-aware supervisor.

    Child RLIMIT_AS/CPU are inherited. Group wall time and sampled summed
    RSS/CPU include descendants. These guards are not a kernel cgroup quota.
    Early leader exit with unfinished descendants is failed, preserving its
    actual returncode. Unconfirmed cleanup keeps a supervisor holding the lock.
    """
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise ValueError('argv must be an explicit nonempty string array; no shell')
    if type(wall_seconds) is not int or not 1 <= wall_seconds <= 600:
        raise ValueError('wall_seconds must be in 1..600')
    if type(memory_bytes) is not int or not 1 <= memory_bytes <= 16 * 1024**3:
        raise ValueError('memory_bytes must be in 1..16GiB')
    if stage not in ('preflight', 'graph', 'compile', 'service', 'reference'):
        raise ValueError('stage must be preflight/graph/compile/service/reference')
    if type(dimensions) is not dict or not dimensions or type(estimate) is not dict or not estimate:
        raise ValueError('Complete original dimensions and a separate resource estimate are required')
    if sys.platform != 'linux' or not Path('/proc/self/stat').is_file():
        raise RuntimeError('Scale supervision requires Linux /proc and child-subreaper support')
    snapshot = json.loads(json.dumps({'dimensions': dimensions, 'estimate': estimate}, allow_nan=False))
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / 'INPUT.json', snapshot)
    environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    settings = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    environment.update(settings)
    record = {'schema': 'arch-explorer.scale-stage.v1', 'status': 'starting', 'stage': stage,
        'argv': list(argv), 'cwd': str(Path(cwd).resolve()), **snapshot,
        'limits': {'wall_seconds': wall_seconds, 'cpu_seconds': wall_seconds,
            'address_space_bytes': memory_bytes, 'blas_threads': 1,
            'group_rss_guard': 'sampled sum of member RSS; shared pages may be counted more than once',
            'group_cpu_guard': 'sampled member CPU and final reaped usage', 'poll_seconds': _POLL_SECONDS,
            'kernel_aggregate_quota': False}, 'environment': settings, 'lock': scale_lock_path(),
        'host': socket.gethostname(), 'returncode': None, 'max_rss_bytes': None,
        'child_user_seconds': None, 'child_system_seconds': None,
        'lock_acquired': False, 'lock_released': False,
        'cleanup': {'status': 'not-started', 'confirmed_no_running': False}}
    started, supervisor, lock, error, events, buffer = time.monotonic(), None, None, None, [], b''
    saved_signals = {}
    def drain():
        nonlocal buffer
        while True:
            try:
                value = os.read(supervisor.stdout.fileno(), 4096)
            except BlockingIOError:
                break
            if not value:
                break
            buffer += value
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            try:
                events.append(json.loads(line))
            except ValueError as exc:
                record.setdefault('protocol_errors', []).append(str(exc))
    def wait_supervisor():
        while supervisor.poll() is None:
            drain()
            if any(e['event'] == 'cleanup-incomplete' for e in events):
                return False
            time.sleep(_POLL_SECONDS)
        drain()
        return True
    def cancelled_by_signal(number, frame):
        raise SystemExit(128 + number)
    try:
        lock = open(record['lock'], 'a+')
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record.update(status='locked', reason='Another paper-scale stage owns the host lock')
            return record
        record['lock_acquired'] = True
        config = directory / 'SUPERVISOR-INPUT.json'
        _write(config, {'record': record, 'argv': argv, 'controller_pid': os.getpid()})
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGTERM, signal.SIGINT):
                saved_signals[number] = signal.getsignal(number)
            signal.signal(signal.SIGTERM, cancelled_by_signal)
        with (directory / 'supervisor-stderr.txt').open('wb') as stderr:
            supervisor = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--supervise',
                str(config), str(lock.fileno())], cwd=cwd, env=environment, pass_fds=(lock.fileno(),),
                stdout=subprocess.PIPE, stderr=stderr, start_new_session=True)
            os.set_blocking(supervisor.stdout.fileno(), False)
            wait_supervisor()
    except BaseException as exc:
        error = exc
        record.update(status='cancelled' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                      error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        if saved_signals:
            for number in saved_signals:
                signal.signal(number, signal.SIG_IGN)
        try:
            if supervisor is not None:
                if error and supervisor.poll() is None:
                    try:
                        os.kill(supervisor.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                finished = wait_supervisor()
                journal = directory / 'SUPERVISION.json'
                try:
                    record.update(json.loads(journal.read_text()))
                except (OSError, ValueError) as exc:
                    record.update(status='failed', journal_error={'type': type(exc).__name__, 'message': str(exc)})
                record['supervisor_pid'] = supervisor.pid
                record['supervision_events'] = events
                complete = next((e for e in events if e['event'] == 'complete' and e['confirmed']), None)
                if not finished:
                    record.update(status='cleanup-incomplete', reason='Supervisor retains the lock until the group ends')
                    record['cleanup']['lock_retained'] = True
                elif complete:
                    record['returncode'] = complete['returncode']
                    record['cleanup'].update(status='confirmed', confirmed_no_running=True,
                        confirmation='Isolated supervisor confirmed group cleanup before completion.')
                else:
                    groups = {e['pgid'] for e in events if e.get('pgid')}
                    if not groups:
                        for path in Path('/proc').iterdir():
                            if not path.name.isdecimal():
                                continue
                            try:
                                args = (path / 'cmdline').read_bytes().split(b'\0')
                                if b'--run-command' in args and str(directory / 'SUPERVISOR-INPUT.json').encode() in args:
                                    groups.add(os.getpgid(int(path.name)))
                            except (FileNotFoundError, ProcessLookupError):
                                pass
                    for pgid in groups:
                        _emergency_cleanup(pgid, record, directory / 'RECOVERY.json')
                    if not groups:
                        record.update(status='failed', reason='Supervisor failed before command launch')
                        record['cleanup'].update(status='not-started', confirmed_no_running=True)
                if error:
                    record.update(status='cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                                  error={'type': type(error).__name__, 'message': str(error)})
            else:
                record['cleanup'].update(status='not-started', confirmed_no_running=True)
        except BaseException as exc:
            # A journal/protocol failure is not permission to release a group.
            # The launch gate guarantees an event before the command can exec.
            error = error or exc
            record.update(status='failed', cleanup_error={'type': type(exc).__name__, 'message': str(exc)})
            if supervisor is not None:
                if supervisor.poll() is None:
                    try:
                        os.kill(supervisor.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    supervisor.wait()
                drain()
                groups = {e['pgid'] for e in events if e.get('pgid')}
                if record.get('process_group_id'):
                    groups.add(record['process_group_id'])
                for pgid in groups:
                    _emergency_cleanup(pgid, record, directory / 'RECOVERY.json')
                if not groups:
                    record['cleanup'].update(status='not-started', confirmed_no_running=True)
            else:
                record['cleanup'].update(status='not-started', confirmed_no_running=True)
        finally:
            if supervisor is not None and supervisor.stdout is not None:
                supervisor.stdout.close()
            # An incomplete cleanup has a separate inherited lock keeper.
            if lock:
                lock.close()
            record['lock_released'] = record['lock_acquired'] and record['cleanup']['confirmed_no_running']
            record['cleanup']['lock_retained'] = record['lock_acquired'] and not record['lock_released']
            record['wall_seconds'] = time.monotonic() - started
            record['outputs'] = [{'path': str(p.relative_to(directory)), 'bytes': p.stat().st_size}
                                 for p in sorted(directory.rglob('*')) if p.is_file()]
            record['logs'] = {}
            for name in ('stdout.txt', 'stderr.txt', 'supervisor-stderr.txt'):
                path = directory / name
                if path.exists():
                    h = hashlib.sha256()
                    with path.open('rb') as stream:
                        for part in iter(lambda: stream.read(1024 * 1024), b''):
                            h.update(part)
                    record['logs'][name] = {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
            _write(directory / 'RESULT.json', record)
            for number, previous in saved_signals.items():
                signal.signal(number, previous)
    if error:
        raise error
    return record


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--supervise':
        _supervise(Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 4 and sys.argv[1] == '--run-command':
        _exec_command(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit('Internal scale supervisor entry point')
