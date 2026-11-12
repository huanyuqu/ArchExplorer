# Concrete Accel-Sim implementation; owned by the component provider.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic device boundaries backed by the actual C++ CPU state executor.

The capture boundary generates files through Fixture, and predictions are read
from its completed, same-program native protocols. Only the source-bound CPU
Driver is executable here. Device snapshots and resource records are explicitly
synthetic; no query, production host, native engine or resource lock is run.
"""
import copy
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from arch_explorer.components.accelsim.microbundle import json_file
from arch_explorer.components.accelsim.microexperiment import prediction_freeze, ref, write_new
from arch_explorer.components.accelsim.microrecords import original

CPU_PROCESS = 'arch-explorer.measurement-state-device-cpu-process.v1'
_FAULTS = (None, 'launch', 'timeout', 'quality', 'allocate-input', 'cleanup')
_NS = 10**9


def _start_tick(pid):
    try:
        # comm can contain spaces and parentheses. Field 22 follows its last ).
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _kill_and_wait(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return process.wait()


def make_capabilities(fixture, fault=None):
    """Return Campaign capabilities for one marked fixture and one fixed fault.

    A fault affects only the first fit case. Roles, ordering, output checks and
    partial cleanup are executed by the same C++ executor as the real adapter.
    No Python substitute generates a timing slot or a resource lifecycle.
    """
    if fault not in _FAULTS:
        raise ValueError('unknown CPU boundary fault')

    class SyntheticCapabilities:
        kind = 'synthetic'

        def __init__(self, campaign):
            from arch_explorer.components.accelsim.microstate import check_cpu_binding
            from arch_explorer.components.accelsim.state_resource import validate_handoff

            self.c = campaign
            self._root = Path(campaign.root).resolve()
            marker = json_file(self._root / 'SYNTHETIC.json')[0]
            if (campaign.kind != self.kind or self._root != Path(fixture.root).resolve()
                    or marker.get('kind') != self.kind
                    or marker.get('no_resource_authority') is not True
                    or marker.get('run_id') != campaign.design['run_id']
                    or marker.get('design') != ref(campaign.design_file)
                    or ref(fixture.design_file) != ref(campaign.design_file)
                    or fixture.design != campaign.design
                    or ref(fixture.bundle.path) != ref(campaign.bundle.path)):
                raise ValueError('capabilities require their own marked synthetic design/root')
            validate_handoff(campaign.bundle, campaign.design, campaign.handoff, synthetic=True)
            lease = json_file(original(fixture.lease_ref))[0]
            expected = {'run_id': campaign.design['run_id'],
                        'design_sha256': campaign.design['design_sha256'],
                        'registry_sha256': campaign.bundle.sha256,
                        'plan_sha256': campaign.design['plan_sha256'],
                        'handoff': campaign.handoff_ref, 'budget': campaign.design['budget']}
            self._soft = (campaign.handoff['absolute_deadline_realtime_ns']
                          - campaign.handoff['close_reserve_s'] * _NS)
            if (any(lease.get(k) != value for k, value in expected.items())
                    or lease.get('deadline_realtime_ns') != self._soft):
                raise ValueError('synthetic lease plan differs from this run/window')
            self._cpu = copy.deepcopy(campaign.design['state_protocol']['cpu'])
            catalog = json_file(original(campaign.design['catalog']))[0]
            # Offline ELF/source/dependency checks reject production executables
            # before Popen. No dlopen or process execution occurs in this check.
            check_cpu_binding(campaign.bundle, self._cpu, catalog['generation'])
            self._clock = max(time.time_ns(), campaign.handoff['issued_realtime_ns'] + 1)
            self._epoch = self._clock
            self._monotonic_epoch = time.monotonic_ns()
            self._acquired = self._released = self._captured = self._predicted = False
            self._halted = False
            self._observed = []
            self._children = []
            self._cpu_reaps = []

        def now(self):
            self._clock = max(time.time_ns(), self._clock + 1)
            return self._clock

        def _open(self):
            if not self._acquired or self._released or self._halted:
                raise ValueError('synthetic boundary is not open; no retry')
            if self.now() >= self._soft:
                raise ValueError('synthetic work deadline reached; cleanup reserve retained')

        def acquire(self):
            if self._acquired or self._released:
                raise ValueError('synthetic plan can be opened only once')
            original(fixture.lease_ref)
            if self.now() >= self._soft:
                raise ValueError('expired synthetic window')
            self._acquired = True
            return copy.deepcopy(fixture.lease_ref)

        def capture(self):
            from arch_explorer.components.accelsim.microcampaign import Collector
            from arch_explorer.components.accelsim.state_device import Journal, PROCESS

            self._open()
            if self._captured:
                raise ValueError('synthetic capture cannot be repeated')
            self._captured = True
            receipts = []

            def host(phase, freeze, cid=None):
                if phase != 'structure' or cid is not None:
                    raise ValueError('capture proxy accepts only the structural file boundary')
                output, receipt = fixture.process('structure', None)
                receipts.append(receipt)
                return output, receipt

            proxy = SimpleNamespace(bundle=self.c.bundle, root=self._root,
                                    design=self.c.design, host=host, journal=Journal(self.c))
            try:
                # Keep the established independent full-byte/address derivation.
                return Collector.structures(proxy)
            finally:
                if receipts:
                    receipt = receipts[-1]
                    raw = json_file(original(receipt))[0]
                    self._clock = max(self._clock, raw['ended_realtime_ns'])
                    frame = {'schema': PROCESS, 'kind': self.kind,
                             'run_id': self.c.design['run_id'], 'index': len(self.c.frames),
                             'case': None, 'phase': 'structure', 'raw': receipt,
                             'ledger': None, 'before': None, 'after': None,
                             'started_realtime_ns': raw['started_realtime_ns'],
                             'ended_realtime_ns': raw['ended_realtime_ns'],
                             'reaped': True, 'execution_status': 'returned', 'error': None}
                    write_new(self._root / f'frame-{len(self.c.frames):02d}.json', frame)
                    self.c.frames.append(frame)

        def predict(self, output, bindings):
            from arch_explorer.components.accelsim.state_device import Journal

            self._open()
            if not self._captured or self._predicted or set(bindings) != set(self.c.bundle.cases):
                raise ValueError('one complete structural prediction freeze required')
            self._predicted = True
            predictions, parameters = fixture.native_predictions, fixture.parameters
            initial = prediction_freeze(self.c.bundle, self.c.design, predictions, parameters,
                                        self._root / 'freezes', stage='initial')
            Journal(self.c).initial_predictions(initial)
            return initial, predictions, parameters

        def _snapshot(self, cid, position, uuid_hex, process=None):
            from arch_explorer.components.accelsim.state_resource import snapshot_policy, validate_snapshot

            policy = snapshot_policy()
            start, end = self.now(), self.now()
            directory = self._root / 'state'
            directory.mkdir(exist_ok=True)
            stdout = directory / f'{cid}-{position}.stdout'
            stderr = directory / f'{cid}-{position}.stderr'
            with stdout.open('xb') as stream:
                stream.write(f'GPU-{uuid_hex}, 100, 200, P0\n'.encode('ascii'))
            with stderr.open('xb'):
                pass
            row = {'schema': 'arch-explorer.measurement-state-snapshot.v1',
                   'kind': self.kind, 'no_resource_authority': True,
                   'run_id': self.c.design['run_id'], 'case': cid, 'position': position,
                   'uuid_hex': uuid_hex, 'argv': policy['query'], 'status': 'returned',
                   'exit_code': 0, 'started_realtime_ns': start, 'ended_realtime_ns': end,
                   'started_monotonic_ns': self._monotonic_epoch + start - self._epoch,
                   'ended_monotonic_ns': self._monotonic_epoch + end - self._epoch,
                   'stdout': ref(stdout), 'stderr': ref(stderr),
                   'values': {'sm_mhz': 100, 'memory_mhz': 200, 'pstate': 'P0'}, 'reason': None}
            validate_snapshot(row, self._snapshot_design(), cid, position, uuid_hex,
                              process, synthetic=True)
            return write_new(directory / f'{cid}-{position}.json', row)

        def _snapshot_design(self):
            return dict(self.c.design,
                        absolute_deadline_realtime_ns=self.c.handoff['absolute_deadline_realtime_ns'],
                        close_reserve_s=self.c.handoff['close_reserve_s'])

        def observe(self, cid, freeze):
            from arch_explorer.components.accelsim.state_device import PROCESS
            from arch_explorer.components.accelsim.state_resource import validate_snapshot

            self._open()
            order = self.c.design['order']['fit'] + self.c.design['order']['heldout']
            if (not self._captured or not self._predicted or len(self._observed) >= len(order)
                    or cid != order[len(self._observed)]):
                raise ValueError('CPU observation must follow the frozen case order once')
            original(freeze)
            executable = original(self._cpu['executable'])
            uuid_hex = json_file(Path(self.c.capture_output) / 'preflight.json')[0]['uuid_hex']
            before = self._snapshot(cid, 'before', uuid_hex)
            start = self.now()
            deadline = min(self._soft, start + 120 * _NS)
            output = self._root / f'cpu-{len(self.c.frames):02d}-{cid}'
            argv = [str(executable), '--bundle', str(self.c.bundle.path.parent),
                    '--case', cid, '--output', str(output), '--run', self.c.design['run_id'],
                    '--freeze', freeze['sha256'], '--start-ns', str(start),
                    '--deadline-ns', str(deadline)]
            if fault is not None and cid == self.c.design['order']['fit'][0]:
                argv += ['--fault', fault]
            stdout = self._root / f'cpu-{len(self.c.frames):02d}-{cid}.stdout'
            stderr = stdout.with_suffix('.stderr')
            self._observed.append(cid)
            timed_out = False
            observed_start = time.time_ns()
            with stdout.open('xb') as out, stderr.open('xb') as err:
                process = subprocess.Popen(argv, cwd=self._root,
                    env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'CUDA_VISIBLE_DEVICES': ''},
                    stdout=out, stderr=err, start_new_session=True)
                self._children.append(process)
                tick = _start_tick(process.pid)
                try:
                    code = process.wait(timeout=(deadline - start) / _NS)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    code = _kill_and_wait(process)
                finally:
                    # Includes interruption/error paths: a CPU child never escapes
                    # into release or a later case, even if no ledger was written.
                    if process.returncode is None:
                        _kill_and_wait(process)
            observed_end = time.time_ns()
            ledger_path = output / 'LEDGER.json'
            ledger_ref = ref(ledger_path) if ledger_path.is_file() else None
            ledger = json_file(ledger_path)[0] if ledger_ref is not None else None
            end = ledger['ended_ns'] if ledger is not None else max(start, observed_end)
            if type(end) is not int or end < start:
                raise ValueError('CPU executor returned a reversed/noninteger clock')
            self._clock = max(self._clock, end)
            status = ('timeout' if timed_out or end > deadline else
                      'returned' if code == 0 and ledger is not None
                      and ledger.get('status') == 'returned' else 'failed')
            error = None if status == 'returned' else f'CPU executor {status}; exit {code}; no retry'
            command = output / 'COMMAND.json'
            if command.is_file() and json_file(command)[0].get('argv') != argv:
                status, error = 'failed', 'actual CPU command differs from frozen argv; no retry'
            raw = {'schema': CPU_PROCESS, 'kind': self.kind, 'no_resource_authority': True,
                   'run_id': self.c.design['run_id'], 'case': cid, 'argv': argv,
                   'executable': copy.deepcopy(self._cpu['executable']), 'pid': process.pid,
                   'start_tick': tick, 'wait_completed': True, 'exit_code': code,
                   'stdout': ref(stdout), 'stderr': ref(stderr), 'ledger': ledger_ref,
                   'clock_injected': True, 'started_realtime_ns': start, 'ended_realtime_ns': end,
                   'observed_started_realtime_ns': observed_start,
                   'observed_ended_realtime_ns': observed_end}
            raw_ref = write_new(self._root / f'cpu-process-{len(self.c.frames):02d}.json', raw)
            self._cpu_reaps.append({'pid': process.pid, 'start_tick': tick, 'wait_completed': True,
                                   'exit_code': code, 'raw': raw_ref})
            self._halted = status != 'returned'
            after = self._snapshot(cid, 'after', uuid_hex, raw)
            validate_snapshot(json_file(original(before))[0], self._snapshot_design(),
                              cid, 'before', uuid_hex, raw, synthetic=True)
            return {'schema': PROCESS, 'kind': self.kind, 'run_id': self.c.design['run_id'],
                    'index': len(self.c.frames), 'case': cid,
                    'phase': self.c.bundle.cases[cid]['partition'], 'raw': raw_ref,
                    'ledger': ledger_ref, 'before': before, 'after': after,
                    'started_realtime_ns': start, 'ended_realtime_ns': end, 'reaped': True,
                    'execution_status': status, 'error': error}

        def release(self):
            # These are CPU waits, not a simulated claim that a GPU host ran.
            if (len(self._cpu_reaps) != len(self._children)
                    or any(child.returncode is None for child in self._children)):
                raise ValueError('CPU process/reap record incomplete')
            self._released = True
            return {'lease': copy.deepcopy(fixture.lease_ref),
                    'raw_processes': [frame['raw'] for frame in self.c.frames], 'reaps': [],
                    'all_owned_hosts_reaped': True, 'common_lock_released': None,
                    'real_locks': 0, 'real_leases': 0, 'ended_realtime_ns': self.now(),
                    'cpu_reaps': copy.deepcopy(self._cpu_reaps)}

    return SyntheticCapabilities
