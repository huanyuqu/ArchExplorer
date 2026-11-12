"""A GDDR6 shared-row controller implementing the unified ExecutionModel protocol.

The instruction frontend is an explicit, deterministic synthetic stream feeder.
Command timing/mode and ordinary-vs-AiM admission follow the fixed AiM source.
Refresh is channel-local all-bank periodic maintenance (see hsndi-detail.md).
"""
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math

from arch_explorer.components.aim_profile import AimCommand, AimProfile, LATENCY, REG, timing_rules
from arch_explorer.components.api import ExecutionEvent, valid_time


@dataclass
class _Item:
    client: str
    command: AimCommand
    arrive: int
    index: int


class AimController:
    """One owner for one channel; instances must be created anew per evaluation.

    ``stream`` takes a tuple of AimCommand; other requests take an AimCommand
    matching operation. Program admission and hardware queues are separately
    bounded by queue_size. Rejected submit has no side effects.
    """
    def __init__(self, profile=None, trace=None):
        self.profile = profile or AimProfile()
        self.rules = timing_rules(self.profile)
        self.trace = trace if trace is not None else []
        self.rows = [None] * 16
        self.deadlines = {}
        self.queues = {k: [] for k in ('read', 'write', 'aim', 'active')}
        self.clients = {}
        self.seen = set()
        self.read_returns = deque()
        self.other_returns = []
        self.tick = -1
        self.wake = None
        self.serial = 0
        self.reg_mode = False
        self.write_mode = False
        self.next_refresh = self.profile.refi
        self.refresh_debt = 0
        self.refresh_end = 0
        self.bus_windows = []
        self.data_pending = []
        self.bytes = dict(external_read=0, external_write=0, bank_read=0, bank_write=0,
                          useful_external=0, mac_operations=0)
        self.accounting = dict(logical=0, accepted=0, issued=0, forwarded=0,
                               returned=0, physical_completed=0, pending=0,
                               mode_blocked_ticks=0, queue_blocked_ticks=0)

    def submit(self, request, now):
        valid_time(now)
        if len(self.clients) >= self.profile.queue_size:
            return False
        if request.request_id in self.seen:
            raise ValueError('duplicate AiM execution request')
        if now < request.available_time or now < self.tick * self.profile.clock_ns:
            raise ValueError('AiM request time precedes availability/controller time')
        if self.wake is not None and now > self.next_event_time():
            raise ValueError('advance the pending AiM edge before submitting at a later time')
        commands = request.payload if request.operation == 'stream' else (request.payload,)
        if not isinstance(commands, tuple) or not commands:
            raise ValueError('AiM stream must be a nonempty tuple of AimCommand')
        for command in commands:
            if not isinstance(command, AimCommand):
                raise TypeError('AiM request requires AimCommand payloads')
            command.validate(self.profile)
        if request.operation != 'stream' and (len(commands) != 1 or commands[0].operation != request.operation):
            raise ValueError('AiM operation/payload mismatch')
        edge = math.ceil(Fraction(now) / self.profile.clock_ns)
        # When quiescent, account for maintenance during an idle gap before
        # accepting new work. No request events can be produced in this gap.
        if self.wake is None:
            for tick in range(self.tick + 1, edge):
                self._step(tick)
        self.seen.add(request.request_id)
        self.clients[request.request_id] = dict(commands=commands, cursor=0, left=len(commands),
                                              physical=len(commands), available=edge)
        self.accounting['logical'] += len(commands)
        self.accounting['pending'] += len(commands)
        self.trace.append(dict(tick=edge, event='submit', request=request.request_id,
                               commands=len(commands)))
        self.wake = max(self.tick + 1, edge) if self.wake is None else min(self.wake, max(self.tick + 1, edge))
        return True

    def next_event_time(self):
        # Effective controller ticks are 1 ns; JSON output is a numeric time.
        return None if self.wake is None else float(self.wake * self.profile.clock_ns)

    @staticmethod
    def _queue(command):
        return 'read' if command.operation == 'RD' else 'write' if command.operation == 'WR' else 'aim'

    def _feed(self, tick):
        # One instruction per client per edge: a declared software frontend,
        # not an inferred HSNDI/IPU host throughput. Backpressure retries do
        # not advance the cursor or duplicate commands/bytes.
        for cid, client in self.clients.items():
            i = client['cursor']
            if tick < client['available'] or i == len(client['commands']):
                continue
            command = client['commands'][i]
            q = self._queue(command)
            if q == 'aim' and (self.queues['read'] or self.queues['write']):
                self.accounting['mode_blocked_ticks'] += 1
                continue
            if q != 'aim' and self.queues['aim']:
                self.accounting['mode_blocked_ticks'] += 1
                continue
            if len(self.queues[q]) >= self.profile.queue_size:
                self.accounting['queue_blocked_ticks'] += 1
                continue
            item = _Item(cid, command, tick, i)
            self.accounting['accepted'] += 1
            self.trace.append(dict(tick=tick, event='accept', request=cid, index=i,
                                   command=command.operation, address=command.address,
                                   useful_bytes=command.useful_bytes))
            # Source forwarding: a queued ordinary write satisfies the read.
            if q == 'read' and any(x.command.address == command.address for x in self.queues['write']):
                self.accounting['forwarded'] += 1
                self.read_returns.append((tick + 1, item))
                self.data_pending.append((tick + 1, item))
                self.trace.append(dict(tick=tick, event='forward', request=cid, index=i))
            else:
                self.queues[q].append(item)
            client['cursor'] += 1

    def _preq(self, item):
        op = item.command.operation
        bank, row, _ = self.profile.decode(item.command.address)
        if op in ('RD', 'WR'):
            return 'ACT' if self.rows[bank] is None else 'PRE' if self.rows[bank] != row else op
        if op == 'MAC16':
            if any(r is not None and r != row for r in self.rows):
                return 'PREA'
            if any(r is None for r in self.rows):
                return 'ACT16'
        return op

    def _ready(self, cmd, bank, tick):
        if tick < max(self.deadlines.get((scope, target, cmd), 0)
                      for scope, target in (('channel', 0), ('group', bank // 4), ('bank', bank))):
            return False
        if cmd in ('RD', 'WR', 'WRGB', 'WRMAC16', 'RDMAC16'):
            start = tick + LATENCY[cmd] - 2
            if any(start < end and begin < start + 2 for begin, end in self.bus_windows):
                return False
        return True

    def _best(self, queue, tick):
        if not queue:
            return None
        def key(item):
            bank = self.profile.decode(item.command.address)[0]
            return (not self._ready(self._preq(item), bank, tick), item.arrive)
        return min(queue, key=key)

    def _issue(self, cmd, bank, row, tick, item=None):
        for (scope, before, after), delay in self.rules.items():
            if before == cmd:
                target = 0 if scope == 'channel' else bank // 4 if scope == 'group' else bank
                key = (scope, target, after)
                self.deadlines[key] = max(self.deadlines.get(key, 0), tick + delay)
        if cmd == 'ACT':
            self.rows[bank] = row
        elif cmd == 'ACT16':
            self.rows[:] = [row] * 16
        elif cmd == 'PRE':
            self.rows[bank] = None
        elif cmd == 'PREA':
            self.rows[:] = [None] * 16
        elif cmd == 'TMOD':
            self.reg_mode = not self.reg_mode
        elif cmd == 'REFab':
            self.refresh_debt -= 1
            self.refresh_end = tick + self.profile.rfc
        entry = dict(tick=tick, event='issue', command=cmd, bank=bank, row=row,
                     request=None if item is None else item.client,
                     index=None if item is None else item.index)
        self.trace.append(entry)
        if item is None or cmd != item.command.operation:
            return
        self.accounting['issued'] += 1
        self.bytes['external_read'] += 32 if cmd in ('RD', 'RDMAC16') else 0
        self.bytes['external_write'] += 32 if cmd in ('WR', 'WRGB', 'WRMAC16') else 0
        self.bytes['bank_read'] += 512 if cmd == 'MAC16' else 32 if cmd == 'RD' else 0
        self.bytes['bank_write'] += 32 if cmd == 'WR' else 0
        self.bytes['mac_operations'] += 512 if cmd == 'MAC16' else 0
        self.bytes['useful_external'] += item.command.useful_bytes if cmd != 'MAC16' else 0
        due = tick + LATENCY[cmd]
        entry['data_ready_tick'] = due
        if cmd != 'MAC16':
            self.bus_windows.append((due - 2, due))
            entry['bus_start_tick'] = due - 2
            entry['bus_end_tick'] = due
        self.data_pending.append((due, item))
        entry['precharge_legal_tick'] = max(self.deadlines.get(('channel', 0, 'PREA'), 0),
                                           self.deadlines.get(('bank', bank, 'PRE'), 0))
        if cmd in ('RD', 'RDMAC16'):
            self.read_returns.append((due, item))
        else:
            self.other_returns.append((due, item))

    def _event(self, tick, cid, kind, phase):
        self.serial += 1
        return ExecutionEvent(float(tick * self.profile.clock_ns), phase, self.serial, cid, kind)

    def _returned(self, tick, item, out):
        self.accounting['returned'] += 1
        self.accounting['pending'] -= 1
        self.trace.append(dict(tick=tick, event='return', request=item.client, index=item.index,
                               command=item.command.operation))
        client = self.clients[item.client]
        client['left'] -= 1
        if client['left'] == 0:
            out.append(self._event(tick, item.client, 'consumer_ready', 0))
            del self.clients[item.client]
            out.append(self._event(tick, item.client, 'queue_available', 1))

    def _step(self, tick):
        self.tick = tick
        out = []
        self.bus_windows[:] = [(a, b) for a, b in self.bus_windows if b > tick]
        data_pending = []
        for due, item in self.data_pending:
            if due <= tick:
                self.accounting['physical_completed'] += 1
                self.trace.append(dict(tick=tick, event='data_complete', request=item.client,
                                       index=item.index, command=item.command.operation))
                self.clients[item.client]['physical'] -= 1
                if self.clients[item.client]['physical'] == 0:
                    out.append(self._event(tick, item.client, 'physical_complete', 0))
                    out.append(self._event(tick, item.client, 'resource_ready', 0))
            else:
                data_pending.append((due, item))
        self.data_pending = data_pending
        # Source order: one FIFO reader response, all due nonread responses,
        # refresh enqueue, command arbitration. No new synchronization task.
        if self.read_returns and self.read_returns[0][0] <= tick:
            _, item = self.read_returns.popleft()
            self._returned(tick, item, out)
        pending = []
        for due, item in self.other_returns:
            if due <= tick:
                self._returned(tick, item, out)
            else:
                pending.append((due, item))
        self.other_returns = pending
        if tick >= self.next_refresh:
            self.refresh_debt += 1
            self.next_refresh += self.profile.refi
            self.trace.append(dict(tick=tick, event='refresh_due'))
        self._feed(tick)
        q = 'active'
        item = self._best(self.queues[q], tick)
        active_ready = item is not None and self._ready(self._preq(item), self.profile.decode(item.command.address)[0], tick)
        if not active_ready:
            if self.refresh_debt:
                cmd = 'PREA' if any(r is not None for r in self.rows) else 'REFab'
                if not self.queues['active'] and tick >= self.refresh_end and self._ready(cmd, 0, tick):
                    self._issue(cmd, 0, 0, tick)
                return out
            if self.queues['aim']:
                q, item = 'aim', self.queues['aim'][0]
            else:
                reads, writes = self.queues['read'], self.queues['write']
                # Source declared watermarks 0.8/0.2 of max_size=32.
                if not self.write_mode and (len(writes) > 0.8 * 32 or not reads):
                    self.write_mode = True
                elif self.write_mode and len(writes) < 0.2 * 32 and reads:
                    self.write_mode = False
                q = 'write' if self.write_mode else 'read'
                item = self._best(self.queues[q], tick)
        if item is None:
            return out
        bank, row, _ = self.profile.decode(item.command.address)
        cmd = self._preq(item)
        if tick < self.refresh_end or not self._ready(cmd, bank, tick):
            return out
        if cmd in ('PRE', 'PREA') and any(cmd == 'PREA' or self.profile.decode(x.command.address)[0] == bank
                                        for x in self.queues['active']):
            return out
        if (item.command.operation in REG) != self.reg_mode:
            cmd = 'TMOD'
        if cmd == 'ACT' and q != 'active' and len(self.queues['active']) >= self.profile.queue_size:
            return out
        self._issue(cmd, bank, row, tick, item)
        if cmd == item.command.operation:
            self.queues[q].remove(item)
        elif cmd == 'ACT' and q != 'active':
            self.queues[q].remove(item)
            self.queues['active'].append(item)
        return out

    def advance(self, now):
        if self.next_event_time() != now:
            raise ValueError('advance must use the advertised AiM clock edge')
        tick = int(Fraction(now) / self.profile.clock_ns)
        out = self._step(tick)
        self.wake = tick + 1 if self.clients or self.refresh_debt or tick < self.refresh_end else None
        return out
