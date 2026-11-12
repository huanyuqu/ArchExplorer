"""Clocked, resumable instruction/transaction composition for unified execution.

One binding owns the core(s), controller(s), queues and their internal response
mailbox. Programs yield before advancing time. Only advance() steps hardware;
no calendar callback runs a sub-simulation or precomputes a program's latency.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from arch_explorer.components.api import ExecutionEvent


@dataclass
class Await:
    predicate: object
    earliest: int
    core: object = None


class InstructionProgramModel:
    def __init__(self, memory, *, clock_ns=Fraction(1), trace=False):
        self.memory = memory
        self.clock_ns = Fraction(clock_ns)
        if self.clock_ns <= 0:
            raise ValueError('positive clock period required')
        if self.clock_ns.denominator == 1:
            self.clock_ns = int(self.clock_ns)
        self.keep_trace = trace
        self.request = None
        self.program = None
        self.waiting = None
        self.next_tick = None
        self.serial = 0
        self.rows = []
        self.operations = []
        self.instructions = []
        self.result = None
        self.last_tick = -1

    def submit(self, request, now):
        if self.request is not None:
            return False
        if request.operation != 'program' or not callable(request.payload):
            raise ValueError('instruction execution requires a resumable program factory')
        if now < request.available_time:
            raise ValueError('request not yet available')
        self.request = request
        self.arrival = now
        tick = ceil(Fraction(now)/self.clock_ns)
        self.program = request.payload(self, tick)
        self.waiting = next(self.program)
        if not isinstance(self.waiting, Await):
            raise TypeError('programs must yield Await boundaries')
        self.next_tick = tick
        return True

    def next_event_time(self):
        if self.next_tick is not None:
            return self.next_tick*self.clock_ns
        # Outstanding physical work continues during unrelated Demand tasks.
        # Refresh phase also advances; a finite next tick is always safe.
        if self.memory.active:
            return (self.last_tick+1)*self.clock_ns
        return None

    def _event(self, tick, kind, request_id='', payload=None, phase=30):
        self.serial += 1
        return ExecutionEvent(tick*self.clock_ns, phase, self.serial, request_id, kind, payload)

    def advance(self, now):
        tick = Fraction(now)/self.clock_ns
        if tick.denominator != 1:
            raise ValueError('advance must be at a component clock edge')
        tick = int(tick)
        if tick < self.last_tick:
            raise ValueError('component time moved backwards')
        self.memory.step(tick)
        self.last_tick = tick
        if self.request is None:
            return []
        # A response delivered by memory.step is consumed in the core phase,
        # preserving source response -> memory issue -> core retire/issue order.
        while True:
            if self.waiting is not None:
                if tick < self.waiting.earliest:
                    break
                if self.waiting.core is not None:
                    self.waiting.core.step(tick, self.memory)
                if not self.waiting.predicate():
                    break
            try:
                self.waiting = self.program.send(tick) if self.waiting is not None else next(self.program)
            except StopIteration as done:
                ident = self.request.request_id
                self.result = done.value
                self.request = self.program = self.waiting = None
                self.next_tick = None
                events = [self._event(tick, 'consumer_ready', ident),
                          self._event(tick, 'queue_available')]
                # A completed consumer is not necessarily a drained controller.
                physical = (self.memory.physically_ready(tick) if hasattr(self.memory,'physically_ready')
                            else not self.memory.active)
                if physical:
                    events.append(self._event(tick, 'physical_complete', ident))
                return events
            if not isinstance(self.waiting, Await):
                raise TypeError('programs must yield Await boundaries')
        self.next_tick = tick+1
        return []


def matrix_program(tiles, height, width, *, operator='', label='', policy=None):
    def program(model, start):
        from arch_explorer.components.stateful.neupims_core import InstructionCore
        options = {} if policy is None else {'policy': policy}
        core = InstructionCore(tiles(), height, width, **options)
        memory = model.memory
        before, logical = list(memory.physical), list(memory.logical)
        end = yield Await(lambda: not core.active, start, core)
        model.operations.append(dict(operator=operator, tiles=len(core.finished),
            logical_bytes=[a-b for a,b in zip(memory.logical,logical)],
            physical_bytes=[a-b for a,b in zip(memory.physical,before)],
            compute_instructions=sum(r['event'] in ('GEMM','GEMM_PRELOAD') for r in core.trace)))
        if model.keep_trace:
            model.instructions.extend(dict(r,operator=operator) for r in core.trace)
        model.rows.append(dict(operator=operator,label=label,task_type='compute',
            resource='npu-instruction-core',start_ns=model.arrival,end_ns=end,tasks=len(core.finished)))
        return end
    return program


class LinearAddressStream:
    """Explicit tensor base addresses for synthetic/general region transfers."""
    def __init__(self, bases, layout):
        self.bases = dict(bases)
        self.layout = layout
        self.cursor = 0

    def region_bursts(self, region):
        size = self.layout.burst_bytes
        bursts = {}
        for address, count in region.spans(self.bases[region.tensor]):
            for block in range(address//size, (address+count-1)//size+1):
                address = block*size
                self.layout.decode(address)
                bursts.setdefault(address, None)
        return tuple(bursts)
