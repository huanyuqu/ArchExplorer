"""Evaluation-local execution contracts. Times may be exact Fractions in ns."""
from dataclasses import dataclass, field
from typing import Protocol
import math


def valid_time(time):
    if isinstance(time, bool) or not math.isfinite(time) or time < 0:
        raise ValueError(f'invalid execution time {time!r}')
    return time


@dataclass(frozen=True)
class ExecutionRequest:
    request_id: str
    component: str
    available_time: object = 0
    operation: str = ''
    payload: object = None
    operator: str = ''
    iteration: int = 0

    def __post_init__(self):
        valid_time(self.available_time)
        if not self.request_id or not self.component:
            raise ValueError('request_id and component are required')


@dataclass(frozen=True, order=True)
class ExecutionEvent:
    time: object
    phase: int
    serial: int
    request_id: str = field(compare=False)
    kind: str = field(compare=False)
    payload: object = field(default=None, compare=False)

    def __post_init__(self):
        valid_time(self.time)
        if type(self.phase) is not int or type(self.serial) is not int:
            raise ValueError('event phase and serial must be integers')
        if self.kind not in {'consumer_ready', 'physical_complete', 'resource_ready', 'queue_available'}:
            raise ValueError(f'unknown event kind {self.kind}')


class ExecutionModel(Protocol):
    def submit(self, request: ExecutionRequest, now) -> bool:
        """False means backpressure, with no mutation or side effects."""
    def next_event_time(self): ...
    def advance(self, now) -> list[ExecutionEvent]: ...


@dataclass(frozen=True)
class DemandAdapter:
    """Stateless demand -> graph node; arbitration remains in the unified driver."""
    demand: dict
    latency_ns: float = 0

    def task(self, name, **kwargs):
        from .engine import Task
        return Task(name, demand=dict(self.demand), latency_ns=self.latency_ns, **kwargs)
