"""Declarative execution constraints and report hooks shared by all mappings.

The scheduler consumes this contract without knowing a mapping's name or its
component mechanisms. Hooks are evaluation-local; they cannot change execution
after scheduling. A component may declare ``assembly_hooks`` on its type; each
hook receives the completed Simulator once, before any mapping is applied.
"""
from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

_PLANNING_EXECUTION = ContextVar('planning_execution', default=True)


@contextmanager
def planning_execution(allowed):
    """Scope whether a mapping may evaluate trial graphs while planning.

    Preparation/assessment uses False. Ordinary evaluate retains True and its
    existing auto-search behavior. A ContextVar also covers nested mapping
    contexts built by request_space without leaking across evaluations.
    """
    if type(allowed) is not bool:
        raise TypeError('planning execution permission must be bool')
    token = _PLANNING_EXECUTION.set(allowed)
    try:
        yield
    finally:
        _PLANNING_EXECUTION.reset(token)


def require_planning_execution():
    if not _PLANNING_EXECUTION.get():
        raise ValueError('automatic mapping choice requires execution; supply explicit mapping tiles for preparation')


@dataclass
class MemoryGroup:
    """A striped aggregate of memory models, in deterministic endpoint order."""
    path: str
    parts: list

    @property
    def capacity(self):
        return sum(part.capacity for part in self.parts)

    @property
    def latency(self):
        return max(part.latency for part in self.parts)

    @property
    def technology(self):
        return getattr(self.parts[0], 'technology', None)


@dataclass(frozen=True)
class PlanConstraints:
    """Execution requirements supplied by a mapping or component composition.

    ``iterations`` fixes the graph's allowed iteration count. ``scheduler``
    selects a required strategy: the normal ``consistent`` request may be
    refined to that strategy, while an explicit incompatible strategy refuses.
    ``check_capacity`` makes residency checks mandatory even when the caller
    disabled optional capacity checking. ``reason`` explains a refusal.
    """
    iterations: int | None = None
    scheduler: str | None = None
    check_capacity: bool = False
    reason: str = ""

    def __post_init__(self):
        if self.iterations is not None and (type(self.iterations) is not int or self.iterations < 1):
            raise ValueError("plan iterations must be a positive integer")
        if self.scheduler is not None and (not isinstance(self.scheduler, str) or not self.scheduler):
            raise ValueError("plan scheduler must be a nonempty strategy name")
        if type(self.check_capacity) is not bool or not isinstance(self.reason, str):
            raise TypeError("plan capacity requirement and reason must be bool/string")


@dataclass
class ExecutionPlan:
    """One evaluation's requirements and component-owned result construction.

    ``detail_reporters`` take (simulator, trace) and return dictionaries merged
    into ordinary Evaluation details. ``result_builder`` optionally takes
    (simulator, trace, include_trace) and returns an entire Evaluation, for a
    component composition with its own accounting. There is no name dispatch.
    """
    constraints: list[PlanConstraints] = field(default_factory=list)
    detail_reporters: list[Callable] = field(default_factory=list)
    result_builder: Callable | None = None
    scheduler: str | None = field(default=None, init=False)

    def require(self, constraint: PlanConstraints):
        if not isinstance(constraint, PlanConstraints):
            raise TypeError("PlanConstraints required")
        for current in self.constraints:
            for name in ("iterations", "scheduler"):
                old, new = getattr(current, name), getattr(constraint, name)
                if old is not None and new is not None and old != new:
                    raise ValueError(f"incompatible plan {name}: {old!r} and {new!r}")
        self.constraints.append(constraint)

    @property
    def check_capacity(self):
        return any(c.check_capacity for c in self.constraints)

    def resolve(self, graph, *, iterations=None, scheduler="consistent"):
        from arch_explorer.simulator.events.engine import SCHEDULERS
        if scheduler not in (*SCHEDULERS, 'conservative'):
            raise ValueError(f'Unknown scheduler {scheduler!r}')
        effective = graph.iterations if iterations is None else iterations
        if type(effective) is not int or effective < 1:
            raise ValueError("iterations must be a positive integer")
        selected = scheduler
        for constraint in self.constraints:
            prefix = (constraint.reason + ": ") if constraint.reason else ""
            if constraint.iterations is not None and effective != constraint.iterations:
                raise ValueError(f"{prefix}requires iterations={constraint.iterations}")
            if constraint.scheduler is not None:
                if selected not in ("consistent", constraint.scheduler):
                    raise ValueError(f"{prefix}requires scheduler={constraint.scheduler}")
                selected = constraint.scheduler
        if selected not in (*SCHEDULERS, 'conservative'):
            raise ValueError(f'Unknown scheduler {selected!r}')
        graph.iterations, self.scheduler = effective, selected
        return selected
