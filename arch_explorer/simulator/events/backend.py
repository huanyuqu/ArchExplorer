"""The ``events`` evaluation backend: ADF -> simulator -> mapped task graph -> schedule.

``evaluate(adf, WorkloadSpec, MappingSpec)`` (registered in
``arch_explorer.adf.evaluation``) builds the ADF, applies the mapping
template, runs the hardware-consistent scheduler and reports latency, an
operator-level breakdown, resource utilization, the provenance of every
parameter the component models consumed, and warnings. Area and cost are not
modeled by this backend yet and are ``None``.

Options: ``scheduler`` (``consistent``, default, or unified ``conservative``),
``systolic_model`` (``native``, default, or ``scalesim``/``auto`` to run
SCALE-Sim; both give identical cycles, see docs/components.md),
``check_capacity`` (raise when resident tensors exceed a memory's capacity;
otherwise warn), ``iterations`` (override the workload's iteration count),
``registry`` (component registry), ``trace`` (attach per-task records).

``Evaluation.details["stats"]`` holds the scheduler statistics (commits,
rollbacks, ...), task and instance counts, the scheduler name and the peak
occupancy of every memory; ``details["trace"]`` (with ``trace=True``) lists
one record per task instance.
"""
from __future__ import annotations

from collections import defaultdict

from arch_explorer.adf.evaluation import Evaluation, MappingSpec, WorkloadSpec, register_backend
from arch_explorer.adf.registry import COMPONENTS, MAPPING_TEMPLATES
from arch_explorer.simulator.events.builder import build_simulator
from arch_explorer.simulator.events.engine import simulate


class CapacityError(ValueError):
    """Resident tensors exceed a memory's capacity."""


def validate_capacity_allocations(simulator, *, check_capacity=True):
    """Reject a mapped allocation that cannot fit, without running its graph.

    Storage tasks are the mapping's actual allocation authority, including
    tiling, placement and implementation temporaries. Each allocation is a
    lower bound on occupancy; adding their sizes would incorrectly assume
    simultaneous lifetimes. Exact overlap remains runtime trace accounting.
    """
    if not (check_capacity or simulator.mapping.plan.check_capacity):
        return
    for task in simulator.mapping.graph.tasks:
        if task.kind != 'storage' or task.memory is None:
            continue
        memory, size = task.memory
        capacity = simulator.memory(memory).capacity
        if size > capacity * (1 + 1e-12):
            raise CapacityError(
                f"Storage allocation {task.name!r} (operator {task.operator!r}) at {memory}: "
                f"required {size} > available {capacity} bytes")


def _prepare_mapping(adf, workload: WorkloadSpec, mapping: MappingSpec, *, scheduler="consistent",
                    systolic_model="native", check_capacity=True, iterations=None, registry=COMPONENTS, trace=False):
    """Build and validate an execution plan, without executing its graph."""
    import arch_explorer.mapping  # noqa: F401  (registers the templates)
    from arch_explorer.mapping.base import check_params
    from arch_explorer.mapping.context import MappingContext
    if not isinstance(workload, WorkloadSpec) or not isinstance(mapping, MappingSpec):
        raise TypeError("evaluate expects a WorkloadSpec and a MappingSpec")
    simulator = build_simulator(adf, registry, options={"systolic_model": systolic_model})
    simulator.execution_trace = trace
    template = MAPPING_TEMPLATES.get(mapping.template)
    if not template.applicable(adf, workload):
        raise ValueError(f"Mapping template {mapping.template} is not applicable to this ADF/workload")
    check_params(mapping.template, template.request_space(adf, workload, mapping.params), mapping.params)
    simulator.mapping = MappingContext(simulator)
    template.apply(simulator, workload, dict(mapping.params))
    graph = simulator.mapping.graph
    if not graph.tasks:
        raise ValueError("The mapping produced no tasks")
    simulator.mapping.plan.resolve(graph, iterations=iterations, scheduler=scheduler)
    validate_capacity_allocations(simulator, check_capacity=check_capacity)
    return simulator


def prepare_mapping(adf, workload: WorkloadSpec, mapping: MappingSpec, *, allow_planning_execution=False, **options):
    """Construct and validate, with no implicit mapping trial evaluation."""
    from arch_explorer.simulator.plan import planning_execution
    with planning_execution(allow_planning_execution):
        return _prepare_mapping(adf, workload, mapping, **options)


def run_mapping(adf, workload: WorkloadSpec, mapping: MappingSpec, **options):
    """Build, validate and simulate; returns ``(simulator, trace)``."""
    simulator = prepare_mapping(adf, workload, mapping, allow_planning_execution=True, **options)
    return simulator, simulate(simulator.mapping.graph, simulator.mapping.plan.scheduler)


def evaluate_events(adf, workload, mapping, *, scheduler="consistent", systolic_model="native",
                    check_capacity=True, iterations=None, registry=COMPONENTS, trace=False) -> Evaluation:
    simulator, result = run_mapping(adf, workload, mapping, scheduler=scheduler, systolic_model=systolic_model,
                                    check_capacity=check_capacity, iterations=iterations, registry=registry, trace=trace)
    return evaluation_from_trace(simulator, result, trace=trace, check_capacity=check_capacity)


def evaluation_from_trace(simulator, result, *, trace=False, check_capacity=True):
    """Apply ordinary accounting and the execution plan's component reports."""
    plan = simulator.mapping.plan
    warnings = list(simulator.warnings)
    for memory, peak in sorted(result.memory_peak.items()):
        capacity = simulator.memory(memory).capacity
        if peak > capacity * (1 + 1e-12):
            message = f"Memory capacity exceeded at {memory}: {peak:.0f} > {capacity:.0f} bytes"
            if check_capacity or plan.check_capacity:
                raise CapacityError(message)
            warnings.append(message)
    if plan.result_builder is not None:
        evaluation = plan.result_builder(simulator, result, trace)
        if not isinstance(evaluation, Evaluation):
            raise TypeError('execution plan result_builder must return Evaluation')
        return evaluation
    latency = result.latency_ns
    if not latency > 0:
        raise RuntimeError(f"Invalid simulated latency {latency}")
    graph = simulator.mapping.graph
    rows = defaultdict(lambda: {"start_ns": float("inf"), "end_ns": 0.0, "tasks": 0, "resources": set()})
    resources = graph.resources
    for record in result.records:
        task = graph.tasks[record.task]
        if record.kind == "storage" or not record.operator:
            continue
        row = rows[(record.operator, task.meta.get("label", ""), record.kind)]
        row["start_ns"] = min(row["start_ns"], record.start_ns if record.start_ns is not None else record.end_ns)
        row["end_ns"] = max(row["end_ns"], record.end_ns)
        row["tasks"] += 1
        for name in record.resources:
            owner = resources[name].owner
            row["resources"].add(simulator.points[owner].component.name if owner in simulator.points
                                 else owner.rsplit("/", 1)[-1])
    breakdown = [{"operator": operator, "label": label, "task_type": kind, "resource": ",".join(sorted(r["resources"])),
                  "start_ns": r["start_ns"], "end_ns": r["end_ns"], "tasks": r["tasks"]}
                 for (operator, label, kind), r in rows.items()]
    breakdown.sort(key=lambda row: (row["start_ns"], row["operator"], row["task_type"]))
    utilization = {}
    for name, used in result.resource_usage.items():
        resource = resources[name]
        utilization[name] = used / (resource.capacity * latency)
    details = {"stats": dict(result.stats, tasks=len(graph.tasks), instances=len(result.records),
                             scheduler=plan.scheduler,
                             memory_peak=dict(result.memory_peak))}
    details.update(simulator.mapping.details)
    details['component_implementations'] = [dict(path=path, **model.component_type.identity())
        for path, model in simulator.points.items() if hasattr(model.component_type, 'identity')]
    for reporter in plan.detail_reporters:
        extra = reporter(simulator, result)
        if not isinstance(extra, dict):
            raise TypeError('execution plan detail reporter must return a dictionary')
        details.update(extra)
    if trace:
        details["trace"] = [dict(record.__dict__) for record in result.records]
    return Evaluation(latency_ns=latency, breakdown=breakdown, utilization=utilization, warnings=warnings,
                      provenance=simulator.provenance.as_list(), details=details)


def label_latency(evaluation, device=None):
    """Sum of operator spans per breakdown label (e.g. QKV, Attention) for kernels run in order."""
    spans = defaultdict(dict)
    for row in evaluation.breakdown:
        if row["task_type"] == "sync":
            continue
        span = spans[row["label"]].setdefault(row["operator"], [float("inf"), 0.0])
        span[0], span[1] = min(span[0], row["start_ns"]), max(span[1], row["end_ns"])
    return {label: sum(end - start for start, end in ops.values()) for label, ops in spans.items()}


register_backend("events", evaluate_events)
