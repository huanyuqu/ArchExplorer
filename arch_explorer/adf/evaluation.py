"""Evaluation request/result types and the backend dispatch used by search and agents.

Search loops, the agent environment and the CLI call :func:`evaluate`; the
simulator registers the ``events`` backend that builds the ADF, applies the
mapping template and runs the discrete-event scheduler. Results always carry
the provenance of the parameters that produced them, and metrics a backend
cannot model are ``None`` rather than zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Callable, Optional

# Built-in names retained for clients that enumerate the shipped examples.
# Installed builders extend the contract through the workload registry.
WORKLOAD_KINDS = {"operator", "transformer", "vit", "clip", "pi05", "graph"}


@dataclass(frozen=True)
class WorkloadSpec:
    """One workload of a suite.

    ``kind="operator"``: ``params`` names an operator (``matmul``, ``mvm``,
    ``softmax`` ...) and its shape. ``kind="transformer"``: ``params`` gives a
    model (registered name or config fields), ``phase`` (prefill/decode),
    ``batch``, ``seq_len`` and the layers to simulate. ``weight`` is the
    suite weight w_w of the search objective.

    ``vit`` and ``clip`` select complete forward graphs using ``params.profile``.
    CLIP is one workload including both towers and similarity outputs. Profile
    validation and graph construction belong to the workload dispatcher; this
    transport type does not duplicate their model constants or semantics.
    ``pi05`` uses the sole declarative model_specs contract. Its executor must
    explicitly advertise that version before any backend call is permitted.
    """
    kind: str
    params: dict
    weight: float = 1.0
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("workload kind must be a nonempty string")
        from arch_explorer.workload.extensions import has_workload
        if not has_workload(self.kind):
            raise ValueError(f"unknown workload kind {self.kind!r}")
        if not isinstance(self.params, dict):
            raise ValueError("workload params must be a JSON object")
        json.dumps(self.params, allow_nan=False)
        if self.kind == "pi05":
            from arch_explorer.model_specs import normalize
            object.__setattr__(self, "params", normalize(self.params))
        if (isinstance(self.weight, bool) or not isinstance(self.weight, (int, float))
                or not math.isfinite(self.weight) or self.weight <= 0):
            raise ValueError("workload weight must be positive")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MappingSpec:
    """A mapping template name plus concrete values for its parameters."""
    template: str
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class Evaluation:
    """Result of simulating one workload on one ADF with one mapping.

    ``breakdown`` rows: ``{"operator", "task_type", "resource", "start_ns",
    "end_ns"}``; backends may add keys such as ``label`` (the operator group,
    e.g. QKV or Attention) and ``tasks``. ``utilization`` maps resource paths
    to busy fraction. ``provenance`` lists ``{"path", "param", "value",
    "unit", "source"}`` for every parameter the evaluators consumed.
    ``details`` holds backend-specific run data such as scheduler statistics
    and, when requested, the per-task trace.
    """
    latency_ns: float
    breakdown: list = field(default_factory=list)
    utilization: dict = field(default_factory=dict)
    area_mm2: Optional[float] = None
    cost_usd: Optional[float] = None
    warnings: list = field(default_factory=list)
    provenance: list = field(default_factory=list)
    backend: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


_BACKENDS: dict = {}


def register_backend(name: str, function: Callable) -> None:
    """Register ``function(adf, workload, mapping, **options) -> Evaluation``."""
    if name in _BACKENDS and _BACKENDS[name] is not function:
        raise ValueError(f"Backend {name!r} is already registered")
    _BACKENDS[name] = function


def backends():
    from arch_explorer.components.extensions import discover_components
    from arch_explorer.components.extensions import providers
    discover_components()
    return sorted(set(_BACKENDS) | {p['id'] for p in providers()})


def supports_model_spec(backend, kind):
    """Only an executor's explicit contract declaration enables a new model."""
    from arch_explorer.model_specs import contract_id
    from arch_explorer.components.extensions import discover_components
    from arch_explorer.components.extensions import get_provider
    discover_components()
    version = contract_id(kind)
    if version is None:
        return True
    contracts = getattr(_BACKENDS.get(backend), "model_spec_contracts", {})
    if backend not in _BACKENDS:
        try:
            contracts = get_provider(backend).capabilities.get('model_spec_contracts', {})
        except NotImplementedError:
            return False
    return contracts.get(kind) == version


def _evaluate_provider(identifier, adf, workload, mapping, **options):
    """Adapt a registered execution capability to the historical Evaluation API."""
    from pathlib import Path
    import tempfile
    from arch_explorer.simulator.api import prepare_request, evaluate_request
    parent = options.pop('output_dir', None)
    if parent is not None:
        Path(parent).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='provider-', dir=parent)) / 'request'
    request = prepare_request(adf, workload, mapping, output_dir=directory,
                              provider=identifier, options=options)
    return Evaluation(**evaluate_request(request)['evaluation'])


def evaluate(adf, workload: WorkloadSpec, mapping: MappingSpec, *, backend: str = "events",
             cost_model=None, **options) -> Evaluation:
    """Run ``backend`` on one workload and mapping.

    When the backend leaves both ``area_mm2`` and ``cost_usd`` unset, they are
    filled from :func:`arch_explorer.cost.estimate` with ``cost_model`` as its
    options (``None`` uses the model defaults, ``False`` skips the estimate),
    and ``details["cost"]`` records the basis and unsupported paths. Metrics
    the cost model cannot estimate stay ``None``.
    """
    from arch_explorer.components.extensions import discover_components
    discovery = discover_components()
    if backend not in _BACKENDS:
        from arch_explorer.components.extensions import get_provider
        try:
            get_provider(backend)
        except NotImplementedError as error:
            raise NotImplementedError(f"No evaluation backend {backend!r}; registered: {backends()}; "
                                      f"extension failures: {discovery['failures']}") from error
    if workload.kind == "pi05":
        from arch_explorer.model_specs import normalize
        normalize(workload.params)  # Also reject mutation of a nested params object.
    if not supports_model_spec(backend, workload.kind):
        raise NotImplementedError(f"unsupported_workload: {workload.kind} full-model executor not registered for {backend}")
    result = (_BACKENDS[backend](adf, workload, mapping, **options) if backend in _BACKENDS
              else _evaluate_provider(backend, adf, workload, mapping, **options))
    if not isinstance(result, Evaluation):
        raise TypeError(f"Backend {backend!r} returned {type(result).__name__}, not Evaluation")
    result.backend = backend
    if cost_model is not False and result.area_mm2 is None and result.cost_usd is None:
        from arch_explorer.cost import estimate
        cost = estimate(adf, options=cost_model or None)
        result.area_mm2, result.cost_usd = cost.area_mm2, cost.cost_usd
        result.details["cost"] = {"model": cost.model, "basis": cost.basis, "raw_cost_usd": cost.raw_cost_usd,
                                  "full_cost_usd": cost.full_cost_usd, "by_category": cost.by_category,
                                  "unsupported": cost.unsupported}
    return result
