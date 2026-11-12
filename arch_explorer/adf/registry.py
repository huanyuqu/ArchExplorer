"""Component and mapping-template registries shared by builder, search and agents.

A component type declares its parameters (unit, whether the quantity is
divided when the component is split, tunable domain) and the conventions its
evaluator follows (HPCA draft, section III-B1 and response E1). The simulator
side implements ``build``; structure primitives only need the declarative
part, so they can split, merge and validate components without simulating.

A mapping template is a parameterised mapping strategy (section IV-C3). Its
``space`` lists the parameters the inner search may choose; ``apply`` places
the workload's tasks on an instantiated simulator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar, Optional

from arch_explorer.adf.model import SpacePoint

ROLES = {"compute", "memory", "network"}
ARBITRATIONS = {"exclusive", "fair_share", "fifo", "priority"}


@dataclass(frozen=True)
class ParamSpec:
    """Declaration of one component parameter.

    ``extensive`` quantities (capacity, bandwidth, unit count) are divided by
    ``split`` and summed by ``merge``; intensive ones (clock, latency, array
    shape) are copied. ``domain`` is the tunable range used by the parameter
    search: a ``(low, high)`` tuple or a list of choices; ``None`` means fixed.
    """
    unit: str
    kind: str = "float"
    extensive: bool = False
    domain: Optional[object] = None
    required: bool = True
    doc: str = ""


@dataclass(frozen=True)
class Conventions:
    """What an evaluator's numbers mean, checked when components are bound.

    ``time_base`` is ``"ns"`` or ``"cycle"``; cycle-based components must
    declare a ``clock_ghz`` parameter so binding can convert to nanoseconds.
    ``arbitration`` states how concurrent tasks share the component; the
    scheduler finds every contending set and the evaluator applies this rule.
    """
    time_base: str = "ns"
    arbitration: str = "exclusive"
    pipelined: bool = False
    includes_startup: bool = True


class ComponentType(ABC):
    api_version: ClassVar[str] = '1'
    name: ClassVar[str]
    role: ClassVar[str]
    params: ClassVar[dict] = {}
    conventions: ClassVar[Conventions] = Conventions()
    execution_modes: ClassVar[tuple] = ("demand",)  # optional "stateful" capability
    assembly_hooks: ClassVar[tuple] = ()  # evaluation-local bind hooks after all points exist

    def validate(self, point: SpacePoint) -> None:
        unknown = set(point.params) - set(self.params)
        if unknown:
            raise ValueError(f"{point.name}: unknown parameters for {self.name}: {sorted(unknown)}")
        for key, spec in self.params.items():
            if key not in point.params:
                if spec.required:
                    raise ValueError(f"{point.name}: {self.name} requires parameter {key}")
                continue
            param = point.params[key]
            if param.unit != spec.unit:
                raise ValueError(f"{point.name}.{key}: unit {param.unit!r} != declared {spec.unit!r}")
            expected = {"int": int, "float": (int, float), "str": str, "bool": bool}[spec.kind]
            if isinstance(param.value, bool) and spec.kind != "bool" or not isinstance(param.value, expected):
                raise ValueError(f"{point.name}.{key}: expected {spec.kind}")
        if self.conventions.time_base == "cycle" and "clock_ghz" not in self.params:
            raise ValueError(f"{self.name} is cycle-based but declares no clock_ghz")

    def split(self, point: SpacePoint, num: int, algo: str = "even") -> list:
        """Partition one component into ``num`` sub-components (structure primitive split)."""
        if type(num) is not int or num < 2:
            raise ValueError("split needs an integer num >= 2")
        if algo != "even":
            raise ValueError(f"{self.name} does not support split algorithm {algo!r}")
        params = {}
        for key, param in point.params.items():
            if self.params[key].extensive:
                value = param.value / num
                if self.params[key].kind == "int":
                    if param.value % num:
                        raise ValueError(f"{point.name}.{key}={param.value} is not divisible by {num}")
                    value = param.value // num
                param = replace(param, value=value, source=f"split({num}) of {param.source or point.name}")
            params[key] = param
        from copy import deepcopy
        from arch_explorer.components.implementations import split_binding
        return [SpacePoint(f"{point.name}_{i}", self.name, dict(params), implementation=deepcopy(split_binding(point,num)))
                for i in range(num)]

    def merge(self, points: list, algo: str = "sum") -> SpacePoint:
        """Coalesce components of this type back into one (structure primitive merge)."""
        if len(points) < 2 or any(p.component != self.name for p in points):
            raise ValueError(f"merge needs at least two {self.name} points")
        if algo != "sum":
            raise ValueError(f"{self.name} does not support merge algorithm {algo!r}")
        from arch_explorer.components.implementations import merge_bindings
        binding=merge_bindings(points)
        first, params = points[0], {}
        for key, param in first.params.items():
            values = [p.params[key].value for p in points]
            if self.params[key].extensive:
                param = replace(param, value=sum(values), source=f"merge of {len(points)} {self.name}")
            elif any(v != param.value for v in values):
                raise ValueError(f"Cannot merge {self.name} points with different {key}")
            params[key] = param
        from copy import deepcopy
        return SpacePoint(first.name, self.name, params, implementation=deepcopy(binding))

    @abstractmethod
    def build(self, point: SpacePoint, context):
        """Return the simulator SpacePoint (with its evaluator) for one instance."""

    def area_mm2(self, point: SpacePoint) -> Optional[float]:
        return None

    def cost_usd(self, point: SpacePoint) -> Optional[float]:
        return None


@dataclass
class ConditionalMappingSpace:
    """A small union of flat products, never a product of inactive parameters.

    ``branches`` are disjoint active parameter dictionaries. ``defaults`` apply
    only to abbreviated direct API calls; complete search/replay mappings must
    supply every active key. ``sampling_defaults`` keep costly paths opt-in.
    ``unavailable`` explains branches excluded by hardware/workload validation.
    """
    branches: list
    defaults: dict
    sampling_defaults: dict
    unavailable: list

    def to_dict(self):
        return {"kind": "conditional", "branches": self.branches,
                "direct_defaults": self.defaults, "sampling_defaults": self.sampling_defaults,
                "unavailable": self.unavailable}


def _choice(value, choices):
    # bool is not an integer mapping choice, despite Python's True == 1.
    return any(value == item and isinstance(value, bool) == isinstance(item, bool) for item in choices)


def mapping_spaces(space, selectors=None):
    """Resolve finite active products for sampling (flat templates unchanged)."""
    selectors = {} if selectors is None else selectors
    if not isinstance(selectors, dict):
        raise ValueError("mapping selectors must be an object")
    if isinstance(space, ConditionalMappingSpace):
        branches = space.branches
        selectors = {**space.sampling_defaults, **selectors}
    else:
        branches = [space]
    result = []
    for branch in branches:
        if any(not isinstance(v, (list, tuple)) or not v for v in branch.values()):
            raise ValueError("mapping space must contain nonempty choice lists")
        if all(k in branch and _choice(v, branch[k]) for k, v in selectors.items()):
            result.append({k: [selectors[k]] if k in selectors else list(v) for k, v in branch.items()})
    if not result:
        unavailable = space.unavailable if isinstance(space, ConditionalMappingSpace) else []
        raise ValueError(f"No supported mapping branch for selectors {selectors}; unavailable: {unavailable}")
    return result


def validate_mapping_params(space, params, *, complete=False):
    """Validate without discarding keys. Return normalized direct-call parameters."""
    if not isinstance(params, dict):
        raise ValueError("mapping parameters must be an object")
    conditional = isinstance(space, ConditionalMappingSpace)
    values = {**space.defaults, **params} if conditional and not complete else dict(params)
    branches = space.branches if conditional else [space]
    diagnostics = []
    for branch in branches:
        extra = sorted(set(values) - set(branch))
        missing = sorted(set(branch) - set(values)) if complete else []
        invalid = {k: values[k] for k in values.keys() & branch.keys() if not _choice(values[k], branch[k])}
        if not extra and not missing and not invalid:
            return values
        diagnostics.append((len(extra) + len(missing) + len(invalid),
                            f"unknown or inactive parameters: {extra}; missing active parameters: {missing}; "
                            f"invalid choices: {invalid}"))
    detail = min(diagnostics, key=lambda row: row[0])[1] if diagnostics else "no supported branches"
    unavailable = space.unavailable if conditional else []
    raise ValueError(f"Invalid mapping: {detail}; unavailable: {unavailable}")


class MappingTemplate(ABC):
    name: ClassVar[str]

    @abstractmethod
    def applicable(self, adf, workload) -> bool:
        """Whether the template's structural requirements hold for this ADF."""

    @abstractmethod
    def space(self, adf, workload) -> dict:
        """Flat name -> choices, or ConditionalMappingSpace for mutually exclusive branches."""

    def request_space(self, adf, workload, params=None):
        """Space for discovery, sampling selectors or a concrete request.

        Normally this is the finite declared space. Templates carrying explicit
        graph-sized parameters can override it to validate a request and return
        correlated ConditionalMappingSpace branches without enumerating every
        possible graph. An override must enforce its physical/domain constraints;
        it must not merely echo unchecked user values as supported choices.
        """
        return self.space(adf, workload)

    @abstractmethod
    def apply(self, simulator, workload, params: dict) -> None:
        """Transform and place the workload task graph on the built simulator."""


class Registry:
    def __init__(self, kind, base):
        self.kind, self.base, self._items = kind, base, {}

    def register(self, cls):
        if not (isinstance(cls, type) and issubclass(cls, self.base)):
            raise TypeError(f"{cls!r} is not a {self.base.__name__}")
        name = getattr(cls, "name", None)
        if not name:
            raise ValueError(f"{cls.__name__} has no name")
        if name in self._items and self._items[name] is not cls:
            raise ValueError(f"{self.kind} {name!r} is already registered")
        if self.base is ComponentType:
            if getattr(cls, 'api_version', '1') != '1':
                raise ValueError(f"{name}: incompatible component extension API version")
            hooks = cls.assembly_hooks
            if not isinstance(hooks, tuple) or any(not callable(hook) for hook in hooks):
                raise TypeError(f"{name}: assembly_hooks must be a tuple of callables")
            if cls.role not in ROLES:
                raise ValueError(f"{name}: role must be one of {sorted(ROLES)}")
            if cls.conventions.arbitration not in ARBITRATIONS:
                raise ValueError(f"{name}: arbitration must be one of {sorted(ARBITRATIONS)}")
        self._items[name] = cls
        if self.base is ComponentType and self is COMPONENTS and getattr(cls, 'implementation_policy', 'legacy') == 'legacy':
            from arch_explorer.components.implementations import register_legacy_implementation
            register_legacy_implementation(cls)
        return cls

    def get(self, name):
        if self.base is ComponentType:
            from arch_explorer.components.extensions import discover_components
            discover_components()
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} {name!r}; registered: {sorted(self._items)}")
        return self._items[name]()

    def names(self):
        if self.base is ComponentType:
            from arch_explorer.components.extensions import discover_components
            discover_components()
        return sorted(self._items)


COMPONENTS = Registry("component", ComponentType)
MAPPING_TEMPLATES = Registry("mapping template", MappingTemplate)
register_component = COMPONENTS.register
register_template = MAPPING_TEMPLATES.register
