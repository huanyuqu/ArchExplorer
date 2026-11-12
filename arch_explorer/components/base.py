"""Simulator-side component instances and the kernels compute units execute.

A component type (``arch_explorer.adf.registry.ComponentType``) declares
parameters and conventions; its ``build`` returns one of the model classes
below for each physical instance. A model owns the instance's resources and
answers the demand queries the mapping layer asks:

* compute: ``kernel_demand(kernel)`` -> occupancy of the unit for a kernel;
* memory: ``port`` resource (bytes at the memory's bandwidth) and ``latency``;
* network: ``route(src, dst)`` -> link resources crossed and latency.

Every parameter a model reads goes through :meth:`PointModel.param`, which
records the value, unit and source in the evaluation provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

PRECISION_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int32": 4, "uint32": 4, "int64": 8, "bool": 1}


def dtype_bytes(dtype: str) -> int:
    if dtype not in PRECISION_BYTES:
        raise ValueError(f"Unknown precision {dtype!r}; known: {sorted(PRECISION_BYTES)}")
    return PRECISION_BYTES[dtype]


@dataclass(frozen=True)
class Kernel:
    """Work one compute unit performs for one task.

    ``op`` is ``gemm`` (``shape = (batch, m, k, n)``, C = A @ B), ``mvm``
    (``(batch, m, k)``, y = A @ x) or a vector kernel with ``shape = (rows,
    cols)`` and per-element instruction counts ``simt``/``sfu`` plus per-row
    ``row_simt``/``row_sfu`` (see :mod:`arch_explorer.workload.operators`).
    """
    op: str
    shape: tuple
    precision: str = "fp16"
    simt: float = 0.0
    sfu: float = 0.0
    row_simt: float = 0.0
    row_sfu: float = 0.0

    @property
    def flops(self) -> float:
        if self.op == "gemm":
            b, m, k, n = self.shape
            return 2.0 * b * m * k * n
        if self.op == "mvm":
            b, m, k = self.shape
            return 2.0 * b * m * k
        rows, cols = self.shape
        return rows * cols * (self.simt + self.sfu) + rows * (self.row_simt + self.row_sfu)


@dataclass
class Demand:
    """Resources a task occupies (amount in each resource's unit) and its latency tail."""
    resources: dict = field(default_factory=dict)
    latency_ns: float = 0.0

    def add(self, other: "Demand") -> "Demand":
        resources = dict(self.resources)
        for name, amount in other.resources.items():
            resources[name] = resources.get(name, 0.0) + amount
        return Demand(resources, self.latency_ns + other.latency_ns)

    def scaled(self, factor: float) -> "Demand":
        return Demand({name: amount * factor for name, amount in self.resources.items()}, self.latency_ns)


class Provenance:
    """Parameters consumed during one evaluation, keyed by type path and name."""

    def __init__(self):
        self.rows = {}

    def record(self, path, param, value, unit, source):
        self.rows.setdefault((path, param), {"path": path, "param": param, "value": value,
                                             "unit": unit, "source": source})

    def derived(self, path, param, value, unit, formula):
        self.record(path, param, value, unit, f"derived: {formula}")

    def as_list(self):
        return [self.rows[key] for key in sorted(self.rows)]


@dataclass
class BuildContext:
    path: str
    type_path: str
    coord: tuple
    provenance: Provenance
    endpoints: int = 0
    options: dict = field(default_factory=dict)


class PointModel:
    """One instantiated SpacePoint."""
    role = ""

    def __init__(self, component, point, context: BuildContext):
        self.component = component
        self.point = point
        self.path = context.path
        self.type_path = context.type_path
        self.coord = context.coord
        self.provenance = context.provenance
        self.conventions = component.conventions
        self.resources = []

    def param(self, key, default=None):
        """Read a parameter and record its provenance; optional ones may be absent."""
        if key not in self.point.params:
            spec = self.component.params.get(key)
            if spec is None or spec.required:
                raise KeyError(f"{self.type_path}: missing parameter {key}")
            return default
        param = self.point.params[key]
        self.provenance.record(self.type_path, key, param.value, param.unit, param.source)
        return param.value

    def positive(self, key, default=None):
        value = self.param(key, default)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{self.type_path}.{key} must be finite and positive, got {value!r}")
        return value

    def resource(self, suffix, capacity, unit, arbitration=None):
        from arch_explorer.components.api import Resource
        resource = Resource(f"{self.path}:{suffix}" if suffix else self.path, capacity,
                            arbitration or self.conventions.arbitration, unit, self.path)
        self.resources.append(resource)
        return resource


class ComputeModel(PointModel):
    """A compute unit occupied for the kernel's duration; capacity 1 ns per ns."""
    role = "compute"

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.unit = self.resource("", 1.0, "ns/ns")
        self.clock_ghz = self.positive("clock_ghz") if self.conventions.time_base == "cycle" else None

    def kernel_time(self, kernel: Kernel) -> float:
        """Duration in the component's ``time_base`` units."""
        raise NotImplementedError(f"{self.component.name} cannot execute {kernel.op} kernels")

    def kernel_demand(self, kernel: Kernel) -> Demand:
        duration = self.kernel_time(kernel)
        if self.conventions.time_base == "cycle":
            duration = duration / self.clock_ghz   # cycles / (cycles per ns)
        return Demand({self.unit.name: duration})

    def precision(self, kernel: Kernel) -> str:
        native = self.param("precision")
        if kernel.precision != native:
            raise NotImplementedError(
                f"{self.type_path} executes {native} kernels; the workload asks for {kernel.precision}")
        return native


class MemoryModel(PointModel):
    role = "memory"

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.capacity = self.positive("capacity")
        self.latency = self.param("latency")
        if not isinstance(self.latency, (int, float)) or self.latency < 0:
            raise ValueError(f"{self.type_path}.latency must be nonnegative")
        self.port = self.resource("port", self.port_bandwidth(), "B/ns")

    def port_bandwidth(self) -> float:
        return self.positive("bandwidth")

    def access(self, size: float) -> Demand:
        return Demand({self.port.name: size}, self.latency)


GATEWAY = -1


class NetworkModel(PointModel):
    """Connects the endpoints (expanded elements) of one container and its gateway."""
    role = "network"

    def __init__(self, component, point, context):
        super().__init__(component, point, context)
        self.endpoints = context.endpoints

    def route(self, src: int, dst: int) -> Demand:
        """Demand per byte moved from endpoint ``src`` to ``dst`` (``GATEWAY`` = container boundary)."""
        raise NotImplementedError

    def _check(self, *endpoints):
        for endpoint in endpoints:
            if endpoint != GATEWAY and not 0 <= endpoint < self.endpoints:
                raise IndexError(f"{self.path}: endpoint {endpoint} out of range")
