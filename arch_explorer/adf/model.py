"""Architecture description (ADF): the recursive SpaceMatrix/SpacePoint tree.

An ADF is a declarative tree. A ``SpaceMatrix`` is a container whose elements
are other matrices or points; its ``networks`` are the communication
SpacePoints that connect those elements (HPCA draft, section III-B2). A
``SpacePoint`` is one hardware component whose behaviour is defined by a
registered component type (see :mod:`arch_explorer.adf.registry`).

``repeat`` instantiates an element several times with identical parameters, so
a 316-core die is one matrix entry rather than 316 copies. Instance ``i`` of
element ``core`` is addressed as ``core[i]``; type-level paths omit the index.

Every parameter value is a :class:`Param` carrying its unit and a provenance
``source``. The source names the datasheet, paper or derivation the value
comes from, so evaluation reports can list where each number originates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Iterator, Union

NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")


@dataclass(frozen=True)
class Param:
    """A parameter value with its unit and provenance."""
    value: Union[int, float, str, bool]
    unit: str = "1"
    source: str = ""

    def __post_init__(self):
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("Parameter values must be finite")
        if not isinstance(self.value, (int, float, str, bool)):
            raise TypeError(f"Unsupported parameter value {self.value!r}")

    def to_dict(self):
        return {"value": self.value, "unit": self.unit, "source": self.source}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError("A parameter must be an object with value, unit and source")
        unknown = set(data) - {"value", "unit", "source"}
        if unknown:
            raise ValueError(f"Unknown parameter fields: {sorted(unknown)}")
        return cls(data["value"], data.get("unit", "1"), data.get("source", ""))


def _check_name(name):
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(f"Invalid element name {name!r}; use letters, digits, '_' or '-'")


def _check_repeat(repeat):
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive integer")


@dataclass
class SpacePoint:
    """A basic hardware component bound to a registered component type."""
    name: str
    component: str
    params: dict = field(default_factory=dict)
    repeat: int = 1

    # Simulation choice is separate from common hardware parameters.
    implementation: object = None

    def to_dict(self):
        data = {"type": "point", "name": self.name, "component": self.component,
                "params": {key: value.to_dict() for key, value in sorted(self.params.items())}}
        if self.repeat != 1:
            data["repeat"] = self.repeat
        if self.implementation is not None:
            data['implementation'] = self.implementation.to_dict()
        return data

    def value(self, key):
        return self.params[key].value


@dataclass
class SpaceMatrix:
    """A container of elements plus the communication points connecting them."""
    name: str
    elements: list = field(default_factory=list)
    networks: list = field(default_factory=list)
    repeat: int = 1
    connections: list = field(default_factory=list)

    def to_dict(self):
        data = {"type": "matrix", "name": self.name,
                "elements": [element.to_dict() for element in self.elements],
                "networks": [network.to_dict() for network in self.networks]}
        if self.repeat != 1:
            data["repeat"] = self.repeat
        if self.connections:
            data['connections'] = [connection.to_dict() for connection in self.connections]
        return data

    def child(self, name):
        for element in (*self.elements, *self.networks):
            if element.name == name:
                return element
        raise KeyError(f"{self.name} has no element {name!r}")


Node = Union[SpaceMatrix, SpacePoint]


def from_dict(data) -> Node:
    """Parse the JSON form produced by ``to_dict``."""
    if not isinstance(data, dict):
        raise TypeError("ADF nodes must be objects")
    kind = data.get("type")
    if kind == "point":
        unknown = set(data) - {"type", "name", "component", "params", "repeat", "implementation"}
        if unknown:
            raise ValueError(f"Unknown point fields: {sorted(unknown)}")
        params = {key: Param.from_dict(value) for key, value in data.get("params", {}).items()}
        from arch_explorer.components.implementations import ImplementationBinding
        binding = ImplementationBinding.from_dict(data['implementation']) if 'implementation' in data else None
        return SpacePoint(data["name"], data["component"], params, data.get("repeat", 1), binding)
    if kind == "matrix":
        unknown = set(data) - {"type", "name", "elements", "networks", "repeat", "connections"}
        if unknown:
            raise ValueError(f"Unknown matrix fields: {sorted(unknown)}")
        networks = [from_dict(item) for item in data.get("networks", [])]
        if any(not isinstance(network, SpacePoint) for network in networks):
            raise ValueError("Matrix networks must be points")
        from .dynamic import Connection
        connections = data.get('connections', [])
        if not isinstance(connections, list):
            raise TypeError('matrix connections must be a list')
        return SpaceMatrix(data["name"], [from_dict(item) for item in data.get("elements", [])],
                           networks, data.get("repeat", 1),
                           [Connection.from_dict(item) for item in connections])
    raise ValueError(f"Unknown ADF node type {kind!r}")


def validate(node: Node, registry=None, *, _root=True) -> None:
    """Check structure; with a registry, also check component types and parameters."""
    _check_name(node.name)
    _check_repeat(node.repeat)
    if isinstance(node, SpacePoint):
        if not isinstance(node.component, str) or not node.component:
            raise ValueError(f"Point {node.name} needs a component type")
        for key, value in node.params.items():
            if not isinstance(value, Param):
                raise TypeError(f"{node.name}.{key} must be a Param")
        if registry is not None:
            from .registry import COMPONENTS
            if registry is COMPONENTS:
                from arch_explorer.components.implementations import resolve_component
                resolve_component(node)
            else:
                registry.get(node.component).validate(node)
        return
    names = set()
    for element in (*node.elements, *node.networks):
        if element.name in names:
            raise ValueError(f"Duplicate element name {element.name!r} in {node.name}")
        names.add(element.name)
        validate(element, registry, _root=False)
    if registry is not None:
        for network in node.networks:
            if registry.get(network.component).role != "network":
                raise ValueError(f"{node.name}.{network.name} is not a network component")
    if _root:
        from .dynamic import validate_connections
        validate_connections(node)


def walk(node: Node, prefix: str = "") -> Iterator[tuple]:
    """Yield ``(type_path, node)`` pairs in depth-first order."""
    path = f"{prefix}/{node.name}" if prefix else node.name
    yield path, node
    if isinstance(node, SpaceMatrix):
        for element in (*node.elements, *node.networks):
            yield from walk(element, path)


def find(root: Node, path: str) -> Node:
    """Return the node at a type-level path such as ``chip/core/sram``."""
    parts = path.split("/")
    if parts[0] != root.name:
        raise KeyError(f"Path {path!r} does not start at root {root.name!r}")
    node = root
    for part in parts[1:]:
        if not isinstance(node, SpaceMatrix):
            raise KeyError(f"{node.name} is a point and has no children")
        node = node.child(part)
    return node


def instance_count(root: Node, path: str) -> int:
    """Number of physical instances of the node at ``path``."""
    count, node = root.repeat, root
    for part in path.split("/")[1:]:
        node = node.child(part)
        count *= node.repeat
    return count


def canonical_json(node: Node) -> str:
    return json.dumps(node.to_dict(), sort_keys=True, separators=(",", ":"))


def fingerprint(node: Node) -> str:
    """Stable identity of an ADF, used for structure deduplication."""
    return hashlib.sha256(canonical_json(node).encode()).hexdigest()
