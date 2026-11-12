"""Copy-on-write ADF edits. The input tree and all its descendants stay untouched."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from arch_explorer.adf import COMPONENTS, SpaceMatrix, SpacePoint, find, from_dict, validate, walk


class StructureError(ValueError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details

    def to_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}


@dataclass(frozen=True)
class EditResult:
    adf: SpaceMatrix
    record: dict


def _node(root, path, kind=None):
    try:
        result = find(root, path)
    except (KeyError, AttributeError, TypeError) as error:
        raise StructureError("path_not_found", str(error), path=path) from error
    if kind and not isinstance(result, kind):
        raise StructureError("wrong_role", f"{path} must be a {kind.__name__}", path=path)
    return result


def _parent(root, path):
    if "/" not in path:
        raise StructureError("root_edit", "This edit requires a non-root element", path=path)
    parent = _node(root, path.rsplit("/", 1)[0], SpaceMatrix)
    node = _node(root, path)
    return parent, parent.networks if node in parent.networks else parent.elements, node


def _component(registry, name):
    try:
        return registry.get(name)
    except KeyError as error:
        raise StructureError("unsupported_component", str(error), component=name) from error


def check(root, registry=COMPONENTS):
    """Registry validation plus domain and element/network placement constraints."""
    try:
        validate(root, registry)
        for path, node in walk(root):
            if isinstance(node, SpaceMatrix):
                for child in node.elements:
                    if isinstance(child, SpacePoint) and registry.get(child.component).role == "network":
                        raise StructureError("wrong_role", "Network points belong in networks", path=path)
            else:
                component = registry.get(node.component)
                for key, param in node.params.items():
                    domain = component.params[key].domain
                    if isinstance(domain, tuple):
                        valid = len(domain) == 2 and domain[0] <= param.value <= domain[1]
                    elif isinstance(domain, list):
                        valid = param.value in domain
                    else:
                        valid = True
                    if not valid:
                        raise StructureError("out_of_domain", f"{path}.{key} outside {domain}", path=path, param=key)
    except StructureError:
        raise
    except KeyError as error:
        raise StructureError("unsupported_component", str(error)) from error
    except (ValueError, TypeError, AttributeError) as error:
        raise StructureError("invalid_adf", str(error)) from error


def apply_edit(root, edit, *, registry=COMPONENTS):
    """Apply one JSON edit atomically; return a new ADF and replayable record."""
    if not isinstance(edit, dict):
        raise StructureError("invalid_edit", "An edit must be a JSON object")
    data = {key: value for key, value in deepcopy(edit).items() if value is not None}
    op = data.pop("op", None)
    if op not in _OPERATIONS:
        raise StructureError("unknown_edit", f"Unknown basic edit {op!r}")
    candidate = deepcopy(root)
    try:
        _OPERATIONS[op](candidate, registry, **data)
        check(candidate, registry)
    except StructureError:
        raise
    except (ValueError, TypeError, KeyError) as error:
        raise StructureError("invalid_edit", str(error), op=op) from error
    return EditResult(candidate, {"op": op, **data})


def replay(root, records, *, registry=COMPONENTS):
    result = deepcopy(root)
    for record in records:
        result = apply_edit(result, record, registry=registry).adf
    return result


def _split(root, registry, point, num, algo="even"):
    parent, siblings, old = _parent(root, point)
    if not isinstance(old, SpacePoint) or siblings is parent.networks:
        raise StructureError("wrong_role", "split requires a compute or memory point")
    children = _component(registry, old.component).split(old, num, algo)
    siblings[siblings.index(old)] = SpaceMatrix(old.name, children, repeat=old.repeat)


def _merge(root, registry, points, algo="sum", name=None):
    if not isinstance(points, list) or len(points) < 2 or len(set(points)) != len(points):
        raise StructureError("invalid_selection", "merge requires at least two distinct sibling paths")
    triples = [_parent(root, path) for path in points]
    parent, siblings, first = triples[0]
    if any(p is not parent or s is not siblings for p, s, _ in triples):
        raise StructureError("invalid_selection", "merge requires siblings in the same collection")
    nodes = [item[2] for item in triples]
    if any(not isinstance(n, SpacePoint) or n.repeat != first.repeat for n in nodes):
        raise StructureError("wrong_role", "merge requires points with equal repeat")
    merged = _component(registry, first.component).merge(nodes, algo)
    merged.name, merged.repeat = name or first.name, first.repeat
    index = siblings.index(first)
    siblings[:] = [n for n in siblings if n not in nodes]
    siblings.insert(index, merged)


def _insert(root, registry, parent, children, name="group", group_size=None):
    container = _node(root, parent, SpaceMatrix)
    if not isinstance(children, list) or not children or len(set(children)) != len(children):
        raise StructureError("invalid_selection", "insert requires distinct child paths")
    selected = [_node(root, path) for path in children]
    if any(path.rsplit("/", 1)[0] != parent or node not in container.elements
           for path, node in zip(children, selected)):
        raise StructureError("invalid_selection", "insert children must be direct non-network children")
    repeat = 1
    if group_size is not None:
        if len(selected) != 1 or type(group_size) is not int or group_size < 1:
            raise StructureError("invalid_group", "group_size requires one repeated child and a positive integer")
        child = selected[0]
        if child.repeat % group_size:
            raise StructureError("not_divisible", f"repeat {child.repeat} is not divisible by {group_size}")
        repeat, child.repeat = child.repeat // group_size, group_size
    index = min(container.elements.index(n) for n in selected)
    container.elements[:] = [n for n in container.elements if n not in selected]
    container.elements.insert(index, SpaceMatrix(name, selected, repeat=repeat))


def _remove(root, registry, level):
    parent, siblings, old = _parent(root, level)
    if not isinstance(old, SpaceMatrix):
        raise StructureError("wrong_role", "remove requires a spatial level")
    for node in (*old.elements, *old.networks):
        node.repeat *= old.repeat
    index = siblings.index(old)
    siblings[index:index + 1] = old.elements
    parent.networks.extend(old.networks)


def _create(root, registry, container, spec=None, component=None):
    target = _node(root, container, SpaceMatrix)
    if (spec is None) == (component is None):
        raise StructureError("invalid_edit", "create takes exactly one of spec or component")
    if component is not None:
        if container == component or container.startswith(component + "/"):
            raise StructureError("cycle", "Cannot move an ancestor into its descendant")
        _, siblings, node = _parent(root, component)
        # Moving across repeated parents would silently clone/drop physical resources.
        def multiplicity(path):
            value, current = root.repeat, root
            for part in path.split("/")[1:]:
                current = current.child(part)
                value *= current.repeat
            return value
        if multiplicity(component.rsplit("/", 1)[0]) != multiplicity(container):
            raise StructureError("repeat_mismatch", "Moving requires equal source/destination parent multiplicity")
        siblings.remove(node)
    else:
        node = from_dict(spec)
    role = _component(registry, node.component).role if isinstance(node, SpacePoint) else "container"
    (target.networks if role == "network" else target.elements).append(node)


def _delete(root, registry, component):
    _, siblings, node = _parent(root, component)
    siblings.remove(node)


_OPERATIONS = {"split": _split, "merge": _merge, "insert": _insert,
               "remove": _remove, "create": _create, "delete": _delete}


def split(root, point, num, algo="even", *, registry=COMPONENTS):
    return apply_edit(root, dict(op="split", point=point, num=num, algo=algo), registry=registry)


def merge(root, points, algo="sum", *, name=None, registry=COMPONENTS):
    return apply_edit(root, dict(op="merge", points=points, algo=algo, name=name), registry=registry)


def insert(root, parent, children, *, name="group", group_size=None, registry=COMPONENTS):
    return apply_edit(root, dict(op="insert", parent=parent, children=children, name=name,
                                group_size=group_size), registry=registry)


def remove(root, level, *, registry=COMPONENTS):
    return apply_edit(root, dict(op="remove", level=level), registry=registry)


def create(root, container, spec=None, component=None, *, registry=COMPONENTS):
    if isinstance(spec, (SpaceMatrix, SpacePoint)):
        spec = spec.to_dict()
    return apply_edit(root, dict(op="create", container=container, spec=spec, component=component), registry=registry)


def delete(root, component, *, registry=COMPONENTS):
    return apply_edit(root, dict(op="delete", component=component), registry=registry)
