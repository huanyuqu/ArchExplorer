"""Pure ADF composition and instance discovery; no routing or timing model.

Networks are resource inventory. Directed connectivity is a mapping parameter,
not something inferred from names or from the order of ADF children.
"""
from copy import deepcopy

from arch_explorer.adf import COMPONENTS, Param, SpaceMatrix, SpacePoint
from arch_explorer.structure import check


def compose(name, devices, networks=(), *, registry=COMPONENTS):
    """Compose named device subtrees and network points without changing inputs.

    ``devices`` maps arbitrary child names to ADF subtrees. Repeated subtrees
    retain their multiplicity. No device family or device-count whitelist.
    """
    if not isinstance(devices, dict) or not devices:
        raise ValueError("devices must be a nonempty mapping of names to ADF subtrees")
    children = []
    for child_name, subtree in devices.items():
        child = deepcopy(subtree)
        if not isinstance(child, (SpaceMatrix, SpacePoint)):
            raise ValueError("devices must contain ADF nodes")
        child.name = child_name
        children.append(child)
    root = SpaceMatrix(name, children, deepcopy(list(networks)))
    check(root, registry)
    return root


def link_resource(name, bandwidth, latency, *, source):
    """A sourced bus resource (B/ns, ns), to bind to directed mapping edges.

    Separate resources represent independent directions. Reusing one resource
    declares shared capacity; it never doubles the cost inventory.
    """
    import math
    if (type(bandwidth) not in (int, float) or not math.isfinite(bandwidth) or bandwidth <= 0
            or type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0):
        raise ValueError("bandwidth must be positive B/ns; latency nonnegative ns")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must identify measured, synthetic or not-modeled parameters")
    return SpacePoint(name, "bus", {"bandwidth": Param(bandwidth, "B/ns", source),
                                    "latency": Param(latency, "ns", source)})


def instance_inventory(adf, *, registry=COMPONENTS, max_instances=200_000):
    """Resolve instance paths without building simulator components.

    Matches the current builder's ``name[i]`` expansion. Type paths still name
    parameter/edit targets. Root/network repeats unsupported by that builder
    are diagnosed rather than silently dropped. Memory alone is not proof of
    a mapping's physical applicability.
    """
    check(adf, registry)
    if not isinstance(adf, SpaceMatrix) or adf.repeat != 1:
        raise ValueError("Instance discovery requires a matrix root with repeat=1")
    result = {}

    def visit(node, path, type_path):
        if len(result) >= max_instances:
            raise ValueError(f"Instance discovery exceeds {max_instances} nodes")
        if path in result:
            raise ValueError(f"Ambiguous instance path {path}")
        row = {"path": path, "type_path": type_path,
               "kind": "container" if isinstance(node, SpaceMatrix) else "point"}
        result[path] = row
        if isinstance(node, SpacePoint):
            row.update(component=node.component, role=registry.get(node.component).role)
            return {row["role"]}
        roles = set()
        for child in node.elements:
            for i in range(child.repeat):
                name = f"{child.name}[{i}]" if child.repeat > 1 else child.name
                roles |= visit(child, f"{path}/{name}", f"{type_path}/{child.name}")
        for child in node.networks:
            if child.repeat != 1:
                raise ValueError(f"{type_path}/{child.name}: networks cannot repeat")
            visit(child, f"{path}/{child.name}", f"{type_path}/{child.name}")
        row["roles"] = sorted(roles)
        return roles

    visit(adf, adf.name, adf.name)
    return result


def discover(adf, *, registry=COMPONENTS):
    """Generic placement candidates and link resources, with no timing claim."""
    inventory = instance_inventory(adf, registry=registry)
    return {"containers": [r for r in inventory.values() if r["kind"] == "container"],
            "networks": [{**r, "parameters": _point(adf, r["type_path"]).to_dict()["params"]}
                         for r in inventory.values() if r.get("role") == "network"],
            "path_semantics": "instance paths for mapping; type paths for edits and parameter domains",
            "note": "Container roles are structural discovery, not backend applicability."}


def _point(adf, path):
    from arch_explorer.adf import find
    return find(adf, path)
