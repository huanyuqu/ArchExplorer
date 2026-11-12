"""Declarative multi-device MappingSpec contract. No execution implementation.

The simulator owns operator-name/dtype/algorithm support and physical routing.
This module validates JSON shape, resource references and partition arithmetic.
"""
from copy import deepcopy
import json

from arch_explorer.adf.registry import COMPONENTS, ConditionalMappingSpace

TEMPLATE = "multi_device"
PARAMETERS = ("device_paths", "placements", "shards", "collectives", "topology_edges", "routing")
UNSUPPORTED = "Explicit multi-device placement/routing execution is not registered; SIM-16 is required"


def _object(value, fields, at):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{at}: expected exactly {sorted(fields)}")


def _text(value, at):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{at}: expected nonempty string")


def _integer(value, at, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{at}: expected integer >= {minimum}")


def _array(value, at, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{at}: expected {'nonempty ' if nonempty else ''}array")


def validate_request(adf, params, *, registry=COMPONENTS):
    """Return a detached, validated parameter object; never invent defaults.

    Integer ranks index ``device_paths`` in caller order. Shard ranges are
    half-open element intervals, complete and nonoverlapping (unequal tails
    allowed). Collective elements are per-rank message payload before any
    algorithm-dependent padding; the backend must resolve dtype byte width.
    """
    from arch_explorer.devices.topology import instance_inventory
    json.dumps(params, allow_nan=False)
    _object(params, PARAMETERS, "multi_device.params")
    inventory = instance_inventory(adf, registry=registry)
    devices = params["device_paths"]
    _array(devices, "device_paths", True)
    for path in devices:
        _text(path, "device_paths[]")
        row = inventory.get(path, {})
        if row.get("kind") != "container" or not {"compute", "memory"} <= set(row.get("roles", [])):
            raise ValueError(f"device_paths: {path!r} must be a container with compute and memory")
    if len(set(devices)) != len(devices):
        raise ValueError("device_paths: duplicate rank target")
    if any(a.startswith(b + "/") for a in devices for b in devices if a != b):
        raise ValueError("device_paths: rank targets may not contain one another")

    def rank(value):
        _integer(value, "rank")
        if value >= len(devices):
            raise ValueError(f"rank {value} is outside device_paths")

    def ranks(values):
        _array(values, "ranks", True)
        for value in values:
            rank(value)
        if len(set(values)) != len(values):
            raise ValueError("ranks must be unique and ordered")

    placements = params["placements"]
    if not isinstance(placements, dict) or not placements:
        raise ValueError("placements must map operator IDs (or '*') to placements")
    for op, placement in placements.items():
        _text(op, "placement operator")
        if "*" in op and op != "*" and (not op.endswith(".*") or op.count("*") != 1):
            raise ValueError("Placement selectors are exact IDs, a trailing '.*' prefix, or '*'")
        _object(placement, ("ranks", "mapping"), "placement")
        ranks(placement["ranks"])
        mapping = placement["mapping"]
        _object(mapping, ("template", "params"), "placement.mapping")
        _text(mapping["template"], "placement.mapping.template")
        if mapping["template"] == TEMPLATE or not isinstance(mapping["params"], dict):
            raise ValueError("placement.mapping must name a local template with parameter object")

    _array(params["shards"], "shards")
    seen = set()
    for shard in params["shards"]:
        _object(shard, ("tensor", "axis", "size", "ranges"), "shard")
        _text(shard["tensor"], "shard.tensor")
        _integer(shard["axis"], "shard.axis")
        _integer(shard["size"], "shard.size", 1)
        if shard["tensor"] in seen:
            raise ValueError("Each tensor has at most one partition axis in this contract")
        seen.add(shard["tensor"])
        _array(shard["ranges"], "shard.ranges", True)
        intervals, owners = [], []
        for interval in shard["ranges"]:
            _object(interval, ("rank", "start", "stop"), "shard range")
            rank(interval["rank"])
            owners.append(interval["rank"])
            _integer(interval["start"], "range.start")
            _integer(interval["stop"], "range.stop", 1)
            if interval["stop"] <= interval["start"]:
                raise ValueError("Shard intervals must be nonempty [start, stop)")
            intervals.append((interval["start"], interval["stop"]))
        if len(set(owners)) != len(owners):
            raise ValueError("Each rank has at most one interval per tensor")
        cursor = 0
        for start, stop in sorted(intervals):
            if start != cursor:
                raise ValueError("Shard intervals have a gap or overlap")
            cursor = stop
        if cursor != shard["size"]:
            raise ValueError("Shard intervals must cover exactly the declared axis size")

    _array(params["collectives"], "collectives")
    for collective in params["collectives"]:
        _object(collective, ("after", "tensor", "operation", "algorithm", "ranks", "dtype", "elements"), "collective")
        for field in ("after", "tensor", "operation", "algorithm", "dtype"):
            _text(collective[field], f"collective.{field}")
        ranks(collective["ranks"])
        _integer(collective["elements"], "collective.elements", 1)

    _text(params["routing"], "routing")
    _array(params["topology_edges"], "topology_edges")
    edges = set()
    for edge in params["topology_edges"]:
        _object(edge, ("src", "dst", "network"), "edge")
        for field in ("src", "dst", "network"):
            _text(edge[field], f"edge.{field}")
        for endpoint in (edge["src"], edge["dst"]):
            if inventory.get(endpoint, {}).get("kind") != "container":
                raise ValueError(f"Edge endpoint {endpoint!r} is not a container instance")
        if edge["src"] == edge["dst"]:
            raise ValueError("An edge must connect different containers")
        if inventory.get(edge["network"], {}).get("role") != "network":
            raise ValueError(f"Edge network {edge['network']!r} is not a network instance")
        key = (edge["src"], edge["dst"], edge["network"])
        if key in edges:
            raise ValueError("Duplicate directed edge")
        edges.add(key)
    return deepcopy(params)


def request_space(adf, params, *, registry=COMPONENTS):
    """A concrete request as existing conditional choices (not an executable template).

    SIM-16 may union these correlated branches; each structured value is ONE
    choice, never a Cartesian product of its internal ranks/edges/intervals.
    """
    params = validate_request(adf, params, registry=registry)
    return ConditionalMappingSpace([{key: [value] for key, value in params.items()}], {}, {}, [])
