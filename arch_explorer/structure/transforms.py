"""High-level programs composed exclusively of replayable basic edits."""
from copy import deepcopy
from dataclasses import dataclass

from arch_explorer.adf import COMPONENTS, SpaceMatrix, SpacePoint, find, from_dict, walk
from arch_explorer.archs.adf_seeds import bus, mesh
from .primitives import StructureError, apply_edit, check, _component, _node, _parent


@dataclass(frozen=True)
class ProgramResult:
    adf: SpaceMatrix
    records: list


class Program:
    def __init__(self, adf, registry):
        self.adf, self.registry, self.records = deepcopy(adf), registry, []
        check(self.adf, registry)

    def edit(self, op, **args):
        result = apply_edit(self.adf, {"op": op, **args}, registry=self.registry)
        self.adf = result.adf
        self.records.append(result.record)

    def finish(self):
        return ProgramResult(self.adf, self.records)


def _spec(value):
    return value.to_dict() if isinstance(value, (SpacePoint, SpaceMatrix)) else deepcopy(value)


def _cores(adf, chip, registry):
    container = _node(adf, chip, SpaceMatrix)
    return [node for node in container.elements if any(isinstance(n, SpacePoint)
            and _component(registry, n.component).role == "compute" for _, n in walk(node))]


def _network(spec, registry):
    data = _spec(spec)
    if _component(registry, data.get("component")).role != "network":
        raise StructureError("wrong_role", "Fabric spec must name a network component")
    return data


def chiplet_transform(adf, chip, cores_per_chiplet, noc_spec, nop_spec, *, registry=COMPONENTS):
    program = Program(adf, registry)
    noc_spec, nop_spec = _network(noc_spec, registry), _network(nop_spec, registry)
    cores = _cores(adf, chip, registry)
    if type(cores_per_chiplet) is not int or cores_per_chiplet < 1 or not cores:
        raise StructureError("invalid_group", "Need compute cores and a positive cores_per_chiplet")
    for network in _node(adf, chip, SpaceMatrix).networks:
        program.edit("delete", component=f"{chip}/{network.name}")
    if len(cores) == 1 and cores[0].repeat > 1:
        program.edit("insert", parent=chip, children=[f"{chip}/{cores[0].name}"],
                     name="chiplet", group_size=cores_per_chiplet)
        groups = [f"{chip}/chiplet"]
    else:
        if any(c.repeat != 1 for c in cores) or len(cores) % cores_per_chiplet:
            raise StructureError("not_divisible", "Explicit cores must divide evenly into chiplets")
        groups = []
        for i in range(0, len(cores), cores_per_chiplet):
            name = f"chiplet_{i // cores_per_chiplet}"
            program.edit("insert", parent=chip, children=[f"{chip}/{c.name}" for c in cores[i:i + cores_per_chiplet]], name=name)
            groups.append(f"{chip}/{name}")
    for path in groups:
        program.edit("create", container=path, spec=_spec(noc_spec))
    program.edit("create", container=chip, spec=_spec(nop_spec))
    return program.finish()


def gsm_to_dmc(adf, chip, shared_memory, parts, *, registry=COMPONENTS, noc_spec=None, bus_spec=None):
    """Privatize shared memory, one partition per direct compute-bearing core."""
    program = Program(adf, registry)
    bus_spec = _network(bus_spec or bus("local_bus"), registry)
    if noc_spec is not None:
        noc_spec = _network(noc_spec, registry)
    memory = _node(adf, shared_memory, SpacePoint)
    if _component(registry, memory.component).role != "memory" or memory.repeat != 1:
        raise StructureError("wrong_role", "shared_memory must be one memory point")
    if shared_memory.rsplit("/", 1)[0] != chip:
        raise StructureError("invalid_selection", "Shared memory must be a direct child of chip")
    cores = _cores(adf, chip, registry)
    if type(parts) is not int or parts < 2 or sum(c.repeat for c in cores) != parts:
        raise StructureError("invalid_group", "parts must match the number of cores and be >= 2")
    program.edit("split", point=shared_memory, num=parts)
    memories = [n.name for n in find(program.adf, shared_memory).elements]
    program.edit("remove", level=shared_memory)
    core_paths = []
    for core in cores:
        if core.repeat == 1:
            core_paths.append(f"{chip}/{core.name}")
        else:
            program.edit("delete", component=f"{chip}/{core.name}")
            for i in range(core.repeat):
                spec = core.to_dict()
                spec.update(name=f"{core.name}_{i}", repeat=1)
                program.edit("create", container=chip, spec=spec)
                core_paths.append(f"{chip}/{spec['name']}")
    for i, (core_path, memory_name) in enumerate(zip(core_paths, memories)):
        name = f"tile_{i}"
        program.edit("insert", parent=chip, children=[core_path, f"{chip}/{memory_name}"], name=name)
        program.edit("create", container=f"{chip}/{name}", spec=_spec(bus_spec or bus("local_bus")))
    for network in _node(adf, chip, SpaceMatrix).networks:
        program.edit("delete", component=f"{chip}/{network.name}")
    endpoint_count = len(find(program.adf, chip).elements)
    program.edit("create", container=chip, spec=_spec(noc_spec or mesh(cols=endpoint_count)))
    return program.finish()


def pim_transform(adf, memory, level, nums, spec, *, registry=COMPONENTS):
    """Split memory recursively; add supplied compute/fabric at the chosen level.

    level is a label (one partition level) or list of labels aligned with nums.
    spec is a compute ADF point, or {compute: point, network: point}.
    Construct on one private tree and validate at completion. Records remain
    ordinary split/insert/create edits, independently replayable in order.
    """
    levels = [level] if isinstance(level, str) else level
    counts = [nums] if type(nums) is int else nums
    if (not isinstance(levels, list) or not isinstance(counts, list) or not levels
            or len(levels) != len(counts) or any(type(n) is not int or n < 2 for n in counts)):
        raise StructureError("invalid_group", "level labels and partition counts >= 2 must align")
    compute = _spec(spec.get("compute", spec))
    network = _spec(spec.get("network", bus("pim_bus")))
    if _component(registry, compute.get("component")).role != "compute":
        raise StructureError("wrong_role", "PIM spec must name a registered compute component")
    if _component(registry, network.get("component")).role != "network":
        raise StructureError("wrong_role", "PIM network spec must name a network component")
    node = _node(adf, memory, SpacePoint)
    if _component(registry, node.component).role != "memory":
        raise StructureError("wrong_role", "PIM requires a memory point")
    program = Program(adf, registry)

    def record(op, **args):
        program.records.append(deepcopy({"op": op, **args}))

    def partition(point, path, depth):
        record("split", point=path, num=counts[depth])
        children = _component(registry, point.component).split(point, counts[depth], "even")
        groups = []
        for i, child in enumerate(children):
            name = f"{levels[depth]}_{i}"
            record("insert", parent=path, children=[f"{path}/{child.name}"], name=name)
            group = f"{path}/{name}"
            if depth + 1 < len(counts):
                groups.append(SpaceMatrix(name, [partition(child, f"{group}/{child.name}", depth + 1)]))
            else:
                record("create", container=group, spec=compute)
                record("create", container=group, spec=network)
                groups.append(SpaceMatrix(name, [child, from_dict(compute)], [from_dict(network)]))
        return SpaceMatrix(point.name, groups, repeat=point.repeat)

    try:
        _, siblings, old = _parent(program.adf, memory)
        # Validate supplied points before allocating thousands of banks. This
        # distinguishes a registered component's invalid spec from unknown IP.
        check(SpaceMatrix("spec", [from_dict(compute)], [from_dict(network)]), registry)
        siblings[siblings.index(old)] = partition(old, memory, 0)
        check(program.adf, registry)
    except StructureError:
        raise
    except (ValueError, KeyError, TypeError) as error:
        raise StructureError("invalid_edit", str(error), op="pim_transform") from error
    return program.finish()


def bond_memory(adf, core, memory_spec, *, registry=COMPONENTS):
    spec = _spec(memory_spec)
    if spec.get("component") != "dram" or spec.get("params", {}).get("technology", {}).get("value") != "bonded":
        raise StructureError("wrong_role", "bond_memory requires dram with technology='bonded'")
    if spec.get("repeat", 1) != 1:
        raise StructureError("invalid_group", "bond_memory attaches exactly one memory per core")
    program = Program(adf, registry)
    node = _node(adf, core)
    target = core
    if isinstance(node, SpacePoint):
        if _component(registry, node.component).role != "compute":
            raise StructureError("wrong_role", "core must contain compute")
        parent = core.rsplit("/", 1)[0]
        name = f"{node.name}_bonded"
        program.edit("insert", parent=parent, children=[core], name=name, group_size=1)
        target = f"{parent}/{name}"
    elif not any(isinstance(n, SpacePoint) and _component(registry, n.component).role == "compute" for _, n in walk(node)):
        raise StructureError("wrong_role", "core must contain compute")
    program.edit("create", container=target, spec=spec)
    return program.finish()


def share_private_memory(adf, chip, cores_per_group, memory_name, group_network,
                         root_network, *, pool_name="shared_sram", registry=COMPONENTS):
    """Pool each group's private memories using only replayable basic edits.

    Direct compute-bearing containers are grouped in ADF order. Repeated cores
    are expanded into explicit instances before regrouping. Selected memories
    must be direct, independent, single-instance points of one registered type.
    Its existing merge contract owns capacity, bandwidth and binding semantics.
    Both network specifications are explicit; this edit supplies no multicast
    execution semantics or timing assumptions. Explicit connection graphs need
    their own port-aware rewrite and are refused before any edit is returned.
    """
    program = Program(adf, registry)
    original_networks = [network.name for network in _node(adf, chip, SpaceMatrix).networks]
    if type(cores_per_group) is not int or cores_per_group < 1:
        raise StructureError("invalid_group", "cores_per_group must be a positive integer")
    group_network, root_network = _network(group_network, registry), _network(root_network, registry)
    if any(isinstance(node, SpaceMatrix) and node.connections for _, node in walk(adf)):
        raise StructureError("explicit_connections", "Pooling requires a port-aware rewrite for explicit connections")
    cores = _cores(adf, chip, registry)
    if not cores or any(not isinstance(core, SpaceMatrix) for core in cores):
        raise StructureError("invalid_selection", "Select direct compute-bearing core containers")
    if sum(core.repeat for core in cores) % cores_per_group:
        raise StructureError("not_divisible", "Core instances must divide evenly into groups")
    memory_type = None
    for core in cores:
        selected = [node for node in core.elements if node.name == memory_name]
        if (len(selected) != 1 or not isinstance(selected[0], SpacePoint)
                or selected[0].repeat != 1
                or _component(registry, selected[0].component).role != "memory"):
            raise StructureError("invalid_selection", "Each core needs one direct independent memory point")
        memory_type = memory_type or selected[0].component
        if selected[0].component != memory_type:
            raise StructureError("wrong_role", "Pooled memories must have the same registered type")
    names = []
    for core in cores:
        if core.repeat == 1:
            names.append(core.name)
            continue
        program.edit("delete", component=f"{chip}/{core.name}")
        for index in range(core.repeat):
            data = core.to_dict()
            data["name"], data["repeat"] = f"{core.name}_{index}", 1
            program.edit("create", container=chip, spec=data)
            names.append(data["name"])
    for index in range(0, len(names), cores_per_group):
        group = f"memory_group_{index // cores_per_group}"
        path = f"{chip}/{group}"
        members = names[index:index + cores_per_group]
        program.edit("insert", parent=chip, children=[f"{chip}/{name}" for name in members], name=group)
        memories = []
        for number, name in enumerate(members):
            old = f"{path}/{name}/{memory_name}"
            data = _node(program.adf, old, SpacePoint).to_dict()
            data["name"] = f"{pool_name}_{number}" if len(members) > 1 else pool_name
            program.edit("delete", component=old)
            program.edit("create", container=path, spec=data)
            memories.append(f"{path}/{data['name']}")
        if len(memories) > 1:
            program.edit("merge", points=memories, name=pool_name)
        program.edit("create", container=path, spec=group_network)
    for name in original_networks:
        program.edit("delete", component=f"{chip}/{name}")
    program.edit("create", container=chip, spec=root_network)
    return program.finish()


TRANSFORMS = {f.__name__: f for f in (chiplet_transform, gsm_to_dmc, pim_transform, bond_memory,
                                    share_private_memory)}


def apply_transform(adf, edit, *, registry=COMPONENTS):
    args = deepcopy(edit)
    op = args.pop("op", None)
    if op not in TRANSFORMS:
        raise StructureError("unknown_edit", f"Unknown transform {op!r}")
    try:
        return TRANSFORMS[op](adf, registry=registry, **args)
    except StructureError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise StructureError("invalid_edit", str(error), op=op) from error
