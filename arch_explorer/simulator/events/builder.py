"""Hardware builder: instantiate an ADF as a multi-level simulatable model (section III-B2).

``build_simulator(adf)`` expands ``repeat``, binds every SpacePoint to its
registered component type, checks the component's declared conventions, and
assigns multi-level space coordinates. Instance paths add ``[i]`` to repeated
elements (``gpu[2]/sm[17]``); ``coord`` lists the endpoint index at each level
from the root down, like the paper's ``((0, 0)->(2,))`` coordinates.

Routing follows the container hierarchy. A transfer between two points climbs
from the source through each enclosing container's network to its gateway,
crosses the lowest common container's network between the two branches, and
descends to the destination. A container without a network is transparent:
crossing it costs nothing, which is reported as a warning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from arch_explorer.adf.model import SpaceMatrix, SpacePoint, validate, walk
from arch_explorer.adf.registry import ARBITRATIONS, COMPONENTS
from arch_explorer.components.base import (GATEWAY, BuildContext, ComputeModel, Demand, MemoryModel,
                                           NetworkModel, Provenance)
from arch_explorer.simulator.plan import MemoryGroup

TIME_BASES = ("ns", "cycle")


@dataclass
class ContainerInstance:
    path: str
    type_path: str
    coord: tuple
    parent: "ContainerInstance | None" = None
    endpoint: int = 0                       # index among the parent's endpoints
    children: list = field(default_factory=list)   # ContainerInstance or PointModel, endpoint order
    networks: list = field(default_factory=list)


def _accumulate(total: dict, demand: Demand, factor: float = 1.0):
    for name, amount in demand.resources.items():
        total[name] = total.get(name, 0.0) + amount * factor


class Simulator:
    """An instantiated ADF: point models, their resources and hierarchy routing."""

    def __init__(self, adf, root, points, provenance, warnings):
        self.adf = adf
        self.root = root
        self.points = points
        self.provenance = provenance
        self.warnings = warnings
        self.execution_instances = {}
        self._parents = {}
        self._routes = {}
        self._index(root)
        self.memory_groups = {}
        self._group_of = {}

    def add_memory_group(self, group, members=()):
        """Register a component-assembled striped aggregate and its member paths."""
        if (not isinstance(group, MemoryGroup) or not group.parts or not isinstance(group.path, str)
                or not group.path or group.path in self.memory_groups or group.path in self.points):
            raise ValueError('memory group must have parts and a unique path')
        parts = [part.path for part in group.parts]
        members = list(members)
        if any(not isinstance(path, str) for path in members) or len(members) != len(set(members)):
            raise ValueError('memory group requires unique string member paths')
        paths = list(dict.fromkeys([*parts, *members]))
        if len(set(parts)) != len(parts) or any(
                path not in self.points or path in self._group_of for path in paths):
            raise ValueError('memory group has unknown, duplicate or overlapping members')
        if any(not isinstance(part, MemoryModel) or self.points[part.path] is not part for part in group.parts):
            raise TypeError('memory group parts must be this simulator\'s memory instances')
        self.memory_groups[group.path] = group
        self._group_of.update({path: group for path in paths})

    def assemble(self):
        """Run each hook declared by the actual component types exactly once."""
        seen = set()
        models = list(self.points.values())
        models.extend(n for container in self.containers() for n in container.networks)
        for model in models:
            hooks = getattr(model.component_type, 'assembly_hooks', ())
            if not isinstance(hooks, tuple) or any(not callable(hook) for hook in hooks):
                raise TypeError('ComponentType.assembly_hooks must be a tuple of callables')
            for hook in hooks:
                if id(hook) not in seen:
                    hook(self)
                    seen.add(id(hook))

    def memory(self, path):
        """The memory point or :class:`MemoryGroup` at ``path``."""
        if path in self.memory_groups:
            return self.memory_groups[path]
        model = self.points[path]
        if not isinstance(model, MemoryModel):
            raise TypeError(f"{path} is not a memory")
        return model

    def group_of(self, path):
        """The declared aggregate holding point ``path``, or ``None``."""
        return self._group_of.get(path)

    def port_demand(self, memory: str, size: float) -> Demand:
        """Demand on the port(s) of a memory or group for ``size`` bytes (no network)."""
        target = self.memory(memory)
        if isinstance(target, MemoryGroup):
            share = size / len(target.parts)
            return Demand({part.port.name: share for part in target.parts})
        return Demand({target.port.name: size})

    def _index(self, container):
        for child in container.children:
            self._parents[child.path] = (container, child.endpoint if isinstance(child, ContainerInstance)
                                         else child.endpoint)
            if isinstance(child, ContainerInstance):
                self._index(child)

    @property
    def resources(self):
        result = {}
        for point in self.points.values():
            for resource in point.resources:
                result[resource.name] = resource
        for container in self.containers():
            for network in container.networks:
                for resource in network.resources:
                    result[resource.name] = resource
        return result

    def containers(self, root=None):
        root = root or self.root
        yield root
        for child in root.children:
            if isinstance(child, ContainerInstance):
                yield from self.containers(child)

    def container(self, path):
        for container in self.containers():
            if container.path == path:
                return container
        raise KeyError(f"No container instance {path!r}")

    def find_points(self, component=None, role=None, within=None):
        """Point instances in endpoint order, optionally filtered."""
        prefix = None if within is None else within.rstrip("/") + "/"
        result = []
        for path, point in self.points.items():
            if component is not None and point.component.name != component:
                continue
            if role is not None and point.role != role:
                continue
            if prefix is not None and not path.startswith(prefix):
                continue
            result.append(point)
        return result

    def nearest(self, path, role=None, component=None):
        """The point of a role/component closest to ``path`` in the hierarchy."""
        candidates = self.find_points(component, role)
        if not candidates:
            raise ValueError(f"Architecture has no {component or role} point")

        def distance(point):
            a, b = path.split("/"), point.path.split("/")
            common = 0
            while common < min(len(a), len(b)) and a[common] == b[common]:
                common += 1
            return (len(a) + len(b) - 2 * common, point.path)
        return min(candidates, key=distance)

    def _chain(self, path):
        """[(container, endpoint), ...] from the innermost container up to the root."""
        chain = []
        while path in self._parents:
            container, endpoint = self._parents[path]
            chain.append((container, endpoint))
            path = container.path
        return chain

    def route(self, src: str, dst: str) -> Demand:
        """Per-byte network demand and latency from point ``src`` to point ``dst``."""
        key = (src, dst)
        if key not in self._routes:
            self._routes[key] = self._route(src, dst)
        cached = self._routes[key]
        return Demand(dict(cached.resources), cached.latency_ns)

    def _route(self, src, dst):
        if src == dst:
            return Demand()
        up, down = self._chain(src), self._chain(dst)
        down_index = {container.path: i for i, (container, _) in enumerate(down)}
        demand = Demand()
        for i, (container, endpoint) in enumerate(up):
            if container.path in down_index:
                demand = demand.add(self._cross(container, endpoint, down[down_index[container.path]][1]))
                for container2, endpoint2 in reversed(down[:down_index[container.path]]):
                    demand = demand.add(self._cross(container2, GATEWAY, endpoint2))
                return demand
            demand = demand.add(self._cross(container, endpoint, GATEWAY))
        raise ValueError(f"{src} and {dst} share no container")

    def _cross(self, container, src, dst) -> Demand:
        if src == dst:
            return Demand()
        if not container.networks:
            # Indexed siblings (channel_0, channel_1, ...) share one warning.
            warning = (f"{re.sub(r'_[0-9]+(?=/|$)', '_*', container.type_path)} has no network; "
                       "transfers inside it are not timed")
            if warning not in self.warnings:
                self.warnings.append(warning)
            return Demand()
        return container.networks[0].route(src, dst)

    def _parts(self, memory):
        target = self.memory(memory)
        return target.parts if isinstance(target, MemoryGroup) else [target]

    def multicast(self, src_memory, dst_memories, size):
        """One immutable source read, a directed network tree, one store per target.

        Every traversed container must declare a route_many-capable network;
        missing network service is a refusal, never a free broadcast.
        """
        import math
        targets=tuple(dst_memories)
        if (not targets or len(set(targets))!=len(targets) or src_memory in targets
                or type(size) not in (int,float) or not math.isfinite(size) or size<=0):
            raise ValueError('multicast requires positive bytes and distinct non-source memories')
        memories=[self.memory(n) for n in (src_memory,*targets)]
        if any(isinstance(m,MemoryGroup) for m in memories):raise ValueError('multicast requires actual single storage points')
        segments={};path_latencies=[]
        for dst in targets:
            up,down=self._chain(src_memory),self._chain(dst)
            down_index={c.path:i for i,(c,_) in enumerate(down)}
            path=[]
            for c,e in up:
                if c.path in down_index:
                    j=down_index[c.path];path.append((c,e,down[j][1]))
                    path.extend((c2,GATEWAY,e2) for c2,e2 in reversed(down[:j]));break
                path.append((c,e,GATEWAY))
            else:raise ValueError('multicast memories share no container')
            latency=0
            for c,a,b in path:
                if a==b:continue
                if len(c.networks)!=1 or not callable(getattr(c.networks[0],'route_many',None)):
                    raise ValueError('multicast requires one declared multicast network at '+c.path)
                key=(c.path,a)
                if key not in segments:segments[key]=(c,set())
                segments[key][1].add(b)
                latency+=c.networks[0].route(a,b).latency_ns
            path_latencies.append(latency)
        total=dict(memories[0].access(size).resources)
        for dst in memories[1:]:total[dst.port.name]=total.get(dst.port.name,0)+size
        for (_,src),(c,destinations) in sorted(segments.items()):
            _accumulate(total,c.networks[0].route_many(src,sorted(destinations)),size)
        return Demand(total,memories[0].latency+max(path_latencies))

    def transfer(self, src_memory: str, dst_memory: str, size: float) -> Demand:
        """Demand of moving ``size`` bytes between two memories or groups (read, network, write)."""
        sources, targets = self._parts(src_memory), self._parts(dst_memory)
        total, latency = {}, 0.0
        share = size / (len(sources) * len(targets))
        for src in sources:
            for dst in targets:
                _accumulate(total, src.access(share))
                path_latency = src.latency
                if dst is not src:
                    route = self.route(src.path, dst.path)
                    _accumulate(total, route, share)
                    total[dst.port.name] = total.get(dst.port.name, 0.0) + share
                    path_latency += route.latency_ns
                latency = max(latency, path_latency)
        return Demand(total, latency)

    def stream(self, memory: str, compute: str, size: float, write: bool = False) -> Demand:
        """Bytes ``compute`` reads from (or with ``write``, writes to) ``memory`` while it computes."""
        parts = self._parts(memory)
        share = size / len(parts)
        total = {}
        for mem in parts:
            total[mem.port.name] = total.get(mem.port.name, 0.0) + share
            _accumulate(total, self.route(compute, mem.path) if write else self.route(mem.path, compute), share)
        return Demand(total)

    def access_latency(self, memory: str, compute: str) -> float:
        return max(mem.latency + self.route(mem.path, compute).latency_ns for mem in self._parts(memory))


def _check_conventions(component):
    conventions = component.conventions
    if conventions.time_base not in TIME_BASES:
        raise ValueError(f"{component.name}: time_base must be one of {TIME_BASES}")
    if conventions.arbitration not in ARBITRATIONS:
        raise ValueError(f"{component.name}: unknown arbitration {conventions.arbitration!r}")
    if conventions.time_base == "cycle" and "clock_ghz" not in component.params:
        raise ValueError(f"{component.name} is cycle-based but declares no clock_ghz to convert cycles to ns")
    if conventions.time_base == "cycle" and component.params["clock_ghz"].unit != "GHz":
        raise ValueError(f"{component.name}.clock_ghz must be in GHz")


def build_simulator(adf, registry=COMPONENTS, max_instances=200_000, options=None) -> Simulator:
    """Instantiate the ADF tree; reject unknown components and incompatible conventions."""
    if not isinstance(adf, SpaceMatrix):
        raise TypeError("The ADF root must be a SpaceMatrix")
    validate(adf, registry)
    if any(getattr(node, 'connections', []) for _, node in walk(adf)):
        raise ValueError('dynamic component connections require a dynamic execution provider, not static Demand routing')
    provenance, warnings, points = Provenance(), [], {}
    options = dict(options or {})
    count = 0
    expected = {}

    def make_point(node: SpacePoint, path, type_path, coord, endpoints, endpoint):
        nonlocal count
        count += 1
        if count > max_instances:
            raise ValueError(f"Architecture exceeds {max_instances} instances")
        if registry is COMPONENTS:
            from arch_explorer.components.implementations import resolve_component
            component = resolve_component(node, execution_domain='events')
        else:
            component = registry.get(node.component)
        _check_conventions(component)
        model = component.build(node, BuildContext(path, type_path, coord, provenance, endpoints, options))
        expected_role = component.role
        base = {"compute": ComputeModel, "memory": MemoryModel, "network": NetworkModel}[expected_role]
        if not isinstance(model, base):
            raise TypeError(f"{component.name}.build must return a {base.__name__}")
        model.endpoint = endpoint
        model.component_type = component
        expected[path] = model
        return model

    def build(node: SpaceMatrix, path, type_path, coord, parent, endpoint):
        container = ContainerInstance(path, type_path, coord, parent, endpoint)
        index = 0
        for element in node.elements:
            for i in range(element.repeat):
                name = f"{element.name}[{i}]" if element.repeat > 1 else element.name
                child_path, child_type = f"{path}/{name}", f"{type_path}/{element.name}"
                child_coord = coord + (index,)
                if isinstance(element, SpaceMatrix):
                    container.children.append(build(element, child_path, child_type, child_coord, container, index))
                else:
                    container.children.append(make_point(element, child_path, child_type, child_coord, 0, index))
                index += 1
        for network in node.networks:
            if network.repeat != 1:
                raise ValueError(f"{type_path}/{network.name}: networks cannot repeat")
            container.networks.append(make_point(network, f"{path}/{network.name}", f"{type_path}/{network.name}",
                                                 coord + ("net",), index, GATEWAY))
        return container

    if adf.repeat != 1:
        raise ValueError("The ADF root cannot repeat")
    root = build(adf, adf.name, adf.name, (), None, 0)
    for path, model in expected.items():
        if model.role != "network":
            points[path] = model
    simulator = Simulator(adf, root, points, provenance, warnings)
    simulator.assemble()
    if not simulator.find_points(role="compute"):
        raise ValueError("Architecture has no compute point")
    return simulator
