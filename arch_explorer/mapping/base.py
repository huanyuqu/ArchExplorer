"""Shared machinery of the operator-placing mapping templates.

A *device* is a minimal container instance whose subtree holds compute points
and a DRAM: a GPU, a many-core chip with its memory, or one rank of a server.
Two kinds of DRAM do not make a container a device:

* a DRAM group (see :class:`arch_explorer.simulator.events.builder.MemoryGroup`):
  a DRAM split into channels/banks, possibly with ``pim_unit`` per bank. It is
  the device's main memory, striped over the parts; its PIM units are used
  only by the ``pim_offload`` template;
* ``bonded`` DRAM: a core's private shard, used by ``dmc_partition`` with
  ``weights="bonded"``.

Device templates (``gsm_tiled``, ``dmc_partition``) place one operator at a
time on one device; ``tensor_parallel`` composes them across devices.
"""
from __future__ import annotations

from arch_explorer.adf.model import SpaceMatrix, SpacePoint
from arch_explorer.adf.registry import COMPONENTS, MappingTemplate
from arch_explorer.workload.llm import workload_from_spec


def _technology(point):
    param = point.params.get("technology")
    return None if param is None else param.value


def is_memory_group(node) -> bool:
    """ADF-level test for a DRAM group: only non-bonded DRAMs (at least two instances) and PIM units."""
    def scan(matrix):
        drams = 0
        for element in matrix.elements:
            if isinstance(element, SpaceMatrix):
                sub = scan(element)
                if sub is None:
                    return None
                drams += sub * element.repeat
            elif element.component == "dram" and _technology(element) != "bonded":
                drams += element.repeat
            elif element.component != "pim_unit":
                return None
        return drams
    return isinstance(node, SpaceMatrix) and (scan(node) or 0) * node.repeat >= 2


def adf_devices(adf, registry=COMPONENTS):
    """Type paths of the minimal containers holding compute and DRAM (ADF level, before build)."""
    found = []

    def holds(node):
        roles, drams = set(), False
        for element in (node.elements if isinstance(node, SpaceMatrix) else []):
            if isinstance(element, SpacePoint):
                component = registry.get(element.component)
                roles.add(component.role)
                drams |= element.component == "dram" and _technology(element) != "bonded"
            elif is_memory_group(element):
                drams = True                      # main memory; its PIM units are not device compute
            else:
                sub_roles, sub_drams = holds(element)
                roles |= sub_roles
                drams |= sub_drams
        return roles, drams

    def visit(node, path, count):
        if not isinstance(node, SpaceMatrix) or is_memory_group(node):
            return False
        inner = [visit(e, f"{path}/{e.name}", count * e.repeat) for e in node.elements]
        if any(inner):
            return True
        roles, drams = holds(node)
        if "compute" in roles and drams:
            found.append((path, count))
            return True
        return False

    visit(adf, adf.name, adf.repeat)
    return found


def sim_devices(sim):
    """Container instances that are devices, in endpoint order."""
    types = {path for path, _ in adf_devices(sim.adf)}
    return [c for c in sim.containers() if c.type_path in types]


class DeviceState:
    """Where each tensor of one device's workload lives and which task makes it available."""

    def __init__(self, sim, container):
        self.sim = sim
        self.container = container
        self.path = container.path
        self.computes = [p.path for p in sim.find_points(role="compute", within=self.path)
                         if sim.group_of(p.path) is None]
        prefix = self.path.rstrip("/") + "/"
        self.pim_groups = [g for path, g in sim.memory_groups.items() if path.startswith(prefix) and g.pims]
        drams = [path for path in sim.memory_groups if path.startswith(prefix)]
        drams += [p.path for p in sim.find_points(component="dram", within=self.path)
                  if sim.group_of(p.path) is None and p.technology != "bonded"]
        if not self.computes or not drams:
            raise ValueError(f"{self.path} is not a device: it needs compute points and a DRAM")
        self.dram = min(drams, key=lambda path: (path.count("/"), path))
        self.ready = {}

    def units_for(self, kernel):
        """The device's compute points of the one ADF element (unit type) that runs ``kernel`` fastest.

        A device may mix unit types (an NPU's systolic arrays and vector
        units). For each type that can execute the kernel, the time of the
        whole kernel on one unit divided by the number of units of that type
        is compared; a single-type device always uses all its units.
        """
        by_type = {}
        for path in self.computes:
            by_type.setdefault(self.sim.points[path].type_path, []).append(path)
        if len(by_type) == 1:
            return self.computes
        best = None
        for paths in by_type.values():
            try:
                time = sum(self.sim.points[paths[0]].kernel_demand(kernel).resources.values()) / len(paths)
            except NotImplementedError:
                continue
            if best is None or time < best[0]:
                best = (time, paths)
        if best is None:
            raise NotImplementedError(f"{self.path}: no compute unit runs {kernel.op} {kernel.precision} kernels")
        return best[1]

    def input(self, ctx, workload, name, operator):
        """Storage task of a tensor not produced by any operator (weights, KV cache, inputs)."""
        if name not in self.ready:
            tensor = workload.tensors[name]
            static = tensor.kind in ("weight", "kv")
            self.ready[name] = ctx.storage(f"{self.path}:{name}", self.dram, tensor.bytes, (), operator,
                                           "Storage", static=static)
        return self.ready[name]

    def deps(self, ctx, workload, names, operator):
        """Task ids that make the named tensors available (storage tasks of their pieces)."""
        result = []
        for name in names:
            ready = self.ready[name] if name in self.ready else self.input(ctx, workload, name, operator)
            result.extend(ready if isinstance(ready, list) else [ready])
        return result


class PlacementTemplate(MappingTemplate):
    """A single-device template; subclasses implement ``place_operator``."""
    single_device = True

    def applicable(self, adf, workload) -> bool:
        if workload.kind in ("vit", "clip", "pi05") and self.name != "gsm_tiled":
            return False
        devices = adf_devices(adf)
        return len(devices) == 1 and devices[0][1] == 1

    def device_state(self, sim, container) -> DeviceState:
        return DeviceState(sim, container)

    def place_operator(self, ctx, state, workload, op, params, deps):
        raise NotImplementedError

    def allocate_outputs(self, ctx, state, workload, op, params, deps):
        """Allocate the operator's outputs when it starts; returns ``{tensor: [storage task ids]}``."""
        return {name: [ctx.storage(f"{state.path}:{name}", state.dram, workload.tensors[name].bytes, deps,
                                   op.name, op.label)] for name in op.outputs}

    @staticmethod
    def publish(state, allocations, done):
        """Consumers wait for the producer (``done``) and keep the output storage alive until they finish."""
        for name, ids in allocations.items():
            state.ready[name] = list(ids) + [done]

    def kernel_start(self, ctx, state, op, deps):
        """Task ids the operator's tasks wait for: ``deps`` (kept, so held storage outlives the operator) plus
        any template-specific ones such as stream order or a launch task; default: ``deps``."""
        return deps

    def kernel_end(self, state, done):
        """Record that the operator finished with task ``done`` (for templates with stream order)."""

    def run_device(self, ctx, state, workload, op, params):
        deps = state.deps(ctx, workload, op.inputs, op.name)
        allocations = self.allocate_outputs(ctx, state, workload, op, params, deps)
        held = [i for ids in allocations.values() for i in ids]
        if op.op == "allreduce":   # a single rank has nothing to reduce
            done = ctx.sync(f"{state.path}:{op.name}.done", deps + held, op.name, op.label)
        else:
            start = self.kernel_start(ctx, state, op, deps + held)
            tasks = self.place_operator(ctx, state, workload, op, params, start)
            done = ctx.sync(f"{state.path}:{op.name}.done", tasks, op.name, op.label)
            self.kernel_end(state, done)
        self.publish(state, allocations, done)
        return done

    def apply(self, simulator, workload, params) -> None:
        devices = sim_devices(simulator)
        if len(devices) != 1:
            raise ValueError(f"{self.name} maps onto one device; the ADF has {len(devices)} (use tensor_parallel)")
        ir = workload_from_spec(workload)
        simulator.mapping.graph.iterations = ir.iterations
        state = self.device_state(simulator, devices[0])
        for op in ir.operators:
            self.run_device(simulator.mapping, state, ir, op, params)


def check_params(template, space, params):
    from arch_explorer.adf.registry import ConditionalMappingSpace, validate_mapping_params
    if isinstance(space, ConditionalMappingSpace):
        return validate_mapping_params(space, params)
    unknown = set(params) - set(space)
    if unknown:
        raise ValueError(f"{template}: unknown mapping parameters {sorted(unknown)}; space: {sorted(space)}")
    for key, value in params.items():
        if value not in space[key]:
            raise ValueError(f"{template}: {key}={value!r} is not in its domain {space[key]}")
