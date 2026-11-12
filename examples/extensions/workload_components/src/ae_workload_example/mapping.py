"""Row slices on two arrays, using only public mapping primitives.

Input tensors are initially resident in shared SRAM. Dependencies, capacity,
resource contention and timing are handled by the real events executor.
"""
from arch_explorer.adf import SpaceMatrix, SpacePoint
from arch_explorer.adf.registry import MappingTemplate
from arch_explorer.components.api import PlanConstraints
from arch_explorer.workload.api import workload_from_spec


def _graph(spec):
    graph = workload_from_spec(spec)
    if graph.dtype != "fp32" or graph.iterations != 1 or not graph.operators:
        raise NotImplementedError("tutorial_rows requires a nonempty, single-iteration FP32 graph")
    shape = next(iter(graph.tensors.values())).shape
    if len(shape) != 2 or any(type(size) is not int or not 1 <= size <= 4096 for size in shape):
        raise NotImplementedError("tutorial_rows maps two-dimensional tensors with dimensions in [1, 4096]")
    for tensor in graph.tensors.values():
        if (tensor.shape != shape or tensor.dtype != "fp32" or tensor.kind != "activation"
                or tensor.alias_of is not None):
            raise NotImplementedError("tutorial_rows requires equal-shaped FP32 activations without aliases")
    for op in graph.operators:
        if (op.attrs.get("rows"), op.attrs.get("cols")) != shape:
            raise NotImplementedError("tutorial_rows requires elementwise row/column semantics")
        kernel = op.kernel(graph.dtype, rows=1)
        if kernel.op != "vector" or kernel.sfu or kernel.row_sfu:
            raise NotImplementedError("tutorial_rows requires a SIMD vector lowering without SFU")
    return graph


class RowTiles(MappingTemplate):
    name = "tutorial_rows"

    def applicable(self, adf, workload):
        # This example intentionally supports one explicit shared-memory shape.
        if (not isinstance(adf, SpaceMatrix) or adf.repeat != 1 or len(adf.elements) != 2
                or any(not isinstance(point, SpacePoint) for point in adf.elements)
                or [(point.component, point.repeat) for point in adf.elements]
                   != [("tutorial_array", 2), ("sram", 1)]
                or len(adf.networks) != 1 or adf.networks[0].component != "bus"):
            return False
        _graph(workload)
        return True

    def space(self, adf, workload):
        return {"tile_rows": [4, 16]}

    def apply(self, simulator, workload, params):
        if (set(params) != {"tile_rows"} or type(params["tile_rows"]) is not int
                or params["tile_rows"] not in (4, 16)):
            raise ValueError("tutorial_rows requires integer tile_rows=4 or 16")
        graph = _graph(workload)
        ctx = simulator.mapping
        ctx.plan.require(PlanConstraints(iterations=1, check_capacity=True,
                                         reason="external graph tensor residency"))
        arrays = simulator.find_points(component="tutorial_array")
        memories = simulator.find_points(component="sram")
        if len(arrays) != 2 or len(memories) != 1:
            raise ValueError("tutorial_rows needs two arrays and one shared SRAM")
        memory = memories[0].path
        producers = graph.producers()
        ready = {name: [ctx.storage(f"input:{name}", memory, tensor.bytes)]
                 for name, tensor in graph.tensors.items() if name not in producers}
        placements = []
        for op in graph.operators:
            deps = list(dict.fromkeys(task for name in op.inputs for task in ready[name]))
            allocations = {name: ctx.storage(f"output:{name}", memory, graph.tensors[name].bytes,
                                            deps, op.name, op.label) for name in op.outputs}
            tasks = []
            for index, start in enumerate(range(0, op.attrs["rows"], params["tile_rows"])):
                rows = min(params["tile_rows"], op.attrs["rows"] - start)
                compute = arrays[index % len(arrays)].path
                size = rows * op.attrs["cols"] * 4
                streams = [(memory, size) for _ in op.inputs] + [(memory, size, "write") for _ in op.outputs]
                task = ctx.map_node(f"{op.name}.rows{start}", compute, [(op.kernel(graph.dtype, rows=rows), 1)],
                                    deps + list(allocations.values()), streams, op.name, op.label,
                                    latency_ns=simulator.access_latency(memory, compute))
                tasks.append(task)
                placements.append({"operator": op.name, "task": task, "compute": compute,
                                   "row_start": start, "rows": rows})
            done = ctx.sync(f"{op.name}.ready", tasks, op.name, op.label)
            for name, allocation in allocations.items():
                ready[name] = [allocation, done]
        consumed = {name for op in graph.operators for name in op.inputs}
        outputs = [name for name in producers if name not in consumed]
        ctx.sync("return", [task for name in outputs for task in ready[name]])
        ctx.details["tutorial_rows"] = {"tile_rows": params["tile_rows"], "placements": placements,
                                         "scope": "synthetic compute and shared SRAM; inputs initially resident"}
