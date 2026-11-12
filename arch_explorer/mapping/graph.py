"""Default single-device placement of registered, materialized DAG operators.

GSM keeps its existing tiled matrix/vector implementations. Other registered
operators select an explicit target program, including declared temporaries.
Programs use the shared storage/publish path and no scheduler trial. The public
execution_graph policy selects implementations/placement without copying a
template's internal device state machine.
"""
from arch_explorer.mapping.base import sim_devices
from arch_explorer.simulator.plan import PlanConstraints
from arch_explorer.workload.llm import workload_from_spec


def _kernel(op, dtype):
    # Operator.kernel owns validation of the common lowering IR.
    return op.kernel(dtype)


def operator_dtype(ir, op):
    operands = op.inputs or op.outputs
    return op.attrs.get("dtype", ir.tensors[operands[0]].dtype)


def _units(state, kernel):
    units = state.units_for(kernel)
    for unit in units:
        state.sim.points[unit].kernel_demand(kernel)
    return units


def _materialized(ir):
    for tensor in ir.tensors.values():
        if tensor.alias_of is not None or tensor.strides is not None:
            raise NotImplementedError("default DAG mapping requires materialized tensors; use an explicit layout mapping")


def graph_applicable(template, adf, spec):
    """Use the same registered lower and physical capability queries as execution."""
    from arch_explorer.simulator.events.builder import build_simulator
    try:
        ir = workload_from_spec(spec)
        _materialized(ir)
        sim = build_simulator(adf, options={"systolic_model": "native"})
        devices = sim_devices(sim)
        if len(devices) != 1:
            return False
        state = template.device_state(sim, devices[0])
        from arch_explorer.mapping.execution_graph import prepare_operator
        from arch_explorer.workload.operators import VECTOR_COSTS
        for op in ir.operators:
            if op.op != "allreduce":
                if op.op in ('matmul', 'mvm', *VECTOR_COSTS):
                    _units(state, _kernel(op, operator_dtype(ir, op)))
                else:
                    prepare_operator(state, ir, op)
        return True
    except (ValueError, TypeError, KeyError, NotImplementedError):
        return False


def place_registered(ctx, state, workload, op, dtype, deps):
    """Consume selected target artifacts, including multi-step temporaries.

    GSM's outer run_device already allocated the original outputs. Pass those
    allocations to the shared DAG step placement instead of charging them twice.
    """
    from arch_explorer.mapping.execution_graph import prepare_operator, place_program
    implementation, program = prepare_operator(state, workload, op)
    allocations = {}
    for name in op.outputs:
        ids = [i for i in deps if ctx.graph.tasks[i].kind == 'storage'
               and ctx.graph.tasks[i].name == f'{state.path}:{name}']
        if len(ids) != 1:
            raise ValueError(f'{op.name}: original output allocation {name!r} is not unique')
        allocations[name] = ids
    return place_program(ctx, state, workload, op, implementation, program,
                         deps=deps, allocations=allocations)


def apply_graph(template, sim, spec, params):
    ir = workload_from_spec(spec)
    _materialized(ir)
    # A prepared graph has no execution-based autotuning phase. Its default
    # is a documented fixed tile, independent of shape or measured latency.
    params = {**template.defaults(), **params}
    for axis in ("tile_m", "tile_n"):
        if params[axis] == "auto":
            params[axis] = 64
    devices = sim_devices(sim)
    if len(devices) != 1:
        raise ValueError("default DAG mapping requires one device")
    ctx = sim.mapping
    ctx.plan.require(PlanConstraints(iterations=ir.iterations, check_capacity=True,
                                     reason="materialized DAG workload"))
    ctx.graph.iterations = ir.iterations
    state = template.device_state(sim, devices[0])
    state.materialized_graph = True
    for op in ir.operators:
        template.run_device(ctx, state, ir, op, params)
    consumed = {name for op in ir.operators for name in op.inputs}
    outputs = [name for op in ir.operators for name in op.outputs if name not in consumed]
    held = [task for name in outputs for task in state.ready[name]]
    ctx.sync("workload.return", [*state.stream, *held])
    ctx.details["workload_graph"] = {
        "graph": ir.to_dict(), "outputs": outputs,
        "placement": "GSM matrix/vector tiling; other registered kernels atomic on one compatible unit",
        "storage": "materialized inputs/outputs; direct consumer dependencies retain allocations; terminal outputs held through return",
    }
