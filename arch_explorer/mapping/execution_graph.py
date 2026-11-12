"""Configurable single-device placement of selected target execution programs.

This is a policy over the existing graph/DeviceState/MappingContext primitives,
not another scheduler. Operator semantics and decompositions come exclusively
from the workload execution registry; compute demand comes from real points.
"""
from dataclasses import asdict, dataclass, replace
import json
from types import SimpleNamespace

from arch_explorer.adf.registry import MappingTemplate, register_template
from arch_explorer.components.base import Kernel
from arch_explorer.mapping.base import DeviceState, PlacementTemplate, sim_devices
from arch_explorer.simulator.plan import PlanConstraints


def _qualified(kind, operator, local):
    # Both names are arbitrary graph identifiers. Encoding the pair avoids
    # collisions such as ('a/b', 'c') versus ('a', 'b/c').
    return '@'+kind+'/'+json.dumps([operator, local], ensure_ascii=True, separators=(',', ':'))


def _parameters(params, ir):
    from arch_explorer.workload.graph import finite_json
    value = finite_json(params)
    if type(value) is not dict or set(value) - {'implementations', 'placement', 'serialize', 'memory'}:
        raise ValueError('execution_graph expects implementations/placement/serialize/memory only')
    result = {'implementations': {}, 'placement': 'first', 'serialize': False, **value}
    choices = result['implementations']
    if type(choices) is not dict or set(choices) - {op.name for op in ir.operators}:
        raise ValueError('execution_graph implementations must address actual operator node names')
    if any(type(v) is not str or not v or v.strip() != v for v in choices.values()):
        raise ValueError('execution_graph implementation IDs must be nonempty strings')
    if result['placement'] not in ('first', 'round_robin') or type(result['serialize']) is not bool:
        raise ValueError('execution_graph placement must be first/round_robin and serialize must be bool')
    if 'memory' in result and (type(result['memory']) is not str or not result['memory']):
        raise ValueError('execution_graph memory must be an explicit memory path')
    return result


def _compatible_units(state, kernel):
    accepted = []
    for path in sorted(state.computes):
        try:
            state.sim.points[path].kernel_demand(kernel)
            accepted.append(path)
        except (NotImplementedError, ValueError, TypeError):
            continue
    if not accepted:
        raise NotImplementedError(f'{state.path}: no actual component accepts {kernel.op}/{kernel.precision}/{kernel.shape}')
    return accepted


def capabilities(state, dtype):
    result = {f'component:{state.sim.points[path].component_type.name}' for path in state.computes}
    for op, shape in [('vector', (1, 1)), ('gemm', (1, 1, 1, 1)), ('mvm', (1, 1, 1))]:
        try:
            _compatible_units(state, Kernel(op, shape, dtype, simt=1))
            result.add('kernel:' + op)
        except NotImplementedError:
            pass
    return tuple(sorted(result))


def prepare_operator(state, ir, op, choice=None):
    from arch_explorer.mapping.graph import operator_dtype
    from arch_explorer.workload.execution import ExecutionContext, lower_implementation, select_implementation
    dtype = operator_dtype(ir, op)
    caps = capabilities(state, dtype)
    implementation = select_implementation(op, 'events', dtype=dtype, tensors=ir.tensors,
                                           capabilities=caps, implementation=choice)
    context = ExecutionContext('events', dtype, ir.tensors, {}, caps)
    program = lower_implementation(implementation, op, context)
    for step in program.steps:
        _compatible_units(state, step.kernel)
    return implementation, program


@dataclass(frozen=True)
class _PlacedStep:
    name: str
    inputs: tuple
    outputs: tuple
    product: Kernel
    label: str
    operator: str
    explicit_deps: tuple = ()
    op: str = 'execution_step'


class _StepPlacement(PlacementTemplate):
    """Use the existing storage/publish machine for each declared program step."""
    name = 'execution_graph_steps'

    def __init__(self, params, allocations=None):
        self.params = params
        self.allocations = {} if allocations is None else allocations
        self.done = {}
        self.selected = []
        self.serial = []
        self.controls = []
        self.index = 0

    def space(self, adf, workload):
        return {}

    def kernel_start(self, ctx, state, op, deps):
        return [*deps, *(self.done[name] for name in op.explicit_deps), *self.serial, *self.controls]

    def allocate_outputs(self, ctx, state, workload, op, params, deps):
        return {name: self.allocations[name] if name in self.allocations else [
            ctx.storage(f'{state.path}:{name}', state.dram, workload.tensors[name].bytes, deps,
                        op.operator, op.label)] for name in op.outputs}

    def place_operator(self, ctx, state, workload, op, params, deps):
        units = _compatible_units(state, op.product)
        index = self.index if self.params['placement'] == 'round_robin' else 0
        unit = units[index % len(units)]
        self.index += 1
        read = sum(workload.tensors[name].bytes for name in op.inputs)
        written = sum(workload.tensors[name].bytes for name in op.outputs)
        task = ctx.map_node(op.name+'@'+unit, unit, [(op.product, 1)], deps,
                            [(state.dram, read), (state.dram, written, 'write')],
                            op.operator, op.label, latency_ns=ctx.sim.access_latency(state.dram, unit))
        self.selected.append({'step': op.name, 'operator': op.operator, 'unit': unit,
                              'kernel': asdict(op.product), 'inputs': list(op.inputs),
                              'outputs': list(op.outputs), 'deps': list(op.explicit_deps),
                              'read_bytes': read, 'written_bytes': written})
        return [task]

    def kernel_end(self, state, done):
        if self.params['serialize']:
            self.serial = [done]


def place_program(ctx, state, ir, op, implementation, program, *, params=None, deps=(), allocations=None,
                  placement=None):
    """Place one explicit multi-step lowering using normal storage dependencies.

    Existing strategies may pass their already allocated original outputs. That
    avoids double allocation while all new temporaries use normal task storage.
    """
    from arch_explorer.workload.execution import implementation_identity
    params = {'placement': 'first', 'serialize': False, **({} if params is None else params)}
    tensors = dict(ir.tensors)
    names = {}
    for tensor in program.tensors:
        name = _qualified('execution', op.name, tensor.name)
        if name in tensors:
            raise ValueError(f'{op.name}: temporary namespace collides with graph tensor {name!r}')
        names[tensor.name] = name
        tensors[name] = replace(tensor, name=name)
    steps = []
    for step in program.steps:
        steps.append(_PlacedStep(_qualified('step', op.name, step.name),
            tuple(names.get(n, n) for n in step.inputs), tuple(names.get(n, n) for n in step.outputs),
            step.kernel, op.label, op.name, tuple(_qualified('step', op.name, n) for n in step.deps)))
    policy = placement or _StepPlacement(params, allocations)
    policy.allocations = {} if allocations is None else allocations
    # Dependencies on actual input tensors are rebuilt per step, so an input
    # need not remain resident after its final real consumer. Extra ordering
    # dependencies from a caller are retained without synthesizing delays.
    input_deps = set(state.deps(ctx, ir, op.inputs, op.name))
    controls = [i for i in deps if i not in input_deps]
    policy.controls = controls
    local = SimpleNamespace(tensors=tensors)
    before = len(policy.selected)
    done = []
    for step in steps:
        task = policy.run_device(ctx, state, local, step, {})
        policy.done[step.name] = task
        done.append(task)
    # Expose the selected stable identity and exact lowered graph for binding.
    report = ctx.details.setdefault('execution_graph', {'schema': 'arch-explorer.execution-graph-plan.v1', 'operators': []})
    report['operators'].append({'operator': op.name, 'implementation': implementation_identity(implementation),
        'temporaries': [{'name': names[t.name], 'shape': list(t.shape), 'dtype': t.dtype, 'bytes': t.bytes}
                        for t in program.tensors], 'steps': policy.selected[before:]})
    return done


class ExecutionGraph(MappingTemplate):
    """A public configurable strategy for materialized single-device DAGs."""
    name = 'execution_graph'

    def space(self, adf, workload):
        return {'implementations': [{}], 'placement': ['first', 'round_robin'], 'serialize': [False, True]}

    def applicable(self, adf, workload):
        # Structural availability does not silently choose an ambiguous program.
        # Concrete selection is checked by request_space/apply with its params.
        from arch_explorer.mapping.base import adf_devices
        from arch_explorer.mapping.gsm import _has_gpu
        devices = adf_devices(adf)
        return not _has_gpu(adf) and len(devices) == 1 and devices[0][1] == 1

    def request_space(self, adf, workload, params=None):
        from arch_explorer.simulator.events.builder import build_simulator
        from arch_explorer.mapping.context import MappingContext
        simulator = build_simulator(adf, options={'systolic_model': 'native'})
        simulator.mapping = MappingContext(simulator)
        result = self._apply(simulator, workload, {} if params is None else params)
        return {key: [value] for key, value in result.items()}

    def apply(self, simulator, workload, params):
        self._apply(simulator, workload, params)

    def _apply(self, sim, spec, params):
        from arch_explorer.mapping.graph import _materialized
        from arch_explorer.workload.api import workload_from_spec
        ir = workload_from_spec(spec)
        _materialized(ir)
        params = _parameters(params, ir)
        devices = sim_devices(sim)
        if len(devices) != 1 or not self.applicable(sim.adf, spec):
            raise ValueError('execution_graph requires one ordinary events device')
        state = DeviceState(sim, devices[0])
        if 'memory' in params:
            sim.memory(params['memory'])  # validate a real storage point/group
            state.dram = params['memory']
        prepared = [(op, *prepare_operator(state, ir, op, params['implementations'].get(op.name)))
                    for op in ir.operators]
        ctx = sim.mapping
        ctx.plan.require(PlanConstraints(iterations=ir.iterations, check_capacity=True,
                                         reason='selected materialized execution DAG'))
        ctx.graph.iterations = ir.iterations
        policy = _StepPlacement(params)
        all_done = []
        for op, implementation, program in prepared:
            all_done.extend(place_program(ctx, state, ir, op, implementation, program,
                                           params=params, placement=policy))
        consumed = {n for op in ir.operators for n in op.inputs}
        outputs = [n for op in ir.operators for n in op.outputs if n not in consumed]
        held = [task for n in outputs for task in state.ready[n]]
        ctx.sync('workload.return', [*all_done, *held])
        ctx.details['execution_graph'].update(graph=ir.to_dict(), outputs=outputs,
                                              memory=state.dram, policy=params)
        return params


register_template(ExecutionGraph)
