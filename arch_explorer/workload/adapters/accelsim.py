"""Native capability, topology, thread options and instruction conversion.

All public tasks reach the existing trace component engine through this
adapter. Neither common mapping nor semantic registration imports it.
"""
import math

from . import (TaskAdapter, AdapterDomain, AdaptedGroup, TargetInput,
               register_task_adapter, register_adapter_domain)
from . import materialized_native

_IMPLEMENTATIONS = None


def supports(op, context):
    from ..compiled_operators import _validate, access_counts
    from arch_explorer.components.sm.implementations.accelsim.lowering_limits import LIMITS
    if context.target != 'accelsim' or context.dtype != 'fp32':
        return False
    if any(context.tensors[n].dtype != 'fp32' for n in (*op.inputs, *op.outputs)):
        return False
    _validate(op,context.tensors)
    return (all(math.prod(context.tensors[n].shape) <= LIMITS['tensor_elements']
                for n in (*op.inputs,*op.outputs))
            and access_counts(op)['reduction_fmas'] <= LIMITS['matrix_fmas'])


def lower_operator(op, context):
    """Original scalar emission; source region indices and reduction order unchanged."""
    import numpy as np
    from ..compiled_operators import _validate, reduction_blocks
    a = _validate(op,context.tensors)
    if not supports(op,context):
        raise ValueError('compiled operator exceeds native target expansion bounds')
    from arch_explorer.components.sm.implementations.accelsim.emitter import ProgramEmitter
    emitter = context.options.get('emitter')
    if (not isinstance(emitter, ProgramEmitter) or emitter.operator is not op
            or any(emitter.graph.tensors[n] != context.tensors[n]
                   for n in (*op.inputs,*op.outputs))):
        raise ValueError('compiled native lowering requires its own evaluation-local emitter')
    if any(n not in emitter.memory.addresses or emitter.memory.dtypes[n] != np.dtype('float32')
           or emitter.memory.addresses[n].shape != context.tensors[n].shape for n in op.inputs):
        raise ValueError('compiled native operand memory differs from its FP32 tensor declaration')
    output = op.outputs[0]
    kernel = emitter.kernel(output)
    index = kernel.index
    if op.op == 'compiled.matmul':
        batch = index // (a['rows'] * a['cols'])
        row = a['row'] + index // a['cols'] % a['rows']
        col = a['col'] + index % a['cols']
        kernel.constant(8, 0)
        for block in reduction_blocks(op):
            for j in range(block['start'],block['stop']):
                kernel.load(9,op.inputs[0],(batch*a['m']+row)*a['k']+j)
                kernel.load(10,op.inputs[1],
                    ((0 if a['shared_b'] else batch)*a['k']+j)*a['n']+col)
                kernel.arithmetic('FFMA',8,9,10,8)
    else:
        addresses = np.empty((a['batch'],a['m'],a['n']),dtype=np.uint32)
        for name,tile in zip(op.inputs,a['tiles']):
            addresses[:,tile['row']:tile['row']+tile['rows'],
                      tile['col']:tile['col']+tile['cols']] = emitter.memory.addresses[name]
        kernel.gather(8,addresses.reshape(-1))
    kernel.finish(output,8)
    return emitter.finish_operator()


def adapt(context):
    from arch_explorer.adf.dynamic import instances
    from arch_explorer.components.implementations import resolve_component
    from arch_explorer.mapping.compilation import storage_locations
    from arch_explorer.components.accelsim.program import _compile
    from ..extensions import _record_source
    from ..reference_api import execute_reference
    from arch_explorer.adf.evaluation import WorkloadSpec, MappingSpec
    from arch_explorer.components.sm.implementations.accelsim.open_program import prepare_program
    physical = dict(instances(context.adf))
    computes = [path for path,point in physical.items() if resolve_component(point).role == 'compute']
    memories = [path for path,point in physical.items() if resolve_component(point).name == 'dram']
    selected_memories = storage_locations(context.adf,context.memory)
    if (computes != [context.compute] or memories != [path for path,_ in selected_memories]
            or set(context.tasks) != {op.name for op in context.program.operators}):
        raise ValueError('native adapter requires one SM cluster and its complete explicit striped DRAM domain/task group')
    _compile(context.adf)
    choices = {op.name:context.implementations.get(op.op,'compiled.accelsim.'+op.op.split('.')[-1])
               for op in context.program.operators}
    spec = WorkloadSpec('graph',context.program.to_dict())
    mapped = MappingSpec('accelsim_program',dict(threads_per_block=context.options['threads_per_block'],
                                               implementations=choices))
    prepared = prepare_program(execute_reference(spec),mapped)
    for subject in prepared.source_objects:
        _record_source(subject)
    return AdaptedGroup(dict(mapping=mapped.to_dict()), dict(
        numerics='CPU emitted instruction interpretation checked against semantic reference',
        native='actual timing-only trace execution; no functional device result',
        boundary='component_io data; address metadata/cache traffic separately reported',
        compute=context.compute,memory=context.memory,
        physical_memory=[path for path,_ in selected_memories],hardware_accuracy=None))


def compose(adf, program, groups):
    from arch_explorer.adf.evaluation import WorkloadSpec, MappingSpec
    if len(groups) != 1:
        raise NotImplementedError('native execution has no bridge between separately adapted task groups')
    group = groups[0]
    if set(group['payload']) != {'mapping'}:
        raise ValueError('native adapter requires an explicit validated program mapping')
    return TargetInput('accelsim_components', WorkloadSpec('graph',program.to_dict()),
                       MappingSpec(**group['payload']['mapping']),group['scope'])


DEFINITION = TaskAdapter('accelsim','1','accelsim',
    ('compiled.matmul','compiled.assemble') + materialized_native.KINDS,adapt,
    parameters={'threads_per_block':{'type':'int','choices':[32,64,128]}})
DOMAIN = AdapterDomain('accelsim',compose)


def install():
    from ..execution import ImplementationDefinition, register_implementation
    from arch_explorer.components.sm.implementations.accelsim.lowering_limits import CAPABILITIES
    global _IMPLEMENTATIONS
    if _IMPLEMENTATIONS is None:
        _IMPLEMENTATIONS = tuple(ImplementationDefinition('compiled.accelsim.'+kind,'1',
            'compiled.'+kind,'accelsim',lower_operator,requirements=CAPABILITIES,
            dtypes=('fp32',),supports=supports) for kind in ('matmul','assemble'))
    for definition in _IMPLEMENTATIONS:
        register_implementation(definition)
    materialized_native.install()
    register_task_adapter(DEFINITION)
    register_adapter_domain(DOMAIN)
