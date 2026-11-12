"""Kernel and component-IO conversions sharing one event execution domain."""
from dataclasses import asdict
from copy import deepcopy

from . import (TaskAdapter, AdapterDomain, AdaptedGroup, TargetInput,
               register_task_adapter, register_adapter_domain)

_IMPLEMENTATIONS = None


def _context(op, context):
    if context.dtype != 'fp32':
        raise ValueError('materialized adapters require explicit FP32 input')
    from ..graph import validate_operator
    validate_operator(op, context.tensors)
    return op.attrs


def lower_kernel(op, context):
    from arch_explorer.components.base import Kernel
    a = _context(op, context)
    if op.op=='storage.view':return Kernel('vector',(1,1),'fp32')
    from ..array_ops import KINDS as ARRAY_KINDS,lower_kernel as lower_array
    if op.op in ARRAY_KINDS:return lower_array(op,context)
    from ..block_ops import KINDS as BLOCK_KINDS
    if op.op in BLOCK_KINDS:
        from arch_explorer.mapping.block_rules import lower_kernel as lower_block
        return lower_block(op,context)
    from ..mla_ops import KINDS as MLA_KINDS
    if op.op in MLA_KINDS:
        from arch_explorer.mapping.rules import validate_access
        counts=validate_access(context.options.get('access'))
        return Kernel('vector',(1,1),'fp32',row_simt=counts['simt_instructions'],row_sfu=counts['sfu_instructions'])
    if op.op == 'compiled.matmul':
        return Kernel('gemm', (a['batch'], a['rows'], a['k'], a['cols']), 'fp32')
    if op.op == 'compiled.assemble':
        return Kernel('vector', (1, a['batch'] * a['m'] * a['n']), 'fp32')
    from ..indexing import KINDS as INDEX_KINDS
    from ..spatial_ops import KINDS as SPATIAL_KINDS
    if op.op in (*INDEX_KINDS,*SPATIAL_KINDS):
        from arch_explorer.mapping.rules import validate_access
        counts=validate_access(context.options.get('access'))
        return Kernel('vector',(1,1),'fp32',row_simt=counts['simt_instructions'],row_sfu=counts['sfu_instructions'])
    from ..tensor_ops import validate
    from ..extensions import _record_source
    _record_source(validate)
    validate(op, context.tensors)
    import math
    size = math.prod(context.tensors[op.outputs[0]].shape)
    if op.op in ('add', 'sub', 'mul', 'scale','scale_tensor','relu'):
        return Kernel('vector', (1, size), 'fp32', simt=1)
    if op.op in ('exp','erf','rsqrt','sqrt'):
        return Kernel('vector', (1, size), 'fp32', sfu=1)
    if op.op == 'div':
        return Kernel('vector', (1, size), 'fp32', simt=1, sfu=1)
    if op.op in ('reduce_max', 'reduce_sum'):
        shape = context.tensors[op.inputs[0]].shape
        return Kernel('vector', (math.prod(shape[:-1]), 1), 'fp32', row_simt=shape[-1]-1)
    if op.op in ('transpose', 'broadcast', 'reshape', 'cast','slice','patchify','concat'):
        return Kernel('vector', (1, size), 'fp32')
    if op.op in ('affine_norm', 'gelu_tanh', 'softmax', 'gelu_exact', 'bias','quick_gelu','l2_norm'):
        from ..full_primitives import COSTS
        simt,sfu,row_simt,row_sfu = COSTS[op.op]
        shape = context.tensors[op.outputs[0]].shape
        return Kernel('vector', (math.prod(shape[:-1]),shape[-1]), 'fp32',simt,sfu,row_simt,row_sfu)
    raise NotImplementedError('no materialized Kernel conversion for '+op.op)


def lower_roofline(op, context):
    from arch_explorer.mapping.rules import validate_access
    from arch_explorer.components.sm.implementations.roofline import RooflineWork
    _context(op, context)
    counts = validate_access(context.options.get('access'))
    return RooflineWork(counts['operations'], counts['read_bytes'] + counts['write_bytes'], counts['boundary'])


def restore_kernel(value):
    if value in ({'schema':'arch-explorer.readonly-view.v1'},{'schema':'arch-explorer.storage-copy.v1'}):return None
    if type(value) is dict and value.get('schema') == 'arch-explorer.events-precision-work.v1':
        if set(value) != {'schema','kernel','conversion_instructions'}:
            raise ValueError('saved precision work fields differ')
        from arch_explorer.components.sm.implementations.events import PrecisionWork
        return PrecisionWork(restore_kernel(value['kernel']),value['conversion_instructions'])
    from arch_explorer.components.base import Kernel
    from ..execution import validate_kernel
    if type(value) is not dict or set(value) != {'op','shape','precision','simt','sfu','row_simt','row_sfu'}:
        raise ValueError('saved adapter Kernel fields differ')
    result = Kernel(**{**value, 'shape': tuple(value['shape'])})
    validate_kernel(result)
    return result


def restore_roofline(value):
    if value=={'schema':'arch-explorer.readonly-view.v1'}:return None
    from arch_explorer.components.sm.implementations.roofline import RooflineWork
    return RooflineWork(**value)


def _check_context(op, execution, before):
    from ..execution import _callback_unchanged
    if not _callback_unchanged(op,execution,before):
        raise ValueError('execution callback mutated its operator, tensor context or options')


def _lower(context, target, traffic):
    from ..execution import (ExecutionContext, _fork_context, _select_in_context, lower_implementation,
                             _implementation_by_id, _callback_checkpoint)
    from ..extensions import _record_source
    from ..graph import finite_json
    from arch_explorer.adf.dynamic import instances
    point = dict(instances(context.adf))[context.compute]
    if 'precision' in point.params and point.value('precision') != 'fp32':
        raise ValueError('materialized FP32 adapter requires FP32 compute precision')
    template=ExecutionContext(target,'fp32',context.program.tensors)
    _record_source(ExecutionContext);_record_source(_fork_context);_record_source(_select_in_context)
    result = {}
    tasks=set(context.tasks)
    for op in context.program.operators:
        if op.name not in tasks:
            continue
        if op.op in ('storage.view','storage.copy'):
            if op.op=='storage.copy' and traffic!='external':raise ValueError('explicit routed storage copies require external event service')
            schema='arch-explorer.storage-copy.v1' if op.op=='storage.copy' else 'arch-explorer.readonly-view.v1'
            result[op.name]=dict(work={'schema':schema},traffic=traffic)
            continue
        choice = context.implementations.get(op.op, 'compiled.'+target+'.'+op.op.split('.')[-1])
        definition=_implementation_by_id(op,target,choice)
        isolated=definition.tensor_scope=='isolated-operands'
        operation=deepcopy(op) if isolated else op
        selection=_fork_context(template,operator=op,tensor_scope=definition.tensor_scope)
        before=_callback_checkpoint(operation,selection)
        definition = _select_in_context(operation,selection,implementation=choice)
        _check_context(operation,selection,before)
        options=dict(access=context.accesses[op.name])
        operation=deepcopy(op) if isolated else op
        execution = _fork_context(template,options=options,operator=op,tensor_scope=definition.tensor_scope)
        before=_callback_checkpoint(operation,execution)
        product = lower_implementation(definition, operation, execution)
        _check_context(operation,execution,before)
        if target == 'events':
            if product.tensors or len(product.steps) != 1:
                raise ValueError('kernel adapter requires one materialized semantic step per task')
            step = product.steps[0]
            if step.inputs != tuple(op.inputs) or step.outputs != tuple(op.outputs):
                raise ValueError('kernel adapter cannot conceal intermediate tensor accesses')
            product = step.kernel
        work = finite_json(asdict(product), tuples=True)
        conversions = context.accesses[op.name].get('conversion_instructions',0)
        if target == 'events' and conversions:
            work = dict(schema='arch-explorer.events-precision-work.v1',kernel=work,conversion_instructions=conversions)
        result[op.name] = dict(work=work, traffic=traffic)
    return AdaptedGroup(dict(tasks=result), dict(boundary='component_io',
        traffic=traffic, numerics='registered semantic reference', hardware_accuracy=None))


def adapt_kernels(context):
    return _lower(context, 'events', 'external')


def adapt_roofline(context):
    return _lower(context, 'roofline', 'included')


def compose(adf, program, groups):
    from arch_explorer.adf.evaluation import WorkloadSpec, MappingSpec
    from arch_explorer.simulator.events.backend import prepare_mapping
    from arch_explorer.mapping.compilation import storage_locations
    tasks = {}
    memories = set()
    for group in groups:
        paths = storage_locations(adf, group['memory'])
        if len(paths) != 1:
            raise ValueError('event adapters require one actual memory; striped domains need a routing adapter')
        memory = paths[0][0]
        memories.add(memory)
        payload = group['payload']
        if type(payload) is not dict or set(payload) != {'tasks'} or set(payload['tasks']) != set(group['tasks']):
            raise ValueError('event adapter product must cover exactly its declared tasks')
        for name, item in payload['tasks'].items():
            if (type(item) is not dict or set(item) not in ({'work', 'traffic'}, {'work', 'traffic', 'access'})
                    or item['traffic'] not in ('external','included')):
                raise ValueError('event adapter requires work and explicit external/included traffic boundary')
            from arch_explorer.mapping.rules import validate_access
            counts = validate_access(group['accesses'][name])
            if 'access' in item and item['access'] != counts:
                raise ValueError('event adapter access differs from the authoritative mapping rule')
            if name in tasks:
                raise ValueError('event adapter task overlap')
            tasks[name] = dict(item, access=counts, compute=group['compute'], memory=memory, adapter=group['adapter'])
    routed='storage_plan' in program.meta
    if not routed and len(memories) != 1:
        raise NotImplementedError('event groups require shared memory; no implicit cross-memory copy bridge')
    from ..compilation import _validated_program_dict
    spec = WorkloadSpec('graph', _validated_program_dict(program))
    if routed and any(item['traffic']!='external' for item in tasks.values()):
        raise ValueError('routed storage v1 requires explicit external Events streams; included Roofline IO is ambiguous')
    mapping = MappingSpec('adapted_routed_events' if routed else 'adapted_events', dict(tasks=tasks))
    simulator = prepare_mapping(adf, spec, mapping)
    from ..extensions import _record_source
    for model in simulator.points.values():
        _record_source(type(model))
        _record_source(model.component_type.implementation.factory)
    return TargetInput('events', spec, mapping, dict(
        boundary='per-task external streams or included component_io; shared lifetime graph',
        groups=[g['id'] for g in groups], hardware_accuracy=None))


from ..tensor_ops import KINDS
from ..indexing import KINDS as INDEX_KINDS
from ..mla_ops import KINDS as MLA_KINDS
from ..spatial_ops import KINDS as SPATIAL_KINDS
from ..array_ops import KINDS as ARRAY_KINDS
from ..block_ops import KINDS as BLOCK_KINDS
SEMANTICS = ('compiled.matmul', 'compiled.assemble','storage.view', *KINDS,*INDEX_KINDS,*MLA_KINDS,*SPATIAL_KINDS,*ARRAY_KINDS,*BLOCK_KINDS)
KERNELS = TaskAdapter('events', '1', 'events', SEMANTICS, adapt_kernels, restore_kernel,layouts=('independent','readonly_strided'))
ROOFLINE = TaskAdapter('roofline', '1', 'events', SEMANTICS, adapt_roofline, restore_roofline,layouts=('independent','readonly_strided'))
DOMAIN = AdapterDomain('events', compose)


def install():
    from ..execution import ImplementationDefinition, register_implementation
    global _IMPLEMENTATIONS
    if _IMPLEMENTATIONS is None:
        _IMPLEMENTATIONS = tuple(ImplementationDefinition('compiled.'+target+'.'+kind.split('.')[-1], '1',
            kind, target, lower, dtypes=('fp32',),tensor_scope='isolated-operands')
            for target, lower in (('events',lower_kernel), ('roofline',lower_roofline))
            for kind in SEMANTICS if kind!='storage.view')
    for definition in _IMPLEMENTATIONS:
        register_implementation(definition)
    register_task_adapter(KERNELS)
    register_task_adapter(ROOFLINE)
    register_adapter_domain(DOMAIN)
