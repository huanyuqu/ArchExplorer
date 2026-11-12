"""Generic materialized graph expansion through source-bound mapping rules."""
import math
from copy import copy, deepcopy

from arch_explorer.workload.graph import finite_json, validate_graph
from arch_explorer.workload.operators import Operator, Tensor, Workload

MAX_STEPS = 512
MAX_ELEMENTS = 1_000_000
MAX_TOTAL_BYTES = 32 * 1024 * 1024


def storage_locations(adf, path):
    """Resolve declared storage membership; routing capability is adapter-owned."""
    from arch_explorer.adf.dynamic import instances
    from arch_explorer.adf.model import SpacePoint, walk
    from arch_explorer.components.implementations import resolve_component
    physical = dict(instances(adf))
    if path in physical:
        memories = [(path,resolve_component(physical[path]))]
    else:
        declared = dict(walk(adf)).get(path)
        if not isinstance(declared,SpacePoint):
            raise ValueError('compile_graph memory must name a physical or repeated storage point')
        memories = [(path,resolve_component(point)) for path,point in physical.items() if point is declared]
    if not memories or any(r.role != 'memory' for _,r in memories):
        raise ValueError('compile_graph memory role differs')
    return memories


def locations(adf, mapping, origin=None):
    return _resolved_locations(adf,parameters(mapping),origin)


def _resolved_locations(adf,p,origin=None):
    """Internal resolver for a request-local, already validated parameter copy."""
    from arch_explorer.adf.dynamic import instances
    from arch_explorer.components.implementations import resolve_component
    path = placement(p, 'compute', origin)
    physical = dict(instances(adf))
    if path not in physical:
        raise ValueError('compile_graph compute must name an actual physical instance')
    compute = (path,resolve_component(physical[path]))
    if compute[1].role != 'compute':
        raise ValueError('compile_graph compute role differs')
    memories = storage_locations(adf,placement(p, 'memory', origin))
    return compute, memories


def parameters(mapping):
    from .rules import MappingRuleBinding
    if mapping.template not in ('compile_graph_v2','compile_graph_v3','compile_graph_v4'):
        raise ValueError('compilation requires compile_graph_v2/v3; migrate historical compilations explicitly')
    fields={'compute','memory','rules'} | ({'operator_rules'} if mapping.template!='compile_graph_v2' else set()) | ({'storage','replicas'} if mapping.template=='compile_graph_v4' else set())
    if type(mapping.params) is not dict or set(mapping.params) != fields:
        raise ValueError(mapping.template+' fields differ; v3 adds exact operator_rules, backend options belong in bindings')
    raw = dict(mapping.params)
    selectors=('rules','operator_rules') if mapping.template!='compile_graph_v2' else ('rules',)
    for key in selectors:
        if type(raw[key]) is not dict:raise ValueError('mapping '+key+' must be a selector object')
        raw[key] = {k:v.to_dict() if type(v) is MappingRuleBinding else v for k,v in raw[key].items()}
    p = finite_json(raw)
    if mapping.template=='compile_graph_v4':
        from .storage_plan import validate_parameters
        validate_parameters(p)
    for key in ('compute', 'memory'):
        values = p[key].values() if type(p[key]) is dict else (p[key],)
        if not p[key] or any(type(v) is not str or not v or v.strip() != v for v in values):
            raise ValueError('compile_graph '+key+' requires an exact physical path')
    for key in selectors:
        for selector,binding in p[key].items():
            if not selector or selector.strip()!=selector:raise ValueError('mapping selectors must be nonempty and unpadded')
            MappingRuleBinding.from_dict(binding)
    return p


def placement(params, key, origin):
    value = params[key]
    if type(value) is dict:
        if origin not in value:
            raise ValueError('compile_graph '+key+' requires every original operator placement')
        return value[origin]
    return value


def _tensor_state(tensors):
    # Tensor fields have already been validated and contain immutable scalar /
    # tuple values. Copy each field dictionary, independently of hook objects.
    return {n:dict(vars(t)) for n,t in tensors.items()}


def _same_state(value,baseline):
    """Compare native fields without equating bool/int/float or signed zeros."""
    kind=type(baseline)
    if type(value) is not kind:return False
    if kind is dict:
        return (value.keys()==baseline.keys() and all(type(k) is str for k in value)
                and all(_same_state(value[k],v) for k,v in baseline.items()))
    if kind in (tuple,list):
        return len(value)==len(baseline) and all(_same_state(a,b) for a,b in zip(value,baseline))
    if kind is float:return value.hex()==baseline.hex()
    return value==baseline


def _rule_call(rule,method,op,tensors,options,state=None):
    """Isolate mutable hook arguments without copying the full program metadata."""
    from .rules import invoke
    operation=deepcopy(op)
    isolated=rule.tensor_scope=='isolated-operands'
    if isolated:
        # The explicit contract exposes only operands and transitive alias
        # backing descriptors. Copies cannot escape and later mutate the DAG.
        selected={};pending=list((*op.inputs,*op.outputs))
        while pending:
            name=pending.pop()
            if name in selected:continue
            tensor=tensors[name];selected[name]=tensor
            if tensor.alias_of is not None:pending.append(tensor.alias_of)
        context={name:copy(tensor) for name,tensor in selected.items()}
        state={name:(dict(vars(tensor)) if state is None else state[name]) for name,tensor in selected.items()}
    else:
        selected=tensors;context=dict(tensors)
        if state is None:state=_tensor_state(tensors)
    local_options=deepcopy(options)
    product=invoke(rule,method,operation,context,local_options)
    if (context.keys()!=state.keys() or selected.keys()!=state.keys() or (not isolated and context!=selected)
            or any(type(t) is not Tensor for t in context.values())):
        raise ValueError('mapping callback changed tensor dictionary membership or value types')
    if (type(operation) is not type(op) or not _same_state(vars(operation),vars(op))
            or not _same_state(local_options,options)
            or any(not _same_state(vars(t),state[n]) for n,t in context.items())):
        raise ValueError('mapping callback mutated its operator, tensors or options')
    return product


from arch_explorer.workload.resource_budget import graph_budget, limit

@graph_budget
def map_workload(graph, mapping):
    """Compose selected expansions without knowing any concrete operator kind."""
    from .rules import MappingExpansion, select_rule, rule_identity, validate_access
    from arch_explorer.workload.graph import validate_operator
    validate_graph(graph)
    p = parameters(mapping)
    for key in ('compute', 'memory'):
        if type(p[key]) is dict and set(p[key]) != {op.name for op in graph.operators}:
            raise ValueError('compile_graph '+key+' must cover exactly the original operator names')
    if graph.iterations != 1 or not graph.operators:
        raise ValueError('compilation requires one nonempty explicit DAG iteration')
    if any(t.dtype not in ('fp32', 'fp16', 'bf16', 'int32', 'int64', 'bool') or t.kind == 'kv'
           or not 1 <= len(t.shape) <= limit('max_rank',4)
           for t in graph.tensors.values()):
        raise ValueError('compilation requires floating/int32/int64/bool non-KV tensors, rank 1..4')
    if any(math.prod(t.shape) > limit('max_tensor_elements',MAX_ELEMENTS) for t in graph.tensors.values()):
        raise ValueError('compilation tensor expansion bound exceeded')
    if set(p['rules']) - {op.op for op in graph.operators}:
        raise ValueError('mapping rules refer to absent original operator kinds')
    if set(p.get('operator_rules',{})) - {op.name for op in graph.operators}:
        raise ValueError('operator_rules refer to absent original operator instances')
    result = Workload(graph.name+'.mapped', tensors={n:Tensor(n,t.shape,t.dtype,t.kind,t.alias_of,t.strides,t.offset)
                      for n,t in graph.tensors.items()}, dtype=graph.dtype,
                      iterations=1, meta=finite_json(graph.meta, tuples=True))
    result.meta.pop('outputs',None)  # Partial expansions do not yet produce every retained output.
    facts = dict(origins={}, rules={}, accesses={}, inspections={})
    names = set()
    original = graph.to_dict()
    tensor_state=_tensor_state(result.tensors)
    materialized_bytes=sum(t.bytes for t in result.tensors.values() if t.alias_of is None)
    produced={n for op in graph.operators for n in op.outputs}
    available=set(graph.tensors)-produced
    for op in graph.operators:
        rule, binding = select_rule(op.op, p.get('operator_rules',{}).get(op.name,p['rules'].get(op.op)))
        options = binding.to_dict()['options']
        expansion = _rule_call(rule,'expand',op,result.tensors,options,tensor_state)
        if (type(expansion) is not MappingExpansion or type(expansion.tensors) is not tuple
                or type(expansion.operators) is not tuple or type(expansion.views) is not tuple or not expansion.operators):
            raise TypeError('mapping expansion requires tensor/operator tuples and nonempty operators')
        replaced=set()
        for tensor in expansion.views:
            if (type(tensor) is not Tensor or tensor.name not in op.outputs or tensor.name in replaced
                    or tensor.alias_of not in op.inputs or (tensor.shape,tensor.dtype)!=(result.tensors[tensor.name].shape,result.tensors[tensor.name].dtype)):
                raise ValueError('view expansion may only replace same-shape/dtype original outputs with input aliases')
            previous=result.tensors[tensor.name]
            replacement=Tensor(tensor.name,tensor.shape,tensor.dtype,tensor.kind,tensor.alias_of,tensor.strides,tensor.offset)
            replaced.add(tensor.name);result.tensors[tensor.name]=replacement
            materialized_bytes+=(replacement.bytes if replacement.alias_of is None else 0)-(previous.bytes if previous.alias_of is None else 0)
            tensor_state[tensor.name]=dict(vars(replacement))
        temporary = set()
        for tensor in expansion.tensors:
            if type(tensor) is not Tensor or tensor.name in result.tensors or tensor.name in temporary:
                raise ValueError('mapping temporary tensor namespace collides')
            if (tensor.dtype not in ('fp32', 'fp16', 'bf16', 'int32') or tensor.kind == 'kv'
                    or not 1 <= len(tensor.shape) <= limit('max_rank',4) or math.prod(tensor.shape) > limit('max_tensor_elements',MAX_ELEMENTS)):
                raise ValueError('mapping temporary must be bounded floating/int32 non-KV storage, rank 1..4')
            temporary.add(tensor.name)
            result.tensors[tensor.name] = Tensor(tensor.name,tensor.shape,tensor.dtype,tensor.kind,tensor.alias_of,tensor.strides,tensor.offset)
            tensor_state[tensor.name]=dict(vars(result.tensors[tensor.name]))
            if tensor.alias_of is None:materialized_bytes+=tensor.bytes
        written = set()
        for step in expansion.operators:
            if not isinstance(step, Operator) or step.name in names:
                raise ValueError('mapping expansion operator names must be unique')
            if set(step.inputs) - (set(op.inputs) | temporary):
                raise ValueError('mapping expansion reads undeclared original inputs')
            if set(step.inputs)-available:
                raise ValueError('mapping expansion contains a forward tensor dependency')
            if set(step.outputs) - (set(op.outputs) | temporary) or written.intersection(step.outputs):
                raise ValueError('mapping expansion overwrites a foreign tensor or duplicate output')
            names.add(step.name)
            validate_operator(step,result.tensors)
            written.update(step.outputs)
            available.update(step.outputs)
            result.operators.append(deepcopy(step))
            facts['origins'][step.name] = op.name
        if written != set(op.outputs) | temporary:
            raise ValueError('mapping expansion must produce every original output and temporary')
        if len(result.operators) > limit('max_steps',MAX_STEPS) or materialized_bytes > limit('max_total_tensor_bytes',MAX_TOTAL_BYTES):
            raise ValueError('compilation materialized expansion bound exceeded')
        for step in expansion.operators:
            facts['accesses'][step.name] = validate_access(_rule_call(rule,'access',step,result.tensors,options,tensor_state))
            inspection = finite_json(_rule_call(rule,'inspect',step,result.tensors,options,tensor_state))
            if type(inspection) is not dict:
                raise TypeError('mapping inspection must return a JSON object')
            facts['inspections'][step.name] = inspection
        facts['rules'][op.name] = dict(operator=op.op, binding=binding.to_dict(), definition=rule_identity(rule))
    if not _same_state(_tensor_state(result.tensors),tensor_state):
        raise ValueError('mapping callback mutated tensor descriptors outside its context')
    actual_bytes=sum(t.bytes for t in result.tensors.values() if t.alias_of is None)
    if actual_bytes!=materialized_bytes:
        raise ValueError('mapping materialized byte accounting differs')
    if actual_bytes > limit('max_total_tensor_bytes',MAX_TOTAL_BYTES):
        raise ValueError('compilation total materialized tensor bound exceeded')
    if graph.to_dict()!=original:
        raise ValueError('mapping callback mutated the original graph')
    if 'outputs' in graph.meta:result.meta['outputs']=finite_json(graph.meta['outputs'])
    if mapping.template=='compile_graph_v4':
        from .storage_plan import apply_storage_plan
        apply_storage_plan(graph,result,facts,p)
    validate_graph(result)
    return result, facts


def inspect_program(graph, mapping, facts):
    """Inspectable derivations; graph edges remain the lifetime authority."""
    p = parameters(mapping)
    producers = {n: op.name for op in graph.operators for n in op.outputs}
    consumers = {n:[] for n in graph.tensors}
    operators = {}
    for op in graph.operators:
        operators[op.name]=op
        for n in dict.fromkeys(op.inputs):
            consumers[n].append(op.name)
    steps = []
    for op in graph.operators:
        origin = facts['origins'][op.name]
        steps.append(dict(name=op.name, origin=origin, semantic=op.op,
            compute=placement(p,'compute',origin), memory=placement(p,'memory',origin),
            dependencies=sorted({producers[n] for n in op.inputs if n in producers}),
            access=facts['accesses'][op.name], inspection=facts['inspections'][op.name]))
    step_memory = {s['name']:s['memory'] for s in steps}
    from arch_explorer.workload.graph import graph_outputs
    returned=set(graph_outputs(graph))
    from arch_explorer.workload.storage import resolve_layout
    layouts={n:resolve_layout(graph.tensors,n) for n in graph.tensors}
    aliases_by_storage={}
    for n,layout in layouts.items():
        aliases_by_storage.setdefault(layout.storage,[]).append(n)
    physical=None
    if mapping.template=='compile_graph_v4':
        from .storage_plan import physical_locations
        physical=physical_locations(graph)
    for step in steps:
        op=operators[step['name']]
        step['address_map']=dict(reads=[dict(tensor=n,**layouts[n].to_dict()) for n in op.inputs],
            writes=[dict(tensor=n,**layouts[n].to_dict()) for n in op.outputs] if op.op!='storage.view' else [],
            indexing=dict(semantic=op.op,attrs=op.attrs),holes='not charged; logical elements only',
            service='direct positive-stride logical loads; no hidden packing')
    roots=[]
    for root in (n for n,t in graph.tensors.items() if t.alias_of is None):
        aliases=aliases_by_storage[root]
        uses=sorted({s for n in aliases for s in consumers[n]+([producers[n]] if n in producers else [])})
        memories=[physical[root]] if physical is not None else sorted({step_memory[s] for s in uses})
        if len(memories)>1:raise ValueError('shared root storage has conflicting memory placements: '+root)
        roots.append(dict(storage=root,bytes=graph.tensors[root].bytes,aliases=aliases,memories=memories,
            producer=producers.get(root),consumers=uses,returned=bool(set(aliases)&returned),
            release='resident for run' if graph.tensors[root].kind=='weight' and root not in producers else
                'return' if set(aliases)&returned else 'all real alias consumers and control dependencies complete'))
    return dict(rules=facts['rules'], steps=steps, allocations=roots,
        storage=[dict(tensor=n, bytes=t.bytes,logical_bytes=t.bytes,allocation_bytes=0 if t.alias_of else t.bytes,
        layout=layouts[n].to_dict(),kind=t.kind,dtype=t.dtype,
        memories=[physical[n]] if physical is not None else sorted({step_memory[s] for s in [*consumers[n], *([producers[n]] if n in producers else [])]}),
        producer=producers.get(n), consumers=consumers[n], release=('resident for run' if t.kind=='weight' and n not in producers else
            'return' if n in returned else 'producer and all real consumers complete'),
        returned=n in returned) for n,t in graph.tensors.items()],
        edge_policy='registered materialized expansion; all real graph readers determine release')


def resolved_parameters(mapping,facts):
    p=parameters(mapping)
    if mapping.template in ('compile_graph_v3','compile_graph_v4'):
        return {**p,'operator_rules':{name:item['binding'] for name,item in facts['rules'].items()}}
    return {**p,'rules':{item['operator']:item['binding'] for item in facts['rules'].values()}}


def migrate_mapping_v2(mapping):
    from arch_explorer.adf.evaluation import MappingSpec
    if type(mapping) is not MappingSpec or mapping.template!='compile_graph_v2':
        raise ValueError('explicit migration requires compile_graph_v2')
    return MappingSpec('compile_graph_v3',{**parameters(mapping),'operator_rules':{}})


from arch_explorer.adf.registry import MappingTemplate, register_template


class CompileGraph(MappingTemplate):
    name = 'compile_graph_v2'

    def applicable(self, adf, workload):
        from arch_explorer.workload.extensions import has_workload
        return has_workload(getattr(workload, 'kind', None))

    def space(self, adf, workload):
        raise ValueError(self.name+' requires explicit placements and rule selectors')

    def request_space(self, adf, workload, params=None):
        from arch_explorer.adf.evaluation import MappingSpec
        from arch_explorer.workload.api import workload_from_spec
        mapping = MappingSpec(self.name, params)
        p = parameters(mapping)
        original = workload_from_spec(workload)
        map_workload(original, mapping)
        for op in original.operators:
            locations(adf, mapping, op.name)
        return {key:[value] for key,value in p.items()}

    def apply(self, simulator, workload, params):
        raise ValueError('compile_graph_v2 must use compile_workload then prepare_compiled/evaluate_request')


register_template(CompileGraph)


class CompileGraphV3(CompileGraph):
    name='compile_graph_v3'


register_template(CompileGraphV3)


class CompileGraphV4(CompileGraph):
    name='compile_graph_v4'

    def request_space(self,adf,workload,params=None):
        space=super().request_space(adf,workload,params)
        paths=set(params['storage'].values())
        paths.update(dst['memory'] for item in params['replicas'] for dst in item['destinations'])
        for path in paths:
            found=storage_locations(adf,path)
            if len(found)!=1 or found[0][0]!=path:
                raise ValueError('v4 requires an actual single storage point: '+path)
        return space


register_template(CompileGraphV4)
