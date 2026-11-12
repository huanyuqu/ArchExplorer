"""Common semantic mapping, component binding and source-bound adaptation.

The framework partitions tasks by declared placements and compatible adapter
contracts. It knows no concrete target, thread option, ISA or topology rule.
"""
import hashlib
import inspect
import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

from .graph import finite_json
from .adapters import (AdapterBinding, AdapterContext, AdaptedGroup, TargetInput,
                       bind_adapter, get_domain, adapter_identity, task_adapters)

SCHEMA = 'arch-explorer.compilation.v3'


def canonical(value):
    return json.dumps(finite_json(value),sort_keys=True,separators=(',',':'),
                      ensure_ascii=True,allow_nan=False).encode('ascii')


def seal(value):
    value = finite_json(value)
    return {**value,'binding':{'sha256':hashlib.sha256(canonical(value)).hexdigest()}}


def check(value, fields, schema):
    if type(value) is not dict or set(value) != fields or value.get('schema') != schema:
        raise ValueError('invalid '+schema+' fields; use explicit version migration for historical compilation')
    if value['binding'] != seal({k:v for k,v in value.items() if k != 'binding'})['binding']:
        raise ValueError('compilation content binding differs')


def _sources(subjects, implementations=()):
    from .extensions import callable_identity
    objects = [callable_identity(fn) for fn in subjects]
    root = Path(__file__).resolve().parents[1]
    names = ('workload/compilation.py','workload/extensions.py','workload/adapters/__init__.py','mapping/compilation.py',
             'mapping/rules.py','mapping/compiled.py','simulator/compilation.py')
    files = {name:root/name for name in names}
    for subject in subjects:
        source = inspect.getsourcefile(subject)
        if source is None:
            raise ValueError('compilation requires inspectable conversion sources')
        path = Path(source).resolve()
        if path.is_relative_to(root):
            files[path.relative_to(root).as_posix()] = path
            continue
        package = path.parent
        if (package/'__init__.py').is_file():
            while (package.parent/'__init__.py').is_file():
                package = package.parent
            closure = [(f'extension:{package.name}/'+p.relative_to(package).as_posix(),p)
                       for p in sorted(package.rglob('*.py'))]
        else:
            closure = [(f'extension:{subject.__module__}',path)]
        for key,path in closure:
            if key in files and files[key] != path:
                raise ValueError('ambiguous compilation extension source: '+key)
            files[key] = path
    return dict(files={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(files.items())},
                callables=sorted(objects,key=canonical),implementations=sorted(implementations,key=canonical))


def _registry(registry):
    from arch_explorer.adf.registry import COMPONENTS
    if registry is not None and registry is not COMPONENTS:
        raise NotImplementedError('saved compilation requires the installed component registry; '
            'process-local custom Registry cannot be reconstructed by bound providers')


def _same_inputs(value, baseline):
    """Compare complete initialized JSON roots without rebuilding numeric lists."""
    kind=type(baseline)
    if type(value) is not kind:
        return False
    if kind is dict:
        return (value.keys()==baseline.keys() and all(type(k) is str for k in value)
                and all(_same_inputs(value[k],v) for k,v in baseline.items()))
    if kind is list:
        if len(value)!=len(baseline):
            return False
        # Public reference initialization produces homogeneous native numeric
        # leaf rows. Check every actual leaf type, then use list value equality
        # in C; no Python JSON tree or numeric array is allocated per group.
        if baseline and type(baseline[0]) in (int,float):
            leaf=type(baseline[0])
            return all(type(v) is leaf for v in value) and value==baseline
        return all(_same_inputs(v,b) for v,b in zip(value,baseline))
    return value==baseline


def _same_json(value,baseline):
    """Compare against an independent validated JSON value, without rebuilding it."""
    kind=type(baseline)
    if kind is list:
        return (type(value) in (list,tuple) and len(value)==len(baseline)
                and all(_same_json(v,b) for v,b in zip(value,baseline)))
    if type(value) is not kind:
        return False
    if kind is dict:
        return (value.keys()==baseline.keys() and all(type(k) is str for k in value)
                and all(_same_json(value[k],b) for k,b in baseline.items()))
    if kind is float:
        return value.hex()==baseline.hex()
    return value==baseline


class _NativeBatch:
    """Columnar checks against independent native snapshots, with strict types.

    Object fromiter stores references without invoking array protocols. Type
    identity is checked before equality, numeric conversion, or traversal of
    any current value. Unusual heterogeneous containers retain the old guard.
    """

    def __init__(self, snapshots, operations=None):
        if operations is None:
            import numpy as np
            from operator import is_
            operations=(np,np.frompyfunc(type,1,1),np.frompyfunc(is_,2,1),np.frompyfunc(len,1,1))
        self.ops=operations
        self.size=len(snapshots)
        self.groups=[]
        self.fallback=None
        grouped={}
        for index,saved in enumerate(snapshots):
            kind,baseline=saved[:2]
            signature=(kind,tuple(sorted(baseline))) if kind is dict else ((kind,*saved[2:]) if kind in (list,tuple) else (kind,))
            grouped.setdefault(signature,[]).append(index)
        if self.size>=128 and len(grouped)>32:
            self.fallback=snapshots
            return
        np=self.ops[0]
        for signature,indices in grouped.items():
            selected=[snapshots[index] for index in indices]
            kind=signature[0]
            item=dict(kind=kind,indices=None if len(indices)==self.size else np.asarray(indices,dtype=np.intp))
            if kind is dict:
                keys=signature[1]
                item.update(keys=frozenset(keys),length=len(keys),children={key:_NativeBatch([saved[1][key] for saved in selected],self.ops) for key in keys})
            elif kind in (list,tuple):
                leaf,length=signature[1:]
                item.update(leaf=leaf,length=length)
                if leaf is float:
                    item['baseline']=np.frombuffer(b''.join(saved[1] for saved in selected),dtype=np.uint64)
                elif leaf is not None:
                    from itertools import chain
                    item['baseline']=self.objects(chain.from_iterable(saved[1] for saved in selected),len(selected)*length)
                elif length<=32:
                    item['children']=[_NativeBatch([saved[1][i] for saved in selected],self.ops) for i in range(length)]
                else:item['fallback']=selected
            elif kind is float:
                item['baseline']=np.frombuffer(b''.join(saved[1] for saved in selected),dtype=np.uint64)
            else:item['baseline']=self.objects((saved[1] for saved in selected),len(selected))
            self.groups.append(item)

    def objects(self, values, count):
        return self.ops[0].fromiter(values,dtype=object,count=count)

    def same_type(self, values, kind):
        return bool(self.ops[2](self.ops[1](values),kind).all())

    def matches(self, values):
        from .execution import _matches_value
        from itertools import chain
        from operator import itemgetter
        np=self.ops[0]
        if len(values)!=self.size:
            return False
        if self.fallback is not None:
            return all(_matches_value(value,saved) for value,saved in zip(values,self.fallback))
        for item in self.groups:
            current=values if item['indices'] is None else values.take(item['indices'])
            kind=item['kind']
            if not self.same_type(current,kind):
                return False
            if kind is dict:
                if not (self.ops[3](current)==item['length']).all():
                    return False
                keys=self.objects(chain.from_iterable(map(dict.keys,current)),len(current)*item['length'])
                if not self.same_type(keys,str) or set(keys)-item['keys']:
                    return False
                for key,child in item['children'].items():
                    if not child.matches(self.objects(map(itemgetter(key),current),len(current))):
                        return False
            elif kind in (list,tuple):
                if not (self.ops[3](current)==item['length']).all():
                    return False
                if item['leaf'] is not None:
                    flat=self.objects(chain.from_iterable(current),len(current)*item['length'])
                    if not self.same_type(flat,item['leaf']):
                        return False
                    actual=np.fromiter(flat,dtype=np.float64,count=len(flat)).view(np.uint64) if item['leaf'] is float else flat
                    if not np.array_equal(actual,item['baseline']):
                        return False
                elif 'fallback' in item:
                    if not all(_matches_value(value,saved) for value,saved in zip(current,item['fallback'])):
                        return False
                else:
                    for index,child in enumerate(item['children']):
                        if not child.matches(self.objects(map(itemgetter(index),current),len(current))):
                            return False
            else:
                actual=np.fromiter(current,dtype=np.float64,count=len(current)).view(np.uint64) if kind is float else current
                if not np.array_equal(actual,item['baseline']):
                    return False
        return True


class _ProgramSnapshot:
    """Independent complete native state of an already validated program."""

    def __init__(self,program,document):
        from .execution import _value_snapshot
        self.program=program
        self.document=document
        self.keys=frozenset(vars(program))
        self.header=_value_snapshot((program.name,program.dtype,program.iterations,program.meta))
        self.names=tuple(program.tensors)
        self.tensors={n:_value_snapshot(vars(t)) for n,t in program.tensors.items()}
        self.operators=tuple((type(op),_value_snapshot(vars(op))) for op in program.operators)
        self.batch=None
        if len(self.tensors)+len(self.operators)>=2048:
            tensors=_NativeBatch(list(self.tensors.values()))
            operators=_NativeBatch([saved for _,saved in self.operators],tensors.ops)
            classes=tensors.objects((cls for cls,_ in self.operators),len(self.operators))
            self.batch=(tensors,operators,classes,tensors.ops[0].frompyfunc(vars,1,1))

    def matches(self,program):
        from .operators import Workload,Tensor
        from .execution import _matches_value
        if (type(program) is not Workload or frozenset(vars(program))!=self.keys
                or not _matches_value((program.name,program.dtype,program.iterations,program.meta),self.header)
                or type(program.tensors) is not dict
                or any(type(name) is not str for name in program.tensors)
                or tuple(program.tensors)!=self.names or type(program.operators) is not list
                or len(program.operators)!=len(self.operators)):
            return False
        if self.batch is not None:
            tensors,operators,classes,fields=self.batch
            current_tensors=tensors.objects(program.tensors.values(),len(self.names))
            current_operators=operators.objects(iter(program.operators),len(self.operators))
            if (not tensors.same_type(current_tensors,Tensor)
                    or not operators.ops[2](operators.ops[1](current_operators),classes).all()):
                return False
            try:
                return tensors.matches(fields(current_tensors)) and operators.matches(fields(current_operators))
            except (KeyError,IndexError,TypeError,ValueError):
                return False
        return (all(type(t) is Tensor and _matches_value(vars(t),self.tensors[n]) for n,t in program.tensors.items())
                and all(type(op) is cls and _matches_value(vars(op),saved)
                        for op,(cls,saved) in zip(program.operators,self.operators)))


_PROGRAM_SNAPSHOT=ContextVar('compiled_program_snapshot',default=None)


@contextmanager
def _snapshot_scope(snapshot):
    token=_PROGRAM_SNAPSHOT.set(snapshot)
    try:
        yield
    finally:
        _PROGRAM_SNAPSHOT.reset(token)


def _validated_program_dict(program):
    snapshot=_PROGRAM_SNAPSHOT.get()
    if snapshot is None or snapshot.program is not program:
        return program.to_dict()
    if not snapshot.matches(program):
        raise ValueError('validated program changed before composition serialization')
    return deepcopy(snapshot.document)


def _program_unchanged(program, common, structure):
    if type(structure) is _ProgramSnapshot:
        return structure.matches(program)
    from .operators import Workload, Tensor, Operator
    from .full_primitives import FullOperator
    if (type(program) is not Workload or type(program.meta) is not dict
            or 'inputs' not in program.meta
            or not _same_inputs(program.meta['inputs'],common['meta']['inputs'])):
        return False
    # The baseline came from the strict public serializer before any adapter
    # received the graph. Compare every serialized field with exact native leaf
    # types; identical content needs no repeated semantic/source validation.
    if (not all(_same_json(getattr(program,k),structure[k]) for k in ('name','dtype','iterations'))
            or not _same_json({k:v for k,v in program.meta.items() if k!='inputs'},structure['meta'])
            or type(program.tensors) is not dict or type(program.operators) is not list
            or list(program.tensors)!=[t['name'] for t in structure['tensors']]
            or len(program.operators)!=len(structure['operators'])):
        return False
    for tensor,saved in zip(program.tensors.values(),structure['tensors']):
        if type(tensor) is not Tensor or any(not _same_json(getattr(tensor,k),v) for k,v in saved.items()):
            return False
    for op,saved in zip(program.operators,structure['operators']):
        expected=FullOperator if saved['type']=='full' else Operator
        if type(op) is not expected or any(not _same_json(getattr(op,k),v) for k,v in saved.items() if k!='type'):
            return False
    return True


def compile_workload(adf, workload, mapping, *, bindings=None, implementations=None, registry=None):
    from arch_explorer.adf.evaluation import WorkloadSpec, MappingSpec
    if type(workload) is not WorkloadSpec or type(mapping) is not MappingSpec:
        raise TypeError('compilation requires WorkloadSpec and MappingSpec')
    from .resource_budget import budget_scope, policy_for_spec
    with budget_scope(policy_for_spec(workload)):
        return _compile_workload(adf,workload,mapping,bindings=bindings,implementations=implementations,registry=registry)


def _compile_workload(adf, workload, mapping, *, bindings=None, implementations=None, registry=None):
    from arch_explorer.adf.evaluation import MappingSpec, WorkloadSpec
    from arch_explorer.mapping.compilation import map_workload, parameters, inspect_program, _resolved_locations, placement, resolved_parameters
    from .api import workload_from_spec
    from .extensions import source_objects, source_trace, callable_identity
    from .reference_api import execute_reference
    from arch_explorer.simulator.plan import planning_execution
    _registry(registry)
    if type(workload) is not WorkloadSpec or type(mapping) is not MappingSpec:
        raise TypeError('compilation requires WorkloadSpec and MappingSpec')
    mapping = MappingSpec(mapping.template, parameters(mapping))
    source_input = finite_json(dict(adf=adf.to_dict(),workload=workload.to_dict(),mapping=mapping.to_dict()))
    if bindings is None:
        bindings = {}
    if type(bindings) is not dict or any(type(key) is not str or not key for key in bindings):
        raise ValueError('bindings must map exact compute paths to adapter bindings')
    choices = finite_json({} if implementations is None else implementations)
    if type(choices) is not dict or any(type(v) is not str or not v for v in choices.values()):
        raise ValueError('implementations requires semantic kind -> exact implementation ID')
    with source_trace() as used, planning_execution(False):
        original = workload_from_spec(workload)
        program,facts = map_workload(original,mapping)
        # Persist resolved selections, including formerly implicit unique rules.
        # A later unrelated registration must not silently select another rule.
        mapping = MappingSpec(mapping.template,resolved_parameters(mapping,facts))
        source_input['mapping'] = mapping.to_dict()
        from .resource_budget import limit
        from .reference_api import initialize_reference_inputs
        from .extensions import _record_source
        _record_source(initialize_reference_inputs)
        reference = (initialize_reference_inputs(workload) if limit('reference_mode','execute')=='inputs'
                     else execute_reference(workload))
        program.meta['inputs'] = {name:reference['tensors'][name].tolist()
                                  for name in reference['input_tensors']}
        if set(choices)-{op.op for op in program.operators}:
            raise ValueError('implementation choices reference absent mapped semantics')
        common = program.to_dict()
        common_structure=_ProgramSnapshot(program,common)
        plan = inspect_program(program,mapping,facts)
        # Placement selectors and component resolution are invariant across
        # mapped tiles of the same source operator and the same physical pair.
        # Keep a request-local copy; mutation guards below still bind the full
        # original mapping and common graph before returning a product.
        location_parameters=parameters(mapping)
        placement_cache={};placements={}
        for op in program.operators:
            origin=facts['origins'][op.name]
            pair=(placement(location_parameters,'compute',origin),placement(location_parameters,'memory',origin))
            if pair not in placement_cache:
                placement_cache[pair]=_resolved_locations(adf,location_parameters,origin)
            placements[op.name]=placement_cache[pair]
        if set(bindings)-{pair[0][0] for pair in placements.values()}:
            raise ValueError('adapter binding refers to an unused compute placement')
        components, selected, groups, group_map = {}, {}, [], {}
        semantics_by_compute={};adapters_by_compute={}
        for step in plan['steps']:
            semantics_by_compute.setdefault(step['compute'],set()).add(step['semantic'])
        for step in plan['steps']:
            compute,memories = placements[step['name']]
            path,component = compute
            # Semantic coverage is checked for all tasks assigned to this component.
            if path not in adapters_by_compute:
                adapters_by_compute[path]=bind_adapter(component,semantics_by_compute[path],bindings.get(path))
            adapter,binding = adapters_by_compute[path]
            selected[path] = binding.to_dict()
            for location,resolved in (compute,*memories):
                components[location] = resolved
            key = canonical(dict(compute=path,memory=step['memory'],adapter=binding.to_dict()))
            if key not in group_map:
                group = dict(id='group-'+str(len(groups)+1),tasks=[],compute=path,memory=step['memory'],
                             adapter=binding.to_dict())
                group_map[key] = (group,adapter,component)
                groups.append(group)
            group = group_map[key][0]
            group['tasks'].append(step['name'])
            step.update(group=group['id'],adapter=binding.to_dict())
        domains = {adapter.execution_domain for _,adapter,_ in group_map.values()}
        if len(domains) != 1:
            raise NotImplementedError('task groups span execution domains without an explicit bridge')
        domain = get_domain(next(iter(domains)))
        subjects = [domain.compose]
        for group,adapter,component in group_map.values():
            group_tasks=set(group['tasks'])
            for op in program.operators:
                if op.name not in group_tasks:continue
                for n in (*op.inputs,*op.outputs):
                    layout='readonly_strided' if program.tensors[n].alias_of is not None else 'independent'
                    if layout not in adapter.layouts:
                        raise NotImplementedError('adapter '+adapter.id+' does not declare '+layout+' tensor access: '+n)
            group['accesses'] = {name:finite_json(facts['accesses'][name]) for name in group['tasks']}
            context = AdapterContext(adf,program,mapping,tuple(group['tasks']),group['compute'],group['memory'],
                                     component,finite_json(group['adapter']['options']),dict(choices),
                                     finite_json(group['accesses']))
            access_before = finite_json(group['accesses'])
            product = adapter.lower(context)
            if (not _program_unchanged(program,common,common_structure)
                    or not _same_json(context.accesses,access_before) or not _same_json(group['accesses'],access_before)):
                raise ValueError('adapter lower mutated the common semantic graph or rule accesses')
            if type(product) is not AdaptedGroup or type(product.payload) is not dict or type(product.scope) is not dict:
                raise TypeError('adapter lower must return AdaptedGroup(payload, scope)')
            group.update(payload=finite_json(product.payload),scope=finite_json(product.scope),
                         definition=adapter_identity(adapter))
            subjects.append(adapter.lower)
            if adapter.restore is not None:
                subjects.append(adapter.restore)
        frozen_groups = finite_json(groups)
        with _snapshot_scope(common_structure):
            product = domain.compose(adf,program,groups)
        if (not _program_unchanged(program,common,common_structure) or not _same_json(groups,frozen_groups)
                or not _same_json(dict(adf=adf.to_dict(),workload=workload.to_dict(),mapping=mapping.to_dict()),source_input)):
            raise ValueError('domain composition mutated the public graph or adapted groups')
    if (type(product) is not TargetInput or type(product.workload) is not WorkloadSpec
            or type(product.mapping) is not MappingSpec or type(product.provider) is not str or not product.provider):
        raise TypeError('domain compose must return TargetInput(provider, WorkloadSpec, MappingSpec, scope)')
    if product.workload.kind != 'graph' or not _same_json(product.workload.params,common):
        raise ValueError('execution domain must consume the exact common mapped semantic graph')
    subjects += [*used,*source_objects(workload,original),
                 *source_objects(WorkloadSpec('graph',common),program),*reference.get('_source_objects',())]
    for resolved in components.values():
        subjects.append(resolved.implementation.factory)
        if resolved.implementation.validate is not None:
            subjects.append(resolved.implementation.validate)
    return seal(dict(schema=SCHEMA,
        source=source_input,
        bindings=selected,implementations=choices,program=common,plan=plan,
        components=[dict(path=path,identity=resolved.identity()) for path,resolved in sorted(components.items())],
        groups=groups,domain=dict(id=domain.id,compose=callable_identity(domain.compose)),
        lowered=dict(provider=product.provider,workload=product.workload.to_dict(),
                     mapping=product.mapping.to_dict(),scope=finite_json(product.scope)),
        sources=_sources(subjects,[value for value in used.identities if value.get('kind')=='implementation'])))


def validate_compilation(document, *, registry=None):
    from arch_explorer.adf.evaluation import MappingSpec, WorkloadSpec
    from arch_explorer.adf.model import from_dict
    fields = {'schema','source','bindings','implementations','program','plan','components',
              'groups','domain','lowered','sources','binding'}
    check(document,fields,SCHEMA)
    source = document['source']
    rebuilt = compile_workload(from_dict(source['adf']),WorkloadSpec(**source['workload']),
        MappingSpec(**source['mapping']),bindings=document['bindings'],
        implementations=document['implementations'],registry=registry)
    if canonical(document) != canonical(rebuilt):
        raise ValueError('compilation source, mapping, adapter, options, component or product differs')
    return rebuilt


def compiler_targets():
    """Compatibility discovery only; no second target selection implementation."""
    return task_adapters()


def register_compiler_target(definition):
    raise NotImplementedError('CompilerTarget v1 is retired; register TaskAdapter and a component.adapters relationship')


def migrate_compilation_v1(document):
    from .compilation_v1 import migrate
    return migrate(document)


def migrate_compilation_v2(document):
    """Explicit new compilation from old public inputs, never relabel a result."""
    from arch_explorer.adf.evaluation import MappingSpec, WorkloadSpec
    from arch_explorer.adf.model import from_dict
    check(document, {'schema','source','bindings','implementations','program','plan','components',
                     'groups','domain','lowered','sources','binding'}, 'arch-explorer.compilation.v2')
    source = document['source']
    old = source['mapping']
    if old.get('template') != 'compile_graph' or set(old.get('params', {})) != {'compute', 'memory', 'tiles'}:
        raise ValueError('unsupported historical v2 mapping')
    p = old['params']
    mapping = MappingSpec('compile_graph_v2', dict(compute=p['compute'], memory=p['memory'], rules={
        'matmul': dict(id='matmul.tiled', version='1', options=dict(tiles=p['tiles']))}))
    return compile_workload(from_dict(source['adf']), WorkloadSpec(**source['workload']), mapping,
                            bindings=document['bindings'], implementations=document['implementations'])
