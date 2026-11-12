"""VAL55 consumers of the frozen SIM44 interface; collection is not acceptance.

All new-API imports occur during execution. No skip, fallback or xfail lets a
pre-feature baseline look like a passing implementation. Run against the lead's
sole installed candidate, from outside the checkout, after installing the fixture.
"""
from contextlib import contextmanager
from copy import deepcopy
import ctypes
import hashlib
import importlib
import json
import math
from pathlib import Path
import socket
import subprocess

import pytest

from operator_attention_expectations import attention, f32
from test_compilation_execution_contract import independently_sweep_storage


REJECT = (ValueError, TypeError, KeyError, NotImplementedError)
INPUTS = Path(__file__).resolve().parents[1]/'fixtures/operator_attention/INPUTS-v3.json'


@pytest.fixture
def api():
    import numpy  # Ordinary CPU numerical dependency; not a simulator warm-up.
    from arch_explorer.components import Kernel  # noqa: F401
    w = importlib.import_module('arch_explorer.workload.api')
    s = importlib.import_module('arch_explorer.simulator.api')
    for name in ('MappingRuleBinding','MappingExpansion','OperatorMapping',
                 'register_operator_mapping','operator_mappings','migrate_compilation_v2'):
        assert callable(getattr(w,name)), name
    return w,s


@contextmanager
def no_backend(monkeypatch):
    """Guard backend mechanisms, leaving public readonly dispatch callable."""
    from arch_explorer.simulator.events import engine, backend
    from arch_explorer import search
    def forbidden(*args,**kwargs):
        raise AssertionError('VAL55 forbidden execution during preparation/replay')
    with monkeypatch.context() as patch:
        for name in ('Popen','run','check_call','check_output'):
            patch.setattr(subprocess,name,forbidden)
        for name in ('CDLL','PyDLL'):
            patch.setattr(ctypes,name,forbidden)
        patch.setattr(socket,'socket',forbidden)
        patch.setattr(engine,'simulate',forbidden)
        patch.setattr(engine.ConsistentScheduler,'run',forbidden)
        if hasattr(backend,'simulate'):
            patch.setattr(backend,'simulate',forbidden)
        patch.setattr(search,'run_search',forbidden)
        yield


def scene(mode='events',index=0):
    from arch_explorer.adf import from_dict,WorkloadSpec,MappingSpec
    assert hashlib.sha256(INPUTS.read_bytes()).hexdigest() == '73ae4bab83b768c945bac34ceed97bc1e733c85d684039b7f99734b06df0cae0'
    row = json.loads(INPUTS.read_text())[mode]
    return (from_dict(row['adf']),WorkloadSpec(**deepcopy(row['workload'])),
            MappingSpec(**deepcopy(row['mappings'][index])))


def node_spec(kind,shapes,output,attrs,values):
    from arch_explorer.adf import WorkloadSpec
    inputs = ['x'+str(i) for i in range(len(shapes))]
    tensors = {name:dict(name=name,shape=list(shape),dtype='fp32',kind='activation',
                         alias_of=None,strides=None) for name,shape in zip(inputs,shapes)}
    tensors['y'] = dict(name='y',shape=list(output),dtype='fp32',kind='activation',alias_of=None,strides=None)
    graph = dict(schema='arch-explorer.workload-graph.v1',name='val55-'+kind,dtype='fp32',iterations=1,
        meta=dict(inputs=dict(zip(inputs,deepcopy(values)))),tensors=list(tensors.values()),
        operators=[dict(name='primitive',op=kind,inputs=inputs,outputs=['y'],
                        attrs=dict(shape=list(output),**attrs),label='VAL55',type='operator')])
    return WorkloadSpec('graph',graph)


def primitive_scene(spec,mode='roofline'):
    from arch_explorer.adf import MappingSpec
    adf,_,mapping = scene(mode)
    return adf,spec,MappingSpec('compile_graph_v2',dict(mapping.params,rules={}))


def flat(values):
    return sum((flat(v) for v in values),[]) if isinstance(values,(list,tuple)) else [values]


def check_values(actual,expected):
    def shape(value):
        if not isinstance(value,(tuple,list)):
            return ()
        assert value and all(shape(v)==shape(value[0]) for v in value)
        return (len(value),)+shape(value[0])
    values = actual.tolist() if hasattr(actual,'tolist') else actual
    assert shape(values)==shape(expected)
    assert len(flat(values)) == len(flat(expected))
    assert flat(values) == pytest.approx(flat(expected),rel=2e-6,abs=3e-7)


def files(directory):
    return {str(p.relative_to(directory)):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
            for p in directory.rglob('*') if p.is_file()}


def test_backend_guard_really_intercepts_and_restores_harmless_sentinels(api,monkeypatch):
    from arch_explorer.simulator.events import engine
    from arch_explorer import search
    owners = [(engine,'simulate'),(engine.ConsistentScheduler,'run'),
              (subprocess,'Popen'),(ctypes,'CDLL'),(socket,'socket'),(search,'run_search')]
    seen = []
    def harmless(*args,**kwargs):
        seen.append('called')
        return 'sentinel'
    for owner,name in owners:
        monkeypatch.setattr(owner,name,harmless)
        assert getattr(owner,name)()=='sentinel'
    size = len(seen)
    with no_backend(monkeypatch):
        for owner,name in owners:
            with pytest.raises(AssertionError,match='VAL55 forbidden execution'):
                getattr(owner,name)()
    assert len(seen)==size
    for owner,name in owners:
        assert getattr(owner,name) is harmless


@pytest.mark.parametrize('mode,index',[('events',0),('events',1),('roofline',0),('roofline',1)])
def test_original_and_mapped_attention_match_independent_scalar_reference(api,monkeypatch,mode,index,tmp_path):
    from arch_explorer.adf import WorkloadSpec
    w,s = api
    adf,spec,mapping = scene(mode,index)
    inputs = spec.params['meta']['inputs']
    expected = attention(inputs['Q'][0],inputs['K'][0],inputs['V'][0])
    with no_backend(monkeypatch):
        original = w.execute_reference(spec)
        compilation = w.compile_workload(adf,spec,mapping)
        assert compilation['schema'] == 'arch-explorer.compilation.v3'
        assert compilation['source']['workload'] == spec.to_dict()
        saved_mapping = compilation['source']['mapping']
        assert saved_mapping['template']==mapping.template
        assert saved_mapping['params']['compute']==mapping.params['compute']
        assert saved_mapping['params']['memory']==mapping.params['memory']
        assert saved_mapping['params']['rules']['matmul']==mapping.params['rules']['matmul']
        assert set(saved_mapping['params']['rules'])=={op['op'] for op in spec.params['operators']}
        mapped = w.execute_reference(WorkloadSpec('graph',compilation['program']),inputs=inputs)
        request = s.prepare_compiled(compilation,output_dir=tmp_path/'prepared',options={'trace':True})
        assert request['schema'] == 'arch-explorer.compiled-request.v3'
    expected_tensors = dict(KT=expected['kt'],scores=expected['scores'],scaled=expected['scaled'],
        row_max=expected['maxima'],shifted=expected['shifted'],exponentials=expected['exponentials'],
        row_sum=expected['sums'],probabilities=expected['normalized'],attention=expected['output'])
    for name,value in expected_tensors.items():
        check_values(original['tensors'][name],[value])
        check_values(mapped['tensors'][name],[value])
    assert original['outputs']['attention'].shape == mapped['outputs']['attention'].shape == (1,3,2)


PRIMITIVES = [
    ('add',[(2,3),(2,3)],(2,3),{},[[[1,2,3],[4,5,6]],[[.5,1,1.5],[2,2.5,3]]],[[1.5,3,4.5],[6,7.5,9]]),
    ('sub',[(2,3),(2,3)],(2,3),{},[[[1,2,3],[4,5,6]],[[.5,1,1.5],[2,2.5,3]]],[[.5,1,1.5],[2,2.5,3]]),
    ('mul',[(2,3),(2,3)],(2,3),{},[[[1,2,3],[4,5,6]],[[2,2,2],[.5,.5,.5]]],[[2,4,6],[2,2.5,3]]),
    ('div',[(2,3),(2,3)],(2,3),{},[[[1,2,3],[4,5,6]],[[2,2,2],[2,2,2]]],[[.5,1,1.5],[2,2.5,3]]),
    ('scale',[(2,3)],(2,3),{'factor':.5},[[[1,2,3],[4,5,6]]],[[.5,1,1.5],[2,2.5,3]]),
    ('exp',[(2,3)],(2,3),{},[[[0,1,-1],[0,2,-2]]],[[1,f32(math.e),f32(math.exp(-1))],[1,f32(math.exp(2)),f32(math.exp(-2))]]),
    ('transpose',[(2,3)],(3,2),{'axes':[1,0]},[[[1,2,3],[4,5,6]]],[[1,4],[2,5],[3,6]]),
    ('broadcast',[(2,1)],(2,3),{},[[[1],[2]]],[[1,1,1],[2,2,2]]),
    ('reduce_sum',[(2,3)],(2,1),{'axis':-1,'keepdims':True},[[[1,2,3],[4,5,6]]],[[6],[15]]),
    ('reduce_max',[(2,3)],(2,),{'axis':1,'keepdims':False},[[[1,9,3],[4,5,6]]],[9,6]),
    ('exp',[(3,)],(3,),{},[[-1,0,1]],[f32(math.exp(-1)),1,f32(math.e)]),
    ('mul',[(1,2,3,1),(1,2,3,1)],(1,2,3,1),{},
        [[[[[1],[2],[3]],[[4],[5],[6]]]],[[[[2],[2],[2]],[[.5],[.5],[.5]]]]],
        [[[[2],[4],[6]],[[2],[2.5],[3]]]]),
    ('broadcast',[(3,)],(2,1,3),{},[[1,2,3]],[[[1,2,3]],[[1,2,3]]]),
    ('transpose',[(1,2,3)],(3,1,2),{'axes':[2,0,1]},[[[[1,2,3],[4,5,6]]]],[[[1,4]],[[2,5]],[[3,6]]]),
]


@pytest.mark.parametrize('kind,shapes,output,attrs,values,expected',PRIMITIVES,ids=[r[0] for r in PRIMITIVES])
def test_materialized_primitive_raw_mapped_semantics_and_roofline_consumption(
        api,monkeypatch,tmp_path,kind,shapes,output,attrs,values,expected):
    from arch_explorer.adf import WorkloadSpec
    w,s = api
    adf,spec,mapping = primitive_scene(node_spec(kind,shapes,output,attrs,values))
    with no_backend(monkeypatch):
        raw = w.execute_reference(spec)
        compilation = w.compile_workload(adf,spec,mapping)
        mapped = w.execute_reference(WorkloadSpec('graph',compilation['program']),inputs=spec.params['meta']['inputs'])
        request = s.prepare_compiled(compilation,output_dir=tmp_path/kind,options={'trace':True})
    check_values(raw['outputs']['y'],expected)
    check_values(mapped['outputs']['y'],expected)
    assert tuple(mapped['outputs']['y'].shape) == output
    assert len(compilation['plan']['steps'])==1
    step = compilation['plan']['steps'][0]
    demand = step['access']
    elements = math.prod(output)
    expected_ops = 0 if kind in ('transpose','broadcast') else (2*elements if kind=='div' else
        4 if kind in ('reduce_sum','reduce_max') else elements)
    assert demand['boundary']=='component_io' and demand['operations']==expected_ops
    assert demand['write_bytes']==4*elements
    expected_reads = 4*sum(math.prod(shape) for shape in shapes)
    if kind=='broadcast':
        assert demand['read_bytes']>=expected_reads
    else:
        assert demand['read_bytes']==expected_reads
    result = s.evaluate_request(request)
    assert result['schema'] == 'arch-explorer.compiled-result.v3'
    assert math.isfinite(result['evaluation']['latency_ns']) and result['evaluation']['latency_ns'] > 0
    # Rates128op/ns and64B/ns are the frozen input's existing teaching hardware.
    # This asserts abstract component_io ownership, not device instruction timing.
    expected_ns = max(expected_ops/128,(demand['read_bytes']+4*elements)/64)
    assert result['evaluation']['latency_ns']==pytest.approx(expected_ns,rel=1e-9,abs=1e-9)
    # A zero-arithmetic materialization still has full output storage and service.
    if kind in ('transpose','broadcast'):
        assert sum(t['bytes'] for t in compilation['plan']['storage'] if t['tensor']=='y') == 4*math.prod(output)


@pytest.mark.parametrize('kind,shapes,output,attrs,values,expected',PRIMITIVES,ids=[r[0] for r in PRIMITIVES])
def test_events_kernel_and_authoritative_access_retain_logical_simt_sfu_demand(
        api,monkeypatch,kind,shapes,output,attrs,values,expected):
    w,_ = api
    adf,spec,mapping = primitive_scene(node_spec(kind,shapes,output,attrs,values),'events')
    with no_backend(monkeypatch):
        compilation = w.compile_workload(adf,spec,mapping)
    assert len(compilation['plan']['steps'])==1
    step = compilation['plan']['steps'][0]
    task = compilation['lowered']['mapping']['params']['tasks'][step['name']]
    assert task['access']==step['access']
    assert set(task)=={'compute','memory','adapter','work','traffic','access'}
    kernel = task['work']
    assert kernel['op']=='vector'
    rows,cols = kernel['shape']
    simt = rows*cols*kernel['simt']+rows*kernel['row_simt']
    sfu = rows*cols*kernel['sfu']+rows*kernel['row_sfu']
    count = math.prod(output)
    expected_simt = (0 if kind in ('transpose','broadcast','exp') else
                     4 if kind in ('reduce_max','reduce_sum') else count)
    expected_sfu = count if kind in ('exp','div') else 0
    assert simt==expected_simt and sfu==expected_sfu
    assert step['access']['operations']==simt+sfu


@pytest.mark.parametrize('mutation',[ 'axis','keepdims','duplicate-axis','broadcast-shape',
    'implicit-broadcast','dtype-fp16','dtype-bf16','mixed-dtype','alias','strides','shape',
    'scale-bool','scale-infinite','rank1-scalar','rank5','zero-dimension'])
def test_extended_precision_and_invalid_shape_layout_or_axis_before_prepare(api,monkeypatch,tmp_path,mutation):
    from arch_explorer.adf import WorkloadSpec
    w,s = api
    spec = node_spec('reduce_sum',[(2,3)],(2,1),{'axis':-1,'keepdims':True},[[[1,2,3],[4,5,6]]])
    raw = spec.params
    tensors = {t['name']:t for t in raw['tensors']}
    if mutation=='axis': raw['operators'][0]['attrs']['axis']=0
    elif mutation=='keepdims': raw['operators'][0]['attrs']['keepdims']=1
    elif mutation=='duplicate-axis':
        raw['operators'][0].update(op='transpose',attrs={'shape':[2,1],'axes':[0,0]})
    elif mutation=='broadcast-shape':
        raw['operators'][0].update(op='broadcast',attrs={'shape':[2,1]})
    elif mutation=='implicit-broadcast':
        spec = node_spec('div',[(2,3),(2,1)],(2,3),{},[[[1,2,3],[4,5,6]],[[1],[2]]])
    elif mutation.startswith('dtype-'):
        raw['dtype']=mutation[6:]
        for t in raw['tensors']: t['dtype']=raw['dtype']
    elif mutation=='mixed-dtype': tensors['y']['dtype']='bf16'
    elif mutation=='alias': tensors['y']['alias_of']='x0'
    elif mutation=='strides': tensors['x0'].update(alias_of='y',strides=[3,1])
    elif mutation=='shape': raw['operators'][0]['attrs']['shape']=[1,2]
    elif mutation.startswith('scale-'):
        if mutation=='scale-infinite':
            with no_backend(monkeypatch),pytest.raises(REJECT,match='(?i)finite|JSON|float'):
                node_spec('scale',[(2,3)],(2,3),{'factor':float('inf')},[[[1,2,3],[4,5,6]]])
            assert not (tmp_path/'must-not-exist').exists()
            return
        spec = node_spec('scale',[(2,3)],(2,3),{'factor':True},[[[1,2,3],[4,5,6]]])
    elif mutation=='zero-dimension': tensors['x0']['shape'][0]=0
    elif mutation=='rank5':
        value = 1.0
        for _ in range(5): value=[value]
        spec = node_spec('exp',[(1,1,1,1,1)],(1,1,1,1,1),{},[value])
    else: spec = node_spec('reduce_sum',[(3,)],(1,),{'axis':-1,'keepdims':False},[[1,2,3]])
    adf,spec,mapping = primitive_scene(spec)
    with no_backend(monkeypatch):
        if mutation.startswith('dtype-'):
            compilation=w.compile_workload(adf,spec,mapping)
            actual=w.execute_reference(WorkloadSpec('graph',compilation['program']),inputs=spec.params['meta']['inputs'])
            assert actual['outputs']['y'].tolist()==[[6.],[15.]]
            s.prepare_compiled(compilation,output_dir=tmp_path/'supported-precision')
            return
        with pytest.raises(REJECT):
            compilation = w.compile_workload(adf,spec,mapping)
            s.prepare_compiled(compilation,output_dir=tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


def test_attention_real_event_dependencies_last_readers_peak_and_readonly_replay(api,monkeypatch,tmp_path):
    w,s = api
    adf,spec,mapping = scene('events')
    with no_backend(monkeypatch):
        compilation = w.compile_workload(adf,spec,mapping)
        request = s.prepare_compiled(compilation,output_dir=tmp_path/'attention',options={'trace':True})
    result = s.evaluate_request(request)
    peak,intervals = independently_sweep_storage(compilation,result)
    assert peak == result['evaluation']['details']['stats']['memory_peak'][mapping.params['memory']]
    assert all(t['bytes']>0 for t in compilation['plan']['storage'])
    computes = {r['operator']:r for r in result['evaluation']['details']['trace'] if r['kind']=='compute'}
    for name in ('scaled','exponentials'):
        interval = next(t for t in intervals if t['tensor']==name)
        ends = [computes[r]['end_ns'] for r in interval['readers']]
        assert len(ends)>=2 and interval['last_reader']==max(ends)
        assert max(ends)>min(ends)
    before = files(tmp_path/'attention')
    with no_backend(monkeypatch):
        assert s.evaluate_request(result)==result
    assert files(tmp_path/'attention')==before
    (tmp_path/'ATTENTION-LIFETIME.json').write_text(json.dumps(dict(peak_bytes=peak,intervals=intervals),indent=2))


def test_mapping_rule_catalog_has_identities_parameters_and_no_execution(api,monkeypatch):
    w,_ = api
    with no_backend(monkeypatch):
        catalog = w.operator_mappings()
        assert catalog == w.operator_mappings()
        assert {'matmul','scale','sub','mul','div','exp','transpose','broadcast','reduce_sum','reduce_max'} <= {row['operator'] for row in catalog}
        assert all({'id','version','operator','expand','access','parameters'} <= set(row) for row in catalog)
        for kind in ('exp','reduce_sum','transpose'):
            selected = w.operator_mappings(kind)
            assert selected and all(row['operator']==kind for row in selected)


def test_attention_impossible_capacity_is_rejected_without_execution(api,monkeypatch,tmp_path):
    from arch_explorer.adf import Param
    from arch_explorer.adf.model import walk
    w,s = api
    adf,spec,mapping = scene('events')
    memory = dict(walk(adf))[mapping.params['memory']]
    previous = memory.params['capacity']
    memory.params['capacity'] = Param(1,'B',previous.source)
    with no_backend(monkeypatch):
        with pytest.raises(REJECT,match='(?i)capacity|allocation|storage'):
            compilation = w.compile_workload(adf,spec,mapping)
            s.prepare_compiled(compilation,output_dir=tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


@pytest.mark.parametrize('change',['version','unknown-rule','nonfinite-options','missing-matmul-tiles','retired-template'])
def test_invalid_mapping_bindings_do_not_fallback_or_write_outputs(api,monkeypatch,tmp_path,change):
    from arch_explorer.adf import MappingSpec
    w,s = api
    adf,spec,mapping = scene()
    params = deepcopy(mapping.params)
    rule = params['rules']['matmul']
    if change=='version': rule['version']='999'
    elif change=='unknown-rule': rule['id']='val55.no-such-rule'
    elif change=='nonfinite-options': rule['options']['tiles']['m']=float('nan')
    elif change=='missing-matmul-tiles': rule['options']={}
    if change=='retired-template':
        mapping = MappingSpec('compile_graph',dict(compute=params['compute'],memory=params['memory'],tiles={'m':2,'k':3,'n':3}))
    else: mapping = MappingSpec(mapping.template,params)
    with no_backend(monkeypatch):
        with pytest.raises(REJECT):
            compilation = w.compile_workload(adf,spec,mapping)
            s.prepare_compiled(compilation,output_dir=tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


def test_real_v2_fixture_requires_explicit_migration_and_keeps_original_identity(api,monkeypatch,tmp_path):
    from arch_explorer.adf import WorkloadSpec
    from operator_attention_expectations import matmul
    w,s = api
    path = INPUTS.with_name('V2-COMPILATION.json')
    before = path.read_bytes(),path.stat().st_mtime_ns
    assert hashlib.sha256(before[0]).hexdigest()=='d5126b69f81a09f279f5751fe327f285598e6e40f29e442c554430ff1bc5bf45'
    old = json.loads(before[0])
    assert old['schema']=='arch-explorer.compilation.v2'
    with no_backend(monkeypatch):
        with pytest.raises(REJECT):
            w.validate_compilation(old)
        migrated = w.migrate_compilation_v2(old)
        assert migrated['schema']=='arch-explorer.compilation.v3'
        assert migrated['source']['adf']==old['source']['adf']
        assert migrated['source']['workload']==old['source']['workload']
        assert migrated['source']['mapping']['template']=='compile_graph_v2'
        data = old['source']['workload']['params']['meta']['inputs']
        expected = matmul(data['A'][0],data['B'][0])
        actual = w.execute_reference(WorkloadSpec('graph',migrated['program']),inputs=data)
        check_values(actual['outputs']['C'],[expected])
        s.prepare_compiled(migrated,output_dir=tmp_path/'new-preparation')
    assert (path.read_bytes(),path.stat().st_mtime_ns)==before


@pytest.mark.parametrize('dtype',['bf16','fp32'])
def test_new_compilation_precision_gate_does_not_remove_legacy_noncompiled_graph_kernel_support(api,dtype):
    from arch_explorer.adf import WorkloadSpec
    w,_ = api
    graph = w.workload_from_spec(WorkloadSpec('operator',dict(op='matmul',m=2,k=3,n=2,dtype=dtype)))
    assert graph.dtype==dtype and all(t.dtype==dtype for t in graph.tensors.values())
    assert all(op.kernel(dtype=dtype).precision==dtype for op in graph.operators)


def test_native_matmul_compile_control_then_low_precision_attention_refuses_without_execution(api,monkeypatch):
    from adapter_boundary_cases import native_scene
    w,_ = api
    adf,_,matmul_spec,mapping = native_scene()
    _,attention_spec,_ = scene('events')
    # SIM45 may extend native FP32 materialized semantics; the current explicit
    # unsupported boundary is native low-precision storage, not attention's name.
    attention_spec.params['dtype']='fp16'
    for tensor in attention_spec.params['tensors']:tensor['dtype']='fp16'
    for operator in attention_spec.params['operators']:
        for field in ('dtype','output_dtype','result_dtype'):
            if field in operator['attrs']:operator['attrs'][field]='fp16'
    path = mapping.params['compute']
    bindings = {path:w.AdapterBinding('accelsim','1',{'threads_per_block':32})}
    with no_backend(monkeypatch):
        control = w.compile_workload(adf,matmul_spec,mapping,bindings=bindings)
        assert control['schema']=='arch-explorer.compilation.v3'
        assert control['bindings'][path]['id']=='accelsim'
        with pytest.raises(REJECT,match='(?i)fp32|precision|dtype|unsupported|native'):
            w.compile_workload(adf,attention_spec,mapping,bindings=bindings)


@pytest.mark.parametrize('case',['large','constant'])
def test_actual_attention_stable_softmax_extremes_and_constant_rows(api,monkeypatch,tmp_path,case):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.archs.attention import attention_scene
    w,s = api
    adf,spec,mapping = attention_scene(input_case=case)
    inputs = spec.params['meta']['inputs']
    batch,queries,depth = len(inputs['Q']),len(inputs['Q'][0]),len(inputs['Q'][0][0])
    keys,width = len(inputs['K'][0]),len(inputs['V'][0][0])
    assert (batch,queries,depth,keys)==(1,3,4,5)
    # Exact dyadic operands produce scaled logits1000,999,... without a rounding tie.
    # Naive uncentered FP32 exp would overflow; the real graph must center first.
    inputs['Q'] = [[[1 if c==0 and case=='large' else 0 for c in range(depth)] for _ in range(queries)]]
    inputs['K'] = [[[2000-2*j if c==0 else 0 for c in range(depth)] for j in range(keys)]]
    inputs['V'] = [[[(j+1)*(c+1)/8-.5 for c in range(width)] for j in range(keys)]]
    spec.params['name']='val55-softmax-'+case
    expected = attention(inputs['Q'][0],inputs['K'][0],inputs['V'][0])
    with no_backend(monkeypatch):
        original = w.execute_reference(spec)
        compilation = w.compile_workload(adf,spec,mapping)
        mapped = w.execute_reference(WorkloadSpec('graph',compilation['program']),inputs=inputs)
        request = s.prepare_compiled(compilation,output_dir=tmp_path/case,options={'trace':True})
    check_values(original['outputs']['attention'],[expected['output']])
    check_values(mapped['outputs']['attention'],[expected['output']])
    normalizer = next(op for op in spec.params['operators'] if op['op']=='div')
    probabilities = original['tensors'][normalizer['outputs'][0]].tolist()[0]
    assert all(math.isfinite(p) and p>=0 for row in probabilities for p in row)
    assert all(sum(row)==pytest.approx(1,abs=2e-7) for row in probabilities)
    if case=='constant':
        assert all(row==pytest.approx([1/keys]*keys,abs=2e-7) for row in probabilities)
    result = s.evaluate_request(request)
    assert result['evaluation']['latency_ns']>0
    with no_backend(monkeypatch):
        assert s.evaluate_request(result)==result
