"""Frozen SIM46 input consumers and independent actual-storage inspection."""
from copy import deepcopy
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from precision_transformer_expectations import lifetime_peak,tensor_bytes

INPUTS=Path(__file__).resolve().parents[1]/'fixtures/multilayer_composition/INPUTS.json'
INPUT_SHA='9e58072da7759672f202af528e387682a6044b74fc9bf311c46aeff3d567e31f'
REJECT=(ValueError,TypeError,KeyError,NotImplementedError)


@lru_cache(maxsize=1)
def frozen_inputs():
    raw=INPUTS.read_bytes();assert hashlib.sha256(raw).hexdigest()==INPUT_SHA
    return json.loads(raw)


@pytest.fixture
def composition_api():
    import numpy
    from arch_explorer.components import Kernel
    w=importlib.import_module('arch_explorer.workload.api')
    s=importlib.import_module('arch_explorer.simulator.api')
    for name in ('compose_workloads','qualified_name','transformer_encoder',
                 'transformer_block_structure','transformer_encoder_input_shapes','migrate_mapping_v2'):
        assert callable(getattr(w,name)),name
    return w,s


def scene(key,index=0):
    from arch_explorer.adf import from_dict,WorkloadSpec,MappingSpec
    row=deepcopy(frozen_inputs()[key])
    return from_dict(row['adf']),WorkloadSpec(**row['workload']),MappingSpec(**row['mappings'][index])


def basic_mapping():
    from arch_explorer.adf import MappingSpec
    return MappingSpec('compile_graph_v3',dict(compute='board/core/sm',memory='board/sram',rules={},operator_rules={}))


def scale_graph(w,factor=2.,*,dtype='fp32',shape=(1,4)):
    graph=w.Workload('repeated_leaf',dtype=dtype)
    x=graph.tensor('x',shape,dtype=dtype);y=graph.tensor('y',shape,dtype=dtype)
    graph.add('apply','scale',[x],[y],shape=list(shape),factor=factor)
    return graph


def fork_spec(w,*,dtype='fp32',retain=False,registered=False):
    """The hand-computed fork in composition_expectations, using public ports."""
    from arch_explorer.adf import WorkloadSpec
    if registered:
        import val57_composition_extension as extension
        extension.register()
        def leaf(factor):return WorkloadSpec(extension.KIND,dict(shape=[1,4],factor=factor))
    else:
        def leaf(factor):return scale_graph(w,factor,dtype=dtype)
    merge=w.Workload('merge',dtype=dtype)
    a=merge.tensor('a',(1,4),dtype=dtype);b=merge.tensor('b',(1,4),dtype=dtype)
    y=merge.tensor('y',(1,4),dtype=dtype)
    merge.add('sum','add',[a,b],[y],shape=[1,4])
    instances=[dict(id=name,workload=leaf(factor),inputs={'x':'x'},outputs={'out':'y'})
               for name,factor in (('make',2.),('left',3.),('right',-1.))]
    instances.append(dict(id='join',workload=merge,inputs={'left':'a','right':'b'},outputs={'out':'y'}))
    bindings=[dict(to=dict(instance='make',port='x'),**{'from':{'input':'ROOT'}}),
        *[dict(to=dict(instance=name,port='x'),**{'from':dict(instance='make',port='out')})
          for name in ('left','right')],
        *[dict(to=dict(instance='join',port=name),**{'from':dict(instance=name,port='out')})
          for name in ('left','right')]]
    outputs={'Y':dict(instance='join',port='out')}
    if retain:outputs['SHARED']=dict(instance='make',port='out')
    return w.compose_workloads(instances=instances,inputs={'ROOT':dict(shape=[1,4],dtype=dtype,
        kind='activation',value=[[1.,2.,-1.,.5]])},bindings=bindings,outputs=outputs,name='fork-and-merge')


@contextmanager
def observe_memory(monkeypatch):
    """Pass-through observation of real allocation/free calls, not a model."""
    from arch_explorer.simulator.events.engine import ConsistentScheduler
    events=[]
    allocate=ConsistentScheduler._allocate_memory
    free=ConsistentScheduler._free
    def allocated(self,instance,time):
        if instance.task.kind=='storage' and instance.task.memory is not None:
            owner,size=instance.task.memory
            events.append(dict(action='allocate',name=instance.task.name,time=time,owner=owner,bytes=size))
        return allocate(self,instance,time)
    def freed(self,instance,time):
        before=instance.freed
        result=free(self,instance,time)
        if not before and instance.freed and instance.task.memory is not None:
            owner,size=instance.task.memory
            events.append(dict(action='free',name=instance.task.name,time=time,owner=owner,bytes=size))
        return result
    with monkeypatch.context() as patch:
        patch.setattr(ConsistentScheduler,'_allocate_memory',allocated)
        patch.setattr(ConsistentScheduler,'_free',freed)
        yield events


def storage_oracle(compilation,result,observed=None):
    """Derive lifetimes from actual edges/task times, never allocator counters."""
    declared={t['name']:t for t in compilation['program']['tensors']}
    operators=compilation['program']['operators']
    producers={n:op['name'] for op in operators for n in op['outputs']}
    readers={name:[op['name'] for op in operators if name in op['inputs']] for name in declared}
    evaluation=result['evaluation'];trace=evaluation['details']['trace']
    tasks={row['operator']:dict(start=row['start_ns'],end=row['end_ns'])
           for row in trace if row['kind']=='compute'}
    allocations={row['name']:row['start_ns'] for row in trace if row['kind']=='storage'}
    rows=[]
    for plan in compilation['plan']['storage']:
        name=plan['tensor'];tensor=declared[name]
        assert plan['producer']==producers.get(name)
        assert set(plan['consumers'])==set(readers[name])
        assert plan['bytes']==tensor_bytes(tensor['shape'],tensor['dtype'])
        if name in producers:
            for reader in readers[name]:assert tasks[reader]['start']>=tasks[producers[name]]['end']
        rows.append(dict(name=name,shape=tensor['shape'],dtype=tensor['dtype'],
            allocation=allocations['board:'+name],readers=readers[name],returned=plan['returned'],
            persistent=name not in producers and tensor['kind'] in ('weight','kv')))
    peak,intervals=lifetime_peak(rows,tasks,evaluation['latency_ns'])
    assert peak==evaluation['details']['stats']['memory_peak']['board/sram']
    if observed is not None:
        intervals_by_name={row['name']:row for row in intervals}
        for row in rows:
            name=row['name'];actual=[e for e in observed if e['name']=='board:'+name]
            allocations=[e for e in actual if e['action']=='allocate']
            assert len(allocations)==1
            assert allocations[0]['bytes']==tensor_bytes(row['shape'],row['dtype'])
            releases=[e['time'] for e in actual if e['action']=='free']
            if row['persistent'] or row['returned']:
                assert all(time>=evaluation['latency_ns'] for time in releases)
            else:
                assert releases==pytest.approx([intervals_by_name[name]['last_reader']])
    return peak,intervals,tasks
