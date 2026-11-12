"""Hand dynamic addresses, integer service, masks and actual shared-root fit."""
from collections import Counter
from copy import deepcopy
import json,math
import pytest
from clip_contract_cases import REJECT,clip_api,close,frozen_inputs
from clip_expectations import first_argmax,lookup_addresses,eot_addresses,lookup_traffic,eot_traffic
from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import independent_layouts
from view_storage_runtime_cases import audit_storage,observe_memory

def compile_hand(w,graph,dtype,mode='events'):
    from arch_explorer.adf import WorkloadSpec,MappingSpec,from_dict
    doc=frozen_inputs();params=deepcopy(doc['cases']['regular-2-'+dtype]['candidates'][0]['materialized']['params'])
    paths=sorted(set(params['compute'].values()));params.update(compute={op.name:paths[i%len(paths)] for i,op in enumerate(graph.operators)},rules={},operator_rules={})
    return from_dict(deepcopy(doc['hardware'][mode])),WorkloadSpec('graph',graph.to_dict()),MappingSpec('compile_graph_v3',params)

def row(layout,name,direction,index):
    unit=4 if layout.root.dtype in ('int32','fp32') else 2
    offset=layout.address(index)
    return dict(direction=direction,tensor=name,storage=layout.root.name,logical_index=list(index),element_offset=offset,byte_offset=offset*unit,bytes=unit)

def same_rows(actual,wanted):
    assert Counter(json.dumps(r,sort_keys=True) for r in actual)==Counter(json.dumps(r,sort_keys=True) for r in wanted)

@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('policy',['materialized','views'])
def test_hand_lookup_argmax_select_two_outputs_and_actual_capacity(clip_api,monkeypatch,tmp_path,dtype,policy):
    from arch_explorer.adf import from_dict
    w,s=clip_api;ids=[[0,4,4],[3,0,4]];table=[[i+1.,2*i+1.,-float(i)] for i in range(5)]
    g=w.Workload('hand-CLIP-dynamic',dtype=dtype)
    for n,shape,kind,dt in [('I',(2,3),'activation','int32'),('W',(5,3),'weight',dtype),('L',(2,3,3),'activation',dtype),
        ('E',(2,),'activation','int32'),('S',(2,3),'activation',dtype),('T',(3,2),'activation',dtype)]:
        if n!='T' or policy=='materialized':g.tensor(n,shape,dtype=dt,kind=kind)
    g.meta['inputs']={'I':ids,'W':table};g.add('lookup','lookup',['I','W'],['L'],shape=[2,3,3])
    g.add('argmax','argmax',['I'],['E'],shape=[2]);g.add('select','select',['L','E'],['S'],shape=[2,3])
    if policy=='views':w.readonly_view(g,'T','S',transform='transpose',shape=(3,2),axes=(1,0))
    else:g.add('transpose','transpose',['S'],['T'],shape=[3,2],axes=[1,0])
    g.meta['outputs']=['S','T'];adf,spec,mapping=compile_hand(w,g,dtype)
    with no_backend(monkeypatch):
        c=w.compile_workload(adf,spec,mapping);program=c['program'];layouts=independent_layouts(program)
        mapped=w.workload_from_spec(type(spec)('graph',program));steps={v['name']:v for v in c['plan']['steps']}
        ref=w.execute_reference(spec);indices=[first_argmax(r) for r in ids];selected=[table[ids[b][i]] for b,i in enumerate(indices)]
        close(ref['tensors']['E'],indices,'int32');close(ref['outputs']['S'],selected,dtype,exact=True)
        expected=[]
        for b,tokens in enumerate(ids):
            for j,index in enumerate(tokens):
                expected.append(row(layouts['I'],'I','read',(b,j)))
                for k in range(3):
                    expected.extend([row(layouts['W'],'W','read',(index,k)),row(layouts['L'],'L','write',(b,j,k))])
        same_rows(w.logical_accesses(mapped,'lookup'),expected)
        with pytest.raises(REJECT):w.logical_accesses(mapped,'select')
        with pytest.raises(REJECT):w.logical_accesses(mapped,'select',values={'E':[0,3]})
        access=w.logical_accesses(mapped,'select',values={'E':ref['tensors']['E']})
        wanted=[]
        for b,index in enumerate(indices):
            wanted.append(row(layouts['E'],'E','read',(b,)))
            for k in range(3):wanted.extend([row(layouts['L'],'L','read',(b,index,k)),row(layouts['S'],'S','write',(b,k))])
        same_rows(access,wanted)
        with pytest.raises(ValueError):w.logical_accesses(mapped,'lookup',limit=len(expected)-1)
        unit=4 if dtype=='fp32' else 2
        for op,reads,writes,issue in [('lookup',24+18*unit,18*unit,30),('argmax',24,8,8),('select',8+6*unit,6*unit,10)]:
            a=steps[op]['access'];assert a['read_bytes']==reads and a['write_bytes']==writes and a['operations']==issue
            assert a['conversion_instructions']==0
        assert layouts['W'].root.bytes==15*unit
        assert (layouts['S'].root.name==layouts['T'].root.name)==(policy=='views')
        request=s.prepare_compiled(c,output_dir=tmp_path/'actual',options={'trace':True})
    with observe_memory(monkeypatch) as observed:result=s.evaluate_request(request)
    physical=audit_storage(c,result,observed);peak=next(iter(physical['peaks'].values()));largest=max(r['bytes'] for r in physical['intervals'])
    for n in ('S','T'):
        r=next(r for r in physical['intervals'] if r['storage']==layouts[n].root.name)
        assert r['returned'] and r['end']==result['evaluation']['latency_ns']
    capacities=[]
    for size in dict.fromkeys((peak,peak-1,largest-1)):
        document=deepcopy(adf.to_dict());memory=mapping.params['memory'].split('/')[-1]
        def change(node):
            if isinstance(node,dict):
                if node.get('name')==memory:node['params']['capacity']['value']=size
                for value in node.values():change(value)
            elif isinstance(node,list):
                for value in node:change(value)
        change(document);device=from_dict(document)
        if size<largest:
            with no_backend(monkeypatch),pytest.raises(REJECT):w.compile_workload(device,spec,mapping)
            phase='compile'
        else:
            with no_backend(monkeypatch):cc=w.compile_workload(device,spec,mapping);rq=s.prepare_compiled(cc,output_dir=tmp_path/f'capacity-{size}',options={'trace':True})
            if size==peak:assert s.evaluate_request(rq)['evaluation']['latency_ns']>0
            else:
                with pytest.raises((ValueError,RuntimeError)):s.evaluate_request(rq)
            phase='execute'
        capacities.append(dict(bytes=size,phase=phase,supported=size==peak))
    (tmp_path/'DYNAMIC-CAPACITY.json').write_text(json.dumps(dict(dtype=dtype,policy=policy,actual=physical,capacity=capacities),indent=2)+'\n')

@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_mask_payload_is_finite_and_each_score_reads_one_integer_mask(clip_api,monkeypatch,tmp_path,dtype):
    w,s=clip_api;g=w.Workload('hand-CLIP-mask',dtype=dtype)
    for n,shape,dt in [('A',(2,3,3),dtype),('M',(3,3),'int32'),('P',(2,3,3),dtype),('Q',(2,3,3),dtype)]:g.tensor(n,shape,dtype=dt)
    g.meta['inputs']={'A':[[[0.,0.,1000.],[0.,0.,1000.],[0.,0.,0.]]]*2}
    g.add('mask','causal_mask',[],['M'],shape=[3,3]);g.add('apply','apply_mask',['A','M'],['P'],shape=[2,3,3])
    g.add('softmax','masked_softmax',['P','M'],['Q'],shape=[2,3,3]);g.meta['outputs']=['Q']
    adf,spec,mapping=compile_hand(w,g,dtype,mode='roofline')
    with no_backend(monkeypatch):
        c=w.compile_workload(adf,spec,mapping);ref=w.execute_reference(spec);close(ref['tensors']['M'],[[0,1,1],[0,0,1],[0,0,0]],'int32')
        expected=[[[1.,0.,0.],[.5,.5,0.],[1/3,1/3,1/3]]]*2
        close(ref['outputs']['Q'],expected,dtype)
        assert all(math.isfinite(v) for v in ref['tensors']['P'].reshape(-1).tolist())
        steps={v['name']:v for v in c['plan']['steps']};unit=4 if dtype=='fp32' else 2
        assert steps['mask']['access']['read_bytes']==0 and steps['mask']['access']['write_bytes']==36
        assert steps['mask']['access']['operations']==9 and steps['mask']['access']['conversion_instructions']==0
        for name in ('apply','softmax'):
            access=steps[name]['access'];assert access['read_bytes']==18*(unit+4) and access['write_bytes']==18*unit
            assert access['conversion_instructions']==(0 if dtype=='fp32' else 36)
        mapped=w.workload_from_spec(type(spec)('graph',c['program']));layouts=independent_layouts(c['program'])
        records=w.logical_accesses(mapped,'softmax')
        assert sum(r['bytes'] for r in records if r['direction']=='read' and r['tensor']=='M')==18*4
        rq=s.prepare_compiled(c,output_dir=tmp_path/'actual',options={'trace':True})
    with observe_memory(monkeypatch) as observed:result=s.evaluate_request(rq)
    physical=audit_storage(c,result,observed)
    (tmp_path/'MASK-SERVICE.json').write_text(json.dumps(dict(actual=physical,latency_ns=result['evaluation']['latency_ns']),indent=2)+'\n')

@pytest.mark.parametrize('failure',['flag','all-blocked','zero-L2','overflow-L2','overflow-exp'])
def test_operator_owned_mask_L2_and_exp_rejections_have_a_passing_reference_control(clip_api,monkeypatch,failure):
    from arch_explorer.adf import WorkloadSpec
    w,s=clip_api
    if failure in ('flag','all-blocked'):
        g=w.Workload('mask-refusal',dtype='fp32');g.tensor('A',(1,2));g.tensor('M',(1,2),dtype='int32');g.tensor('Q',(1,2))
        g.meta['inputs']={'A':[[0.,0.]],'M':[[0,1]]};g.add('softmax','masked_softmax',['A','M'],['Q'],shape=[1,2])
        bad=[[0,2]] if failure=='flag' else [[1,1]]
        with no_backend(monkeypatch):
            close(w.execute_reference(WorkloadSpec('graph',g.to_dict()))['outputs']['Q'],[[1.,0.]],'fp32',exact=True)
            g.meta['inputs']['M']=bad
            with pytest.raises((ValueError,FloatingPointError)):w.execute_reference(WorkloadSpec('graph',g.to_dict()))
    else:
        g=w.Workload('floating-refusal',dtype='fp32');shape=(1,) if failure=='overflow-exp' else (1,2)
        g.tensor('X',shape);g.tensor('Y',shape);g.meta['inputs']={'X':[0.] if failure=='overflow-exp' else [[3.,4.]]}
        g.add('primitive','exp' if failure=='overflow-exp' else 'l2_norm',['X'],['Y'],shape=list(shape))
        with no_backend(monkeypatch):
            assert w.execute_reference(WorkloadSpec('graph',g.to_dict()))
            g.meta['inputs']['X']=[1000.] if failure=='overflow-exp' else [[0.,0.]] if failure=='zero-L2' else [[1e30,1e30]]
            with pytest.raises((ValueError,FloatingPointError)):w.execute_reference(WorkloadSpec('graph',g.to_dict()))
