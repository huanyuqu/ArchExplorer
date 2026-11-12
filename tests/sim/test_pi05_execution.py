"""Independent small semantics and real source-size timing (no checkpoint inference)."""
from copy import deepcopy
from collections import Counter
import json
import math
import subprocess
import sys
import numpy as np
import pytest
from arch_explorer.adf import from_dict, evaluate, WorkloadSpec, MappingSpec
from arch_explorer.adf.registry import MAPPING_TEMPLATES
from arch_explorer.agent_env import ExplorationEnv, replay_trace
from arch_explorer.model_specs.pi05 import expand
from arch_explorer.workload.pi05 import PiGraph, pi05_workload
from arch_explorer.workload.full_primitives import reference
from arch_explorer.simulator.events.backend import CapacityError
from tools.generate_pi05 import configuration

SMALL=dict(profile='openpi_pi05_dummy',num_steps=2,action_horizon=3,max_token_len=5)


def run(params=None, scheduler='consistent', trace=False, cfg=None, **options):
    cfg=cfg or configuration(**(params or SMALL))
    return evaluate(from_dict(cfg['seed_adf']),WorkloadSpec(**cfg['workloads'][0]),
                    MappingSpec('gsm_tiled',{}),scheduler=scheduler,trace=trace,**options)


def primitive(op, inputs, shape, dtype='fp32', **attrs):
    g=PiGraph(expand(SMALL)); names=[]
    for i,x in enumerate(inputs): names.append(g.w.tensor(str(i),x.shape,dtype='fp32'))
    g.primitive('out',op,names,shape,dtype,**attrs)
    return reference(g.w.operators[-1],inputs)


def test_mask_positions_not_causal_and_all_masked():
    camera=np.array([1,0],np.int8)
    np.testing.assert_array_equal(primitive('copy',[camera],(2,3),'int8',layout='camera mask broadcast'),[[1,1,1],[0,0,0]])
    valid=np.array([[0,1,0,1],[0,0,0,0]],dtype=np.int8)
    np.testing.assert_array_equal(primitive('positions',[valid],(2,4),'int32'),[[-1,0,0,1],[-1,-1,-1,-1]])
    np.testing.assert_array_equal(primitive('suffix_positions',[valid],(2,3),'int32'),[[2,3,4],[0,1,2]])
    expected=np.array([[[False]*4]*4,[[False]*4]*4]);expected[0,1,1]=expected[0,1,3]=expected[0,3,1]=expected[0,3,3]=True
    np.testing.assert_array_equal(primitive('prefix_mask',[valid],(2,4,4),'int8'),expected)
    mask=primitive('suffix_mask',[valid],(2,3,7),'int8')
    for batch in range(2):
        for query in range(3): np.testing.assert_array_equal(mask[batch,query],list(valid[batch])+[1,1,1])
    logits=np.arange(32,dtype=np.float32).reshape(2,4,4)
    masked=primitive('attention_mask',[logits,expected],logits.shape,heads=1,big_neg=-2.3819763e38)
    probs=primitive('softmax',[masked],masked.shape)
    np.testing.assert_allclose(probs[1],np.full((4,4),.25))


def test_rope_gqa_against_scalar_rotations_and_head_loops():
    q=np.arange(24,dtype=np.float32).reshape(1,3,2,4)/7
    pos=np.array([[0,2,5]],dtype=np.int32)
    y=primitive('rope',[q,pos],q.shape,max_wavelength=10000)
    expected=np.empty_like(q)
    for t in range(3):
        for head in range(2):
            for d in range(2):
                theta=float(pos[0,t])/(10000**(d/2))
                a,b=q[0,t,head,d],q[0,t,head,d+2]
                expected[0,t,head,d]=a*math.cos(theta)-b*math.sin(theta)
                expected[0,t,head,d+2]=b*math.cos(theta)+a*math.sin(theta)
    np.testing.assert_allclose(y,expected,atol=1e-6)
    left=y.transpose(0,2,1,3).reshape(2,3,4); right=np.arange(20,dtype=np.float32).reshape(1,4,5)/9
    g=PiGraph(expand(SMALL));a=g.w.tensor('a',left.shape);b=g.w.tensor('b',right.shape)
    g.mm('gqa',a,b,2,3,4,5,group=2,output_dtype='fp32')
    actual=reference(g.w.operators[-1],[left,right])
    expected=np.array([[[sum(float(left[h,i,k])*float(right[0,k,j]) for k in range(4)) for j in range(5)] for i in range(3)] for h in range(2)])
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)


def test_adarms_gate_time_flow_and_bfloat16():
    x=np.array([[[1.,-2.,3.,4.]]],np.float32)
    norm=primitive('rms',[x],x.shape,epsilon=1e-6)
    np.testing.assert_allclose(norm,x/math.sqrt(7.5+1e-6),rtol=1e-6)
    scale=np.array([[[.2,.1,-.2,0]]],np.float32);shift=scale+1;gate=scale-1
    affine=primitive('ada_affine',[norm,scale,shift],x.shape)
    residual=primitive('gated_add',[x,affine,gate],x.shape)
    np.testing.assert_allclose(residual,x+(x/math.sqrt(7.5+1e-6)*(1+scale)+shift)*gate,rtol=1e-6)
    times=np.array([1.,.5],np.float32)
    embed=primitive('sincos',[times],(2,4),min_period=.004,max_period=4.)
    angles=times[:,None]*np.array([2*np.pi/.004,2*np.pi/4],np.float32)
    np.testing.assert_allclose(embed,np.concatenate([np.sin(angles),np.cos(angles)],axis=-1),atol=2e-4)
    value=x.copy(); clock=np.ones(1,np.float32)
    for _ in range(10):
        value=primitive('flow',[value,np.ones_like(x)],x.shape,dt=-.1)
        clock=primitive('scale_add',[clock],(1,),value=-.1)
    np.testing.assert_allclose(value,x-1,atol=2e-6);assert clock[0]<.05
    vals=np.array([1+1/256,1+3/256,1.001],np.float32)
    np.testing.assert_array_equal(primitive('cast',[vals],(3,),'bf16'),[1.,1+1/64,1.])
    # NHWC patch lowering is a physical reorder, not a flattened image view.
    image=np.arange(48).reshape(1,4,4,3)
    patches=primitive('copy',[image],(1,4,12),layout='NHWC patches stride=patch VALID')
    expected=np.stack([image[0,y:y+2,z:z+2].reshape(-1) for y in (0,2) for z in (0,2)])[None]
    np.testing.assert_array_equal(patches,expected)


@pytest.mark.parametrize('params',[{}, {**SMALL,'iterations':2}, {**SMALL,'dtype':'float32','batch':2}])
def test_source_weights_graph_and_cache_identity(params):
    w=pi05_workload(params);s=expand(params);p=s['params'];ops={o.name:o for o in w.operators}
    assert w.iterations==1
    assert sum(t.bytes//4 for t in w.tensors.values() if t.kind=='weight' and t.dtype=='fp32')==sum(s['storage']['weight_elements'].values())
    assert len([n for n in ops if n.endswith('.stem.mm')])==3*p['iterations']
    assert len([n for n in ops if '.prefix.layer' in n and n.endswith('.attn.qk')])==s['language']['layers']*p['iterations']
    assert len([n for n in ops if '.expert.layer' in n and n.endswith('.attn.qk')])==s['language']['layers']*p['iterations']*p['num_steps']
    assert len([n for n in ops if n.endswith('.flow')])==p['iterations']*p['num_steps']
    caches=w.meta['full_model']['prefix_cache_ids'];assert len(set(n for layers in caches for pair in layers for n in pair))==2*s['language']['layers']*p['iterations']
    for i,layers in enumerate(caches):
        assert sum(w.tensors[n].bytes for pair in layers for n in pair)==s['storage']['prefix_kv_bytes']
        for pair in layers:
            for n in pair:
                readers=[o for o in w.operators if n in o.inputs and o.op=='concat']
                assert len(readers)==p['num_steps']
                assert all(w.tensors[o.outputs[0]].shape[1]==s['prefix_tokens']+p['action_horizon'] for o in readers)
        out=w.meta['full_model']['outputs'][i];assert w.tensors[out].shape==(p['batch'],p['action_horizon'],p['action_dim'])
        assert w.tensors[out].dtype=='fp32'
    assert all(w.tensors[o.outputs[0]].dtype=='fp32' for o in w.operators if o.op in ('rms','sincos','flow'))
    activation_dtype={'bfloat16':'bf16','float32':'fp32'}[params.get('dtype','bfloat16')]
    qks=[o for o in w.operators if o.name.endswith('.qk')]
    visual=[o for o in qks if '.vision' in o.name]
    language=[o for o in qks if '.vision' not in o.name]
    assert visual and language
    assert all(w.tensors[o.outputs[0]].dtype==activation_dtype for o in visual)
    assert all(w.tensors[o.outputs[0]].dtype=='fp32' for o in language)
    assert w.tensors['weights.vision.stem.weight'].dtype=='fp32'
    assert s['storage']['weight_storage_bytes'] is None


def test_capacity_lifetimes_scheduler_equality_and_iterations():
    cfg=configuration(**{**SMALL,'iterations':2})
    x=run(cfg=cfg,trace=True);y=run(cfg=cfg,scheduler='conservative',trace=True)
    assert x.latency_ns==y.latency_ns
    assert [(r['task'],r['start_ns'],r['end_ns']) for r in x.details['trace']]==[(r['task'],r['start_ns'],r['end_ns']) for r in y.details['trace']]
    f=x.details['full_model'];t=f['tensors'];assert sum(r['latency_ns'] for r in f['request_timing'])==x.latency_ns
    for i,layers in enumerate(f['prefix_cache_ids']):
        output=t[f['outputs'][i]]
        for pair in layers:
            for n in pair: assert t[n]['free_ns']>=output['ready_ns']
    converted=t['weights.vision.block0.attn.q.weight.bf16']
    assert converted['allocation_ns']>0 and converted['free_ns']==x.latency_ns
    assert t[f['prefix_cache_ids'][0][0][0]]['free_ns']<t[f['prefix_cache_ids'][1][0][0]]['allocation_ns']
    def shrink(node):
        if node.get('component')=='dram':node['params']['capacity']['value']=1024
        for c in node.get('elements',[]):shrink(c)
    shrink(cfg['seed_adf'])
    with pytest.raises(CapacityError):run(cfg=cfg,check_capacity=False)


def test_support_domain_and_noise_seed():
    cfg=configuration(**SMALL);w=WorkloadSpec(**cfg['workloads'][0]);adf=from_dict(cfg['seed_adf'])
    assert MAPPING_TEMPLATES.get('gsm_tiled').applicable(adf,w)
    for name in ('dmc_partition','pim_offload','multi_device'):
        assert not MAPPING_TEMPLATES.get(name).applicable(adf,w)
    def half(node):
        if node.get('component')=='systolic_array':node['params']['precision']['value']='fp16'
        for c in node.get('elements',[]):half(c)
    half(cfg['seed_adf']);assert not MAPPING_TEMPLATES.get('gsm_tiled').applicable(from_dict(cfg['seed_adf']),w)
    a=pi05_workload(SMALL);b=pi05_workload({**SMALL,'noise_seed':998})
    assert a.operators==b.operators and a.tensors==b.tensors


@pytest.mark.parametrize('mode',['none','mapping','parameters_and_mapping'])
def test_real_env_modes_two_candidates_restore_submit_replay(mode,tmp_path):
    cfg=configuration(**SMALL);cfg['trace_path']=str(tmp_path/'trace.jsonl');cfg['max_evaluations']=2
    env=ExplorationEnv(cfg)
    assert 'error' not in env.checkpoint(name='original')
    assert 'error' not in env.set_mappings(mappings=[dict(template='gsm_tiled',params=dict(tile_m=64,tile_n=64,rows_per_cta=4))])
    first=env.evaluate(optimize=mode);assert 'error' not in first,str(first)[:400]
    assert 'error' not in env.restore(name='original')
    assert 'error' not in env.set_mappings(mappings=[dict(template='gsm_tiled',params=dict(tile_m=128,tile_n=128,rows_per_cta=4))])
    second=env.evaluate(optimize=mode);assert 'error' not in second,str(second)[:400]
    assert env.used==2 and len(env.attempts)==2
    assert 'error' in env.evaluate(optimize=mode)
    assert env.used==2
    winner=tmp_path/'winner.json';assert 'error' not in env.submit(path=str(winner))
    assert replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines()).get_state()==env.get_state()
    for args in [('evaluate',str(winner),'--output',str(tmp_path/'winner-replay.json')),('env','replay',str(env.trace_path),'--output',str(tmp_path/'replayed.json'))]:
        process=subprocess.run([sys.executable,'-m','arch_explorer',*args],capture_output=True,text=True)
        assert process.returncode==0,process.stderr[:500]

    assert json.loads(winner.read_text())['best']==json.loads((tmp_path/'winner-replay.json').read_text())['best']
