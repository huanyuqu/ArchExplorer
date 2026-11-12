"""CPU-only grammar, causal oracles and refusal tests. No hardware fixtures.

Raw-shaped dictionaries below test individual parsers, not device provenance.
The public fit interface requires full original-reference closure instead.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import pytest

from arch_explorer.simulator.accelsim.microbundle import oracle, BUDGET
from arch_explorer.simulator.accelsim.microrecords import validate_sample, load_attempt
from arch_explorer.simulator.accelsim.microcapture import validate_addresses
from arch_explorer.simulator.accelsim.microexperiment import (_vertices, make_design, validate_design, fit_intervals, sample_quality, Journal, ref, write_new)
from arch_explorer.simulator.accelsim.microcampaign import check_handoff, require_raw_budget


def model():
    cases={}
    for family in ('dep','quad','chase'):
        for n,part in [(10,'fit'),(20,'fit'),(25,'heldout'),(30,'fit')]:
            cid=f'{family}-{n}'
            cases[cid]={'id':cid,'family':family,'steps':n,'partition':part,'function':f'sim30_{family}_{n}','grid':[1,1,1],'block':[32,1,1],'text_sha256':'c'*64,'input':{'sha256':'d'*64},'auxiliary':{'sha256':'e'*64},'expected':{'sha256':'f'*64}}
    return SimpleNamespace(sha256='a'*64,document={'cubin':{'sha256':'b'*64}},cases=cases)


def raw_sample():
    b=model();c=b.cases['dep-10'];ms=struct.unpack('<f',struct.pack('<f',0.012345))[0]
    return b,c,{'schema':'arch-explorer.sim30-host-sample.v1','registry_sha256':b.sha256,'case':c['id'],'family':c['family'],'partition':c['partition'],'steps':c['steps'],'function':c['function'],'cubin_sha256':'b'*64,'text_sha256':'c'*64,'input_sha256':'d'*64,'auxiliary_sha256':'e'*64,'expected_sha256':'f'*64,'grid':[1,1,1],'block':[32,1,1],'dynamic_shared':0,'context':'0xf00','stream':0,'pid':123,'freeze_sha256':'1'*64,'lease_sha256':'2'*64,'sample_index':0,'kind':'hardware','mode':'timing','status':'returned','launch_return':0,'synchronize_return':0,'start_event':'0xaaa','end_event':'0xbbb','event_elapsed_ms':ms,'event_elapsed_ms_float32_bits':struct.unpack('<I',struct.pack('<f',ms))[0],'elapsed_ns':ms*1e6,'unit':'ns','correct':True,'output_sha256':'f'*64,'all_function_handles_live':True,'hidden_warmup_launches':0,'helper_launches':0,'hardware_core_mhz':None,'clock_state':'unknown','cache_state':'uncontrolled; sample order retained'}


def test_event_float32_unit_and_mode_closure():
    b,c,s=raw_sample()
    assert validate_sample(b,c,s,'timing')==s['elapsed_ns']
    s.update(mode='structure',start_event=None,end_event=None,event_elapsed_ms=None,event_elapsed_ms_float32_bits=None,elapsed_ns=None,unit='not_timed')
    assert validate_sample(b,c,s,'structure') is None
    with pytest.raises(ValueError):validate_sample(b,c,s,'timing')


@pytest.mark.parametrize('key,value',[
    ('cubin_sha256','0'*64),('input_sha256','0'*64),('family','chase'),('unit','ms'),
    ('elapsed_ns',0.012345),('kind','synthetic'),('status','failed'),('launch_return',1),
    ('correct',False),('hidden_warmup_launches',1),('hardware_core_mhz',1980),
    ('clock_state','measured'),('start_event','0xbbb'),('event_elapsed_ms_float32_bits',0),
    ('event_elapsed_ms_float32_bits',0x7fc00000),('partition','heldout'),('unknown',True)])
def test_semantic_sample_negatives_not_hash_failures(key,value):
    b,c,s=raw_sample();s[key]=value
    with pytest.raises(ValueError):validate_sample(b,c,s,'timing')


def test_free_form_hardware_labels_do_not_enter_solver():
    with pytest.raises(ValueError):load_attempt(model(),{'kind':'hardware','status':'verified','elapsed_ns':123,'sample':{'sample_index':0}})
    with pytest.raises(ValueError):sample_quality(model(),[{'kind':'hardware'}]*3)


@pytest.mark.parametrize('family,count',[('dep',32),('quad',128)])
def test_closed_form_numeric_oracle_against_iteration(family,count):
    inputs=list(range(count));values=inputs[:]
    for _ in range(37):values=[(v*1664525+1013904223)&0xffffffff for v in values]
    main=struct.pack('<'+'I'*count,*inputs);aux=struct.pack('<2I',1664525,1013904223)
    assert oracle({'family':family,'steps':37},main,aux)==struct.pack('<'+'I'*count,*values)


def test_dependent_addresses_are_derived_from_input_not_observed_output(monkeypatch):
    import arch_explorer.simulator.accelsim.microcapture as mc
    table=[(29*i+37)%4096 for i in range(4096)];starts=list(range(32))
    main=struct.pack('<4096I',*table);aux=struct.pack('<32I',*starts)
    case={'family':'chase','steps':2};expected=oracle(case,main,aux)
    b=SimpleNamespace(inputs=lambda c:(main,aux,expected))
    allocation={'mul':0,'add':0,'buffers':[{'name':name,'address':addr,'bytes':size,'element_bytes':4,'output':name=='output'} for name,addr,size in [('input',0x100000,len(main)),('auxiliary',0x200000,len(aux)),('output',0x300000,len(expected))]]}
    instructions={i:{'opcode_short':op} for i,op in enumerate(['LDG','LDG','LDG','STG'])}
    indices=[starts,[table[x] for x in starts]]
    addresses=[[0x200000+4*i for i in range(32)]]+[[0x100000+4*x for x in row] for row in indices]+[[0x300000+4*i for i in range(32)]]
    stream=[{'id':i,'mask':0xffffffff,'addresses':a} for i,a in enumerate(addresses)]
    monkeypatch.setattr(mc,'operands',lambda ins:([],[],4))  # Isolate MREF oracle, no trace or capture claim.
    validate_addresses(b,case,instructions,{(0,0,0,0):stream},allocation)
    broken=deepcopy(stream);broken[2]['addresses'][7]+=4
    with pytest.raises(ValueError):validate_addresses(b,case,instructions,{(0,0,0,0):broken},allocation)
    with pytest.raises(ValueError):validate_addresses(b,case,instructions,{(0,0,0,0):stream[:-1]},allocation)


def design(tmp_path):
    b=model();reg=write_new(tmp_path/'registry.json',{'synthetic_cpu_fixture':True});b.path=Path(reg['path']);b.sha256=reg['sha256']
    evidence=write_new(tmp_path/'cpu-evidence.json',{'note':'synthetic parser fixture, never hardware'})
    return b,make_design(b,evidence,evidence,{'capture':evidence,'timing':evidence},evidence,evidence,evidence)


def test_frozen_order_budget_and_heldout_does_not_fit(tmp_path):
    b,d=design(tmp_path);validate_design(b,d)
    polluted={cid:[] for cid in d['order']['fit']};polluted[d['order']['heldout'][0]]=[]
    with pytest.raises(ValueError,match='heldout'):fit_intervals(b,d,polluted)
    changed=deepcopy(d);changed['budget']=dict(BUDGET,host_processes=14)
    with pytest.raises(ValueError):validate_design(b,changed)
    j=Journal(tmp_path/'journal',b,d)
    with pytest.raises(ValueError):j.initial_predictions({'path':'/absent','sha256':'0'*64})
    with pytest.raises(ValueError):j.structures({})
    j.fail('CPU test failure','unstarted')
    with pytest.raises(ValueError):j.append('registered',{})


def test_conditional_interval_math_does_not_estimate_physical_clocks():
    v=_vertices([(Fraction(0),Fraction(10)),(Fraction(100),Fraction(210))],Fraction(1))
    assert min(b for a,b in v)==Fraction(198,100)
    assert max(b for a,b in v)==Fraction(202,100)
    assert not _vertices([(Fraction(0),Fraction(100)),(Fraction(100),Fraction(1))],Fraction(1))


def test_old_or_incomplete_handoff_cannot_start_any_host(tmp_path,monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('CPU refusal must not start a host'))
    path=write_new(tmp_path/'old.json',{'schema':'arch-explorer.sim28-lease.v1'})
    with pytest.raises(ValueError):check_handoff({'run_id':'new','created_realtime_ns':1},path)


def test_host_budget_is_not_subsidized_by_unused_capture_budget(tmp_path):
    host=tmp_path/'host-00-fixture';host.mkdir()
    with (host/'large.raw').open('wb') as f:f.truncate(BUDGET['host_raw_bytes']+1)
    with pytest.raises(ValueError):require_raw_budget(tmp_path)


def test_three_samples_cannot_drop_duplicates_or_disagree_about_case(monkeypatch):
    import arch_explorer.simulator.accelsim.microexperiment as me
    monkeypatch.setattr(me,'load_attempt',lambda b,a:a)  # Only isolate quality arithmetic; not a provenance test.
    rows=[{'kind':'hardware','status':'verified','elapsed_ns':100+i,'sample':{'sample_index':i,'case':'cpu-only-fixture','pid':1,'context':'c'},'process':{'freeze_sha256':'f'}} for i in range(3)]
    assert me.sample_quality(None,rows)['status']=='passed'
    broken=deepcopy(rows);broken[2]['sample']['sample_index']=1
    with pytest.raises(ValueError):me.sample_quality(None,broken)
    broken=deepcopy(rows);broken[2]['sample']['case']='another-cpu-fixture'
    with pytest.raises(ValueError):me.sample_quality(None,broken)
    broken=deepcopy(rows);broken[2]['elapsed_ns']=1000
    assert me.sample_quality(None,broken)['status']=='refused'


def test_original_timing_roles_are_closed_before_accessing_files():
    row={'schema':'arch-explorer.sim30-observation-originals.v1','case':'dep-10','mode':'timing','files':{'unknown':{'path':'/missing','sha256':'0'*64}}}
    with pytest.raises(ValueError,match='complete original timing roles'):load_attempt(model(),row)


def test_heldout_comparison_consumes_exact_rational_interval_strings(tmp_path,monkeypatch):
    import arch_explorer.simulator.accelsim.microcampaign as mc
    monkeypatch.setattr(mc,'sample_quality',lambda *a:{'status':'passed'})
    monkeypatch.setattr(mc,'load_attempt',lambda b,a:{'elapsed_ns':a})
    prediction=write_new(tmp_path/'synthetic-prediction.json',{'evaluation':{'latency_ns':10}})
    result=mc.compare(None,{'cpu-fixture':{'prediction':prediction}},{'cpu-fixture':[10,11,12]},{'heldout_service_intervals_ns':{'cpu-fixture':['31/3','35/3']}})
    assert result['rows'][0]['service_mean_inside_conditional_interval'] is True


def test_cleared_capture_preload_needs_mapped_frozen_tool_not_timing(tmp_path):
    from arch_explorer.simulator.accelsim.microrecords import validate_loader_environment,ENV_KEYS
    tool=tmp_path/'SYNTHETIC-NOT-LOADED.so';tool.write_bytes(b'CPU parser fixture, not a library')
    t=ref(tool);launched={k:None for k in ENV_KEYS};launched['LD_PRELOAD']=str(tool)
    observed=dict(launched,LD_PRELOAD=None);maps=['0-1 r-xp 0 0:0 0 '+str(tool)]*2
    validate_loader_environment('structure',launched,observed,maps,t)
    with pytest.raises(ValueError):validate_loader_environment('timing',launched,observed,maps,t)
    with pytest.raises(ValueError):validate_loader_environment('structure',launched,observed,['',''],t)
    unknown=dict(observed,LD_LIBRARY_PATH='another library')
    with pytest.raises(ValueError):validate_loader_environment('structure',launched,unknown,maps,t)
    wrong=dict(observed,LD_PRELOAD='/different-tracer.so')
    with pytest.raises(ValueError):validate_loader_environment('structure',launched,wrong,maps,t)


@pytest.mark.parametrize('capacity',[65536,102400,131072,196608])
def test_fixed_shared_scenario_is_explicit_and_capacity_consistent(tmp_path,capacity):
    from arch_explorer.simulator.accelsim.microcampaign import native_assumption_scenario
    from arch_explorer.simulator.accelsim.config import compile_adf
    data={'schema':'arch-explorer.sim30-live-preflight.v1','status':'passed','gpu':'synthetic CPU fixture, never queried','uuid_hex':'0'*32,'cc':[12,0],'limits':{'sm_count':110,'warp_size':32,'max_threads_block':1024,'registers_sm':65536,'registers_block':65536,'shared_sm':capacity,'shared_block':49152}}
    p=write_new(tmp_path/'synthetic-facts.json',data)
    a=native_assumption_scenario(p['path']);shared=a.child('cluster').child('l1_shared')
    mode=shared.params['gpgpu_adaptive_cache_config']
    assert mode.value==0 and mode.source.startswith('synthetic') and 'unknown' in mode.source
    assert shared.value('gpgpu_shmem_size')==capacity
    compile_adf(a)
