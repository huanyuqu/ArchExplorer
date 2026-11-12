"""Public frontend contracts; the provider double is never native evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from arch_explorer import demo
from arch_explorer.adf import WorkloadSpec, MappingSpec
from arch_explorer.archs import dmc_adf


@pytest.fixture
def provider(monkeypatch):
    state={'calls':[], 'fail_at':None, 'supported':True, 'bad':None}
    specs={'adf':dmc_adf(1).to_dict(), 'workload':WorkloadSpec('operator',{'op':'matmul','m':4,'k':8,'n':8}).to_dict(),
           'mappings':[MappingSpec('dmc_partition',{'split':x,'weights':'local','activations':'local'}).to_dict() for x in ('m','n')],
           'provider':'test.frontend', 'options':{},
           'scope':{'source':'FRONTEND TEST DOUBLE; no native execution'}}
    api=SimpleNamespace(REQUEST_SCHEMA=demo.REQUEST_SCHEMA,RESULT_SCHEMA=demo.RESULT_SCHEMA)
    def prepare(adf,workload,mapping,*,output_dir,provider,options=None):
        assert isinstance(workload,WorkloadSpec) and isinstance(mapping,MappingSpec)
        return {'schema':api.REQUEST_SCHEMA,'provider':{'id':provider,'version':'1'},
                'adf':adf.to_dict(),'workload':workload.to_dict(),'mapping':mapping.to_dict(),
                'options':deepcopy(options),'payload':{'output_dir':output_dir}}
    def evaluate(request):
        state['calls'].append(deepcopy(request))
        if state['fail_at']==len(state['calls']):raise RuntimeError('injected provider failure')
        value={'schema':api.RESULT_SCHEMA,'provider':deepcopy(request['provider']),
               'request':deepcopy(request),'evaluation':{'backend':'test.frontend','latency_ns':12.0,'details':{'source':'FRONTEND TEST DOUBLE'}}}
        if state['bad']=='request':value['request']['mapping']={}
        if state['bad']=='provider':value['provider']['id']='different.provider'
        if state['bad']=='backend':value['evaluation']['backend']=''
        if state['bad']=='latency':value['evaluation']['latency_ns']=True
        return value
    api.prepare_request=prepare;api.assess_request=lambda r:{'supported':state['supported'],'gaps':[] if state['supported'] else ['injected unknown capability']}
    api.evaluate_request=evaluate
    monkeypatch.setattr(demo,'program_api',lambda:api)
    return specs,state


def prepare(root,provider):return demo.prepare_demo(root,specs=provider[0],engine='EXPLICIT_TEST_DOUBLE')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v))


def test_preparation_never_evaluates_and_new_shape_is_preserved(tmp_path,provider):
    specs,state=provider;specs['workload']['params']['m']=7
    prepare(tmp_path/'demo',provider)
    assert not state['calls']
    assert read(tmp_path/'demo/mapping-000/request.json')['workload']['params']['m']==7
    assert not (tmp_path/'demo/comparison.json').exists()


def test_success_and_partial_failure_have_distinct_artifacts(tmp_path,provider):
    prepare(tmp_path/'success',provider);result=demo.run_demo(tmp_path/'success')
    assert result['status']=='completed' and len(result['candidates'])==2
    state=provider[1];state['calls']=[];state['fail_at']=2
    prepare(tmp_path/'partial',provider)
    with pytest.raises(RuntimeError,match='injected'):demo.run_demo(tmp_path/'partial')
    assert (tmp_path/'partial/mapping-000/result.json').is_file()
    assert not (tmp_path/'partial/comparison.json').exists()
    assert read(tmp_path/'partial/run.json')['status']=='failed'


@pytest.mark.parametrize('bad',['request','provider','backend','latency'])
def test_wrong_provider_result_cannot_complete(tmp_path,provider,bad):
    provider[1]['bad']=bad;prepare(tmp_path/'demo',provider)
    with pytest.raises(ValueError):demo.run_demo(tmp_path/'demo')
    assert not (tmp_path/'demo/comparison.json').exists()


def test_unsupported_is_rechecked_before_any_evaluation(tmp_path,provider):
    prepare(tmp_path/'demo',provider);provider[1]['supported']=False
    with pytest.raises(ValueError,match='Unsupported'):demo.run_demo(tmp_path/'demo')
    assert not provider[1]['calls'] and not (tmp_path/'demo/comparison.json').exists()


def test_run_rejects_duplicate_candidate_request(tmp_path,provider):
    prepare(tmp_path/'demo',provider);p=tmp_path/'demo/demo.json';m=read(p)
    m['candidates'][1]['request']=m['candidates'][0]['request'];write(p,m)
    with pytest.raises(ValueError):demo.run_demo(tmp_path/'demo')
    assert not provider[1]['calls']


def test_run_rejects_candidate_mapping_label_changed_after_prepare(tmp_path,provider):
    prepare(tmp_path/'demo',provider);p=tmp_path/'demo/demo.json';m=read(p)
    m['candidates'][0]['mapping']['params']['split']='not-the-prepared-mapping';write(p,m)
    with pytest.raises(ValueError):demo.run_demo(tmp_path/'demo')
    assert not provider[1]['calls']


def test_outside_reference_is_rejected(tmp_path,provider):
    prepare(tmp_path/'demo',provider);p=tmp_path/'demo/demo.json';m=read(p)
    m['candidates'][0]['request']='../outside.json';write(p,m)
    with pytest.raises(ValueError,match='leaves'):demo.run_demo(tmp_path/'demo')
    assert not provider[1]['calls']
