"""Preparation associations are checked before any backend work; no native."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from arch_explorer import demo
from arch_explorer.adf import MappingSpec,WorkloadSpec
from arch_explorer.archs import dmc_adf


def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2))


@pytest.fixture
def opaque(monkeypatch):
    state={'assess':0,'evaluate':0,'same':False,'unsupported_last':False,'boolean_result':False}
    api=SimpleNamespace(REQUEST_SCHEMA=demo.REQUEST_SCHEMA,RESULT_SCHEMA=demo.RESULT_SCHEMA)
    specs={'adf':dmc_adf(1).to_dict(),'workload':WorkloadSpec('operator',{'op':'matmul','m':4,'k':8,'n':8}).to_dict(),
           'mappings':[MappingSpec('dmc_partition',{'split':v,'weights':'local','activations':'local'}).to_dict() for v in ('m','n')],
           'provider':'test.opaque','options':{},
           'scope':{'source':'OPAQUE FRONTEND TEST DOUBLE; no native'}}
    def prepare(a,w,m,**kw):
        if state['same'] and 'first_request' in state:
            return deepcopy(state['first_request'])
        request={'schema':api.REQUEST_SCHEMA,'provider':{'id':kw['provider'],'version':'1'},
                 'adf':a.to_dict(),'workload':w.to_dict(),'mapping':m.to_dict(),'options':deepcopy(kw['options']),
                 'payload':{'token':m.params['split'],'value':1}}
        state['first_request']=deepcopy(request)
        return request
    def assess(r):
        state['assess']+=1
        supported=not(state['unsupported_last'] and state['assess']==2)
        return {'supported':supported,'gaps':[] if supported else ['injected missing capability']}
    def evaluate(r):
        state['evaluate']+=1;request=deepcopy(r)
        if state['boolean_result']:request['payload']['value']=True
        return {'schema':api.RESULT_SCHEMA,'provider':deepcopy(r['provider']),
                'request':request,'evaluation':{'backend':'test.opaque','latency_ns':3.0,'details':{'source':'OPAQUE TEST DOUBLE'}}}
    api.prepare_request=prepare;api.assess_request=assess;api.evaluate_request=evaluate
    monkeypatch.setattr(demo,'program_api',lambda:api)
    return specs,state


def prepared(tmp_path,opaque):
    root=tmp_path/'demo';demo.prepare_demo(root,specs=opaque[0]);opaque[1]['assess']=0
    return root


@pytest.mark.parametrize('change',['last-request','type-flip','duplicate-content','alias','symlink','drop','extra','reorder','mapping-view','all-views','missing-preparation'])
def test_entire_set_validates_before_first_assessment(tmp_path,opaque,change):
    root=prepared(tmp_path,opaque);mp=root/'demo.json';m=load(mp);second=root/'mapping-001/request.json'
    if change in ('last-request','type-flip'):
        r=load(second);r['payload']['value']=True if change=='type-flip' else 2;save(second,r)
    elif change=='duplicate-content':second.write_bytes((root/'mapping-000/request.json').read_bytes())
    elif change=='alias':m['candidates'][1]['request']='mapping-001/../mapping-000/request.json'
    elif change=='symlink':
        second.unlink();second.symlink_to(root/'mapping-000/request.json')
    elif change=='drop':m['candidates'].pop()
    elif change=='extra':m['candidates'].append(deepcopy(m['candidates'][0]))
    elif change=='reorder':m['candidates'].reverse()
    elif change=='mapping-view':
        v=load(root/'mappings.json');v[0]['params']['split']='changed';save(root/'mappings.json',v)
    elif change=='all-views':
        s=load(root/'specs.json');s['workload']['params']['m']=7;save(root/'specs.json',s);save(root/'workload.json',s['workload'])
    elif change=='missing-preparation':(root/'preparation.json').unlink()
    save(mp,m)
    with pytest.raises(ValueError):demo.run_demo(root)
    assert opaque[1]['assess']==opaque[1]['evaluate']==0
    assert not (root/'comparison.json').exists()


def test_opaque_requests_and_formatting_are_valid(tmp_path,opaque):
    root=prepared(tmp_path,opaque)
    for p in root.glob('mapping-*/request.json'):
        p.write_text(json.dumps(load(p),sort_keys=True,separators=(',',':')))
    result=demo.run_demo(root)
    assert result['status']=='completed' and opaque[1]['assess']==opaque[1]['evaluate']==2


def test_provider_cannot_return_same_request_for_distinct_mappings(tmp_path,opaque):
    opaque[1]['same']=True
    with pytest.raises(ValueError,match='differs from its scene candidate|duplicate prepared request'):
        demo.prepare_demo(tmp_path/'demo',specs=opaque[0])
    assert opaque[1]['evaluate']==0


def test_last_unsupported_prevents_every_evaluation(tmp_path,opaque):
    root=prepared(tmp_path,opaque);opaque[1]['unsupported_last']=True
    with pytest.raises(ValueError,match='Unsupported'):demo.run_demo(root)
    assert opaque[1]['assess']==2 and opaque[1]['evaluate']==0


def test_result_true_cannot_replace_prepared_integer_one(tmp_path,opaque):
    root=prepared(tmp_path,opaque);opaque[1]['boolean_result']=True
    with pytest.raises(ValueError,match='does not preserve'):demo.run_demo(root)
    assert not (root/'comparison.json').exists()
