"""Deferred sole-candidate Events runs. CPU values and services stay separate."""
from copy import deepcopy
import hashlib,json,os
from pathlib import Path
import numpy as np
import pytest
from onnx_contract_cases import scene,formal_spec
from test_onnx_import_contract import onnx_api,compare_original_values
from onnx_import_expectations import ABS_TOL,REL_TOL
from multilayer_contract_cases import observe_memory
from view_storage_runtime_cases import audit_storage
from view_storage_contract_cases import independent_layouts,assert_layout
from test_mla_numeric_storage_contract import audit_access
from test_operator_mapping_attention_contract import no_backend

@pytest.mark.parametrize('key',['mlp','attention'])
@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('policy',['materialized','views'])
def test_actual_Events_and_separate_original_mapped_CPU_values(onnx_api,monkeypatch,tmp_path,key,index,policy):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.onnx_import import describe_onnx
    w,s=onnx_api[1:];adf,spec,mapping,expected=scene(key,index,policy)
    description=describe_onnx(spec)
    with no_backend(monkeypatch):
        compiled=w.compile_workload(adf,spec,mapping)
        original=w.execute_reference(spec)
        assert compare_original_values(original,description,expected['stages'])
        mapped=w.execute_reference(WorkloadSpec('graph',compiled['program']))
        for output in description['outputs']:
            wanted=expected['outputs'][output['name']]
            actual=np.asarray(mapped['outputs'][output['tensor']]).reshape(wanted.shape)
            np.testing.assert_allclose(actual,wanted,atol=ABS_TOL,rtol=REL_TOL)
        tensor_objects=w.workload_from_spec(WorkloadSpec('graph',compiled['program'])).tensors
        for name,layout in independent_layouts(compiled['program']).items():
            assert_layout(w,tensor_objects,name,layout)
    access=audit_access(compiled)
    request=s.prepare_compiled(compiled,output_dir=tmp_path/'execution',options={'trace':True})
    with observe_memory(monkeypatch) as observed:result=s.evaluate_request(request)
    storage=audit_storage(compiled,result,observed)
    peaks,intervals=storage['peaks'],storage['intervals']
    assert result['evaluation']['latency_ns']>0
    returned=set(compiled['program']['meta']['outputs'])
    assert len(returned)==len(description['outputs'])==len(expected['outputs'])>=2
    assert all(row['tensor'] in returned for row in description['outputs'])
    facts=dict(schema='arch-explorer.val62.physical.v1',case=key,index=index,layout=policy,
        CPU_original_values_checked=True,CPU_mapped_values_checked=True,
        Events_executed=True,component_numeric_output_not_claimed=True,
        actual_memory_observed=True,peaks=peaks,intervals=intervals,access=access,
        observations=observed,compilation_binding=compiled['binding'],
        views_peak_reduction_not_assumed=True)
    for name,value in [('FACTS.json',facts),('RESULT.json',result)]:
        (tmp_path/name).write_text(json.dumps(value,indent=2)+'\n')
    with no_backend(monkeypatch):assert s.evaluate_request(result)==result

def test_actual_capacity_uses_prior_observed_peak_and_one_byte_less(onnx_api,tmp_path):
    from arch_explorer.adf import Param,find
    w,s=onnx_api[1:];path=Path(os.environ['VAL62_PHYSICAL_FACTS']);raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL62_PHYSICAL_FACTS_SHA256']
    facts=json.loads(raw);assert facts['actual_memory_observed'] and facts['Events_executed']
    adf,spec,mapping,_=scene(facts['case'],facts['index'],facts['layout'])
    peaks=facts['peaks'];assert peaks and all(int(v)==v and v>0 for v in peaks.values())
    for memory,size in peaks.items():find(adf,memory).params['capacity']=Param(int(size),'B','VAL62 actual independent root interval peak')
    compiled=w.compile_workload(adf,spec,mapping)
    just=s.evaluate_request(s.prepare_compiled(compiled,output_dir=tmp_path/'just-fit',options={'trace':True}))
    assert just['evaluation']['details']['stats']['memory_peak']==peaks
    memory=next(iter(peaks));find(adf,memory).params['capacity']=Param(int(peaks[memory])-1,'B','VAL62 previous interval peak minus one byte')
    with pytest.raises((ValueError,RuntimeError,MemoryError)) as error:
        bad=w.compile_workload(adf,spec,mapping)
        s.evaluate_request(s.prepare_compiled(bad,output_dir=tmp_path/'one-short',options={'trace':True}))
    assert 'capacity' in str(error.value).lower()
    (tmp_path/'CAPACITY.json').write_text(json.dumps(dict(memory=memory,peak=peaks[memory],
        source_sha256=hashlib.sha256(raw).hexdigest(),error_type=type(error.value).__name__,
        diagnostic=str(error.value),actual_just_fit=True,actual_one_short_refused=True),indent=2)+'\n')
