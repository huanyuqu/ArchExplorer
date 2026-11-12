"""Installed same-source external FP32 rule retains the two input reads."""
import json
import numpy as np
from test_onnx_import_contract import onnx_api
from test_operator_mapping_attention_contract import scene,no_backend

def test_external_unknown_square_reuses_generic_mapping_and_actual_service(onnx_api,monkeypatch,tmp_path):
    from arch_explorer.adf import MappingSpec,WorkloadSpec
    from ae_operator_mappings.semantics import KIND
    w,s=onnx_api[1:];adf,_,_=scene('events')
    graph=w.Workload('val62-unknown-fp32',dtype='fp32')
    graph.tensor('X',(2,3));graph.tensor('Y',(2,3))
    values=np.asarray([[1.,-2.,.5],[0.,.25,-.5]],dtype=np.float32)
    graph.meta['inputs']={'X':values.tolist()};graph.meta['outputs']=['Y']
    graph.add('square',KIND,['X'],['Y'],shape=[2,3])
    spec=WorkloadSpec('graph',graph.to_dict())
    mapping=MappingSpec('compile_graph_v3',dict(compute='board/core/sm',memory='board/sram',
        rules={KIND:w.MappingRuleBinding('example.square.materialized','1').to_dict()},operator_rules={}))
    with no_backend(monkeypatch):
        compiled=w.compile_workload(adf,spec,mapping)
        np.testing.assert_array_equal(w.execute_reference(spec)['outputs']['Y'],values*values)
        mapped=w.execute_reference(WorkloadSpec('graph',compiled['program']))
        np.testing.assert_array_equal(mapped['outputs']['Y'],values*values)
    step=next(row for row in compiled['plan']['steps'] if row['semantic']=='mul')
    assert step['access']['read_bytes']==2*values.nbytes and step['access']['write_bytes']==values.nbytes
    assert [v['tensor'] for v in step['address_map']['reads']]==['X','X']
    request=s.prepare_compiled(compiled,output_dir=tmp_path/'actual',options={'trace':True})
    result=s.evaluate_request(request);assert result['evaluation']['latency_ns']>0
    (tmp_path/'EXTENSION.json').write_text(json.dumps(dict(unknown_kind=KIND,read_bytes=2*values.nbytes,
        write_bytes=values.nbytes,actual_Events=True,CPU_independent_values=True),indent=2)+'\n')
