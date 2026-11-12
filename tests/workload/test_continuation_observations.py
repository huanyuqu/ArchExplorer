"""QA discriminator controls; no product execution or fake product API."""
from copy import deepcopy
import numpy as np
import pytest
from continuation_observations import compare_value,shaped,check_cast,check_ledger,original_roots,bind_reference,digest


def receipts():
    workload=dict(kind='onnx',params=dict(source='hand-control'))
    mapping=dict(template='compile_graph_v3',params=dict(tile=[2,3,4]))
    item=dict(workload=workload,mapping=mapping,
              compiled_result=dict(request=dict(compilation=dict(source=dict(workload=deepcopy(workload),mapping=deepcopy(mapping))))))
    return dict(evaluations_used=3,max_evaluations=3,evaluation_attempts=[
        dict(index=0,valid=True,execution_attempted=True,score=dict(objective=12)),
        dict(index=1,valid=False,execution_attempted=False,score=None),
        dict(index=2,valid=True,execution_attempted=True,score=dict(objective=8))],
        best=dict(score=dict(objective=8),workloads=[item]))


def test_integer_exactness_cannot_be_hidden_by_float_error_math():
    compare_value([2**53+1],[2**53+1],dtype='int64',shape=[1])
    with pytest.raises(ValueError,match='differs exactly'):
        compare_value([2**53],[2**53+1],dtype='int64',shape=[1])
    with pytest.raises(ValueError,match='float/bool'):
        compare_value([float(2**53+1)],[2**53+1],dtype='int64',shape=[1])
    with pytest.raises(ValueError,match='outside declared'):
        shaped(np.array([2**63],dtype=np.uint64),[1],'int64')
    with pytest.raises(ValueError,match='boolean'):compare_value([1],[True],dtype='bool',shape=[1])
    assert compare_value([False,True],[False,True],dtype='bool',shape=[2])['exact']


def test_float_storage_and_cast_are_separate_from_numeric_error():
    observed=compare_value([1.],[1.+2.**-7],dtype='bf16',shape=[1])
    assert observed['storage_grid'] and not observed['exact'] and observed['passed_fp32_tolerance'] is None
    with pytest.raises(ValueError,match='storage grid'):compare_value([1.+2.**-8],[1.],dtype='bf16',shape=[1])
    with pytest.raises(ValueError,match='nonfinite'):compare_value([float('inf')],[1.],dtype='fp32',shape=[1])
    check_cast(np.array([2**53+1],dtype=np.int64),[2**53],target_dtype='fp32',target_shape=[1])
    check_cast([-1.9,2.9],[-1,2],target_dtype='int64',target_shape=[2])
    with pytest.raises(ValueError,match='zero sign'):check_cast([-0.],[0.],target_dtype='bf16',target_shape=[1])


def test_ledger_tracks_failed_units_and_effective_winner():
    result=check_ledger(receipts(),limit=3)
    assert result['used']==3 and result['invalid_attempts']==1 and result['executed_attempts']==2
    for change,reason in [('drop','count'),('budget','saved budget'),('index','order'),('score','objectives'),('mapping','bound'),('nonfinite','nonfinite')]:
        d=receipts()
        if change=='drop':d['evaluation_attempts'].pop()
        elif change=='budget':d['max_evaluations']=12
        elif change=='index':d['evaluation_attempts'][1]['index']=2
        elif change=='score':d['best']['score']['objective']=7
        elif change=='mapping':d['best']['workloads'][0]['mapping']['params']['tile']=[3,2,3]
        elif change=='nonfinite':d['evaluation_attempts'][0]['score']['objective']=float('nan')
        with pytest.raises(ValueError,match=reason):check_ledger(d,limit=3)


def test_winner_boolean_cannot_alias_a_numeric_objective():
    d=receipts();d['evaluation_attempts'][2]['score']['objective']=1;d['best']['score']['objective']=True
    with pytest.raises(ValueError,match='finite number'):check_ledger(d,limit=3)


def test_actual_receipts_read_bound_results_without_inventing_scores(tmp_path):
    import hashlib,json
    d=receipts();d['objective_config']=dict(cost_budget_usd=None,area_budget_mm2=None)
    source=d['best']['workloads'][0]['compiled_result']['request']['compilation']['source']
    d['best']['workloads'][0]['workload']['weight']=1.
    source['workload']['weight']=1.
    for index,latency in ((0,12),(2,8)):
        compiled=dict(schema='arch-explorer.compiled-result.v3',binding=dict(sha256='result'+str(index)),
            request=dict(binding=dict(sha256='request'+str(index)),compilation=dict(binding=dict(sha256='compile'+str(index)),source=source)),
            evaluation=dict(latency_ns=latency))
        path=tmp_path/(str(index)+'.json');raw=json.dumps(compiled).encode();path.write_bytes(raw)
        row=d['evaluation_attempts'][index];row.pop('score');row.update(workload=0,
            public_result=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),
            result_binding=compiled['binding'],request_binding=compiled['request']['binding'],
            compilation_binding=compiled['request']['compilation']['binding'],effective_mapping=source['mapping'])
    assert check_ledger(d,limit=3)['winner_present']
    d['best']['score']['objective']=9
    with pytest.raises(ValueError,match='winner'):check_ledger(d,limit=3)
    d['best']['score']['objective']=8
    path.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='digest'):check_ledger(d,limit=3)


def test_original_root_bytes_use_actual_integer_bool_and_weight_types():
    onnx=pytest.importorskip('onnx');h=onnx.helper;T=onnx.TensorProto
    graph=h.make_graph([h.make_node('Constant',[],['scalar'],value=h.make_tensor('v',T.INT64,[],[7]))],
        'metadata-control',[h.make_tensor_value_info('tokens',T.INT64,[3,4]),h.make_tensor_value_info('mask',T.BOOL,[2,3])],
        [h.make_tensor_value_info('scalar',T.INT64,[])],
        initializer=[h.make_tensor('conv-weight',T.FLOAT,[1,1,1,1],[1.])])
    model=h.make_model(graph,ir_version=8,opset_imports=[h.make_opsetid('',17)])
    r=original_roots(model,{'tokens':[3,4],'mask':[2,3]})
    assert r['tokens']['bytes']==96 and r['mask']['bytes']==6
    assert r['scalar']['bytes']==8 and r['scalar']['elements']==1
    assert r['conv-weight']['dtype']=='fp32' and r['conv-weight']['bytes']==4


def test_reference_binding_is_portable_and_preserves_raw_failures(tmp_path):
    import json
    package=tmp_path/'relocated-inputs';package.mkdir()
    (package/'INPUTS.json').write_text('{}')
    (package/'model.onnx').write_bytes(b'QA metadata binding control; not executed as a model')
    rows=[dict(path=n,bytes=digest(package/n)['bytes'],sha256=digest(package/n)['sha256']) for n in ('INPUTS.json','model.onnx')]
    (package/'MANIFEST.json').write_text(json.dumps(dict(files=rows)))
    record=tmp_path/'reference'/'RESULT.json';record.parent.mkdir()
    case=dict(id='control',model='model.onnx',output_names=['Y'])
    report=dict(case_id='control',inputs=rows,nodes=[dict(outputs=['Y'],output_dtypes=['int64'])],
                output_names=['Y'],status='failed')
    np.savez(record.with_name('INDEPENDENT.npz'),Y=np.array([2**53+1],dtype=np.int64))
    record.write_text(json.dumps(report))
    facts,types,values=bind_reference(case,package,record)
    assert facts['raw_reference_status']=='failed' and facts['reference_requires_review'] is True
    assert types['Y']=='int64' and values['Y'][0]==2**53+1
    values.close()
    report['inputs']=[*rows,rows[0]];record.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='unique'):bind_reference(case,package,record)
