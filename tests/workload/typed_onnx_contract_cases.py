"""SIM52 frozen data wiring; runtime helpers only call public product APIs."""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path

INPUT_SHA='4465f47136b92df34847b6002d96324d9dfe18e69e47d3b9bc6cff756e17ceb1'
DEFAULT_BUDGET=dict(schema='arch-explorer.resource-budget.v1',max_source_bytes=16777216,
 max_nodes=256,max_rank=4,max_dimension=4096,max_tensor_elements=1000000,
 max_root_bytes=33554432,max_total_tensor_bytes=33554432,max_steps=512,
 max_matrix_fmas=100000000,max_tiles=4096,max_host_bytes=8589934592,reference_mode='execute')
ITEM_BYTES={'fp32':4,'fp16':2,'bf16':2,'int32':4,'int64.shape':8}


def formal_data(root):
    root=Path(root);raw=(root/'INPUTS.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INPUT_SHA:raise ValueError('Frozen SIM52 INPUTS identity changed')
    document=json.loads(raw)
    assert not document['answers_included'] and len(document['cases'])==12
    assert document['resource_budget']==DEFAULT_BUDGET
    return document


def row_for(root,case_id):
    data=formal_data(root)
    rows=[r for r in data['cases'] if r['id']==case_id]
    assert len(rows)==1
    return data,deepcopy(rows[0])


def root_storage(row):
    records={}
    for name,shape in row['input_shapes'].items():
        records[name]=dict(logical_shape=shape,storage_dtype=row['dtype'],
                           elements=math.prod(shape),bytes=math.prod(shape)*ITEM_BYTES[row['dtype']])
    return records


def imported(root,case_id,*,reference_mode='execute',budget=None):
    from arch_explorer.workload import api as w
    root=Path(root);data,row=row_for(root,case_id)
    selected=deepcopy(data['resource_budget'] if budget is None else budget)
    selected['reference_mode']=reference_mode
    normalized=w.normalize_budget(selected)
    preflight=w.onnx_preflight(root/row['model'],input_shapes=row['input_shapes'],
        external_root=root/row['external_root'],resource_budget=selected)
    spec=w.import_onnx(root/row['model'],inputs=row['inputs'],input_shapes=row['input_shapes'],
        external_root=root/row['external_root'],resource_budget=selected)
    assert set(spec.params)=={'name','source','inputs','input_shapes','resource_budget'}
    assert spec.params['resource_budget']==normalized==preflight['budget']
    assert preflight['admitted'] is True and preflight['violations']==[]
    expected=root_storage(row)
    from onnx_root_storage_expectations import frozen_roots
    all_roots=frozen_roots(root/row['model'],row)
    assert preflight['totals']['root_bytes']==sum(v['bytes'] for v in all_roots.values())
    assert preflight['totals']['weight_bytes']==sum(v['bytes'] for v in all_roots.values() if v['role']=='weight')
    for name,record in all_roots.items():
        for field,value in record.items():assert preflight['values'][name][field]==value
    assert set(preflight['inputs'])==set(expected)
    for name,value in expected.items():
        for field,wanted in value.items():assert preflight['inputs'][name][field]==wanted
    graph=w.workload_from_spec(spec)
    assert graph.meta['resource_budget']==normalized
    return spec,preflight,w.describe_onnx(spec)


def scene(root,case_id,index=0,policy='views',*,reference_mode='execute'):
    from arch_explorer.adf import MappingSpec
    from arch_explorer.components.shared_memory import shared_memory_hardware
    from arch_explorer.workload import api as w
    data=formal_data(root);spec,preflight,description=imported(root,case_id,reference_mode=reference_mode)
    adf,_=shared_memory_hardware(**data['hardware']['params'])
    mapping=w.layout_candidates(spec,MappingSpec(**data['mappings'][index]))[policy]
    return adf,spec,mapping,preflight,description


def budget_refusals():
    from arch_explorer.workload import api as w
    assert w.normalize_budget({})==DEFAULT_BUDGET
    errors=[]
    for field,value in [('max_nodes',True),('max_rank',33),('max_dimension',2**31),
                         ('max_host_bytes',17*1024**3),('reference_mode','skip'),('unexpected',1)]:
        try:w.normalize_budget({field:value})
        except ValueError as e:
            assert str(e)
            errors.append(dict(field=field,value=value,type=type(e).__name__,diagnostic=str(e)))
        else:raise AssertionError('Missing budget refusal: '+field)
    return errors
