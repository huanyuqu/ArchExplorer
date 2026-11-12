"""Protocol-only static folding controls, no product backend substituted."""
from copy import deepcopy
import numpy as np
import pytest
from static_control_observations import classify,cast_observability,inspect_consumer_shapes


def control():
    onnx=pytest.importorskip('onnx');h=onnx.helper;T=onnx.TensorProto
    nodes=[h.make_node('Shape',['X'],['dims']),
        h.make_node('Constant',[],['target'],value=h.make_tensor('v',T.INT64,[2],[3,2])),
        h.make_node('Reshape',['X','target'],['Y']),h.make_node('Cast',['dims'],['float_dims'],to=T.FLOAT)]
    model=h.make_model(h.make_graph(nodes,'static-control',[h.make_tensor_value_info('X',T.FLOAT,[2,3])],
        [h.make_tensor_value_info('Y',T.FLOAT,[3,2])]),ir_version=8,opset_imports=[h.make_opsetid('',17)])
    oracle=dict(dims=np.array([2,3],dtype=np.int64),target=np.array([3,2],dtype=np.int64),
                Y=np.arange(6,dtype=np.float32).reshape(3,2),float_dims=np.array([2,3],dtype=np.float32))
    types=dict(dims='int64',target='int64',Y='fp32',float_dims='fp32')
    desc=dict(values={n:dict(tensor='tensor.Y' if n=='Y' else None,dtype=types[n],logical_shape=list(a.shape),
        role='output' if n=='Y' else 'shape') for n,a in oracle.items()},
        nodes=[dict(id=f'n{i:05d}.{n.op_type}',op_type=n.op_type,inputs=list(n.input),outputs=list(n.output),
                    constant_folded=n.op_type!='Reshape') for i,n in enumerate(nodes)],
        outputs=[dict(name='Y',tensor='tensor.Y')])
    return model,oracle,types,desc


def test_null_values_are_not_cpu_passes_and_cast_is_separate():
    model,oracle,types,desc=control();r=classify(desc,types,oracle,model)
    assert [v['value'] for v in r['materialized']]==['Y']
    assert {v['value'] for v in r['static']}=={'dims','target','float_dims'}
    assert r['static_cpu_values_counted_as_passed']==0 and not r['all_original_values_cpu_observable']
    assert all(not v['cpu_value_returned'] and not v['folded_value_numerically_verified'] for v in r['static'])
    target=next(v for v in r['static'] if v['value']=='target')
    assert target['consumers'][0]['op_type']=='Reshape'
    assert target['consumers'][0]['source_attributes_bound'] and not target['consumers'][0]['product_attributes_checked']
    cast=cast_observability(desc['nodes'][3],desc['values'])
    assert not cast['observable'] and cast['cpu_cast_passed'] is False


@pytest.mark.parametrize('change,reason',[('output-null','public output'),('shape','shape'),('consumer','identity')])
def test_inconsistent_static_metadata_refused(change,reason):
    model,oracle,types,desc=control();desc=deepcopy(desc)
    if change=='output-null':desc['values']['Y']['tensor']=None
    elif change=='shape':desc['values']['dims']['logical_shape']=[1]
    else:desc['nodes'][2]['inputs']=['X','dims']
    with pytest.raises(ValueError,match=reason):classify(desc,types,oracle,model)


def test_static_consumer_graph_shape_attribute_is_checked_separately():
    model,oracle,types,desc=control();coverage=classify(desc,types,oracle,model)
    graph=dict(tensors=[dict(name='tensor.Y',shape=[3,2])],operators=[dict(name='n00002.Reshape.reshape',
        op='reshape',inputs=['X'],outputs=['tensor.Y'],attrs=dict(shape=[3,2]))])
    rows=inspect_consumer_shapes(coverage,desc,graph)
    row=next(r for r in rows if r['static_value']=='target')
    assert row['targets'][0]['shape_attributes_checked'][0]['value']==[3,2]
    assert row['uninspected_attribute_values'] and not row['source_integer_or_shape_value_returned']
    graph['operators'][0]['attrs']['shape']=[2,3]
    with pytest.raises(ValueError,match='shape attribute'):inspect_consumer_shapes(coverage,desc,graph)
