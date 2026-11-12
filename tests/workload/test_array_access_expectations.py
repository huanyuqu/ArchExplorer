"""Hand literal read traces discriminate duplicates, eager reads and widths."""
from copy import deepcopy
import pytest
from array_access_expectations import expected,compare_counts,compare_indices


def test_gather_reads_index_once_and_keeps_repeated_selected_reads():
    tensors=[dict(name='x',shape=[2,3],dtype='int64'),dict(name='i',shape=[2],dtype='int32'),
             dict(name='y',shape=[2,2],dtype='int64')]
    op=dict(op='gather',inputs=['x','i'],outputs=['y'],attrs=dict(axis=-1,shape=[2,2]))
    r=expected(op,tensors,{'i':[-1,-1]})
    assert r['reads']==[('i',(0,)),('i',(1,)),('x',(0,2)),('x',(0,2)),('x',(1,2)),('x',(1,2))]
    assert r['read_by_tensor']=={'i':8,'x':32} and r['write_bytes']==32 and r['simt_instructions']==10
    compare_counts(deepcopy(r),r);compare_indices(r['reads'],r)
    bad=deepcopy(r);bad['read_bytes']=24
    with pytest.raises(ValueError,match='read_bytes'):compare_counts(bad,r)


def test_where_reads_both_branches_and_repeated_binary_input_twice():
    tensors=[dict(name='c',shape=[2],dtype='bool'),dict(name='x',shape=[2],dtype='bf16'),dict(name='y',shape=[2],dtype='bf16')]
    r=expected(dict(op='where',inputs=['c','x','x'],outputs=['y'],attrs=dict(shape=[2])),tensors,{})
    assert r['reads']==[('c',(0,)),('x',(0,)),('x',(0,)),('c',(1,)),('x',(1,)),('x',(1,))]
    assert r['read_by_tensor']=={'c':2,'x':8} and r['write_bytes']==4
    assert r['conversion_instructions']==r['sfu_instructions']==0


def test_argmax_axis_order_and_scalar_index_storage():
    tensors=[dict(name='x',shape=[2,2],dtype='fp32'),dict(name='y',shape=[2],dtype='int64')]
    r=expected(dict(op='argmax_axis',inputs=['x'],outputs=['y'],attrs=dict(axis=0,keepdims=False,shape=[2])),tensors,{})
    assert r['reads']==[('x',(0,0)),('x',(1,0)),('x',(0,1)),('x',(1,1))]
    assert r['read_bytes']==16 and r['write_bytes']==16 and r['simt_instructions']==4
    tensors=[dict(name='x',shape=[3],dtype='fp16'),dict(name='i',shape=[1],dtype='int64'),dict(name='y',shape=[1],dtype='fp16')]
    r=expected(dict(op='gather',inputs=['x','i'],outputs=['y'],attrs=dict(axis=0,indices_scalar=True,shape=[1])),tensors,{'i':[-3]})
    assert r['reads']==[('i',(0,)),('x',(0,))] and r['write_bytes']==2
    with pytest.raises(ValueError,match='bounds'):
        expected(dict(op='gather',inputs=['x','i'],outputs=['y'],attrs=dict(axis=0,indices_scalar=True,shape=[1])),tensors,{'i':[3]})
