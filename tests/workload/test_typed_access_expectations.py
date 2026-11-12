from typed_access_expectations import expected_cast,audit_matmul_access
from typed_onnx_contract_cases import root_storage


def test_same_dtype_cast_still_reads_and_writes_payload():
    assert expected_cast([2,3],'bf16','bf16')==dict(elements=6,read_bytes=12,write_bytes=12)
    assert expected_cast([2,3],'fp32','fp16')==dict(elements=6,read_bytes=24,write_bytes=12)
    assert expected_cast([],'bf16','fp32')==dict(elements=1,read_bytes=2,write_bytes=4)


def test_two_reads_of_same_root_are_both_charged_in_low_precision():
    step=dict(name='synthetic',semantic='compiled.matmul',address_map=dict(
        indexing=dict(attrs=dict(batch=1,rows=2,cols=2,k=3)),
        reads=[dict(tensor='shared',dtype='bf16',units='elements')]*2,
        writes=[dict(tensor='out',dtype='bf16',units='elements')]),
        access=dict(read_by_tensor={'shared':48},read_bytes=48,write_bytes=8,operations=52,conversion_instructions=28))
    assert audit_matmul_access(step)['fmas']==12


def test_mixed_model_root_byte_counts_are_physical_storage():
    for dtype,size in [('fp32',4),('fp16',2),('bf16',2)]:
        row=dict(dtype=dtype,input_shapes={'X':[2,5],'R':[2,7]})
        assert sum(v['bytes'] for v in root_storage(row).values())==24*size
