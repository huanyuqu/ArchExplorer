"""Independent typed byte/FMA arithmetic for actual public access records."""
import math
from typed_onnx_contract_cases import ITEM_BYTES


def expected_cast(shape,source_dtype,target_dtype):
    n=math.prod(shape)
    return dict(elements=n,read_bytes=n*ITEM_BYTES[source_dtype],write_bytes=n*ITEM_BYTES[target_dtype])


def audit_cast_access(step):
    assert step['semantic']=='cast'
    reads=step['address_map']['reads'];writes=step['address_map']['writes']
    assert len(reads)==len(writes)==1
    source,target=reads[0],writes[0]
    assert source['shape']==target['shape']
    expected=expected_cast(source['shape'],source['dtype'],target['dtype'])
    access=step['access']
    for field in ('read_bytes','write_bytes'):assert access[field]==expected[field]
    assert access['read_by_tensor']=={source['tensor']:expected['read_bytes']}
    conversions=0 if source['dtype']==target['dtype'] else expected['elements']*((source['dtype']!='fp32')+(target['dtype']!='fp32'))
    assert access['conversion_instructions']==conversions
    return dict(step=step['name'],source_dtype=source['dtype'],target_dtype=target['dtype'],
                conversion_instructions=conversions,**expected)


def audit_matmul_access(step):
    assert step['semantic']=='compiled.matmul'
    attrs=step['address_map']['indexing']['attrs']
    fmas=attrs['batch']*attrs['rows']*attrs['cols']*attrs['k']
    reads=step['address_map']['reads'];writes=step['address_map']['writes']
    assert len(reads)==2 and len(writes)==1
    expected={}
    for d in reads:
        assert d['units']=='elements'
        expected[d['tensor']]=expected.get(d['tensor'],0)+ITEM_BYTES[d['dtype']]*fmas
    access=step['access']
    assert access['read_by_tensor']==expected
    assert access['read_bytes']==sum(expected.values())
    n=attrs['batch']*attrs['rows']*attrs['cols']
    conversions=fmas*sum(d['dtype']!='fp32' for d in reads)+n*(writes[0]['dtype']!='fp32')
    assert access['conversion_instructions']==conversions
    assert access['operations']==2*fmas+conversions
    assert access['write_bytes']==n*ITEM_BYTES[writes[0]['dtype']]
    return dict(step=step['name'],fmas=fmas,read_by_tensor=expected,write_bytes=access['write_bytes'],conversion_instructions=conversions)
