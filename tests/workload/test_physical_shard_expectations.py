"""Hand-specified cache/output overlap and directed communication controls."""
from copy import deepcopy
import pytest
from physical_shard_expectations import audit,SCHEMA


def sample():
    return dict(schema=SCHEMA,capacities={'a':12,'b':12,'bridge':4},links=[['a','bridge'],['bridge','b']],
        tensors={'cache':dict(elements=8,dtype='fp16',root=True),'out':dict(elements=2,dtype='fp16',root=False)},
        buffers=[
            dict(id='a0',tensor='cache',start=0,stop=4,tile='a',offset=0,bytes=8,born=0,ready=0,dead=10,kind='root'),
            dict(id='b0',tensor='cache',start=4,stop=8,tile='b',offset=0,bytes=8,born=0,ready=0,dead=10,kind='root'),
            dict(id='copy',tensor='cache',start=0,stop=2,tile='b',offset=8,bytes=4,born=1,ready=3,dead=5,kind='copy'),
            dict(id='out',tensor='out',start=0,stop=2,tile='b',offset=8,bytes=4,born=5,ready=6,dead=10,kind='output')],
        transfers=[dict(id='t0',source='a0',target='copy',start=0,stop=2,begin=1,end=3,bytes=4,route=['a','bridge','b'])],
        reads=[dict(buffer='copy',tile='b',at=4,start=0,stop=2),
               dict(buffer='b0',tile='b',at=7,start=4,stop=8),dict(buffer='out',tile='b',at=8,start=0,stop=2)])


def test_exact_fit_and_copy_charged_per_hop():
    result=audit(sample())
    assert result['peak_bytes']=={'a':8,'b':12,'bridge':0}
    assert result['allocated_copy_bytes']==4 and result['total_hop_bytes']==8
    assert result['last_read']=={'a0':3,'copy':4,'b0':7,'out':8}
    short=sample();short['capacities']['b']=11
    with pytest.raises(ValueError,match='capacity'):audit(short)


def test_live_output_and_temporary_are_both_charged():
    d=sample();d['capacities']['b']=16
    d['tensors']['temporary']=dict(elements=1,dtype='fp32',root=False)
    d['buffers'].append(dict(id='temp',tensor='temporary',start=0,stop=1,tile='b',offset=12,
                            bytes=4,born=6,ready=6,dead=8,kind='scratch'))
    d['reads'].append(dict(buffer='temp',tile='b',at=7,start=0,stop=1))
    assert audit(d)['peak_bytes']['b']==16
    d['capacities']['b']=15
    with pytest.raises(ValueError,match='capacity'):audit(d)


@pytest.mark.parametrize('mutation,reason',[
    ('early-free','freed before last reader'),('alias','alias'),('zero-copy','byte width'),
    ('no-transfer','partition'),('wrong-route','directed link'),('free-remote-read','remote read'),
    ('hole','partition'),('duplicate-transfer','duplicate transfer'),('copy-too-late','readiness'),
])
def test_invalid_physical_observations_refused(mutation,reason):
    d=deepcopy(sample())
    if mutation=='early-free':d['buffers'][1]['dead']=7
    elif mutation=='alias':d['buffers'][3].update(offset=0)
    elif mutation=='zero-copy':d['buffers'][2]['bytes']=0
    elif mutation=='no-transfer':d['transfers']=[]
    elif mutation=='wrong-route':d['transfers'][0]['route']=['a','b']
    elif mutation=='free-remote-read':d['reads'][0]['tile']='a'
    elif mutation=='hole':d['buffers'][1].update(start=5,bytes=6)
    elif mutation=='duplicate-transfer':d['transfers'].append(deepcopy(d['transfers'][0]))
    elif mutation=='copy-too-late':d['transfers'][0]['end']=4
    with pytest.raises(ValueError,match=reason):audit(d)
