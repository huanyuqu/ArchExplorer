import pytest
from arch_explorer.adf import evaluate, MappingSpec, WorkloadSpec
from arch_explorer.devices.pim import neupims
from arch_explorer.simulator.events.neupims_memory import AddressStream
from arch_explorer.components.neupims_transactions import switch_column_channel


def small_workload():
    model=dict(name='synthetic',hidden=128,heads=1,kv_heads=1,head_dim=128,intermediate=128,
               layers=1,vocab=128,gated=False,norm='layernorm',activation='gelu',source='synthetic test')
    return WorkloadSpec('transformer',dict(model=model,phase='decode',batch=2,seq_len=32,layers=1))


def test_address_stream_matches_independent_element_enumeration():
    stream=AddressStream(64,4);cursor=0
    for count in (1,31,32,1023,14000,1):
        expected={}
        for _ in range(count):
            cursor=(cursor+2)%stream.limit
            expected.setdefault(switch_column_channel(cursor)&~63,None)
        assert stream.bursts(count)==tuple(expected)
        assert stream.cursor==cursor


@pytest.mark.parametrize('width',[64,128])
def test_public_instruction_program(width):
    e=evaluate(neupims(width),small_workload(),MappingSpec('pim_offload',{'npu_execution':'instructions'}),
               cost_model=False,trace=True)
    d=e.details['neupims_instructions']
    assert e.latency_ns>0 and len(d['operations'])==4
    assert len([r for r in e.breakdown if r['resource']=='npu-instruction-core'])==4
    assert {r['label'] for r in e.breakdown}=={'LayerNorm+QKV','PIM attention','Projection+FFN'}
    assert d['physical_bytes'][0]>=d['logical_bytes'][0]
    assert d['core_count']==1
