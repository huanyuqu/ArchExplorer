"""Hand-derived command boundaries, not fits to an upstream latency table."""
import dataclasses
import pytest

from arch_explorer.components.pim_commands import (RankRefresh, combined_times, rank_activations,
                                                  softmax_cycles, write_response)
from arch_explorer.simulator.events.engine import Task, TaskGraph, simulate
from arch_explorer.adf import MappingSpec, Param, WorkloadSpec, evaluate
from arch_explorer.adf.model import walk
from arch_explorer.devices.pim import neupims


def test_independent_rank_windows_and_active_refresh():
    times = rank_activations(32,4,2,6,30)
    assert times == [(0,0),(1,1),(30,0),(31,1),(60,0),(61,1),(90,0),(91,1)]
    ref = RankRefresh(260,3900,2)
    # Rank1 refresh began at 27300. Rank0 can already activate at 27463.
    times = rank_activations(32,4,2,6,30,start=27463,available=ref.available)
    assert times[-1] == (27650,1)
    assert times[0] == (27463,0)
    with pytest.raises(ValueError):
        rank_activations(12,4,2,6,30)


@pytest.mark.parametrize('scheduler',['consistent','conservative'])
def test_absolute_calendar_continues_across_operations(scheduler):
    refresh = RankRefresh(260,3900,2)
    graph = TaskGraph()
    first = graph.add(Task('first operation',latency_ns=1900))
    gate = graph.add(Task('next operation admission',end_time=lambda t:refresh.admit(t,100)),[first])
    active = graph.add(Task('rank0 wait',end_time=lambda t:refresh.available(t,0)),[gate])
    result = simulate(graph,scheduler)
    assert result.records[gate].end_ns == 1951
    assert result.records[active].end_ns == 2210
    assert refresh.available(100,0)==100  # no fictitious refresh at zero
    assert refresh.available(3900,1)==4160
    assert refresh.available(3900,0)==3900
    with pytest.raises(ValueError):
        refresh.admit(0,1950)


def test_return_is_not_bank_ready_or_write_drain():
    b = combined_times(100,32,2,14,2)
    assert b == {'issue':100,'returned':116,'bank_ready':166,'consumer_ready':117}
    assert combined_times(100,8,2,14,2)['bank_ready']==118
    assert combined_times(100,1,2,14,2)['bank_ready']==114 # six-command pipeline fill
    assert write_response(50,100,1,True)==51
    assert write_response(50,100,1)==100
    assert softmax_cycles(1536*2,128)==122 # 2*(24+1)+3*24


def _evaluate(batch=1, seq=512, changes=None):
    arch = neupims(channels=2,banks=32)
    for _,node in walk(arch):
        if getattr(node,'component',None)=='pim_unit':
            for key,value in (changes or {}).items():
                node.params[key]=dataclasses.replace(node.params[key],value=value)
    spec=WorkloadSpec('transformer',dict(model='gpt3-6.7b',phase='decode',batch=batch,seq_len=seq,layers=1))
    return evaluate(arch,spec,MappingSpec('pim_offload',{'tp':4}),trace=True,cost_model=False)


def test_head_pipeline_counts_and_dependency_boundaries():
    result = _evaluate(seq=1536)
    c=result.details['pim_commands'][0]['counts']
    assert c['gwrite']==2+24
    assert c['headers']==96+96
    assert c['combined']==384+96
    assert c['comp']==3072+3072
    assert c['gact']==768+768
    assert c['precharge']==3040+3072
    assert c['return_bytes']==24576+6144
    assert c['softmax']==8 and c['partial_add']==32
    assert c['write_bytes']==25088+10240
    rows=result.details['trace']
    qk=[r for r in rows if r['operator']=='l0.attn_qk' and '.consumer.' in r['name']]
    soft=[r for r in rows if '.softmax.' in r['name'] and r['kind']=='compute']
    assert min(r['start_ns'] for r in soft)<max(r['end_ns'] for r in qk)
    av=[r for r in rows if r['operator']=='l0.attn_av' and r['kind']=='compute']
    assert min(r['start_ns'] for r in av)>=max(r['end_ns'] for r in soft)
    adds=[r for r in av if '.partial-add.' in r['name']]
    assert len(adds)==32
    assert all(r['end_ns']-r['start_ns']>=2 for r in adds)


def test_pipeline_off_still_waits_for_all_qk():
    result=_evaluate(changes={'attention_pipeline':'off'})
    rows=result.details['trace']
    qk_end=max(r['end_ns'] for r in rows if r['operator']=='l0.attn_qk' and '.consumer.' in r['name'])
    assert all(r['start_ns']>=qk_end for r in rows
               if r['operator']=='l0.softmax' and r['kind']=='compute')


def test_invalid_calendar_cannot_move_backwards_or_hold_resources():
    with pytest.raises(ValueError):
        Task('invalid',demand={'unit':1},end_time=lambda t:t)
    graph=TaskGraph(); graph.add(Task('backwards',end_time=lambda t:t-1))
    with pytest.raises(ValueError): simulate(graph)
