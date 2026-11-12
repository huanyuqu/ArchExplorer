"""Hand timing, full-graph conservation and independent numeric shard oracles."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate, from_dict, find
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.mapping.context import MappingContext
from arch_explorer.mapping.directed import Copy, DirectedTransport
from arch_explorer.mapping.gsm import GsmTiled
from arch_explorer.simulator.events.builder import build_simulator
from arch_explorer.simulator.events.engine import simulate
from arch_explorer.components.base import Demand
from arch_explorer.simulator.events.backend import CapacityError, run_mapping
from storage_lifetime import assert_held_through_casts, storage_events
from arch_explorer.workload.llm import workload_from_spec
from tools.generate_multi_device_execution import configuration


def evaluate_config(c, **kwargs):
    return evaluate(from_dict(c['seed_adf']), WorkloadSpec(**c['workloads'][0]),
                    MappingSpec('multi_device', c['mapping_selectors']['multi_device']), **kwargs)


def transport_fixture(count=4, shared=False):
    c = configuration('operator', count)
    p = c['mapping_selectors']['multi_device']
    edges = [p['topology_edges'][0], p['topology_edges'][-2]] if count == 4 else p['topology_edges']
    if shared:
        edges[-1] = dict(edges[-1], network=edges[0]['network'])
    sim = build_simulator(from_dict(c['seed_adf']))
    sim.mapping = ctx = MappingContext(sim)
    states = [GsmTiled().device_state(sim, sim.container(path)) for path in p['device_paths']]
    net = DirectedTransport(ctx, edges, dict(zip(p['device_paths'], [s.dram for s in states])))
    return ctx, net, states, p['device_paths']


@pytest.mark.parametrize('shared', [False, True])
def test_link_sharing_and_latency_is_not_serialization(shared):
    ctx, net, states, paths = transport_fixture(shared=shared)
    outputs = []
    for src, dst in ((0, 1), (2, 3)):
        buf = ctx.storage('source', states[src].dram, 100)
        outputs.append(net.send(paths[src], paths[dst], Copy(buf, [buf], states[src].dram, 100)))
    ctx.sync('return', [v.ready for v in outputs]+[h for v in outputs for h in v.held])
    result = simulate(ctx.graph, 'consistent')
    # Distinct explicit non-GPU DRAM ports; fixed link propagation is separate.
    assert all(ctx.sim.points[s.dram].port.capacity == 64 for s in states)
    assert result.latency_ns == pytest.approx(200/64 + (8 if shared else 4) + 3)
    second = simulate(ctx.graph, 'conservative')
    assert [r.__dict__ for r in result.records] == [r.__dict__ for r in second.records]
    assert result.memory_peak == second.memory_peak


def test_multihop_reverse_rejected_and_propagation_releases_link():
    ctx, net, states, paths = transport_fixture()
    net.edges.append(dict(src=paths[1], dst=paths[2], network=net.edges[0]['network']))
    assert len(net.route(paths[0], paths[3])) == 3
    with pytest.raises(ValueError, match='No directed route'):
        net.route(paths[3], paths[0])
    first = ctx.storage('source0', states[0].dram, 25)
    v = net.send(paths[0], paths[1], Copy(first,[first],states[0].dram,25))
    # Second sender is released while first message is propagating.
    later = ctx.sync('release', [], latency_ns=2)
    second = ctx.storage('source2', states[2].dram, 25, [later])
    net.edges[-2] = dict(net.edges[-2], network=net.edges[0]['network'])
    w = net.send(paths[2], paths[3], Copy(second,[second],states[2].dram,25))
    ctx.sync('return',[v.ready,w.ready,*v.held,*w.held])
    result = simulate(ctx.graph)
    records = {r.task:r for r in result.records}
    one,two = [m['hops'][0] for m in net.messages]
    assert records[two['serialization_task']].start_ns < records[one['arrival_task']].end_ns


def test_compute_overlaps_transfer_and_fifo_has_arrival_order():
    ctx, net, states, paths = transport_fixture()
    model=net.networks[net.edges[0]['network']]
    # This is a component convention in the fixture, not a simulated delay.
    model.bus=replace(model.bus,arbitration='fifo')
    model.resources=[model.bus]
    first,arrive1=net.hop(net.edges[0],100,[],'one')
    second,arrive2=net.hop(net.edges[0],25,[],'two')
    kernel=ctx.sim.points[states[2].computes[0]]
    compute=ctx.task('independent compute',Demand({kernel.unit.name:6}))
    result=simulate(ctx.graph)
    records={r.task:r for r in result.records}
    assert records[first].start_ns==0 and records[first].release_ns==4
    assert records[second].start_ns==4 and records[second].release_ns==5
    assert records[arrive1].end_ns==7 and records[arrive2].end_ns==8
    assert records[compute].start_ns==0 and records[compute].end_ns==6


def test_rank_names_topology_and_precision_not_family_dispatched():
    c=configuration('operator',3,tp=True)
    before=evaluate_config(c)
    text=json.dumps(c)
    for i,name in enumerate(('green','six','unclassified')):
        text=text.replace('rank'+str(i),name)
    renamed=json.loads(text)
    after=evaluate_config(renamed)
    assert before.latency_ns==after.latency_ns
    assert after.area_mm2==before.area_mm2 and after.cost_usd==before.cost_usd
    # The newly selectable FIFO component has no invented area/cost model.
    for network in renamed['seed_adf']['networks']:
        network['component']='fifo_link'
    unknown=evaluate_config(renamed)
    assert unknown.area_mm2 is None and unknown.cost_usd is None


def test_initial_capacity_and_bad_tensor_shape_are_not_mapping_options():
    c=configuration('operator',2,tp=True)
    adf=from_dict(c['seed_adf'])
    p=find(adf,'system/rank0/dram')
    p.params['capacity']=replace(p.params['capacity'],value=1)
    c['seed_adf']=adf.to_dict()
    with pytest.raises(ValueError,match='capacity'):
        evaluate_config(c)


@pytest.mark.parametrize('kind', ['operator','vit','clip'])
def test_one_rank_exact_original_evaluation(kind):
    c=configuration(kind,1,**({'profile':'synthetic'} if kind in ('vit','clip') else {}))
    a=from_dict(c['seed_adf']); w=WorkloadSpec(**c['workloads'][0])
    p=c['mapping_selectors']['multi_device']
    local=next(iter(p['placements'].values()))['mapping']
    old=evaluate(a,w,MappingSpec(**local),trace=True).to_dict()
    new=evaluate_config(c,trace=True).to_dict()
    assert old == new


@pytest.mark.parametrize('kind,count', [('vit',2),('clip',4)])
@pytest.mark.parametrize('dtype', ['fp16', 'fp32'])
def test_full_graph_operators_and_terminal_results(kind,count,dtype):
    c=configuration(kind,count,profile='synthetic')
    c['workloads'][0]['params']['dtype'] = dtype
    def precision(node):
        if node.get('component') == 'systolic_array':
            node['params']['precision']['value'] = dtype
        for child in node.get('elements', []):
            precision(child)
    precision(c['seed_adf'])
    ir=workload_from_spec(WorkloadSpec(**c['workloads'][0]))
    e=evaluate_config(c,trace=True)
    report=e.details['multi_device']
    assert [o['operator'] for o in report['operators']] == [o.name for o in ir.operators]
    assert report['outputs'] == ir.meta['full_model']['outputs']
    assert len({o['rank'] for o in report['operators']}) == count
    assert report['messages']
    if dtype == 'fp16':
        sim, trace = run_mapping(from_dict(c['seed_adf']), WorkloadSpec(**c['workloads'][0]),
                                 MappingSpec('multi_device', c['mapping_selectors']['multi_device']))
        records, frees = storage_events(sim.mapping.graph, trace)
    for row in report['operators']:
        if row['op'] != 'matmul':
            continue
        op = next(op for op in ir.operators if op.name == row['operator'])
        tensor = ir.tensors[op.outputs[0]]
        elements = math.prod(tensor.shape)
        if tensor.dtype in ('fp16', 'bf16'):
            assert row['fp32_matrix_scratch_bytes'] == row['matrix_output_bytes'] == 4 * elements
            assert row['explicit_output_casts'] == elements and row['fused_output_casts'] == 0
            assert row['output_cast_read_bytes'] == 4 * elements and row['output_cast_write_bytes'] == 2 * elements
            assert_held_through_casts(sim.mapping.graph, records, frees,
                                     row['fp32_matrix_scratch_task'], row['output_cast_tasks'])
        else:
            assert row['fp32_matrix_scratch_bytes'] == 0 and row['explicit_output_casts'] == 0
    assert all(m['wire_bytes'] == m['logical_bytes']*len(m['hops']) for m in report['messages'])
    other=evaluate_config(c,scheduler='conservative',trace=True)
    # portions counts internal scheduler advance calls, not physical waits.
    assert [{k:v for k,v in r.items() if k != 'portions'} for r in e.details['trace']] == [
        {k:v for k,v in r.items() if k != 'portions'} for r in other.details['trace']]
    assert e.details['stats']['memory_peak'] == other.details['stats']['memory_peak']


@pytest.mark.parametrize('count',[2,4])
def test_nondivisible_k_tp_numeric_and_work_conservation(count):
    c=configuration('operator',count,tp=True)
    e=evaluate_config(c,trace=True)
    p=c['mapping_selectors']['multi_device']
    ranges=p['shards'][0]['ranges']
    # Independent arithmetic: distinct partials, no reuse of mapping kernels.
    rng=np.random.default_rng(12)
    a=rng.normal(size=(8,5)).astype(np.float32); b=rng.normal(size=(5,8)).astype(np.float32)
    partials=[a[:,r['start']:r['stop']]@b[r['start']:r['stop']] for r in ranges]
    np.testing.assert_allclose(sum(partials),a@b,rtol=1e-5,atol=1e-6)
    report=e.details['multi_device']
    assert sum(o['flops'] for o in report['operators']) == 2*8*5*8
    assert [o['shape'][2] for o in report['operators']] == [r['stop']-r['start'] for r in ranges]
    collective=report['collectives'][0]
    assert collective['reduction_additions'] == (count-1)*64
    assert collective['dtype']=='fp32'
    for row in report['operators']:
        assert row['partial'] and row['output_bytes'] == row['matrix_output_bytes'] == 64 * 4
        assert row['explicit_output_casts'] == row['fused_output_casts'] == 0
    assert not any('.partial_store' in t['name'] for t in e.details['trace'])
    assert len(report['messages'])==2*(count-1)
    assert all(x['logical_bytes']==64*4 for x in report['messages'])
    assert all(v['bytes']==64*2 for v in report['output_placements'])


def collective_configuration(operation, count=4, elements=7):
    c=configuration('operator',count)
    c['workloads']=[dict(kind='operator',params=dict(op='allreduce',bytes=4*elements,dtype='fp32'))]
    p=c['mapping_selectors']['multi_device']
    local=next(iter(p['placements'].values()))['mapping']
    p['placements']={'*':dict(ranks=list(range(count)),mapping=local)}
    p['collectives']=[dict(after='allreduce',tensor='out',operation=operation,
          algorithm={'allreduce':'tree','allgather':'all_pairs','reduce_scatter':'direct'}[operation],
          ranks=list(range(count)),dtype='fp32',elements=elements)]
    return c


@pytest.mark.parametrize('operation',['allreduce','allgather','reduce_scatter'])
def test_collective_contributions_independent_numpy_oracle(operation):
    c=collective_configuration(operation)
    report=evaluate_config(c).details['multi_device']
    ledger=report['collectives'][0]
    source=np.arange(28,dtype=np.float32).reshape(4,7)
    if operation=='allreduce':
        actual=source[0].copy()
        for row in ledger['contributions']:
            actual += source[row['src']]
        np.testing.assert_array_equal(actual,source.sum(axis=0))
        assert ledger['reduction_additions']==21
    elif operation=='allgather':
        for rank in range(4):
            actual=np.full(28,np.nan)
            rows=[r for r in ledger['contributions'] if r['dst']==rank]
            assert sorted(r['src'] for r in rows)==list(range(4))
            for r in rows:actual[r['start']:r['stop']]=source[r['src']]
            np.testing.assert_array_equal(actual,source.ravel())
        assert ledger['reduction_additions']==0
    else:
        for rank in range(4):
            start,stop=ledger['output_ranges'][str(rank)]
            actual=source[rank,start:stop].copy()
            rows=[r for r in ledger['contributions'] if r['dst']==rank]
            assert sorted([rank]+[r['src'] for r in rows])==list(range(4))
            for r in rows:actual += source[r['src'],r['start']:r['stop']]
            np.testing.assert_array_equal(actual,source.sum(axis=0)[start:stop])
        assert ledger['reduction_additions']==21
    assert ledger['padding_elements']==0


def test_buffer_capacity_cannot_be_disabled():
    c=collective_configuration('allreduce',2,16)
    adf=from_dict(c['seed_adf'])
    for rank in range(2):
        point=find(adf,f'system/rank{rank}/dram')
        point.params['capacity']=replace(point.params['capacity'],value=128)
    c['seed_adf']=adf.to_dict()
    with pytest.raises(CapacityError,match='capacity exceeded'):
        evaluate_config(c,check_capacity=False)


@pytest.mark.parametrize('change',[
    lambda p:p['shards'][0].update(size=6),
    lambda p:p['collectives'][0].update(elements=65),
    lambda p:p['placements']['matmul']['mapping']['params'].update(tile_m=3),
    lambda p:p.update(topology_edges=[]),
    lambda p:p['placements'].update(typo={'ranks':[0],'mapping':{'template':'gsm_tiled','params':{}}}),
])
def test_physical_request_validation_before_execution(change):
    c=configuration('operator',2,tp=True)
    change(c['mapping_selectors']['multi_device'])
    with pytest.raises(ValueError):evaluate_config(c)


def test_real_env_two_candidates_budget_submit_and_cli_replay(tmp_path):
    c=configuration('operator',2,tp=True)
    c['trace_path']=str(tmp_path/'tools.jsonl')
    env=ExplorationEnv(c)
    first=env.evaluate()
    assert 'error' not in first,first
    new=deepcopy(c['mapping_selectors']['multi_device'])
    new['placements']['matmul']['mapping']['params']['tile_n']=128
    response=env.set_mappings(mappings=[dict(template='multi_device',params=new)])
    assert 'error' not in response,response
    second=env.evaluate()
    assert 'error' not in second,second
    assert env.used==2
    winner=tmp_path/'winner.json'
    result=env.submit(path=str(winner))
    assert 'error' not in result,result
    run=subprocess.run([sys.executable,'-m','arch_explorer','evaluate',str(winner)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    replay=subprocess.run([sys.executable,'-m','arch_explorer','env','replay',str(env.trace_path),
                           '--output',str(tmp_path/'replayed.json')],capture_output=True,text=True)
    assert replay.returncode==0,replay.stderr
