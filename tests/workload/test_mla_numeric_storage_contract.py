"""Actual candidate-only Events execution and physical-root lifetime checks.

These consumers are deliberately not run against a hand-composed source tree.
The source stage records them as deferred until the lead's unique candidate.
"""
from collections import defaultdict
from contextlib import contextmanager
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re

import pytest
from mla_contract_cases import SCENARIOS, compare_stages, frozen_inputs, mla_api, scene
from mla_independent_expectations import RootLifetime, physical_peak
from mla_multicast_expectations import hierarchical_transfer
from multilayer_contract_cases import observe_memory
from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import independent_layouts


@contextmanager
def observe_taskgraphs(monkeypatch):
    from arch_explorer.simulator.events.engine import ConsistentScheduler
    graphs=[];original=ConsistentScheduler.__init__
    def initialized(self,graph,*args,**options):
        answer=original(self,graph,*args,**options)
        graphs.append(graph);return answer
    with monkeypatch.context() as patch:
        patch.setattr(ConsistentScheduler,'__init__',initialized)
        yield graphs


def audit_transfer_demands(adf,compilation,result,graphs):
    """Actual task work vs independent integer tree paths and memory ports."""
    assert len(graphs)==1,'one v4 global component graph expected'
    tasks=graphs[0].tasks
    tasks={task.id:task for task in (tasks.values() if isinstance(tasks,dict) else tasks)}
    operators={op['name']:op for op in compilation['program']['operators']}
    tensors={tensor['name']:tensor for tensor in compilation['program']['tensors']}
    facts=[]
    for transfer in result['evaluation']['details']['compilation']['transfers']:
        op=operators[transfer['operator']];assert op['op']=='storage.copy'
        source=tensors[op['inputs'][0]]
        size=math.prod(source['shape'])*4
        assert transfer['bytes']==size
        assert transfer['source'] not in transfer['destinations']
        expected=hierarchical_transfer(adf.to_dict(),transfer['source'],transfer['destinations'],size,
                                       multicast=transfer['multicast'])
        jobs=[tasks[ident] for ident in transfer['tasks']]
        assert all(job.kind=='comm' for job in jobs)
        actual=defaultdict(float)
        for job in jobs:
            for name,amount in job.demand.items():actual[name]+=amount
        assert dict(actual)==expected['demand_bytes'],(transfer,dict(actual),expected['demand_bytes'])
        expected_tails=([float(expected['startup_tail_ns'])] if transfer['multicast'] else
                        [float(hierarchical_transfer(adf.to_dict(),transfer['source'],[dest],size,
                            multicast=False)['startup_tail_ns']) for dest in transfer['destinations']])
        assert sorted(job.latency_ns for job in jobs)==sorted(expected_tails)
        facts.append(dict(operator=transfer['operator'],actual_task_ids=transfer['tasks'],
                          demand=dict(actual),startup_tails_ns=expected_tails))
    assert facts
    return facts


def audit_access(compilation):
    """Independent scalar-FMA count and per-input byte conservation."""
    records = []
    for step in compilation['plan']['steps']:
        access = step['access']
        assert all(value >= 0 for value in access.get('read_by_tensor',{}).values())
        assert sum(access.get('read_by_tensor',{}).values()) == access['read_bytes']
        if step['semantic'] == 'compiled.matmul':
            attrs = step['address_map']['indexing']['attrs']
            fmas = attrs['batch']*attrs['rows']*attrs['cols']*attrs['k']
            inputs = step['address_map']['reads']
            assert len(inputs) == 2
            expected = {}
            for descriptor in inputs:
                assert descriptor['dtype'] == 'fp32'
                assert descriptor['units'] == 'elements'
                expected[descriptor['tensor']] = expected.get(descriptor['tensor'],0)+4*fmas
            assert access['read_by_tensor'] == expected
            assert access['operations'] == 2*fmas
            assert access['write_bytes'] == 4*attrs['batch']*attrs['rows']*attrs['cols']
            records.append(dict(step=step['name'],fmas=fmas,read_by_tensor=expected))
        elif step['semantic'] == 'storage.view':
            assert access['read_bytes'] == access['write_bytes'] == access['operations'] == 0
    assert records,'MLA must include charged scalar-FMA matmuls'
    return records


def audit_physical(compilation,result,observed):
    program,plan = compilation['program'],compilation['plan']
    layouts = independent_layouts(program)
    roots = {value.root.name:value.root for value in layouts.values()}
    allocations = {row['storage']:row for row in plan['allocations']}
    assert set(allocations) == set(roots)
    locations = program['meta']['storage_plan']['locations']
    assert set(locations) == set(roots)
    finish = result['evaluation']['latency_ns']
    trace = result['evaluation']['details']['trace']
    by_memory = defaultdict(list)
    for name,root in roots.items():
        declared = allocations[name]
        assert declared['bytes'] == root.bytes
        assert declared['memories'] == [locations[name]]
        aliases = {n for n,value in layouts.items() if value.root.name == name}
        readers = {op['name'] for op in program['operators'] if aliases.intersection(op['inputs'])}
        writes = {op['name'] for op in program['operators'] if aliases.intersection(op['outputs'])}
        active = [r for r in trace if r['operator'] in readers|writes and r['kind'] != 'storage']
        events = [event for event in observed if event['name'].endswith(':'+name)]
        allocated = [event for event in events if event['action'] == 'allocate']
        assert len(allocated) == 1,(name,events)
        allocation = allocated[0]
        assert allocation['bytes'] == root.bytes
        frees = [event['time'] for event in events if event['action'] == 'free']
        returned = bool(declared['returned'])
        if returned or root.resident:
            assert all(time >= finish for time in frees),(name,frees,finish)
            release = finish
        else:
            assert len(frees) == 1,(name,frees)
            release = frees[0]
            assert release >= max((r['end_ns'] for r in active),default=allocation['time'])
        # A zero-duration control root has no resident interval to integrate.
        if release > allocation['time']:
            by_memory[allocation['owner']].append(RootLifetime(
                name,root.bytes,Fraction(allocation['time']),Fraction(release),
                tuple((r['operator'],Fraction(r['start_ns']),Fraction(r['end_ns'])) for r in active),
                Fraction(finish) if returned else None))
    peaks = {memory:physical_peak(rows)['peak_bytes'] for memory,rows in by_memory.items()}
    reported = result['evaluation']['details']['stats']['memory_peak']
    for memory,peak in peaks.items():
        assert reported[memory] == peak,(memory,reported[memory],peak)
        physical_peak(by_memory[memory],capacity=peak)
        with pytest.raises(MemoryError):physical_peak(by_memory[memory],capacity=peak-1)
    return dict(peaks=peaks,actual_free_observed=True,
                capacity_note='Independent interval bound; actual engine just-fit/one-byte-short consumes separate mutated ADF cases.')


@pytest.mark.parametrize('key',SCENARIOS)
@pytest.mark.parametrize('policy',['materialized','views'])
def test_actual_v4_components_and_physical_roots(mla_api,monkeypatch,tmp_path,key,policy):
    from arch_explorer.adf import MappingSpec
    w,s = mla_api;adf,spec,mapping = scene(key)
    candidates = w.layout_candidates(spec,mapping)
    selected = candidates[policy]
    if isinstance(selected,dict):selected = MappingSpec(**selected)
    compiled = w.compile_workload(adf,spec,selected)
    serialized = compiled.to_dict() if hasattr(compiled,'to_dict') else compiled
    access_facts = audit_access(serialized)
    request = s.prepare_compiled(compiled,output_dir=str(tmp_path/'run'),options={'trace':True})
    with observe_memory(monkeypatch) as observed,observe_taskgraphs(monkeypatch) as graphs:
        result = s.evaluate_request(request)
    facts = audit_physical(serialized,result,observed)
    facts['independent_actual_transfer_demands']=audit_transfer_demands(adf,serialized,result,graphs)
    transfers = result['evaluation']['details']['compilation']['transfers']
    assert transfers
    for transfer in transfers:
        assert transfer['bytes'] > 0
        assert transfer['multicast'] == ('multicast' in key)
        assert transfer['source'] != transfer['destinations']
        assert transfer['tasks']
    with no_backend(monkeypatch):
        assert s.evaluate_request(result) == result
    # CPU numerical reference is a distinct claim; expected values are scalar QA.
    facts['CPU_reference'] = compare_stages(w.execute_reference(spec),spec.params)
    facts.update(scenario=key,layout=policy,component_result=True,
                 independent_scalar_FMA_access=access_facts,
                 latency_ns=result['evaluation']['latency_ns'],
                 transfer_records=transfers,assume_peak_or_latency_reduction=False)
    (tmp_path/'FACTS.json').write_text(json.dumps(facts,indent=2)+'\n')
    (tmp_path/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


def test_actual_engine_capacity_just_fit_and_one_byte_short(mla_api,tmp_path):
    """Consume previously observed QA root intervals; change only capacities."""
    import hashlib
    from arch_explorer.adf import Param,find
    w,s=mla_api
    source=Path(os.environ['VAL61_PHYSICAL_FACTS']);raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==os.environ['VAL61_PHYSICAL_FACTS_SHA256']
    facts=json.loads(raw);assert facts['actual_free_observed']
    key,policy=facts['scenario'],facts['layout']
    adf,spec,mapping=scene(key);mapping=w.layout_candidates(spec,mapping)[policy]
    peaks=facts['peaks'];assert peaks
    capacities={}
    for memory,peak in peaks.items():
        assert peak==int(peak) and peak>0
        # A repeated ADF point has one parameter shared by its actual instances.
        # Its minimum representable capacity is the maximum of their peaks.
        type_path=re.sub(r'\[\d+\]','',memory)
        capacities[type_path]=max(capacities.get(type_path,0),int(peak))
    for memory,peak in capacities.items():
        find(adf,memory).params['capacity']=Param(peak,'B','QA independent actual root-interval maximum by ADF type')
    compiled=w.compile_workload(adf,spec,mapping)
    result=s.evaluate_request(s.prepare_compiled(compiled,output_dir=str(tmp_path/'just-fit'),options={'trace':True}))
    assert all(result['evaluation']['details']['stats']['memory_peak'][memory]<=peak for memory,peak in peaks.items())
    selected=next(memory for memory in capacities if memory.endswith('/sram'))
    find(adf,selected).params['capacity']=Param(capacities[selected]-1,'B','QA independent type maximum minus exactly one byte')
    with pytest.raises(Exception) as failure:
        failed=w.compile_workload(adf,spec,mapping)
        s.evaluate_request(s.prepare_compiled(failed,output_dir=str(tmp_path/'one-byte-short'),options={'trace':True}))
    diagnostic=str(failure.value)
    assert 'capacity' in diagnostic.lower() and str(capacities[selected]-1) in diagnostic
    (tmp_path/'CAPACITY.json').write_text(json.dumps(dict(source_facts_sha256=hashlib.sha256(raw).hexdigest(),
        just_fit_passed=True,one_byte_short_rejected=True,memory=selected,expected_peak=capacities[selected],
        per_instance_peaks=peaks,representable_type_capacities=capacities,
        error_type=type(failure.value).__name__,diagnostic=diagnostic),indent=2)+'\n')
