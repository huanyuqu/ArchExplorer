"""Installed compilation behavior, independent storage sweep and source refusal."""
from copy import deepcopy
import hashlib
import json

import pytest

from test_public_compilation_contract import (
    compiler_api, prohibit_execution, roofline_input, two_reader_graph_input,
)
from compilation_expectations import reference_matmul


def events_input(*, capacity=2**20):
    from arch_explorer.adf import Param
    from arch_explorer.components.api import ImplementationBinding
    adf, _, _, mapping = roofline_input(capacity=capacity)
    source = 'explicit synthetic QA rate, not calibrated hardware'
    adf.elements[0].implementation = ImplementationBinding('sm.events', '1', {
        'clock_ghz': Param(1, 'GHz', source), 'lanes': Param(8, 'count', source),
        'sfu_lanes': Param(1, 'count', source)})
    graph, spec = two_reader_graph_input()
    return adf, graph, spec, mapping


def independently_sweep_storage(compilation, result):
    """Use declared residency plus real readers, not scheduler free counters."""
    trace = result['evaluation']['details']['trace']
    finish = result['evaluation']['latency_ns']
    compute = {r['operator']: r for r in trace if r['kind'] == 'compute'}
    storage = {r['name']: r for r in trace if r['kind'] == 'storage'}
    declared = {t['name']:t for t in compilation['program']['tensors']}
    actions, intervals = [], []
    for tensor in compilation['plan']['storage']:
        start = storage['board:' + tensor['tensor']]['start_ns']
        readers = tensor['consumers']
        last_reader = max(compute[r]['end_ns'] for r in readers) if readers else finish
        persistent = (tensor['producer'] is None and declared[tensor['tensor']]['kind'] in ('weight','kv'))
        end = finish if persistent else last_reader
        assert start <= end
        if tensor['producer']:
            producer_end = compute[tensor['producer']]['end_ns']
            assert all(compute[r]['start_ns'] >= producer_end for r in readers)
        intervals.append(dict(tensor=tensor['tensor'], bytes=tensor['bytes'],
                              allocation=start, last_reader=last_reader, release=end,
                              persistent=persistent, readers=readers))
        # Existing memory semantics release before allocate at the same instant.
        actions.extend([(start, 1, tensor['bytes']), (end, 0, -tensor['bytes'])])
    current = peak = 0
    for _, _, change in sorted(actions):
        current += change
        peak = max(peak, current)
    assert current == 0
    return peak, intervals


def test_compiled_two_readers_execute_with_last_reader_peak_and_readonly_replay(compiler_api, tmp_path):
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import execute_reference
    w, s = compiler_api
    adf, graph, spec, mapping = events_input()
    compilation = w.compile_workload(adf, spec, mapping)
    values = graph.meta['inputs']
    t = reference_matmul(values['A'][0], values['B'][0])
    expected = {name: [reference_matmul(t, values[weight][0])]
                for name, weight in [('Y', 'D'), ('Z', 'E')]}
    actual = execute_reference(WorkloadSpec('graph', compilation['program']), inputs=values)
    assert {name: actual['outputs'][name].tolist() for name in expected} == expected
    request = s.prepare_compiled(compilation, output_dir=tmp_path/'run', options={'trace': True})
    result = s.evaluate_request(request)
    peak, intervals = independently_sweep_storage(compilation, result)
    assert peak == result['evaluation']['details']['stats']['memory_peak']['board/dram']
    intermediate = next(r for r in intervals if r['tensor'] == 'T')
    readers = [r for r in result['evaluation']['details']['trace']
               if r['kind'] == 'compute' and r['operator'] in intermediate['readers']]
    assert len(readers) >= 2
    assert intermediate['last_reader'] == max(r['end_ns'] for r in readers)
    assert intermediate['last_reader'] > min(r['end_ns'] for r in readers)
    before = {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in (tmp_path/'run').rglob('*') if p.is_file()}
    assert s.evaluate_request(result) == result
    after = {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
             for p in (tmp_path/'run').rglob('*') if p.is_file()}
    assert before == after
    (tmp_path/'LIFETIME.json').write_text(json.dumps(dict(
        peak_bytes=peak, intervals=intervals, returned_outputs=expected), indent=2))


def test_compiled_impossible_capacity_refuses_without_execution(compiler_api, monkeypatch, tmp_path):
    w, s = compiler_api
    adf, _, spec, mapping = events_input(capacity=1)
    with prohibit_execution(monkeypatch, s):
        with pytest.raises((ValueError, NotImplementedError), match='(?i)capacity|allocation|storage'):
            compilation = w.compile_workload(adf, spec, mapping)
            s.prepare_compiled(compilation, output_dir=tmp_path/'impossible')


@pytest.mark.parametrize('mutation', ['evaluation', 'mapping', 'adapter'])
def test_compiled_saved_result_tamper_is_rejected(compiler_api, tmp_path, mutation):
    w, s = compiler_api
    adf, _, spec, mapping = roofline_input()
    compilation = w.compile_workload(adf, spec, mapping)
    result = s.evaluate_request(s.prepare_compiled(compilation, output_dir=tmp_path/'run'))
    changed = deepcopy(result)
    if mutation == 'evaluation':
        changed['evaluation']['latency_ns'] += 1
    elif mutation == 'mapping':
        changed['request']['compilation']['source']['mapping']['params']['rules']['matmul']['options']['tiles']['k'] += 1
    else:
        changed['request']['compilation']['bindings']['board/sm']['id'] = 'events'
    assert not s.assess_request(changed)['supported']
    with pytest.raises(ValueError):
        s.evaluate_request(changed)
