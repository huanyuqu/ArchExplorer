"""Source/hand-derived clocks and byte accounting; no accuracy fitting targets."""
import copy
import json

import pytest

from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate, from_dict
from arch_explorer.components.aim_profile import AimCommand as C, AimProfile, PROFILE
from arch_explorer.components.aim_controller import AimController
from arch_explorer.devices import get
from arch_explorer.mapping.search import mapping_space, best_mapping
from arch_explorer.simulator.events.engine import Task, TaskGraph, simulate
from arch_explorer.simulator.events.protocol import ExecutionRequest as R

SYNTHETIC = '[synthetic] 105ns = original static210 at its stated0.5ns; fixed service demonstration, not runtime/device nRFC'


def P(**options):
    return AimProfile(refresh_service_ns=105., refresh_source=SYNTHETIC, **options)


def detail(channels=1):
    return get('hsndi_detail', cores=1, channels=channels,
               refresh_service_ns=105., refresh_source=SYNTHETIC)


def run(commands, profile=None):
    model = AimController(profile or P())
    assert model.submit(R('x', 'channel', operation='stream', payload=tuple(commands)), 0)
    events = drain(model)
    return model, events


def drain(model):
    events = []
    for _ in range(20000):
        now = model.next_event_time()
        if now is None:
            return events
        events.extend(model.advance(now))
    raise AssertionError('controller failed to drain')


def issues(model):
    return [(r['command'], r['tick']) for r in model.trace if r['event'] == 'issue']


def test_row_hit_miss_and_return_recovery_are_distinct():
    p = P()
    model, events = run([C('RD'), C('RD', 32), C('RD', p.encode(0, 1))])
    assert issues(model) == [('ACT', 0), ('RD', 36), ('RD', 38), ('PRE', 54), ('ACT', 89), ('RD', 125)]
    assert [r['tick'] for r in model.trace if r['event'] == 'return'] == [88, 90, 177]
    assert next(e.time for e in events if e.kind == 'consumer_ready') == 177
    assert model.bytes['external_read'] == 96
    assert model.rows[0] == 1 and all(r is None for r in model.rows[1:])


def test_act16_register_modes_and_hand_gemv():
    model, events = run([C('WRMAC16'), C('WRGB'), C('MAC16'), C('RDMAC16')])
    assert issues(model) == [('TMOD', 0), ('WRMAC16', 32), ('WRGB', 34), ('TMOD', 35),
                             ('ACT16', 67), ('MAC16', 123), ('TMOD', 125), ('RDMAC16', 157)]
    assert model.rows == [0]*16
    assert model.bytes == dict(external_read=32, external_write=64, bank_read=512,
                               bank_write=0, useful_external=96, mac_operations=512)
    assert next(e.time for e in events if e.kind == 'consumer_ready') == 159


def test_write_turnaround_and_single_row_conflict():
    p = P()
    model, _ = run([C('WR'), C('MAC16', p.encode(0, 1)), C('RD', p.encode(0, 2))])
    issued = issues(model)
    assert ('WR', 28) in issued
    # WR recover = CWL6 + BL2 + WR33; single-row PREA cannot precede 69.
    assert next(t for cmd, t in issued if cmd == 'PREA') >= 69
    assert [cmd for cmd, _ in issued].count('ACT16') == 1
    assert model.bytes['bank_read'] == 544
    windows = [(r['bus_start_tick'], r['bus_end_tick']) for r in model.trace if 'bus_start_tick' in r]
    assert all(b <= c or d <= a for i, (a, b) in enumerate(windows) for c, d in windows[i+1:])


def test_refresh_due_with_response_same_tick_and_postponed_active():
    # Effective clock: start7712 + ACT/RD36 + response52 = tREFI7800ns.
    model = AimController(P())
    assert model.submit(R('x', 'c', operation='RD', payload=C('RD')), 7712)
    events = drain(model)
    trace = model.trace
    at88 = [r['event'] for r in trace if r['tick'] == 7800]
    assert at88[:3] == ['data_complete', 'return', 'refresh_due']
    assert ('PREA', 7800) in issues(model)
    assert ('REFab', 7832) in issues(model)
    assert model.tick == 7832 + 105
    delayed = AimController(P())
    assert delayed.submit(R('x', 'c', operation='RD', payload=C('RD')), 7787)
    drain(delayed)
    assert ('RD', 7823) in issues(delayed)  # active ordinary read has priority
    assert ('PREA', 7841) in issues(delayed)
    assert ('REFab', 7876) in issues(delayed)


def test_queue_full_is_pure_and_retry_has_no_duplicate_bytes():
    model = AimController(P(queue_size=1))
    request = R('a', 'c', operation='RD', payload=C('RD'))
    assert model.submit(request, 0)
    snapshot = copy.deepcopy(model.__dict__)
    second = R('b', 'c', operation='WR', payload=C('WR', 32))
    assert model.submit(second, 0) is False
    assert model.__dict__ == snapshot
    drain(model)
    assert model.submit(second, model.tick)
    drain(model)
    assert model.bytes['external_read'] == model.bytes['external_write'] == 32
    assert len([r for r in model.trace if r['event'] == 'return']) == 2


def test_state_persists_across_operators_but_factory_resets():
    created = []
    graph = TaskGraph()
    def factory():
        model = AimController(P())
        created.append(model)
        return model
    graph.bind_execution('c', factory, resources=('bank',))
    a = graph.add(Task('a', operator='first', execution=R('a', 'c', operation='RD', payload=C('RD'))))
    graph.add(Task('b', operator='second', execution=R('b', 'c', operation='RD', payload=C('RD', 32))), [a])
    x, y = (simulate(graph, 'conservative') for _ in range(2))
    assert x.latency_ns == y.latency_ns
    assert created[0] is not created[1]
    assert issues(created[0]) == issues(created[1]) == [('ACT', 0), ('RD', 36), ('RD', 89)]


@pytest.mark.parametrize('channels,m,k', [(1,16,16), (2,35,17)])
def test_public_mapping_discovery_replay_capacity_bytes(channels, m, k):
    a = detail(channels)
    w = WorkloadSpec('operator', dict(op='mvm', m=m, k=k, batch=2, dtype='bf16'))
    candidates = mapping_space(a, w, 'aim_shared')
    assert len(candidates) == 2
    selected = next(x for x in candidates if x.params['host_traffic'] == 'synthetic_rw')
    result = evaluate(a, w, selected, trace=True)
    stored = json.loads(json.dumps(dict(adf=a.to_dict(), workload=w.to_dict(), mapping=selected.to_dict())))
    again = evaluate(from_dict(stored['adf']), WorkloadSpec(**stored['workload']),
                     MappingSpec(**stored['mapping']), trace=True)
    assert result.to_dict() == again.to_dict()
    json.dumps(result.to_dict())
    report = result.details['aim']
    assert len(report['controllers']) == channels
    assert report['logical_operations'] == 4*m*k
    assert sum(c['bytes']['mac_operations'] for c in report['controllers'].values()) == report['padded_operations']
    for channel in report['controllers'].values():
        assert channel['refresh_service_ns'] == channel['refresh_effective_ns'] == 105
        assert SYNTHETIC in channel['refresh_source']
        parameters = {row['name']: row for row in channel['parameters']}
        assert parameters['nRFC']['effective_value'] is None
        assert parameters['nRFC']['defined'] is False
        assert parameters['nRFC']['category'] == 'caller_configuration'
        assert parameters['nREFI']['index'] == 24 and parameters['nREFI']['static_literal'] == 105
        assert parameters['nREFI']['selected_value'] == 7800
        assert parameters['nRFCpb']['index'] == 26 and parameters['nRFCpb']['static_literal'] == 3333
        assert parameters['nRFCpb']['exclusion']
        counts = channel['requests']
        assert counts['logical'] == counts['accepted'] == counts['returned'] == counts['physical_completed']
        assert counts['issued'] + counts['forwarded'] == counts['logical']
        assert counts['pending'] == 0
        assert channel['bytes']['external_write'] > 0
        assert channel['rows_reserved'] * 32768 <= channel['capacity_bytes']
        assert any(r.get('command') == 'ACT16' for r in channel['trace'])
    winner = best_mapping(a, w, templates=['aim_shared'])
    assert evaluate(a, w, winner.best).latency_ns == winner.evaluation.latency_ns
    assert result.details['stats']['scheduler'] == 'conservative'
    assert any('cast' in warning for warning in result.warnings)


@pytest.mark.parametrize('bad', [dict(k=1025), dict(dtype='fp16'), dict(iterations=2)])
def test_discovery_and_execute_reject_same_unsupported(bad):
    a = detail()
    w = WorkloadSpec('operator', dict(dict(op='mvm',m=16,k=16,dtype='bf16'), **bad))
    assert mapping_space(a,w,'aim_shared') == []
    with pytest.raises(ValueError):
        evaluate(a,w,MappingSpec('aim_shared',dict(profile=PROFILE,host_traffic='none')))


def test_invalid_payload_address_and_mapping_keys():
    for command in (C('HACK'), C('REFpb'), C('RD', 1), C('RD', 2**29), C('MAC16', 2048), C('RD', useful_bytes=33)):
        with pytest.raises(ValueError):
            run([command])
    a = detail()
    w = WorkloadSpec('operator',dict(op='mvm',m=16,k=16,dtype='bf16'))
    with pytest.raises(ValueError):
        evaluate(a,w,MappingSpec('aim_shared',dict(profile=PROFILE,host_traffic='none',tp=1)))
    assert mapping_space(get('neupims'), w, 'aim_shared') == []


def test_pending_clock_cannot_be_skipped_and_invalid_profile():
    with pytest.raises(ValueError):
        P(refi=1)
    model = AimController(P())
    model.submit(R('a','c',operation='RD',payload=C('RD')),0)
    snapshot = copy.deepcopy(model.__dict__)
    with pytest.raises(ValueError):
        model.submit(R('b','c',operation='RD',payload=C('RD')),100)
    assert model.__dict__ == snapshot


def test_forwarding_and_wire_occupancy_do_not_duplicate_bytes():
    # The first write has already moved to active. Only the second write is
    # still queued when the read arrives, which is the source forwarding rule.
    model, _ = run([C('WR'), C('WR', address=32), C('RD', address=32)])
    assert any(r['event']=='forward' for r in model.trace)
    assert model.bytes['external_read'] == 0
    assert model.bytes['external_write'] == 64
    assert model.accounting['logical'] == model.accounting['returned'] == 3
    assert model.accounting['issued'] == 2 and model.accounting['forwarded'] == 1
    # Read latency and register read latency differ. Explicit bus reservation
    # must also protect data windows, not only command issue slots.
    model = AimController(P())
    model.bus_windows = [(50,52)]
    assert not model._ready('RDMAC16',0,50)
    assert model._ready('RDMAC16',0,52)


def test_trace_layout_covers_each_tail_product_once():
    # Execute the schedule's mathematical indexing on small integer tensors;
    # this checks striping/padding independently of the timing formulas.
    m, k, channels = 35, 17, 2
    a = detail(channels)
    w = WorkloadSpec('operator',dict(op='mvm',m=m,k=k,dtype='bf16'))
    result = evaluate(a,w,MappingSpec('aim_shared',dict(profile=PROFILE,host_traffic='none')))
    expected = [sum((r-c)*(c+1) for c in range(k)) for r in range(m)]
    actual = [0]*m
    visits = [[0]*k for _ in range(m)]
    for channel_index, report in enumerate(result.details['aim']['controllers'].values()):
        columns = {}
        for event in report['trace']:
            if event['event'] != 'issue' or event['command'] != 'MAC16':
                continue
            row = event['row']
            column = columns.get(row,0)
            columns[row] = column + 1
            for bank in range(16):
                r = (row*channels+channel_index)*16+bank
                for c in range(column*16,min(k,(column+1)*16)):
                    if r < m:
                        actual[r] += (r-c)*(c+1)
                        visits[r][c] += 1
    assert actual == expected
    assert all(count == 1 for row in visits for count in row)


@pytest.mark.parametrize('second_bank,act_tick', [(4,5),(1,6)])
def test_defined_cross_bank_activation_and_full_read_drain(second_bank,act_tick):
    # Independent source derivation: channel RRDS5; same-group RRDL6.
    # Both banks have RCD36 and read-response CL50+BL2 at 1ns/tick.
    model,events=run([C('RD'),C('RD',P().encode(second_bank,0))])
    assert issues(model)==[('ACT',0),('ACT',act_tick),('RD',36),('RD',act_tick+36)]
    assert [e['tick'] for e in model.trace if e['event']=='return']==[88,act_tick+88]
    assert next(e.time for e in events if e.kind=='consumer_ready')==act_tick+88
    assert model.accounting['returned']==model.accounting['physical_completed']==2
    assert model.accounting['pending']==0 and not model.queues['active']
    assert not model.read_returns and not model.data_pending


@pytest.mark.parametrize('value', [None,0,-1,float('nan'),float('inf'),True,7800])
def test_refresh_unknown_or_invalid_is_not_a_default(value):
    with pytest.raises(ValueError):
        get('hsndi_detail',cores=1,channels=1,refresh_service_ns=value,refresh_source=SYNTHETIC)


def test_explicit_refresh_units_source_and_effect_on_service():
    with pytest.raises(ValueError,match='refresh_service_ns'):
        AimProfile()
    with pytest.raises(ValueError,match='refresh_source'):
        get('hsndi_detail',refresh_service_ns=105)
    for ns in (20,105,105.1):
        p=AimProfile(refresh_service_ns=ns,refresh_source='[synthetic] service-time sweep, mechanism only')
        model=AimController(p)
        model.submit(R('a','c',operation='RD',payload=C('RD')),7712)
        drain(model)
        assert ('REFab',7832) in issues(model)
        assert model.tick==7832+p.rfc
        assert p.rfc>=ns and p.rfc<ns+1


def test_detailed_adf_contract_rejects_aggregate_and_mutated_units():
    from arch_explorer.adf.model import Param, SpaceMatrix
    def points(node):
        if isinstance(node,SpaceMatrix):
            for child in (*node.elements,*node.networks):
                yield from points(child)
        else:
            yield node
    w=WorkloadSpec('operator',dict(op='mvm',m=16,k=16,dtype='bf16'))
    mapping=MappingSpec('aim_shared',dict(profile=PROFILE,host_traffic='none'))
    bad_adfs=[get('hsndi',cores=1,channels=1)]
    for key,value,unit,source in [('trfc',0,'ns',SYNTHETIC),('trfc',105,'cycle',SYNTHETIC),
                                  ('trfc',105,'ns',''),('trcd',18,'ns','old'),
                                  ('trfc',None,None,None)]:
        a=detail()
        bank=next(p for p in points(a) if p.component=='dram')
        if value is None:
            del bank.params[key]
        else:
            bank.params[key]=Param(value,unit,source)
        bad_adfs.append(a)
    a=detail()
    pim=next(p for p in points(a) if p.component=='pim_unit')
    pim.params['clock_ghz']=Param(0.5,'MHz','incorrect unit')
    bad_adfs.append(a)
    for a in bad_adfs:
        assert mapping_space(a,w,'aim_shared')==[]
        with pytest.raises(ValueError):
            evaluate(a,w,mapping)
