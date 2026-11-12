from arch_explorer.components.neupims_transactions import Channel,Transaction,Timing,decode,switch_column_channel


def advance(c,start,end):
    for now in range(start,end+1):c.step(now)


def test_address_permutation():
    assert decode(switch_column_channel(64))==(1,0,0,0,0)
    assert decode(switch_column_channel(32*64))==(0,0,0,0,1)
    assert decode(switch_column_channel(32768))==(0,0,1,0,0)


def test_closed_hit_conflict_and_shared_bus():
    c=Channel();first=Transaction('GWRITE',0,0)
    assert c.submit(first,0)
    advance(c,0,111)
    assert [(r['time'],r['command']) for r in c.trace]==[(1,'ACT'),(15,'GWRITE')]
    hit=Transaction('GWRITE',0,112);c.submit(hit,112);advance(c,112,209)
    assert hit.issued==113 and hit.returned==209
    # row is above rank/group/bank/channel/column fields
    conflict=Transaction('GWRITE',1<<20,210);c.submit(conflict,210);advance(c,210,335)
    assert [(r['time'],r['command']) for r in c.trace[-3:]]==[(211,'PRE'),(225,'ACT'),(239,'GWRITE')]
    assert len({r['time'] for r in c.trace})==len(c.trace)


def test_write_response_does_not_release_bank_and_queue_pressure():
    c=Channel(queue_size=1);write=Transaction('WRITE',0,0);read=Transaction('READ',0,0)
    assert c.submit(write,0) and not c.submit(Transaction('WRITE',64,0),0)
    advance(c,0,1)
    assert write.returned==1 and write.issued is None
    advance(c,2,15)
    assert write.issued==15 and write.ready==37
    assert c.submit(read,16)
    advance(c,16,45)
    assert read.issued==29 and read.returned==45


def test_refresh_waits_for_actual_precharge_eligibility():
    c=Channel(Timing(refi=40,rfc=5),queue_size=1)
    c.submit(Transaction('WRITE',0,0),0)
    advance(c,0,55)
    first=[r for r in c.trace if r['command']=='REF' and r['rank']==0][0]
    # WR at15 => PRE no earlier than37; tRP14 => REF at51, not nominal20.
    assert first['time']==51


def test_pim_and_normal_compete_one_bus():
    c=Channel();read=Transaction('READ',32768,0);pim=Transaction('GWRITE',0,0)
    c.submit(pim,0);c.submit(read,0);advance(c,0,160)
    assert c.trace[0]['command']=='ACT' and c.trace[0]['bank']==0
    # timing.cc restricts GWRITE's own normal bank for 96 ticks; another
    # bank may read while the GWRITE is in flight. Only the bus is shared.
    assert pim.issued<read.issued<pim.ready
    assert len({r['time'] for r in c.trace})==len(c.trace)


def test_pim_group_activations_and_return_before_bank_ready():
    c=Channel()
    c.submit(Transaction('HEADER',1<<20,0,comps=32),0)
    first=Transaction('COMPS_READRES',1<<20,0,comps=32)
    second=Transaction('COMPS_READRES',1<<20,0,comps=32)
    c.submit(first,0);c.submit(second,0);advance(c,0,220)
    acts=[(r['time'],r['rank'],r['bank']) for r in c.trace if r['command']=='G_ACT']
    assert acts==[(2,0,0),(3,1,0),(32,0,4),(33,1,4),(62,0,8),(63,1,4+4),(92,0,12),(93,1,12)]
    assert (first.issued,first.returned,first.ready)==(107,123,173)
    assert second.issued==173


def test_pim_normal_row_collision_uses_same_state():
    c=Channel();c.submit(Transaction('READ',0,0),0);advance(c,0,30)
    c.submit(Transaction('HEADER',0,31,comps=8),31)
    c.submit(Transaction('COMPS_READRES',0,31,comps=8),31);advance(c,31,200)
    pre=next(r for r in c.trace if r['command']=='PRE')
    header=next(r for r in c.trace if r['command']=='HEADER')
    assert pre['time']==35 and header['time']>pre['time']
    assert c.state[0][0].row is None and c.state[0][0].pim_row==0


def test_two_bounded_queue_levels_and_recovery():
    c=Channel(queue_size=2,command_queue_size=2);submitted=[]
    for now in range(5000):
        if len(submitted)<50:
            tx=Transaction('READ',len(submitted)<<20,now)
            if c.submit(tx,now):submitted.append(tx)
        c.step(now)
        assert len(c.ingress)<=2 and all(len(q)<=2 for q in c.normal)
    assert c.rejected>0 and len(submitted)==50
    assert all(t.returned is not None for t in submitted)


def test_read_coalescing_and_write_forwarding():
    c=Channel();a=Transaction('READ',0,0);b=Transaction('READ',0,0)
    c.submit(a,0);c.submit(b,0);advance(c,0,40)
    assert a.returned==b.returned==31 and c.counts['RD']==1
    write=Transaction('WRITE',64,41);forward=Transaction('READ',64,41)
    c.submit(write,41);c.submit(forward,41);advance(c,41,60)
    assert write.returned==forward.returned==42 and write.issued is None
    # Fewer than nine buffered writes are retained until a sourced drain
    # condition holds; an idle timer is deliberately not invented.
    assert len(c.writes)==1


def test_source_four_head_microtrace():
    # Independently retained f299af3 WP-SIM-07 k128_heads4 raw commands.
    c=Channel()
    txs=[Transaction('GWRITE',0,0),Transaction('HEADER',1<<20,1,comps=32,readouts=4)]
    txs += [Transaction('COMPS_READRES',1<<20,i+2,comps=8) for i in range(4)]
    for now in range(288):
        if now<len(txs):assert c.submit(txs[now],now)
        c.step(now)
    assert [(r['time'],r['command']) for r in c.trace]==[
        (1,'ACT'),(15,'GWRITE'),(111,'HEADER'),(112,'G_ACT'),(113,'G_ACT'),(142,'G_ACT'),
        (143,'G_ACT'),(172,'G_ACT'),(173,'G_ACT'),(202,'G_ACT'),(203,'G_ACT'),
        (217,'COMPS_READRES'),(235,'COMPS_READRES'),(253,'COMPS_READRES'),(271,'COMPS_READRES')]
    assert txs[-1].returned==287 and txs[1].returned is None
