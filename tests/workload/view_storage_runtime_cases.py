"""Observe physical allocations; derive their owners from independent root walks."""
from collections import defaultdict
from copy import deepcopy
import itertools

import pytest

from multilayer_contract_cases import observe_memory
from view_storage_contract_cases import independent_layouts


def audit_storage(compilation,result,observed):
    program=compilation['program'];layouts=independent_layouts(program)
    ops={row['name']:row for row in program['operators']}
    tensors={row['name']:row for row in program['tensors']}
    producers={n:op['name'] for op in ops.values() for n in op['outputs']}
    outputs=set(program['meta']['outputs'])
    roots={value.root.name:value.root for value in layouts.values()}
    for row in compilation['plan']['storage']:
        expected=layouts[row['tensor']]
        assert row['logical_bytes']==expected.read_bytes
        assert row['allocation_bytes']==(0 if expected.shared else expected.root.bytes)
        assert row['layout']['allocation_bytes']==expected.root.bytes
    for step in compilation['plan']['steps']:
        op=ops[step['name']];address=step['address_map']
        assert address['indexing']==dict(semantic=op['op'],attrs=op['attrs'])
        assert set(address)=={'reads','writes','indexing','holes','service'}
        assert address['holes']=='not charged; logical elements only'
        assert address['service']=='direct positive-stride logical loads; no hidden packing'
        for direction,names in [('reads',op['inputs']),('writes',[] if op['op']=='storage.view' else op['outputs'])]:
            assert [row['tensor'] for row in address[direction]]==names
            for row in address[direction]:
                expected=layouts[row['tensor']]
                assert row['storage']==expected.root.name and row['shape']==list(expected.shape)
                assert row['strides']==list(expected.steps) and row['offset']==expected.first
                assert row['allocation_bytes']==expected.root.bytes
        if op['op']=='storage.view':
            assert all(step['access'][key]==0 for key in ('read_bytes','write_bytes','operations','conversion_instructions'))
        elif op['op'] in ('scale','exp','add','mul'):
            assert step['access']['read_bytes']==sum(layouts[n].read_bytes for n in op['inputs'])
            assert step['access']['write_bytes']==sum(layouts[n].read_bytes for n in op['outputs'])
    plans={row['storage']:row for row in compilation['plan']['allocations']}
    assert set(plans)==set(roots) and len(plans)==len(compilation['plan']['allocations'])
    evaluation=result['evaluation'];finish=evaluation['latency_ns']
    trace=evaluation['details']['trace'];ends=defaultdict(list);starts=defaultdict(list)
    for op in ops.values():
        if op['op']=='storage.view':
            control=[row for row in trace if row['operator']==op['name'] and row['kind']!='storage']
            assert len(control)==2 and all(row['kind']=='sync' for row in control)
            assert all(row['start_ns']==row['end_ns'] and not row['resources'] for row in control)
    for task in trace:
        if task['operator'] in ops and task['kind']!='storage':
            ends[task['operator']].append(task['end_ns']);starts[task['operator']].append(task['start_ns'])
    intervals=[]
    for name,root in roots.items():
        aliases={n for n,value in layouts.items() if value.root.name==name}
        uses={op['name'] for op in ops.values() if aliases.intersection(op['inputs']+op['outputs'])}
        plan=plans[name]
        assert set(plan['aliases'])==aliases and set(plan['consumers'])==uses
        assert plan['producer']==producers.get(name) and plan['bytes']==root.bytes
        held=bool(outputs&aliases);assert plan['returned']==held
        # Existing public task graph names storage as <device>:<root>. Match
        # the whole root suffix: nested qualified names may contain colons.
        actual=[e for e in observed if e['name'].endswith(':'+name)]
        allocations=[e for e in actual if e['action']=='allocate']
        assert len(allocations)==1,(name,actual)
        allocation=allocations[0];assert allocation['bytes']==root.bytes
        owner=allocation['owner'];free=[e['time'] for e in actual if e['action']=='free']
        last=max((max(ends[op]) for op in uses),default=allocation['time'])
        if root.resident or held:
            assert all(time>=finish for time in free)
            release=finish
        else:
            assert free==pytest.approx([last]),(name,free,last)
            release=last
        if name in producers:
            for op in ops.values():
                if name in op['inputs']:assert min(starts[op['name']])>=max(ends[producers[name]])
        intervals.append(dict(storage=name,owner=owner,bytes=root.bytes,start=allocation['time'],
            end=release,aliases=sorted(aliases),returned=held,resident=root.resident,last_consumer=last))
    assert len([e for e in observed if e['action']=='allocate'])==len(roots)
    # Independent half-open physical root intervals, with frees before new
    # allocations at the same timestamp. Alias logical bytes never enter it.
    changes=[]
    for row in intervals:
        if row['start']<row['end']:
            changes.extend([(row['start'],1,row['owner'],row['bytes']),
                            (row['end'],0,row['owner'],-row['bytes'])])
    live=defaultdict(int);peaks=defaultdict(int)
    for _,_,owner,delta in sorted(changes):
        live[owner]+=delta;assert live[owner]>=0
        peaks[owner]=max(peaks[owner],live[owner])
    assert dict(peaks)==evaluation['details']['stats']['memory_peak']
    return dict(intervals=intervals,peaks=dict(peaks),task_end={n:max(v) for n,v in ends.items()})


def fork_graph(w,*,dtype='fp32',retain=False,swap=False,weight=False):
    g=w.Workload('val58-physical-fork',dtype=dtype)
    g.tensor('X',(4,6),dtype=dtype,kind='weight' if weight else 'activation')
    g.meta['inputs']={'X':[[float(r*6+c)/24 for c in range(6)] for r in range(4)]}
    w.readonly_view(g,'S','X',transform='slice',shape=(2,3),starts=(1,1),stops=(4,6),steps=(2,2))
    w.readonly_view(g,'T','S',transform='transpose',shape=(3,2),axes=(1,0))
    w.readonly_view(g,'U','T',transform='transpose',shape=(2,3),axes=(1,0))
    for name in ('L','R','Y'):g.tensor(name,(2,3),dtype=dtype)
    for name,source,repeated in [('left','S',swap),('right','U',not swap)]:
        # Two operand reads versus one make service asymmetric in both real
        # backends; transcendental instruction costs alone can tie Roofline.
        g.add(name,'mul' if repeated else 'scale',[source,source] if repeated else [source],
              ['L' if name=='left' else 'R'],**dict(shape=[2,3],**({} if repeated else {'factor':2.})))
    g.add('merge','add',['L','R'],['Y'],shape=[2,3])
    g.meta['outputs']=['Y','S'] if retain else ['Y']
    return g


def address_rows(layout,tensor,direction):
    width=layout.root.bytes//layout.root.elements
    return [dict(direction=direction,tensor=tensor,storage=layout.root.name,
        logical_index=list(index),element_offset=layout.address(index),
        byte_offset=width*layout.address(index),bytes=width)
        for index in itertools.product(*(range(dim) for dim in layout.shape))]
