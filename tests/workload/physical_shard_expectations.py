"""Independent dense-interval shard observation audit (QA protocol v1).

Consumes observations, never plans placement or invents product Events.
Coordinates are flat logical elements, physical addresses and time are integers.
Lifetimes are [born,dead), ready/read are instants, and transfers [begin,end].
An adapter must bind this document to actual product source/Events separately.
"""
from collections import defaultdict

SCHEMA='arch-explorer.qa-shard-observations.v1'
WIDTH={'fp32':4,'fp16':2,'bf16':2,'int64':8,'int32':4,'bool':1}


def integer(value,where,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError(where+': invalid integer')
    return value


def partition(intervals,size,where):
    cursor=0
    for lo,hi in sorted(intervals):
        if lo!=cursor or hi<=lo:raise ValueError(where+': partition hole/overlap')
        cursor=hi
    if cursor!=size:raise ValueError(where+': partition does not cover tensor')


def audit(document):
    if document.get('schema')!=SCHEMA:raise ValueError('unsupported shard observation schema')
    tensors=document['tensors'];capacities=document['capacities'];buffers={};origins=defaultdict(list)
    for tile,capacity in capacities.items():integer(capacity,'capacity',1)
    for name,t in tensors.items():
        integer(t['elements'],'tensor.elements',1)
        if t['dtype'] not in WIDTH or type(t['root']) is not bool:raise ValueError('tensor type/root differs')
    for b in document['buffers']:
        bid=b['id'];name=b['tensor']
        if bid in buffers:raise ValueError('duplicate buffer id')
        if name not in tensors or b['tile'] not in capacities:raise ValueError('unknown tensor/tile')
        for key in ('start','stop','offset','born','ready','dead','bytes'):integer(b[key],'buffer.'+key)
        if not b['start']<b['stop']<=tensors[name]['elements']:raise ValueError('buffer interval outside tensor')
        if not b['born']<=b['ready']<b['dead']:raise ValueError('buffer readiness/lifetime differs')
        if b['kind'] not in ('root','output','scratch','copy'):raise ValueError('unknown buffer kind')
        if b['kind']=='root' and not tensors[name]['root']:raise ValueError('root buffer for produced tensor')
        if b['kind'] in ('output','scratch') and tensors[name]['root']:raise ValueError('produced buffer overwrites root identity')
        nbytes=(b['stop']-b['start'])*WIDTH[tensors[name]['dtype']]
        if b['bytes']!=nbytes:raise ValueError('buffer byte width/extent differs')
        if b['offset']+nbytes>capacities[b['tile']]:raise ValueError('physical address outside capacity')
        buffers[bid]=b
        if b['kind'] in ('root','output','scratch'):origins[name].append((b['start'],b['stop']))
    for name,t in tensors.items():partition(origins[name],t['elements'],'origins.'+name)
    peaks={tile:0 for tile in capacities};peak_times={tile:None for tile in capacities}
    # Half-open allocation intervals: frees at t precede allocations at t.
    for tile in capacities:
        own=[b for b in buffers.values() if b['tile']==tile]
        for at in sorted({b[k] for b in own for k in ('born','dead')}):
            live=[b for b in own if b['born']<=at<b['dead']]
            occupied=sum(b['bytes'] for b in live)
            if occupied>capacities[tile]:raise ValueError('simultaneous live bytes exceed tile capacity')
            if occupied>peaks[tile]:peaks[tile]=occupied;peak_times[tile]=at
            for i,left in enumerate(live):
                for right in live[i+1:]:
                    if max(left['offset'],right['offset'])<min(left['offset']+left['bytes'],right['offset']+right['bytes']):
                        raise ValueError('simultaneous physical alias overlaps live buffer')
    last=defaultdict(lambda:None);copy_ranges=defaultdict(list);hops=defaultdict(int);transfer_ids=set()
    links={tuple(edge) for edge in document['links']}
    for transfer in document['transfers']:
        if transfer['id'] in transfer_ids:raise ValueError('duplicate transfer id')
        transfer_ids.add(transfer['id'])
        src=buffers[transfer['source']];dst=buffers[transfer['target']]
        begin=integer(transfer['begin'],'transfer.begin');end=integer(transfer['end'],'transfer.end')
        integer(transfer['bytes'],'transfer.bytes')
        lo=integer(transfer['start'],'transfer.start');hi=integer(transfer['stop'],'transfer.stop')
        if src['tensor']!=dst['tensor'] or dst['kind']!='copy' or not lo<hi:raise ValueError('copy tensor/kind differs')
        if not max(src['start'],dst['start'])<=lo<hi<=min(src['stop'],dst['stop']):raise ValueError('copy range differs')
        if not src['ready']<=begin<=end<src['dead'] or not dst['born']<=begin<=end<=dst['ready']:
            raise ValueError('copy outside source/destination lifetime or readiness')
        route=transfer['route']
        if not isinstance(route,list) or len(route)<2 or route[0]!=src['tile'] or route[-1]!=dst['tile']:
            raise ValueError('copy route endpoints differ')
        nbytes=(hi-lo)*WIDTH[tensors[src['tensor']]['dtype']]
        if transfer['bytes']!=nbytes:raise ValueError('transfer bytes differ')
        for edge in zip(route,route[1:]):
            if edge not in links:raise ValueError('copy route is not a declared directed link')
            hops[edge]+=nbytes
        copy_ranges[dst['id']].append((lo-dst['start'],hi-dst['start']))
        last[src['id']]=max(end,last[src['id']] if last[src['id']] is not None else end)
    for b in buffers.values():
        if b['kind']=='copy':partition(copy_ranges[b['id']],b['stop']-b['start'],'copy.'+b['id'])
    for read in document['reads']:
        b=buffers[read['buffer']];at=integer(read['at'],'read.at')
        integer(read['start'],'read.start');integer(read['stop'],'read.stop')
        if read['tile']!=b['tile']:raise ValueError('remote read requires a charged destination copy')
        if not b['ready']<=at<b['dead']:raise ValueError('read outside ready lifetime; freed before last reader')
        if not b['start']<=read['start']<read['stop']<=b['stop']:raise ValueError('read outside fragment')
        last[b['id']]=max(at,last[b['id']] if last[b['id']] is not None else at)
    return dict(schema=SCHEMA,peak_bytes=peaks,peak_times=peak_times,
                last_read= dict(last),allocated_copy_bytes=sum(b['bytes'] for b in buffers.values() if b['kind']=='copy'),
                hop_bytes=[dict(source=a,target=b,bytes=n) for (a,b),n in sorted(hops.items())],
                total_hop_bytes=sum(hops.values()),
                scope='dense interval observations; adapter authenticity and numerical Events checked separately')
