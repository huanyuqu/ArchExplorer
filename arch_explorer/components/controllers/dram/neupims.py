"""Component-owned memory adapters; no aggregate stream is added.

The circular address cursor follows f299af3 Common.cc::from_instruction.
Its upstream hash-set traversal is replaced by explicit first-touch order.
This is a declared software mapping choice, not a measured timing parameter.
"""
from collections import deque
import heapq
from math import ceil

from arch_explorer.components.transaction_controller import Channel, Transaction, Timing
from arch_explorer.components.execution_profiles import NEUPIMS_LAYOUT, NEUPIMS_CONTROLLER, NEUPIMS_INSTRUCTIONS


class AddressStream:
    def __init__(self, hidden, tp, layout=NEUPIMS_LAYOUT):
        self.layout=layout
        self.limit=NEUPIMS_INSTRUCTIONS.address_ring_bytes(hidden,tp)
        if self.limit <= 0 or self.limit%layout.burst_bytes:raise ValueError('address ring must contain whole bursts')
        self.cursor=0

    def bursts(self, elements):
        if elements<1:raise ValueError('nonempty memory instruction required')
        # Advance by elements exactly as the source (including the end burst).
        # Generate at burst granularity rather than enumerating every element.
        first=(self.cursor+2)%self.limit
        count=elements*2
        seen={}
        while count:
            take=min(count,self.limit-first)
            for block in range(first//self.layout.burst_bytes,(first+take-1)//self.layout.burst_bytes+1):
                seen.setdefault(self.layout.stripe(block*self.layout.burst_bytes),None)
            count-=take;first=0
        self.cursor=(self.cursor+2*elements)%self.limit
        return tuple(seen)


class BandwidthMemory:
    """Ablation A: addressed bursts, independent channel bandwidth/latency.

    No row/refresh timing or early write responses. Bandwidth is supplied by
    the device; cursor and in-flight drain survive operation boundaries.
    """
    def __init__(self, hidden, tp, bandwidth, latency, channels=None, *, layout=NEUPIMS_LAYOUT):
        channels=layout.channels if channels is None else channels
        if channels!=layout.channels or bandwidth<=0 or latency<0:
            raise ValueError('bandwidth memory requires matching layout and positive bandwidth/nonnegative latency')
        self.addresses=AddressStream(hidden,tp,layout);self.bandwidth=bandwidth/channels
        self.latency=latency;self.free=[0.]*channels;self.events=[];self.serial=0
        self.logical=[0,0];self.physical=[0,0];self.trace=[]

    def submit(self,region,write,now,callback):
        if region.element_bytes!=2:raise ValueError('source address stream requires 2-byte elements')
        addrs=self.addresses.region_bursts(region) if hasattr(self.addresses,'region_bursts') else self.addresses.bursts(region.bytes//2)
        self.logical[write]+=region.bytes;self.physical[write]+=len(addrs)*self.addresses.layout.burst_bytes
        end=now
        for address in addrs:
            ch=self.addresses.layout.decode(address)[0]
            self.free[ch]=max(now,self.free[ch])+self.addresses.layout.burst_bytes/self.bandwidth
            end=max(end,self.free[ch]+self.latency)
        self.serial+=1
        heapq.heappush(self.events,(ceil(end),self.serial,callback))
        return True

    def step(self,now):
        while self.events and self.events[0][0]<=now:
            at,_,callback=heapq.heappop(self.events);callback(at)

    @property
    def active(self):return bool(self.events)


def run_core(core,memory,start):
    """Execute at integer core ticks; return consumer completion, not drain."""
    now=ceil(start)
    while core.active:
        memory.step(now);core.step(now,memory);now+=1
    return now-1


class TransactionMemory:
    """Persistent channel controllers shared by host instructions and attention.

    Core request queues are unbounded as in the source; each channel admits one
    request per core tick into its finite transaction queue. The command queue
    is bounded separately per bank. Link legs are one core tick apiece.
    """
    def __init__(self,hidden=None,tp=None,timing=Timing(),trace=False,*,layout=NEUPIMS_LAYOUT,
                 policy=NEUPIMS_CONTROLLER,addresses=None,queue_size=32,command_queue_size=128):
        self.layout=layout;self.policy=policy
        self.addresses=addresses if addresses is not None else AddressStream(hidden,tp,layout)
        self.channels=[Channel(timing,queue_size,layout.ranks,layout.banks,command_queue_size,
                               layout=layout,policy=policy) for _ in range(layout.channels)]
        for c in self.channels:c.keep_trace=trace
        self.pending=[deque() for _ in range(layout.channels)];self.events=[];self.serial=0;self.last=-1
        self.logical=[0,0];self.physical=[0,0];self.trace=[];self.keep_trace=trace
        self.outstanding=0;self.transactions=0
        self.active_channels=set()
        self.idle_version=[0]*layout.channels
        self.idle_wakes=[(min(c.deadline),i,0) for i,c in enumerate(self.channels)]
        heapq.heapify(self.idle_wakes)

    def later(self,at,callback):
        self.serial+=1;heapq.heappush(self.events,(ceil(at),self.serial,callback))

    def enqueue(self,tx,channel,now,callback):
        self.active_channels.add(channel)
        self.idle_version[channel]+=1
        self.transactions+=1;self.outstanding+=1
        def returned(at):
            def consume(ready):
                self.outstanding-=1
                if self.keep_trace:
                    tx.audit=dict(kind=tx.kind,address=tx.address,arrive=tx.arrive,
                        accepted=tx.accepted,issued=tx.issued,returned=tx.returned,bank_ready=tx.ready,consumer_ready=ready)
                    self.trace.append(tx.audit)
                if callback:callback(ready)
            self.later(at+1,consume)
        tx.callback=returned;self.pending[channel].append((ceil(now)+1,tx))

    def submit(self,region,write,now,callback):
        if hasattr(self.addresses,'region_bursts'):
            addrs=self.addresses.region_bursts(region)
        else:
            if region.element_bytes!=2:raise ValueError('source address stream requires 2-byte elements')
            addrs=self.addresses.bursts(region.bytes//2)
        self.logical[write]+=region.bytes;self.physical[write]+=len(addrs)*self.layout.burst_bytes
        left=[len(addrs)]
        def complete(at):
            left[0]-=1
            if left[0]==0:callback(at)
        for address in addrs:
            tx=Transaction('WRITE' if write else 'READ',address,ceil(now))
            self.enqueue(tx,self.layout.decode(address)[0],now,complete)
        return True

    def pim(self,channel,kind,row,now,comps=0,readouts=1,callback=None):
        address=self.layout.encode(channel=channel,row=row)
        tx=Transaction(kind,address,ceil(now),comps=comps,readouts=readouts)
        self.enqueue(tx,channel,now,callback)
        return tx

    def step(self,now):
        for tick in range(self.last+1,int(now)+1):
            while self.events and self.events[0][0]<=tick:
                at,_,callback=heapq.heappop(self.events);callback(at)
            while self.idle_wakes and self.idle_wakes[0][0]<=tick:
                _,i,version=heapq.heappop(self.idle_wakes)
                if version==self.idle_version[i]:self.active_channels.add(i)
            # An idle channel has no response, ingress, command or cursor work.
            # Its only observable next transition is its refresh deadline.
            # Busy channels still step every tick: failed arbitration may move
            # the bank cursor, so it is never skipped or approximated.
            for i in sorted(self.active_channels):
                c=self.channels[i]
                pending=self.pending[i]
                if pending and pending[0][0]<=tick:
                    if c.submit(pending[0][1],tick):pending.popleft()
                c.step(tick)
                if not pending and not c.active:
                    self.active_channels.remove(i)
                    self.idle_version[i]+=1
                    heapq.heappush(self.idle_wakes,(max(tick+1,min(c.deadline)),i,self.idle_version[i]))
        self.last=int(now)

    @property
    def active(self):
        return bool(self.outstanding or self.events or any(self.pending) or any(c.active for c in self.channels))

    def physically_ready(self, now):
        if self.active:return False
        return all(now>=max(c.bus,c.pim_ready,c.gwrite_ready,*c.busy,
            *(max(b.column,b.precharge,b.pim_column,b.pim_precharge) for rank in c.state for b in rank))
            for c in self.channels)

    def wait(self,predicate,start):
        now=ceil(start)
        while not predicate():self.step(now);now+=1
        return now-1

    def report(self):
        counts={}
        for c in self.channels:
            for k,v in c.counts.items():counts[k]=counts.get(k,0)+v
        return dict(commands=counts,transactions=self.transactions,logical_bytes=self.logical,
                    requested_burst_bytes=self.physical,
                    issued_dram_bytes=[self.layout.burst_bytes*counts.get('RD',0),self.layout.burst_bytes*counts.get('WR',0)],
                    pending_unique_writes=sum(len(c.pending_writes) for c in self.channels),
                    forwarded_reads=sum(c.forwarded for c in self.channels),
                    merged_reads=sum(c.merged_reads for c in self.channels),
                    merged_writes=sum(c.merged_writes for c in self.channels),
                    pending_unique_reads=sum(len(c.pending_reads) for c in self.channels),
                    physical_bytes=self.physical,backpressure=sum(c.rejected for c in self.channels),
                    maximum_ingress=max(c.max_queue for c in self.channels),transaction_queue_capacity=self.channels[0].capacity,
                    ingress_classes=['read','write','pim'],per_bank_command_queue_capacity=self.channels[0].command_capacity,address_cursor=self.addresses.cursor,
                    boundary='arrival / accept / issue / callback / bank-ready / consumer are distinct')
