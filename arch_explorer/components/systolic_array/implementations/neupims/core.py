"""Component-owned cycle-stepped NPU instruction executor with explicit memory callbacks.

The memory adapter owns transactions; this core never charges aggregate DRAM
traffic. A callback reports consumer readiness separately from physical drain.
"""
from collections import deque
from dataclasses import dataclass, field
from arch_explorer.components.neupims_instructions import issue_compute
from arch_explorer.components.execution_profiles import NEUPIMS_INSTRUCTIONS


@dataclass
class LiveTile:
    tile: object
    slot: int
    acc_slot: int
    loads: int = 0
    computes: int = 0
    stores: int = 0
    ready: set = field(default_factory=set)
    output: dict = field(default_factory=dict)


class InstructionCore:
    """One FIFO compute pipeline; two scratchpad and accumulator banks.

    `memory.submit(region, write, now, callback)` must return False without
    side effects on backpressure, otherwise invoke callback at response time.
    step is called once per core tick, after memory completion delivery.
    """
    def __init__(self, tiles, height, width, *, policy=NEUPIMS_INSTRUCTIONS, pipeline=issue_compute):
        if any(type(v) is not int or v<1 for v in (height,width)):
            raise ValueError('positive integer core dimensions required')
        self.policy=policy
        self.issue_compute=pipeline
        self.pending=deque(tiles);self.height=height;self.width=width
        self.live=[];self.loadq=deque();self.computeq=deque();self.storeq=deque()
        self.pipeline=deque();self.slot=0;self.acc_slot=0
        self.trace=[];self.finished=[];self.acc_ready={0:{},1:{}}

    @property
    def active(self):
        return bool(self.pending or self.live or self.pipeline or self.loadq or self.computeq or self.storeq)

    def _admit(self,now):
        if not self.pending:return
        tile=self.pending[0]
        if tile.footprint>min(self.policy.spad_bytes,self.policy.accum_bytes)//2:
            raise ValueError('instruction tile exceeds physical half-buffer capacity')
        slot=self.slot^1;acc=self.acc_slot if tile.accum else self.acc_slot^1
        if any(t.slot==slot and (t.loads or t.computes) for t in self.live):return
        if not tile.accum and any(t.acc_slot==acc and (t.loads or t.computes or t.stores) for t in self.live):return
        self.pending.popleft();self.slot=slot;self.acc_slot=acc
        if not tile.accum:self.acc_ready[acc]={}
        t=LiveTile(tile,slot,acc,output=self.acc_ready[acc]);self.live.append(t)
        for inst in tile.instructions:
            if inst.opcode=='MOVIN':self.loadq.append((t,inst));t.loads+=1
            elif inst.opcode=='MOVOUT':self.storeq.append((t,inst));t.stores+=1
            else:
                self.computeq.append((t,inst));t.computes+=1
                t.output[inst.dest]=t.output.get(inst.dest,0)+1
        self.trace.append(dict(event='tile-admit',time=now,index=tile.index,slot=slot,acc_slot=acc,bytes=tile.footprint))

    def step(self,now,memory):
        # Upstream pops at most one finished matrix instruction per tick.
        if self.pipeline and self.pipeline[0][1]<=now:
            start,end,size,t,inst,issued=self.pipeline.popleft()
            t.computes-=1;t.output[inst.dest]-=1
            self.trace.append(dict(event='compute-retire',time=now,index=t.tile.index,dest=inst.dest))
        for q,write in ((self.loadq,False),(self.storeq,True)):
            while q:
                t,inst=q[0]
                if write and t.output.get(inst.dest,1)!=0:break
                def completed(at,t=t,inst=inst,write=write):
                    if write:t.stores-=1
                    else:t.loads-=1;t.ready.add(inst.dest)
                    self.trace.append(dict(event='store-response' if write else 'load-response',time=at,
                                           index=t.tile.index,dest=inst.dest,bytes=inst.region.bytes))
                if not memory.submit(inst.region,write,now,completed):break
                q.popleft()
                self.trace.append(dict(event='MOVOUT' if write else 'MOVIN',time=now,index=t.tile.index,
                                       dest=inst.dest,bytes=inst.region.bytes))
                # load instructions drain into the core request queues; stores
                # issue one per tick (NeuPIMSystolicWS.cc:95,201).
                if write:break
        if self.computeq:
            t,inst=self.computeq[0]
            if all(src in t.ready for src in inst.sources):
                self.computeq.popleft()
                previous=self.pipeline[-1][:3] if self.pipeline else None
                start,end=self.issue_compute(now,inst,self.height,self.width,previous,self.policy)
                self.pipeline.append((start,end,inst.size,t,inst,now))
                self.trace.append(dict(event=inst.opcode,time=now,start=start,finish=end,size=inst.size,
                                       index=t.tile.index,dest=inst.dest,shape=inst.shape))
        for t in list(self.live):
            if not (t.loads or t.computes or t.stores):
                self.live.remove(t);self.finished.append((t.tile.index,now))
                self.trace.append(dict(event='tile-dequeue',time=now,index=t.tile.index))
        self._admit(now)
