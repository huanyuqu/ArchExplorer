"""Unified addressed controller endpoint, independent of NPU/NeuPIMs programs.

Normal and PIM requesters bind to the SAME instance. Queue refusal is pure;
the controller's legacy rejected counter is not changed by protocol refusal.
Consumer and physical recovery notifications are deliberately distinct.
"""
from math import ceil
from fractions import Fraction
from arch_explorer.components.transaction_controller import Channel, Transaction
from arch_explorer.components.api import ExecutionEvent


class ControllerExecutionModel:
    def __init__(self, controller: Channel, clock_ns=1):
        self.controller=controller
        self.clock_ns=Fraction(clock_ns)
        if self.clock_ns<=0:raise ValueError('positive controller clock required')
        if self.clock_ns.denominator==1:self.clock_ns=int(self.clock_ns)
        self.pending={}
        self.physical={}
        self.resource_sent=set()
        self.out=[]
        self.serial=0
        self.tick=-1
        self.wake=None

    def submit(self, request, now):
        c=self.controller
        kind=request.operation
        queue=c.writes if kind=='WRITE' else c.reads if kind=='READ' else c.pim_requests
        if len(queue)>=c.capacity:return False
        if now<request.available_time:raise ValueError('request has not arrived')
        tick=ceil(Fraction(now)/self.clock_ns)
        if tick>self.tick+1:
            if self.pending or c.active:
                raise ValueError('controller must advance through observable pending work')
            for at in range(self.tick+1,tick):c.step(at)
            self.tick=tick-1
        payload=dict(request.payload or {})
        tx=Transaction(kind,arrive=tick,**payload)
        ident=request.request_id
        if ident in self.pending:raise ValueError('duplicate controller request')
        def response(at):
            self.out.append((at,ident,'consumer_ready'))
        tx.callback=response
        physical=c.pending_writes.get(tx.address)
        if physical is None and kind=='READ':
            physical=next(iter(c.pending_reads.get(tx.address,())),None)
        if not c.submit(tx,tick):raise AssertionError('capacity preflight diverged')
        self.pending[ident]=tx
        self.physical[ident]=physical if physical is not None else tx
        self.wake=tick if self.wake is None else min(self.wake,tick)
        return True

    def next_event_time(self):
        return None if self.wake is None else self.wake*self.clock_ns

    def advance(self, now):
        tick=Fraction(now)/self.clock_ns
        if tick.denominator!=1:raise ValueError('advance must be on clock edge')
        tick=int(tick)
        if tick<self.tick:raise ValueError('controller time moved backwards')
        c=self.controller
        # Tick stepping is necessary: unsuccessful bank scans change cursor.
        for at in range(self.tick+1,tick+1):c.step(at)
        self.tick=tick
        events=[]
        def emit(ident,kind,phase):
            self.serial+=1
            events.append(ExecutionEvent(now,phase,self.serial,ident,kind))
        for at,ident,kind in self.out:
            if at!=tick:raise ValueError('missed observable controller response')
            emit(ident,kind,10)
        self.out.clear()
        for ident,consumer in list(self.pending.items()):
            tx=self.physical[ident]
            ready=tx.ready
            if ready is None or tx.issued is None:continue
            if tick>=ready and ident not in self.resource_sent:
                emit(ident,'resource_ready',20)
                self.resource_sent.add(ident)
            # Data response may trail the command resource's next issue time.
            if tx.kind=='READ':ready=max(ready,tx.issued+c.t.cl+c.t.burst)
            if tx.kind in ('GWRITE','COMPS_READRES'):
                ready=max(ready,tx.issued+c.t.cl+c.t.burst)
            if tick>=ready:
                emit(ident,'physical_complete',20)
                del self.pending[ident]
                del self.physical[ident]
        emit('','queue_available',20)
        self.wake=tick+1 if self.pending or c.active else None
        return events
