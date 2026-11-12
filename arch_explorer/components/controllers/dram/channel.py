"""Addressed transactions and bank timing for the optional explicit executor.

Units: integer core/DRAM ticks, bytes. Defaults are the upstream
HBM2_8Gb_x128_dualpim.ini configuration, not hardware calibration.
"""
from dataclasses import dataclass,field
from collections import deque
from arch_explorer.components.execution_profiles import NEUPIMS_LAYOUT, NEUPIMS_CONTROLLER


@dataclass(frozen=True)
class Timing:
    rcd: int=14
    rp: int=14
    ras: int=34
    rrd: int=6
    rrd_s: int=4
    faw: int=30
    cl: int=14
    wl: int=4
    burst: int=2
    ccd: int=2
    rtp: int=5
    wr: int=16
    wtr: int=8
    wtr_s: int=6
    rtrs: int=2
    rfc: int=260
    refi: int=3900
    gwrite: int=96

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in self.__dict__.values()) or self.refi < 1 or self.burst < 1:
            raise ValueError('nonnegative integer timings and positive refresh/burst required')


@dataclass
class Transaction:
    kind: str
    address: int
    arrive: int
    callback: object=None
    comps: int=0
    accepted: int|None=None
    issued: int|None=None
    returned: int|None=None
    ready: int|None=None
    response_sent: bool=False
    readouts: int=1
    admitted: bool=False
    audit: dict|None=None
    command_arrival: int=0


@dataclass
class Bank:
    row: int|None=None
    activate: int=0
    column: int=0
    precharge: int=0
    pim_row: int|None=None
    pim_activate: int=0
    pim_column: int=0
    pim_precharge: int=0


class Channel:
    """Shared normal/GWRITE command bus with persistent bank and refresh state.

    Requests enter a bounded FIFO. Readiness selection chooses PIM GWRITE
    before normal commands; ties within a class preserve arrival order.
    PRE/ACT consume the same one-command-per-tick bus as RD/WR/GWRITE/REF.
    """
    def __init__(self,timing=Timing(),queue_size=32,ranks=2,banks=16,command_queue_size=128, *, layout=None, policy=NEUPIMS_CONTROLLER):
        layout = layout or NEUPIMS_LAYOUT
        if layout.ranks != ranks or layout.banks != banks:
            raise ValueError('ranks/banks must match the explicit address layout')
        if any(type(v) is not int or v < 1 for v in (queue_size, command_queue_size)):
            raise ValueError('positive integer queue capacities required')
        self.layout=layout;self.policy=policy
        self.groups=layout.groups;self.group_size=layout.banks_per_group
        self.t=timing;self.capacity=queue_size;self.ranks=ranks;self.banks=banks
        self.state=[[Bank() for _ in range(banks)] for _ in range(ranks)]
        self.normal=[deque() for _ in range(ranks*banks)];self.pim=deque()
        self.reads=deque();self.writes=deque();self.pim_requests=deque();self.responses=[];self.trace=[]
        self.pending_reads={};self.pending_writes={};self.write_draining=0
        self.command_capacity=command_queue_size;self.bank_cursor=0
        self.deadline=[(r+1)*timing.refi//ranks for r in range(ranks)]
        self.busy=[0]*ranks;self.acts=[deque() for _ in range(ranks)]
        self.rank_act=[0]*ranks;self.rd=0;self.wr=0;self.bus=0
        self.group_act=[[0]*self.groups for _ in range(ranks)]
        self.group_rd=[[0]*self.groups for _ in range(ranks)];self.group_wr=[[0]*self.groups for _ in range(ranks)]
        self.pim_ready=0;self.gwrite_ready=0;self.reserved=None
        self.counts={};self.keep_trace=True;self.max_queue=0;self.rejected=0
        self.forwarded=0;self.merged_reads=0;self.merged_writes=0

    @property
    def ingress(self):return (*self.reads,*self.writes,*self.pim_requests)

    def _q(self,tx):
        if tx.kind not in ('READ','WRITE'):return self.pim
        _,r,b,_,_=self.layout.decode(tx.address)
        return self.normal[r*self.banks+b]

    @property
    def active(self):
        return bool(self.ingress or self.responses or self.pim or any(self.normal))

    def submit(self,tx,now):
        if tx.kind not in ('READ','WRITE','GWRITE','HEADER','COMPS_READRES'):raise ValueError('unsupported transaction kind')
        if not self.policy.pim_enabled and tx.kind not in ('READ','WRITE'):raise ValueError('PIM disabled by controller policy')
        if tx.arrive>now:raise ValueError('transaction has not arrived')
        ingress=self.writes if tx.kind=='WRITE' else self.reads if tx.kind=='READ' else self.pim_requests
        if len(ingress)>=self.capacity:self.rejected+=1;return False
        _,rank,bank,_,_=self.layout.decode(tx.address)
        if rank>=self.ranks or bank>=self.banks:raise ValueError('address outside controller')
        tx.accepted=now
        if tx.kind=='WRITE':
            self.responses.append((now+1,tx))
            if tx.address not in self.pending_writes:
                self.pending_writes[tx.address]=tx;ingress.append(tx)
            else:self.merged_writes+=1
        elif tx.kind=='READ':
            if tx.address in self.pending_writes:
                self.responses.append((now+1,tx));self.forwarded+=1
            else:
                group=self.pending_reads.setdefault(tx.address,[]);group.append(tx)
                if len(group)==1:ingress.append(tx)
                else:self.merged_reads+=1
        else:ingress.append(tx)
        self.max_queue=max(self.max_queue,len(ingress))
        return True

    def _schedule_transaction(self,now):
        # NeuPIMSController::ScheduleTransaction, source thresholds 8 and 2.
        if not self.write_draining and (len(self.writes)>=self.capacity or
                (len(self.writes)>self.policy.write_threshold and not self.pim and not any(self.normal))):
            self.write_draining=len(self.writes)
        if not self.pim_requests:
            queue=self.writes if self.write_draining else self.reads
        elif self.write_draining and len(self.pim)>=self.policy.pim_queue_target:queue=self.writes
        elif len(self.pim)<self.policy.pim_queue_target or not self.reads:queue=self.pim_requests
        else:queue=self.reads
        for tx in queue:
            q=self._q(tx)
            if len(q)>=self.command_capacity:continue
            if tx.kind=='WRITE':
                if tx.address in self.pending_reads:
                    self.write_draining=0;break
                self.write_draining-=1
            queue.remove(tx);tx.command_arrival=now+1;q.append(tx);break

    def _record(self,now,kind,rank,bank=None,row=None):
        if self.keep_trace:self.trace.append(dict(time=now,command=kind,rank=rank,bank=bank,row=row))
        self.counts[kind]=self.counts.get(kind,0)+1;self.bus=now+1

    def _pre(self,now,r,b):
        bank=self.state[r][b]
        if now<bank.precharge:return False
        self._record(now,'PRE',r,b,bank.row);bank.row=None;bank.activate=max(bank.activate,now+self.t.rp)
        return True

    def _pim_pre(self,now,r,b):
        bank=self.state[r][b]
        if now<bank.pim_precharge:return False
        self._record(now,'PIM_PRE',r,b,bank.pim_row);bank.pim_row=None
        bank.pim_activate=max(bank.pim_activate,now+self.t.rp)
        return True

    def _estimate(self,tx):
        _,r,b,row,_=self.layout.decode(tx.address);t=self.t
        if tx.kind=='GWRITE':
            old=self.state[r][b].row
            return t.gwrite+(0 if old==row else t.rp+t.rcd+(t.rtp if old is not None else 0))
        pre=sum(s.pim_row is not None and s.pim_row!=row for rank in self.state for s in rank)
        groups=sum(any(s.pim_row!=row for s in rank[b:b+self.group_size]) for rank in self.state for b in range(0,self.banks,self.group_size))
        return pre*t.rtp+t.faw*max(0,groups//self.ranks-1)+t.rcd+t.burst*max(tx.comps,self.policy.min_comps)+t.ccd*(tx.readouts-1)

    def _pim(self,tx,now):
        _,r,b,row,_=self.layout.decode(tx.address);t=self.t
        # Source GetReadyInPIMQueue blocks the next PIM packet until the
        # GWRITE response. Normal traffic on other banks remains eligible.
        if now<self.gwrite_ready:return False
        if tx.kind=='HEADER':
            for ri,rank in enumerate(self.state):
                for bi,s in enumerate(rank):
                    if s.row==row:
                        if self._pre(now,ri,bi):return True
                    if s.pim_row is not None and s.pim_row!=row:
                        if self._pim_pre(now,ri,bi):return True
            if any(s.row==row or (s.pim_row is not None and s.pim_row!=row) for rank in self.state for s in rank):return False
            self.reserved=row;self._q(tx).popleft();tx.issued=now;tx.ready=now+1
            self._record(now,'HEADER',-1,row=row)
            # Header is a control packet, not a DRAM data response.
            if tx.callback:tx.callback(now)
            return True
        # COMPS_READRES activates complete groups, with per-rank tFAW spacing.
        for ri,rank in enumerate(self.state):
            for first in range(0,self.banks,self.group_size):
                group=rank[first:first+self.group_size]
                if all(s.pim_row==row for s in group):continue
                if any(s.pim_row is not None for s in group):
                    raise ValueError('PIM row change requires HEADER')
                if now<max(self.busy[ri],self.rank_act[ri],self.group_act[ri][first//self.group_size],
                           *(s.pim_activate for s in group)):continue
                if now>=self.deadline[ri]:continue
                self._record(now,'G_ACT',ri,first,row)
                for s in group:
                    s.pim_row=row;s.pim_column=max(s.pim_column,now+t.rcd)
                    s.pim_precharge=max(s.pim_precharge,now+t.ras)
                    s.pim_activate=max(s.pim_activate,now+t.ras+t.rp)
                self.rank_act[ri]=now+t.faw
                return True
        if any(s.pim_row!=row or now<s.pim_column for rank in self.state for s in rank):return False
        if now<max(self.pim_ready,*self.busy):return False
        self._q(tx).popleft();tx.issued=now
        tx.ready=now+t.burst*(max(tx.comps,self.policy.min_comps)+1);self.pim_ready=tx.ready
        self._record(now,'COMPS_READRES',-1,row=row)
        for rank in self.state:
            for s in rank:s.pim_precharge=max(s.pim_precharge,now+t.rtp)
        self.rd=max(self.rd,now+t.burst+t.rtrs)
        self.wr=max(self.wr,now+t.cl+t.burst-t.wl+t.rtrs)
        self.gwrite_ready=max(self.gwrite_ready,now+t.burst)
        self.responses.append((now+t.cl+t.burst,tx));return True

    def step(self,now):
        for end,tx in list(self.responses):
            if end<=now:
                self.responses.remove((end,tx));tx.returned=now;tx.response_sent=True
                if tx.callback:tx.callback(now)
        self._schedule_transaction(now)
        if now<self.bus:return
        for r,deadline in enumerate(self.deadline):
            if now<deadline:continue
            # Refresh waits for row closure and actual bank timing, not a fixed offset.
            opened=[b for b,s in enumerate(self.state[r]) if s.row is not None]
            if opened:
                for b in opened:
                    if self._pre(now,r,b):return
                continue
            pim_opened=[b for b,s in enumerate(self.state[r]) if s.pim_row is not None]
            if pim_opened:
                for b in pim_opened:
                    if self._pim_pre(now,r,b):return
                continue
            if now>=max(self.busy[r],*(max(s.activate,s.pim_activate) for s in self.state[r])):
                self._record(now,'REF',r);self.busy[r]=now+self.t.rfc
                self.deadline[r]+=self.t.refi
                self.reserved=None
                return
        candidates=([self.pim[0]] if self.pim else [])
        if not candidates and not any(self.normal):return
        for offset in range(self.ranks*self.banks):
            q=self.normal[(self.bank_cursor+offset)%(self.ranks*self.banks)]
            if q:candidates.append(q[0])
        for tx in candidates:
            if tx.command_arrival>now:continue
            _,r,b,row,_=self.layout.decode(tx.address);bank=self.state[r][b]
            if tx.kind in ('READ','WRITE'):self.bank_cursor=(r*self.banks+b+1)%(self.ranks*self.banks)
            if tx.kind in ('GWRITE','HEADER') and not tx.admitted:
                # Positive slack relative to nominal deadlines; actual refresh
                # remains constrained by bank timing and may issue later.
                if min(self.deadline)-1-now<=self._estimate(tx):continue
                tx.admitted=True
            if tx.kind in ('HEADER','COMPS_READRES'):
                if self._pim(tx,now):return
                continue
            if tx.kind in ('READ','WRITE') and row==self.reserved:continue
            if now<self.busy[r] or now>=self.deadline[r]:continue
            if bank.pim_row==row:
                if self._pim_pre(now,r,b):return
                continue
            if bank.row is not None and bank.row!=row:
                if self._pre(now,r,b):return
                continue
            if bank.row is None:
                acts=self.acts[r]
                while acts and now-acts[0]>=self.t.faw:acts.popleft()
                if now<max(bank.activate,self.rank_act[r],self.group_act[r][b//self.group_size]) or len(acts)>=self.policy.activation_window:continue
                self._record(now,'ACT',r,b,row);bank.row=row
                bank.column=now+self.t.rcd;bank.precharge=now+self.t.ras
                bank.activate=now+self.t.ras+self.t.rp
                bank.pim_activate=max(bank.pim_activate,now+self.t.rrd)
                for g in range(self.groups):self.group_act[r][g]=max(self.group_act[r][g],now+(self.t.rrd if g==b//self.group_size else self.t.rrd_s))
                acts.append(now)
                return
            eligible=max(bank.column,self.wr if tx.kind=='WRITE' else self.rd,
                         self.group_wr[r][b//self.group_size] if tx.kind=='WRITE' else self.group_rd[r][b//self.group_size])
            if tx.kind=='GWRITE':eligible=max(eligible,self.gwrite_ready)
            if now<eligible:continue
            self._q(tx).popleft();tx.issued=now
            cmd={'READ':'RD','WRITE':'WR','GWRITE':'GWRITE'}[tx.kind]
            self._record(now,cmd,r,b,row)
            if tx.kind=='WRITE':
                self.pending_writes.pop(tx.address,None)
                tx.ready=now+self.t.wl+self.t.burst+self.t.wr
                bank.precharge=max(bank.precharge,tx.ready)
                for ri in range(self.ranks):
                    for g in range(self.groups):
                        rd=(self.t.wl+self.t.burst+(self.t.wtr if g==b//self.group_size else self.t.wtr_s)
                            if ri==r else self.t.wl+self.t.burst+self.t.rtrs-self.t.cl)
                        self.group_rd[ri][g]=max(self.group_rd[ri][g],now+rd)
                        self.group_wr[ri][g]=max(self.group_wr[ri][g],now+self.t.ccd)
                # Dual buffers remove same-bank/group write-to-PIM waits,
                # but a channel-wide COMP also touches the other groups.
                self.pim_ready=max(self.pim_ready,now+self.t.wl+self.t.burst+self.t.wtr_s)
            elif tx.kind=='GWRITE':
                tx.ready=now+self.t.gwrite;bank.precharge=max(bank.precharge,tx.ready)
                bank.column=max(bank.column,tx.ready)
                self.gwrite_ready=tx.ready;self.pim_ready=max(self.pim_ready,tx.ready)
                for g in range(self.groups):
                    self.group_rd[r][g]=max(self.group_rd[r][g],now+self.t.burst)
                    self.group_wr[r][g]=max(self.group_wr[r][g],now+self.t.cl+self.t.burst-self.t.wl+self.t.rtrs)
                self.responses.append((tx.ready,tx))
            else:
                tx.ready=now+self.t.ccd
                for ri in range(self.ranks):
                    for g in range(self.groups):
                        self.group_rd[ri][g]=max(self.group_rd[ri][g],now+self.t.burst+(self.t.rtrs if ri!=r else 0))
                        self.group_wr[ri][g]=max(self.group_wr[ri][g],now+self.t.cl+self.t.burst-self.t.wl+self.t.rtrs)
                bank.precharge=max(bank.precharge,now+self.t.rtp)
                self.pim_ready=max(self.pim_ready,now+self.t.burst)
                for request in self.pending_reads.pop(tx.address):
                    request.issued=now;request.ready=tx.ready
                    self.responses.append((now+self.t.cl+self.t.burst,request))
            if tx.audit is not None:tx.audit.update(issued=tx.issued,bank_ready=tx.ready)
            return
