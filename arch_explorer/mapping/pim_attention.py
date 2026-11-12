"""Explicit channel command dependencies and host head consumers.

NeuPIMs f299af3 src/operations/{NeuPIMSLogitSoftmax,NeuPIMSAttend}.cc.
This is an analytical command model, not an embedded upstream simulator.
Normal DRAM command arbitration and NPU instruction issue queues remain
unmodeled; calendar nodes describe waits, not resource utilization.
"""
import math
from functools import partial

from arch_explorer.components.base import Demand, Kernel, dtype_bytes
from arch_explorer.components.pim import gwrite_time, refresh_time, tile_time
from arch_explorer.components.pim_commands import RankRefresh, rank_activations, softmax_cycles
from arch_explorer.mapping.base import PlacementTemplate


class AttentionProgram:
    def __init__(self, ctx, state, workload, qk, softmax, av):
        self.ctx, self.state, self.workload = ctx, state, workload
        self.qk, self.softmax, self.av = qk, softmax, av
        self.seq, self.dim = qk.attrs['n'], qk.attrs['k']
        self.heads = qk.attrs['batch'] // workload.meta['batch']
        self.e = dtype_bytes(workload.dtype)
        self.vectors = [ctx.sim.points[p] for p in state.computes
                        if ctx.sim.points[p].component.name == 'vector_unit']
        if not self.vectors:
            raise ValueError('NeuPIMs head program requires vector_unit host components')
        self.serial = 0
        self.counts = dict(gwrite=0, headers=0, combined=0, comp=0, precharge=0, gact=0,
                           softmax=0, partial_add=0, return_bytes=0, write_bytes=0)
        if not hasattr(state, 'pim_command_tail'):
            state.pim_command_tail = {}

    def name(self, op, stage):
        self.serial += 1
        return f'{self.state.path}:{op.name}.c{self.channel}.{stage}.{self.serial}'

    def wait(self, op, stage, deps, delay):
        return self.ctx.sync(self.name(op, stage), deps, op.name, op.label, latency_ns=delay)

    def work(self, op, stage, deps, duration):
        # Every bank datapath works in lockstep. Channel command dependencies
        # also cover activation/calendar waits and persist across Operations.
        return self.ctx.task(self.name(op, stage), Demand({u.unit.name:duration for u in self.units}),
                             deps, 'compute', op.name, op.label)

    def calendar(self, op, stage, deps, fn):
        return self.ctx.calendar(self.name(op, stage), deps, fn, op.name, op.label)

    def admission(self, op, deps, estimate):
        if self.refresh is None:
            return deps
        return [self.calendar(op, 'refresh-admit', deps, partial(self.refresh.admit, estimate=estimate))]

    def activation_calendar(self):
        # Capture values, never the mutable mapping cursor: evaluation happens
        # after the entire graph is built, possibly for another channel/layer.
        banks, act, ranks = self.banks, int(self.model.act_banks), self.model.ranks
        timing, tick, refresh = self.t, self.tick, self.refresh
        def end(time):
            acts = rank_activations(banks,act,ranks,timing.trrd,timing.tfaw,tick,time,
                                   refresh.available if refresh else None)
            return acts[-1][0]+timing.trcd
        return end

    def gwrite(self, op, deps):
        delay = ((self.model.tccd_s+self.t.tccd)*32 if self.commands else gwrite_time(self.t))
        deps = self.admission(op, deps, self.t.trp+self.t.trcd+delay)
        if self.t.refresh and self.refresh is None:
            delay = refresh_time([(1,delay)],*self.t.refresh)
        self.counts['gwrite'] += 1
        return self.work(op, 'gwrite', deps, delay)

    def row(self, op, deps, comps, readouts):
        """Separate PRE readiness, next-combined readiness, and result return."""
        self.counts['headers'] += 1
        self.counts['combined'] += readouts
        self.counts['comp'] += comps*readouts
        self.counts['return_bytes'] += readouts*self.banks*self.e
        if not self.commands:
            duration = tile_time(self.t, self.banks, int(self.model.act_banks), self.model.macs_per_ns,
                                 self.e, self.model.command_elements, readouts, self.model.ranks, self.tick)
            deps = self.admission(op, deps, duration)
            if self.t.refresh and self.refresh is None:
                duration = refresh_time([(1,duration)], *self.t.refresh)
            ready = self.work(op, 'atomic-row', deps, duration)
            self.pre_ready = self.combined_ready = ready
            self.open_row = True
            return [ready]*readouts
        groups = self.banks // int(self.model.act_banks)
        pre = self.banks if self.open_row else 0
        estimate = pre*self.model.trtp + self.t.tfaw*max(0, groups//2-1) + self.t.trcd
        estimate += max(self.model.burst, self.model.tccd_s)*max(comps*readouts,6)
        estimate += max(self.model.burst,self.t.tccd)*(readouts-1)
        deps = self.admission(op, deps, estimate)
        if self.t.refresh and self.refresh is None:
            act = rank_activations(self.banks,int(self.model.act_banks),self.model.ranks,
                                   self.t.trrd,self.t.tfaw,self.tick)[-1][0]
            duration = pre*self.tick+self.tick+act+self.t.trcd
            duration += (readouts-1)*self.t.tccd*(max(comps,6)+1)+self.t.tcl+self.model.burst
            delay = refresh_time([(1,duration)],*self.t.refresh)-duration
            deps = [self.wait(op,'mean-refresh',deps,delay)]
        if pre:
            deps = [self.work(op, 'pim-precharge', deps, pre*self.tick)]
            self.counts['precharge'] += pre
        header = self.work(op, 'header', deps, self.tick)
        activated = self.calendar(op, 'rank-activate', [header], self.activation_calendar())
        self.counts['gact'] += groups
        deps = [activated] + ([self.combined_ready] if self.combined_ready is not None else [])
        returned = []
        for _ in range(readouts):
            issue = self.work(op, 'combined-issue', deps, self.tick)
            # issue task ends one tick after the issue timestamp.
            returned.append(self.wait(op, 'consumer', [issue], self.t.tcl+self.model.burst))
            self.combined_ready = self.wait(op, 'bank-ready', [issue],
                                            self.t.tccd*(max(comps,6)+1)-self.tick)
            self.pre_ready = self.wait(op, 'pre-ready', [issue], self.model.trtp-self.tick)
            deps = [self.combined_ready]
        self.open_row = True
        return returned

    def host(self, op, stage, deps, elements, index):
        vu = self.vectors[index % len(self.vectors)]
        size = elements*self.e
        lanes = vu.positive('lanes')
        cycles = softmax_cycles(size, lanes) if stage == 'softmax' else math.ceil(size/lanes)
        task = self.ctx.task(self.name(op, stage), Demand({vu.unit.name:cycles/vu.clock_ghz}),
                             deps, 'compute', op.name, op.label)
        self.counts['softmax' if stage == 'softmax' else 'partial_add'] += 1
        # NPUTensor::get_all_addrs includes the end address: observed byte
        # transactions are ceil(payload/64)+1, not ideal tensor payload bytes.
        size = (math.ceil(size/self.t.column_bytes)+1)*self.t.column_bytes
        self.counts['write_bytes'] += size
        early = self.model.write_response == 'early'
        physical = self.ctx.task(self.name(op, 'write-physical'), self.ctx.sim.stream(self.state.dram, vu.path, size, write=True),
                                 [task], 'comm', '' if early else op.name, 'DRAM write drain' if early else op.label)
        if self.model.write_response == 'early':
            # Acceptance uses the modeled memory request boundary. The drain
            # stays in the graph; response dependency alone may finish earlier.
            return self.wait(op, 'write-response', [task], self.tick)
        return physical

    def run(self):
        ctx, state = self.ctx, self.state
        root = state.deps(ctx, self.workload, self.qk.inputs, self.qk.name)
        root += state.deps(ctx, self.workload, [self.av.inputs[-1]], self.av.name)
        allocations = {}
        for op in (self.qk,self.softmax,self.av):
            allocations[op.name] = PlacementTemplate.allocate_outputs(None, ctx, state, self.workload, op, {}, root)
        held = [i for a in allocations.values() for ids in a.values() for i in ids]
        soft = []; qk_ends = []; final = []
        requests = self.workload.meta['batch']
        for request in range(requests):
            self.channel = request % len(state.pim_channels)
            self.units = [ctx.sim.points[p] for p in state.pim_channels[self.channel]]
            self.model = self.units[0]
            self.model.precision(Kernel(
                'mvm',(1,self.seq,self.dim),self.workload.dtype))
            if self.model.row_buffer != 'dual':
                raise ValueError('command attention currently requires dual row buffers')
            self.t, self.banks, self.tick = self.model.timing(), len(self.units), 1/self.model.clock
            self.commands = self.model.completion_mode == 'commands'
            self.refresh = (RankRefresh(*self.t.refresh, self.model.ranks, self.tick)
                            if self.t.refresh and self.model.refresh_mode == 'rank_staggered' else None)
            if self.banks % (self.model.ranks*int(self.model.act_banks)):
                raise ValueError('command attention requires complete per-rank activation groups')
            row_elements = self.t.row_bytes//self.e
            per_group = row_elements//self.dim
            if per_group < 1 or self.dim % (self.model.command_elements or 1):
                raise ValueError('QK head dimension must fit the bank row and whole COMP commands')
            self.pre_ready, self.combined_ready, self.open_row = state.pim_command_tail.get(self.channel, (None,None,False))
            deps = root+held+([self.pre_ready] if self.pre_ready is not None else [])
            per_head = []
            for first in range(0,self.heads,per_group):
                count = min(per_group,self.heads-first)
                gw = self.gwrite(self.qk,deps)
                deps = [gw]
                for tile in range(math.ceil(self.seq/self.banks)):
                    ret = self.row(self.qk,deps,self.dim//(self.model.command_elements or 1),count)
                    deps = [self.pre_ready]
                per_head.extend(ret)
            qk_done = ctx.sync(self.name(self.qk,'done'),per_head,self.qk.name,self.qk.label)
            qk_ends.append(qk_done)
            for h, ready in enumerate(per_head):
                deps = [ready] if self.model.attention_pipeline == 'heads' else [qk_done]
                soft.append(self.host(self.softmax,'softmax',deps,self.seq,request*self.heads+h))
            state.pim_command_tail[self.channel] = (self.pre_ready,self.combined_ready,self.open_row)
        all_soft = ctx.sync(self.name(self.softmax,'done'),soft,self.softmax.name,self.softmax.label)
        # Original Operation barrier: AV begins only when all QK/softmax tiles finish.
        for request in range(requests):
            self.channel = request % len(state.pim_channels)
            self.units = [ctx.sim.points[p] for p in state.pim_channels[self.channel]]
            self.model = self.units[0]
            self.t, self.banks, self.tick = self.model.timing(), len(self.units), 1/self.model.clock
            self.commands = self.model.completion_mode == 'commands'
            self.refresh = (RankRefresh(*self.t.refresh,self.model.ranks,self.tick)
                            if self.t.refresh and self.model.refresh_mode == 'rank_staggered' else None)
            self.pre_ready,self.combined_ready,self.open_row = state.pim_command_tail[self.channel]
            deps = [all_soft,self.pre_ready]
            row_elements = self.t.row_bytes//self.e
            chunks = math.ceil(self.seq/row_elements)
            for h in range(self.heads):
                partials = [[] for _ in range(math.ceil(self.dim/self.banks))]
                for chunk in range(chunks):
                    gw = self.gwrite(self.av,deps)
                    deps = [gw]
                    comps = math.ceil(min(row_elements,self.seq-chunk*row_elements)/self.model.command_elements)
                    for tile in range(len(partials)):
                        ret = self.row(self.av,deps,comps,1)
                        partials[tile].extend(ret)
                        deps = [self.pre_ready]
                for part in partials:
                    if chunks > 1 and self.model.attention_pipeline == 'heads':
                        final.append(self.host(self.av,'partial-add',part,self.dim,request*self.heads+h))
                    else:
                        final.extend(part)
            state.pim_command_tail[self.channel]=(self.pre_ready,self.combined_ready,self.open_row)
        done = ctx.sync(self.name(self.av,'done'),final,self.av.name,self.av.label)
        for op, ready in ((self.qk,ctx.sync('qk-all',qk_ends)),(self.softmax,all_soft),(self.av,done)):
            PlacementTemplate.publish(state,allocations[op.name],ready)
        ctx.details.setdefault('pim_commands',[]).append(dict(counts=self.counts,
            refresh_epoch_ns=0, boundary='callback/consumer distinct from next command readiness',
            limitations=['normal/PIM shared command-bus arbitration not modeled',
                         'rank refresh issued at nominal deadline; normal traffic postponement unknown',
                         'GWRITE normal-row preparation and NPU issue queues not modeled',
                         'calendar activation waits excluded from resource utilization']))
