"""IPU FP32 partials and explicit pairwise tree events, not a Poplar plan.

See docs/manycore.md for arithmetic sources and the conservative capacity policy.
"""
import math

from arch_explorer.components.base import Kernel
from arch_explorer.mapping.manycore import BlockedSchedule, output_blocks, block_inputs, block_output


def k_partitions(k, tile_k, requested):
    """Balanced contiguous groups of K blocks; no empty partials are created."""
    blocks = math.ceil(k/tile_k)
    count = min(requested, blocks)
    return [(i*blocks//count*tile_k, min(k, (i+1)*blocks//count*tile_k)) for i in range(count)]


def tree_pairs(count):
    """Each level sends right subtree roots to left roots; odd tails carry."""
    stride = 1
    while stride < count:
        yield [(i+stride, i) for i in range(0, count, 2*stride) if i+stride < count]
        stride *= 2


class SplitSchedule(BlockedSchedule):
    def __init__(self, sim, workload, params, component):
        super().__init__(sim, workload, params, component)
        if component != "ipu_tile" or any(u.vector_rate is None for u in self.units):
            raise NotImplementedError("split-K requires sourced IPU FP32 vector arithmetic")
        self.detail.update(schedule="split_k", reduction="tree", partition_plan={}, reductions=[],
                           reduction_additions=0, reduction_transfer_bytes=0,
                           reduction_arithmetic="FP32 add: vector_ops_per_cycle * clock_ghz; Graphcore tile-level parallelism 2021",
                           plan_identity="analytical mapping; not recovered Poplin/SparQ plan")
        self.detail["unmodeled"].append("FP32 reassociation rounding; compiler-specific matmul/reduction plan")
        sim.warnings.append("ipu_split_k: explicit FP32 binary tree; changed addition order, not bitwise equivalence or recovered SDK plan")

    def matrix_workspace(self, nr, nc, kk):
        # Distinct input staging, accumulator and one incoming FP32 partial.
        # All are conservatively held through the complete output wave.
        return super().matrix_workspace(nr, nc, kk) + (nr*nc*4 if self.p["partitions"] > 1 else 0)

    def record(self, tile, job, kernel, start, inputs, read_acc, write, task, partition=0):
        self.jobs.append(dict(operator=self.op.name, tile=tile, output_block=list(job), partition=partition,
                              shape=list(kernel.shape), k_start=start, input_bytes=inputs,
                              output_bytes=write, accumulator_read_bytes=read_acc, task=task,
                              isolated_compute_ns=self.units[tile].kernel_time(kernel)/self.units[tile].clock_ghz))

    def matrix_wave(self, op, wave, ranges, previous):
        p = len(ranges)
        held = [self.ctx.storage(f"partial.{op.name}.{len(self.jobs)}.{i}", self.mem[i].path,
                                 self.workspace, [previous], op.name) for i in range(len(wave)*p)]
        for i in range(len(held)):
            self.detail["capacity_peak_bytes"][i] = max(self.detail["capacity_peak_bytes"][i], self.resident[i]+self.workspace)
        init=[]
        for g,j in enumerate(wave):
            for part in range(p):
                i=g*p+part; size=j[2]*j[4]*4
                init.append(self.local(i,0,size,[previous,held[i]],op.name,"AccumulatorInit"))
                self.detail["accumulator_bytes_written"] += size
        previous=self.ctx.sync(f"{op.name}.init.{len(self.jobs)}",init,op.name,"Init")
        rounds=max(math.ceil((end-start)/self.p['tile_k']) for start,end in ranges)
        for step in range(rounds):
            work=[]; loads=[]
            for g,j in enumerate(wave):
                for part,(start,end) in enumerate(ranges):
                    k=start+step*self.p['tile_k']
                    if k>=end: continue
                    length=min(self.p['tile_k'],end-k); i=g*p+part; read=0
                    for name,rect in block_inputs(op,j,k,length):
                        for src,size in self.rect(name,rect).items():
                            read+=size
                            loads.append(self.move(src,i,size,[previous,held[i]],op.name,"Gather"))
                    work.append((i,part,j,k,length,read))
            self.events.append(dict(phase="gather",operator=op.name,tasks=loads))
            ready=self.ctx.sync(f"{op.name}.gather.{len(self.jobs)}",loads,op.name,"Exchange")
            computes=[]
            for i,part,j,k,length,read in work:
                acc=j[2]*j[4]*4
                kernel=Kernel("gemm_accumulate",(1,j[2],length,j[4]),"fp16")
                task=self.local(i,read+acc,acc,[ready,held[i]],op.name,"Compute",kernel)
                computes.append(task)
                self.detail['partial_blocks']+=1; self.detail['macs']+=j[2]*j[4]*length
                self.detail['accumulator_bytes_read']+=acc; self.detail['accumulator_bytes_written']+=acc
                self.record(i,j,kernel,k,read,acc,acc,task,part)
            self.events.append(dict(phase="compute",operator=op.name,tasks=computes))
            previous=self.barrier(computes,op.name,"compute_sync")
        for level,pairs in enumerate(tree_pairs(p)):
            transfers=[]; pending=[]
            for g,j in enumerate(wave):
                for child,parent in pairs:
                    src,dst=g*p+child,g*p+parent; size=j[2]*j[4]*4
                    transfer=self.move(src,dst,size,[previous,held[src],held[dst]],op.name,"ReduceTransfer")
                    transfers.append(transfer); pending.append((src,dst,j,size,transfer))
                    self.detail['reduction_transfer_bytes']+=size
            self.events.append(dict(phase="reduce_transfer",operator=op.name,tasks=transfers,level=level))
            ready=self.ctx.sync(f"{op.name}.reduce_join.{len(self.jobs)}.{level}",transfers,op.name,"ReduceTransfer")
            adds=[]
            for src,dst,j,size,transfer in pending:
                kernel=Kernel('fp32_add',(j[2]*j[4],),'fp32')
                task=self.local(dst,2*size,size,[ready,held[dst]],op.name,'ReduceAdd',kernel)
                adds.append(task)
                self.detail['reduction_additions']+=j[2]*j[4]
                self.detail['accumulator_bytes_read']+=2*size
                self.detail['accumulator_bytes_written']+=size
                self.detail['reductions'].append(dict(operator=op.name,output_block=list(j),level=level,
                    source=src,destination=dst,bytes=size,additions=j[2]*j[4],transfer=transfer,task=task,
                    isolated_compute_ns=self.units[dst].kernel_time(kernel)/self.units[dst].clock_ghz))
            self.events.append(dict(phase="reduce_add",operator=op.name,tasks=adds,level=level))
            previous=self.barrier(adds,op.name,'reduce_sync')
        finals=[]
        for g,j in enumerate(wave):
            i=g*p; cells=j[2]*j[4]
            finals.append(self.local(i,4*cells,2*cells,[previous,held[i]],op.name,'FinalStore'))
            self.detail['accumulator_bytes_read']+=4*cells
        previous=self.ctx.sync(f"{op.name}.final.{len(self.jobs)}",finals,op.name,'FinalStore')
        scatters=[]
        for g,j in enumerate(wave):
            i=g*p
            for dst,size in self.rect(op.outputs[0],block_output(op,j)).items():
                scatters.append(self.move(i,dst,size,[previous,held[i]],op.name,'Scatter'))
        self.events.append(dict(phase='scatter',operator=op.name,tasks=scatters))
        return self.barrier([*scatters,*held],op.name,'exchange_sync')

    def vector_wave(self,op,rows,previous):
        held=[]; loads=[]
        n=op.attrs['cols']; tensor=self.w.tensors[op.inputs[0]]
        for i,row in enumerate(rows):
            held.append(self.ctx.storage(f"vector.{op.name}.{row}",self.mem[i].path,self.workspace,[previous],op.name))
            self.detail['capacity_peak_bytes'][i]=max(self.detail['capacity_peak_bytes'][i],self.resident[i]+self.workspace)
            rect=(row//tensor.shape[-2],row%tensor.shape[-2],1,0,n)
            for src,size in self.rect(op.inputs[0],rect).items():
                loads.append(self.move(src,i,size,[previous,held[i]],op.name,'Gather'))
        self.events.append(dict(phase='gather',operator=op.name,tasks=loads))
        ready=self.ctx.sync(f"{op.name}.join.{rows[0]}",loads,op.name,'Exchange')
        computes=[]
        for i,row in enumerate(rows):
            kernel=op.kernel('fp16',rows=1)
            task=self.local(i,n*2,n*2,[ready,held[i]],op.name,'Compute',kernel)
            computes.append(task); self.record(i,(row,0,1,0,n),kernel,0,n*2,0,n*2,task)
        self.events.append(dict(phase='compute',operator=op.name,tasks=computes))
        previous=self.barrier(computes,op.name,'compute_sync')
        scatters=[]; tensor=self.w.tensors[op.outputs[0]]
        for i,row in enumerate(rows):
            rect=(row//tensor.shape[-2],row%tensor.shape[-2],1,0,n)
            for dst,size in self.rect(op.outputs[0],rect).items():
                scatters.append(self.move(i,dst,size,[previous,held[i]],op.name,'Scatter'))
        self.events.append(dict(phase='scatter',operator=op.name,tasks=scatters))
        return self.barrier([*scatters,*held],op.name,'exchange_sync')

    def run(self):
        self._layouts()
        previous=self.barrier(self.holds,'program','initial_sync')
        for op in self.w.operators:
            self.op=op
            if op.op in ('matmul','mvm'):
                ranges=k_partitions(op.attrs['k'],self.p['tile_k'],self.p['partitions'])
                p=len(ranges)
                if p>self.active:
                    raise ValueError(f"split-K needs {p} simultaneous partial workspaces; only {self.active} tiles fit capacity")
                self.detail['partition_plan'][op.name]=dict(requested=self.p['partitions'],effective=p,ranges=ranges,
                                                           output_groups_per_wave=self.active//p)
                jobs=list(output_blocks(op,self.p['tile_m'],self.p['tile_n']))
                if len(jobs)>8192: raise ValueError('explicit manycore schedule exceeds 8192 output blocks')
                for offset in range(0,len(jobs),self.active//p):
                    previous=self.matrix_wave(op,jobs[offset:offset+self.active//p],ranges,previous)
            else:
                for offset in range(0,op.attrs['rows'],self.active):
                    previous=self.vector_wave(op,list(range(offset,min(offset+self.active,op.attrs['rows']))),previous)
            if len(self.ctx.graph.tasks)>200000: raise ValueError('explicit manycore task budget exceeded (200000)')
        self.detail['capacity_peak_bytes']=[max(a,b) for a,b in zip(self.detail['capacity_peak_bytes'],self.resident)]
        self.detail['isolated_service_ns']=[dict(sram=(b['local_read']+b['local_write'])/self.mem[i].port.capacity,
            send=b['sent']/self.net.bandwidth if self.net.bandwidth else None,
            receive=b['received']/self.net.bandwidth if self.net.bandwidth else None) for i,b in enumerate(self.bytes)]
        self.detail['remote_service_ns']=(sum(b['remote_read'] for b in self.bytes)/self.remote[0].port.capacity
                                        if self.w.meta.get('kv_storage')=='remote' else None)
        self.detail['parameters']={u.type_path:{k:v.to_dict() for k,v in u.point.params.items()}
                                   for u in [self.units[0],self.mem[0],self.net,*self.remote]}
