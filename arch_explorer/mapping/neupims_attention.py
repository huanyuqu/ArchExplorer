"""Deterministic addressed QK/softmax/AV program for shared controllers.

Command multiplicity and loop order follow the original operations. KV rows
are allocated contiguously after the circular normal-address range, with the
source PIMTensor stride. This explicit disjoint layout is not a claim about
the original whole-model allocator base address.
"""
from math import ceil
from arch_explorer.components.neupims_instructions import Region,SOURCE_SPAD_BYTES
from arch_explorer.components.pim_commands import softmax_cycles


def attention_steps(memory,whole,start,vector_free=None):
    from arch_explorer.components.stateful.instruction_execution import Await
    from arch_explorer.components.execution_profiles import NEUPIMS_INSTRUCTIONS
    profile=NEUPIMS_INSTRUCTIONS
    batch=whole.meta['batch'];seq=whole.meta['seq_len'];model=whole.meta['model']
    heads=model['heads']//whole.meta['tp'];dim=model['head_dim'];banks=memory.layout.ranks*memory.layout.banks;row_elements=memory.layout.columns*memory.layout.burst_bytes//profile.element_bytes
    if dim>row_elements or dim%16:raise ValueError('whole-COMP head dimension must fit a bank row')
    # Both source operations materialize their return slots, not the KV cache.
    # Reject input rather than increasing the configured half-SPAD capacity.
    if heads*ceil(seq/banks)*banks*2>SOURCE_SPAD_BYTES//2:
        raise ValueError('attention result slots exceed half SPAD')
    if model['kv_heads']!=model['heads']:raise ValueError('source program does not specify grouped-query attention')
    free=vector_free if vector_free is not None else [start]*profile.vector_slots
    rows_next=[memory.addresses.limit//memory.layout.row_stride+1]*memory.layout.channels
    locations=[]
    chunks=ceil(seq/row_elements);qtiles=ceil(seq/banks);pergroup=row_elements//dim
    qstride=ceil(model['hidden']/row_elements)
    for req in range(batch):
        ch=req%memory.layout.channels;base=rows_next[ch];vbase=base+qtiles*qstride
        rows_next[ch]=vbase+ceil(model['hidden']/banks)*chunks
        if rows_next[ch]>memory.layout.rows:raise ValueError('KV allocation exceeds channel row capacity')
        locations.append((ch,base,vbase))
    soft_left=[batch*heads];qk_end=[start];soft_end=[start];av_end=[start]
    def host(index,elements,ready,soft,done):
        cycles=softmax_cycles(elements*2,profile.vector_bytes_per_tick) if soft else ceil(elements*2/profile.vector_bytes_per_tick)
        begin=max(ready,free[index%profile.vector_slots]);finish=begin+cycles;free[index%profile.vector_slots]=finish
        region=Region('softmax' if soft else 'partial-add',0,0,1,elements,elements,2)
        memory.later(finish,lambda at:memory.submit(region,True,at,done))
    def soft_done(at):soft_left[0]-=1;soft_end[0]=max(soft_end[0],at)
    for req,(ch,base,_) in enumerate(locations):
        for first in range(0,heads,pergroup):
            count=min(pergroup,heads-first)
            memory.pim(ch,'GWRITE',0,start)
            for tile in range(qtiles):
                row=base+tile*qstride+first//pergroup
                memory.pim(ch,'HEADER',row,start,comps=(dim//16)*count,readouts=count)
                for h in range(count):
                    def qdone(at,index=req*heads+first+h,last=tile==qtiles-1):
                        qk_end[0]=max(qk_end[0],at)
                        if last:host(index,seq,at,True,soft_done)
                    memory.pim(ch,'COMPS_READRES',row,start,comps=dim//16,callback=qdone)
    qdone=yield Await(lambda:soft_left[0]==0,start)
    # All QK/softmax request tiles finish before Attend Operation starts.
    left=[batch*heads*ceil(dim/banks)]
    def final(at):left[0]-=1;av_end[0]=max(av_end[0],at)
    for req,(ch,_,base) in enumerate(locations):
        for h in range(heads):
            for chunk in range(chunks):
                memory.pim(ch,'GWRITE',0,qdone)
                comps=ceil(min(row_elements,seq-chunk*row_elements)/16)
                for tile in range(ceil(dim/banks)):
                    row=base+(h*ceil(dim/banks)+tile)*chunks+chunk
                    memory.pim(ch,'HEADER',row,qdone,comps=1<<(comps-1).bit_length())
                    def adone(at,index=req*heads+h,last=chunk==chunks-1):
                        if last:
                            if chunks>1:host(index,dim,at,False,final)
                            else:final(at)
                    memory.pim(ch,'COMPS_READRES',row,qdone,comps=comps,callback=adone)
    end=yield Await(lambda:left[0]==0,qdone+1)
    return end,dict(qk=(start,qk_end[0]),softmax=(start,soft_end[0]),av=(qdone,av_end[0]),
                    kv_rows=locations,kv_next=rows_next)


def attention(memory,whole,start,vector_free=None):
    """Compatibility direct runner; unified mappings use attention_steps."""
    steps=attention_steps(memory,whole,start,vector_free)
    wait=next(steps)
    while True:
        end=memory.wait(wait.predicate,wait.earliest)
        try:wait=steps.send(end)
        except StopIteration as done:return done.value
