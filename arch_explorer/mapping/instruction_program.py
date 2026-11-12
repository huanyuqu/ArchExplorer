"""NeuPIMs profile compiler into ordinary TaskGraph program/Demand nodes."""
from dataclasses import replace
from math import ceil
from arch_explorer.adf.registry import COMPONENTS, MAPPING_TEMPLATES
from arch_explorer.components.neupims_instructions import Region, matmul_tiles, SOURCE_SPAD_BYTES
from arch_explorer.mapping.base import sim_devices
from arch_explorer.mapping.pim import PimOffload, segment
from arch_explorer.mapping.pim_attention import AttentionProgram
from arch_explorer.components.stateful.instruction_execution import Await, InstructionProgramModel, matrix_program
from arch_explorer.components.stateful.neupims_memory import TransactionMemory, BandwidthMemory
from arch_explorer.simulator.plan import PlanConstraints


def vector_program(op, whole, vectors):
    def program(model, start):
        memory = model.memory
        left = [len(op.inputs)]
        def ready(at): left[0] -= 1
        for name in op.inputs:
            count = whole.tensors[name].bytes//2
            memory.submit(Region(name,0,0,1,count,count,2),False,start,ready)
        now = yield Await(lambda: left[0] == 0, start)
        duration = max(v.kernel_time(op.kernel(whole.dtype,rows=op.attrs['rows']/len(vectors))) for v in vectors)
        now = yield Await(lambda: True, ceil(now+duration))
        left[0] = len(op.outputs)
        for name in op.outputs:
            count = whole.tensors[name].bytes//2
            memory.submit(Region(name,0,0,1,count,count,2),True,now,ready)
        end = yield Await(lambda: left[0] == 0, now+1)
        model.rows.append(dict(operator=op.name,label=op.label,task_type='compute',
            resource='vector-and-transactions',start_ns=start,end_ns=end,tasks=len(vectors)))
    return program


def attention_program(ops, whole):
    def program(model, start):
        from .neupims_attention import attention_steps
        end, spans = yield from attention_steps(model.memory, whole, start)
        for op, key in zip(ops, ('qk','softmax','av')):
            a,b = spans[key]
            model.rows.append(dict(operator=op.name,label=op.label,task_type='compute',
                resource='shared-pim-host',start_ns=a,end_ns=b,tasks=1))
        return end
    return program


def map_program(sim, spec, params):
    from arch_explorer.components.stateful.neupims_program import prepare_program, ProgramReport
    whole, _, width, state, dram = prepare_program(sim.adf,spec,params,registry=COMPONENTS,simulator=sim)
    shared = params['memory_execution'] == 'transactions'
    trace = getattr(sim, 'execution_trace', False)
    ctx = sim.mapping
    ctx.plan.require(PlanConstraints(iterations=1, reason='instruction program'))
    def create():
        memory = (TransactionMemory(whole.meta['model']['hidden'],params['tp'],trace=trace) if shared else
                  BandwidthMemory(whole.meta['model']['hidden'],params['tp'],
                                  sum(p.port.capacity for p in dram.parts),dram.latency))
        return InstructionProgramModel(memory,trace=trace)
    # Transactions own physical memory, networks and compute for this single
    # device. The bandwidth ablation interleaves nonoverlapping aggregate DAG
    # pieces with programs; no transaction controller or duplicate refresh fee.
    ctx.bind_execution('instruction-system',create,sim.resources if shared else ())
    info = dict(shared=shared, params=params, whole=whole, legacy_groups=[])
    ctx.plan.result_builder = ProgramReport(info)
    ops = [replace(o,label='PIM attention' if segment(o.name)=='attention' else segment(o.name))
           for o in whole.operators]
    vectors = [sim.points[p] for p in state.computes if sim.points[p].component.name=='vector_unit']
    last=[];i=0
    while i<len(ops):
        op=ops[i];group=[op];payload=None
        if op.name.endswith('.attn_qk'):
            group=ops[i:i+3]
            if shared:payload=attention_program(group,whole)
            i+=2
        elif op.op=='matmul':
            a=op.attrs
            def tiles(a=a):
                return matmul_tiles(a['m'],a['k'],a['n'],width,SOURCE_SPAD_BYTES,transposed=True)
            payload=matrix_program(tiles,width,width,operator=op.name,label=op.label)
        elif shared and op.op!='allreduce':
            payload=vector_program(op,whole,vectors)
        elif op.op=='allreduce':
            i+=1;continue
        if payload is not None:
            last=[ctx.program(op.name,'instruction-system',payload,last,op.name,op.label)]
        else:
            # Compile the old aggregate piece into this same graph; no nested
            # simulate(), absolute-time calendar, or isolated latency sum.
            inner=MAPPING_TEMPLATES.get('gsm_tiled')
            local=inner.device_state(sim,sim_devices(sim)[0])
            local.pim_channels=PimOffload.channels(sim,local)
            begin=len(ctx.graph.tasks)
            if len(group)==3:AttentionProgram(ctx,local,whole,*group).run()
            else:inner.run_device(ctx,local,whole,op,{})
            end=len(ctx.graph.tasks)
            for node in range(begin,end):
                # The profile permits exactly one iteration. Local static
                # allocations become one-shot allocations at this piece's entry.
                ctx.graph.tasks[node].static=False
                if not ctx.graph.preds[node]:
                    for pred in last:ctx.graph.add_edge(pred,node)
            nodes=list(range(begin,end))
            info['legacy_groups'].append((group,nodes))
            last=[ctx.sync(op.name+'.boundary',nodes)]
        i+=1
