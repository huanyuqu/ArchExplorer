"""Opt-in NeuPIMs instruction program, separate from generic systolic models.

The source configuration runs one core/compute FIFO despite specifying eight
arrays. The IR preserves its width/8 instruction size. No empirical efficiency
or reference latency enters this executor.
"""
from math import ceil

from arch_explorer.adf.evaluation import Evaluation
from arch_explorer.adf.registry import MAPPING_TEMPLATES
from arch_explorer.components.neupims_instructions import SOURCE_SPAD_BYTES
from arch_explorer.components.neupims_transactions import Timing
from arch_explorer.components.base import Kernel
from arch_explorer.mapping.base import sim_devices
from arch_explorer.mapping.pim import PimOffload
from arch_explorer.workload.llm import workload_from_spec
from arch_explorer.components.api import build_simulator

SOURCE='NeuPIMs f299af3 configs/systolic_ws_128x128_dev.json; src/operations/MatMul.cc; src/NeuPIMSystolicWS.cc'


def prepare_program(adf, spec, params, *, registry, iterations=None, simulator=None):
    """Shared input-only preflight for discovery and execution; no timing work."""
    shared = params.get('memory_execution') == 'transactions'
    whole=workload_from_spec(spec,tp=params['tp'])
    if (params['sub_batches']!=1 or params['attention']!='pim' or params['inner']!='gsm_tiled'
            or whole.dtype!='fp16' or whole.iterations!=1 or iterations not in (None,1)):
        raise ValueError('explicit NeuPIMs program requires fp16, one iteration/sub-batch, PIM and gsm_tiled')
    sim=simulator if simulator is not None else build_simulator(adf,registry)
    arrays=[m for m in sim.points.values() if m.component.name=='systolic_array']
    if not arrays or len(sim_devices(sim))!=1:raise ValueError('one NeuPIMs NPU device required')
    width=int(arrays[0].positive('rows'))
    if any(m.positive('rows')!=width or m.positive('cols')!=width or m.clock_ghz!=1 for m in arrays):
        raise ValueError('source instruction profile requires equal square arrays at 1 GHz')
    if width not in (64,128):raise ValueError('source profile supports widths 64 and 128')
    inner=MAPPING_TEMPLATES.get('gsm_tiled');state=inner.device_state(sim,sim_devices(sim)[0])
    channels=PimOffload.channels(sim,state)
    if len(channels)!=32 or any(len(ch)!=32 for ch in channels):
        raise ValueError('source address mapping requires 32 channels and 32 banks/channel')
    dram=sim.memory(state.dram)
    if sum(t.bytes for t in whole.tensors.values())>dram.capacity:
        raise ValueError('workload resident allocation exceeds DRAM capacity')
    if shared:
        model=sim.points[channels[0][0]];timing=model.timing()
        # This is an explicit source profile, not a silent timing override.
        expected=dict(trcd=14,trp=14,tras=34,trrd=6,tfaw=30,tccd=2,tcl=14,row_bytes=1024,column_bytes=64)
        if any(getattr(timing,k)!=v for k,v in expected.items()) or timing.refresh!=(260,3900):
            raise ValueError('shared controller currently requires the source HBM2 dual-buffer timing profile')
        if any(sim.points[p].timing()!=timing or sim.points[p].row_buffer!='dual'
               or sim.points[p].ranks!=2 or sim.points[p].clock!=1 or sim.points[p].command_elements!=16
               for ch in channels for p in ch):
            raise ValueError('shared controller requires homogeneous source-profile PIM banks')
    from arch_explorer.components.stateful.neupims_memory import AddressStream
    addresses = AddressStream(whole.meta['model']['hidden'], params['tp'])
    if addresses.limit <= 0:
        raise ValueError('source address ring must contain positive whole bursts')
    if not any(sim.points[p].component.name == 'vector_unit' for p in state.computes):
        raise ValueError('source instruction profile requires host vector units')
    if not shared:
        # The A ablation always calls AttentionProgram, including on a preset
        # whose optional aggregate command flag was disabled. Mirror its input checks.
        dim = whole.meta['model']['head_dim']
        # AttentionProgram touches only request-assigned channels in this ablation.
        for ch in channels[:min(whole.meta['batch'], len(channels))]:
            model = sim.points[ch[0]]
            model.precision(Kernel('mvm', (1, whole.meta['seq_len'], dim), whole.dtype))
            if model.row_buffer != 'dual':
                raise ValueError('command attention currently requires dual row buffers')
            if len(ch) % (model.ranks * int(model.act_banks)):
                raise ValueError('command attention requires complete per-rank activation groups')
            if dim > model.timing().row_bytes // 2 or dim % (model.command_elements or 1):
                raise ValueError('QK head dimension must fit the bank row and whole COMP commands')
    if shared:
        model = whole.meta['model']
        heads = model['heads'] // whole.meta['tp']
        dim, seq, batch = model['head_dim'], whole.meta['seq_len'], whole.meta['batch']
        if dim > 512 or dim % 16:
            raise ValueError('whole-COMP head dimension must fit a bank row')
        if heads * ceil(seq / 32) * 32 * 2 > SOURCE_SPAD_BYTES // 2:
            raise ValueError('attention result slots exceed half SPAD')
        if model['kv_heads'] != model['heads']:
            raise ValueError('source program does not specify grouped-query attention')
        limit = model['hidden'] * model['hidden'] * 5 * 2 // params['tp']
        rows_per_request = ceil(seq / 32) * ceil(model['hidden'] / 512) + ceil(model['hidden'] / 32) * ceil(seq / 512)
        if (limit >> 20) + 1 + ceil(batch / 32) * rows_per_request > 32768:
            raise ValueError('KV allocation exceeds channel row capacity')
    return whole, sim, width, state, dram


class ProgramReport:
    """Evaluation-local ownership of the profile compiler's reporting inputs."""
    def __init__(self, info):
        self.info = info

    def __call__(self, sim, result, trace=False):
        return _report_program(self.info, sim, result, trace)


def report_program(sim, result, trace=False):
    """Bounded compatibility for the original reporting helper signature."""
    reporter = sim.mapping.plan.result_builder
    if not isinstance(reporter, ProgramReport):
        raise ValueError('no instruction profile report bound to this execution plan')
    return reporter(sim, result, trace)


def _report_program(info,sim,result,trace=False):
    shared=info['shared']
    model=sim.execution_instances['instruction-system']
    memory=model.memory
    now=result.latency_ns;rows=list(model.rows)
    instructions=model.instructions;operations=model.operations
    for ops,nodes in info['legacy_groups']:
        for op in ops:
            rec=[r for r in result.records if r.task in nodes and r.operator==op.name and r.kind!='storage']
            if rec:rows.append(dict(operator=op.name,label=op.label,task_type='compute',resource='legacy-components',
                start_ns=min(r.start_ns if r.start_ns is not None else r.end_ns for r in rec),
                end_ns=max(r.end_ns for r in rec),tasks=len(rec)))
    rows.sort(key=lambda r:(r['start_ns'],r['end_ns']))
    if shared:
        sim.provenance.record('neupims-transactions','timing',Timing().__dict__,'core ticks',
            '[config/derived] f299af3 NewtonSim HBM2_8Gb_x128_dualpim.ini, configuration.cc and timing.cc')
    sim.provenance.record('neupims-instructions','spad_bytes',SOURCE_SPAD_BYTES,'B',
                          '[config/derived] '+SOURCE+'; Common.cc initialize_config reads sram_size for both scratchpads')
    sim.provenance.record('neupims-instructions','core_count',1,'count','[config] '+SOURCE)
    details={'neupims_instructions':dict(operations=operations,logical_bytes=memory.logical,
        physical_bytes=memory.physical,address_order='first-touch; upstream uses robin_hood hash traversal',
        core_count=1,spad_bytes=SOURCE_SPAD_BYTES,weighted_matmul_transposed=True,source=SOURCE),
        'stats':dict(result.stats,scheduler='conservative',tasks=sum(r['tasks'] for r in rows))}
    if trace:details['instruction_trace']=instructions
    from arch_explorer.components.execution_profiles import NEUPIMS_INSTRUCTIONS, PROFILE_SOURCE
    details['execution'] = dict(
        component='instruction-system', scheduler='conservative',
        ownership=dict(sim.mapping.graph.execution_owners),
        profile=dict(NEUPIMS_INSTRUCTIONS.__dict__), source=PROFILE_SOURCE,
        state_persists_across_operators=True, evaluation_local=True,
        completion_boundary='consumer_ready; physical recovery may remain pending',
        capacity_contract='two slots; each tile footprint <= min(spad, accum)/2',
        admission_capacity_bytes=dict(spad=NEUPIMS_INSTRUCTIONS.spad_bytes,
                                      accumulator=NEUPIMS_INSTRUCTIONS.accum_bytes))
    if trace:
        details['execution']['records']=[dict(r.__dict__) for r in result.records]
    limitations=['Optional source-configuration instruction mapping; not a hardware accuracy claim.',
                 'Normal address order is deterministic first-touch, not upstream hash iteration.',
                 'Transposed matrix output preserves logical tensor bounds; upstream MOVOUT does not transpose its output tensor.',
                 'No implicit bias instructions are inserted for workloads without bias tensors.']
    if shared:
        details['neupims_transactions']=memory.report()
        details['neupims_transactions']['pending_at_consumer_boundary']=memory.active
        if trace:
            details['transaction_trace']=memory.trace
            details['command_trace']=[dict(r,channel=i) for i,c in enumerate(memory.channels) for r in c.trace]
        limitations+=['KV rows use a declared disjoint contiguous mapping, not the original allocator base.',
            'Per-bank FIFO command order omits upstream precharge look-ahead arbitration.',
            'Normal requested burst bytes can exceed issued bytes due to forwarding/coalescing or pending writes.',
            'Vector arithmetic outside attention retains generic vector component semantics.',
            'Single sub-batch, one core, no tensor-parallel peer traffic. Layer boundary is consumer completion.']
    else:limitations+=['Ablation A retains aggregate memory service and analytical attention.']
    return Evaluation(latency_ns=now,breakdown=rows,provenance=sim.provenance.as_list(),details=details,
                      warnings=limitations)


def evaluate_program(adf,spec,mapping,*,registry,trace=False,iterations=None,**options):
    """Compatibility entry; all execution now goes through run_mapping."""
    from arch_explorer.adf.evaluation import evaluate
    return evaluate(adf,spec,mapping,backend='events',cost_model=False,registry=registry,
                    trace=trace,iterations=iterations,**options)
