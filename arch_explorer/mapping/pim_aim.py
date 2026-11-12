"""Explicit AiM GEMV mapping and synthetic concurrent host transaction driver."""
from math import ceil

from arch_explorer.adf.registry import MappingTemplate, register_template
from arch_explorer.components.aim_profile import AimCommand, AimProfile, PROFILE, SOURCE, timing_rules, parameter_sources
from arch_explorer.simulator.events.builder import build_simulator
from arch_explorer.simulator.events.engine import Task
from arch_explorer.simulator.events.protocol import ExecutionRequest
from arch_explorer.workload.operators import operator_workload
from arch_explorer.mapping.base import adf_devices
from arch_explorer.simulator.plan import PlanConstraints


def prepare(sim, workload):
    devices = adf_devices(sim.adf)
    if len(devices) != 1 or devices[0][1] != 1:
        raise ValueError('aim_shared supports one device with independent local memory controllers')
    if workload.kind != 'operator':
        raise ValueError('aim_shared currently supports operator GEMV only')
    ir = operator_workload(workload.params)
    if ir.dtype != 'bf16' or ir.operators[0].op != 'mvm' or ir.iterations != 1:
        raise ValueError('aim_shared requires BF16 mvm and one iteration; no implicit host cast')
    dims = ir.operators[0].attrs
    if not 1 <= dims['k'] <= 1024:
        raise ValueError('AiM GEMV K must fit the 1024-BF16 global buffer; multi-row accumulation unsupported')
    channels = {}
    profiles = {}
    pims = sim.find_points(component='pim_unit')
    if not pims:
        raise ValueError('aim_shared requires AiM PIM banks')
    for p in pims:
        if set(p.point.params) != {'row_buffer', 'act_banks', 'clock_ghz', 'ops_per_cycle', 'precision'}:
            raise ValueError('AiM detailed subset does not consume additional aggregate/PIM policy parameters')
        pim_units = dict(row_buffer='1', act_banks='count', clock_ghz='GHz',
                         ops_per_cycle='ops/cycle', precision='1')
        if any(p.point.params[k].unit != unit for k, unit in pim_units.items()):
            raise ValueError('AiM PIM parameters require explicit GHz/ops-per-cycle/count units')
        if (p.row_buffer, p.act_banks, p.clock, p.ops_per_cycle, p.param('precision')) != ('shared', 16, 0.5, 32, 'bf16'):
            raise ValueError('AiM effective profile requires explicit detailed ADF: single-row ACT16 BF16, 0.5GHz bank issue rate; aggregate hsndi is incompatible')
        bank = p.bank
        refresh = None if bank is None else bank.point.params.get('trfc')
        if refresh is None or refresh.unit != 'ns':
            raise ValueError('detailed AiM requires explicit trfc in ns with source (refresh_service_ns)')
        profile = AimProfile(refresh_service_ns=refresh.value, refresh_source=refresh.source)
        expected = profile.bank_values()
        if set(bank.point.params) != set(expected) or any(bank.param(k) != v for k, v in expected.items()):
            raise ValueError('AiM source profile requires matching GDDR6 bank geometry/timing; no silent override')
        units = dict(technology='1',capacity='B',bandwidth='B/ns',row_bytes='B',column_bytes='B')
        if any(bank.point.params[k].unit != units.get(k,'ns') for k in expected):
            raise ValueError('AiM bank parameters require explicit B/B-per-ns/ns units')
        path = p.path.rsplit('/', 2)[0]
        if path in profiles and profiles[path].refresh_service_ns != profile.refresh_service_ns:
            raise ValueError('AiM shared channel requires a common refresh service time across banks')
        profiles.setdefault(path, profile)
        channels.setdefault(path, []).append(p)
    if any(len(ps) != 16 for ps in channels.values()):
        raise ValueError('each AiM controller requires exactly 16 banks')
    if ceil(dims['m'] / (16 * len(channels))) + 1 > 16384:
        raise ValueError('matrix plus synthetic-host row exceeds bank capacity')
    return ir, channels, profiles


class AimShared(MappingTemplate):
    name = 'aim_shared'

    def applicable(self, adf, workload):
        try:
            prepare(build_simulator(adf), workload)
            return True
        except (ValueError, TypeError, KeyError, NotImplementedError):
            return False

    def space(self, adf, workload):
        if not self.applicable(adf, workload):
            return {}
        # Explicit template choice opts in. No dormant NeuPIMs/aggregate keys.
        return dict(profile=[PROFILE], host_traffic=['none', 'synthetic_rw'])

    def apply(self, sim, workload, params):
        if set(params) != {'profile', 'host_traffic'} or params['profile'] != PROFILE or params['host_traffic'] not in ('none', 'synthetic_rw'):
            raise ValueError('aim_shared requires exactly profile and host_traffic')
        ir, channels, profiles = prepare(sim, workload)
        ctx = sim.mapping
        ctx.plan.require(PlanConstraints(iterations=1, reason='AiM shared program'))
        m, k, batch = (ir.operators[0].attrs[key] for key in ('m', 'k', 'batch'))
        count = len(channels)
        blocks = ceil(m / (16 * count))
        warnings = [
            'AiM detailed subsystem: third-party 0f28a07b defined effective initialization; nRFC invalid in source, explicit caller refresh service required. Not measured HSNDI accuracy.',
            'Host boundary: BF16 vector supplied at controller ingress; weights pre-resident/padded. '
            'IPU compute, cast, host scheduling and exchange latency are unknown and excluded.',
            'Synthetic stream feeder: one command/client/clock, bounded admission; channel-local periodic refresh. '
            'No claim of the upstream host ISA/frontend or full simulator equivalence.',
        ]
        sim.warnings.extend(warnings)
        report = dict(profile=PROFILE, source=SOURCE, clock_ns=1., scheduler='conservative',
                      partial=True, paper_configuration_confirmed=False, host_traffic=params['host_traffic'],
                      logical_operations=2*batch*m*k, padded_operations=0,
                      logical_weight_bytes=m*k*2, controllers={}, boundary=warnings)
        ctx.details['aim'] = report
        for ci, (path, ps) in enumerate(sorted(channels.items())):
            profile = profiles[path]
            trace = []
            channel_report = dict(trace=trace, rows_reserved=blocks + 1, capacity_bytes=16*32*1024**2,
                                  global_buffer_bytes=2048, result_transfer_bytes=32,
                                  matrix_resident_bytes=blocks*32768,
                                  parameters=parameter_sources(profile),
                                  refresh_service_ns=profile.refresh_service_ns,
                                  refresh_source=profile.refresh_source, refresh_effective_ns=profile.rfc)
            report['controllers'][path] = channel_report
            def factory(channel_report=channel_report, trace=trace, profile=profile):
                from arch_explorer.components.aim_controller import AimController
                model = AimController(profile, trace)
                channel_report['bytes'] = model.bytes
                channel_report['requests'] = model.accounting
                return model
            resources = [r.name for p in ps for model in (p, p.bank) for r in model.resources]
            ctx.bind_execution(path, factory, resources)
            # Padded bank row layout. Storage persists across all request batches.
            held = [ctx.storage(f'{p.path}:matrix', p.bank.path, blocks*2048, (), 'weights', 'Storage', static=True)
                    for p in ps]
            prior = []
            for b in range(batch):
                commands = []
                for block in range(blocks):
                    first_output = (block*count + ci)*16
                    if first_output >= m:
                        continue
                    # Register bias/reset, vector load, all-bank column MAC,
                    # final BF16 result read. Tail rows/columns are zero padded.
                    address = profile.encode(0, block)
                    commands.append(AimCommand('WRMAC16', address))
                    for col in range(ceil(k/16)):
                        commands.append(AimCommand('WRGB', profile.encode(0, block, col), min(32, (k-col*16)*2)))
                    for col in range(ceil(k/16)):
                        commands.append(AimCommand('MAC16', profile.encode(0, block, col)))
                    commands.append(AimCommand('RDMAC16', address, min(16, m-first_output)*2))
                    report['padded_operations'] += ceil(k/16)*512
                if not commands:
                    continue
                name = f'{path}:gemv{b}'
                task = Task(name, operator=f'gemv{b}', execution=ExecutionRequest(name, path, operation='stream', payload=tuple(commands)),
                            meta={'label': 'AiM GEMV'})
                done = ctx.graph.add(task, prior + held)
                # Separate host and PIM requestors share the very same binding.
                # Independent reads/writes target a reserved row, not model data.
                host = []
                if params['host_traffic'] == 'synthetic_rw':
                    host_held = ctx.storage(f'{path}:host{b}', ps[0].bank.path, 2048, prior, f'host{b}', 'Host scratch')
                    host_name = f'{path}:host{b}'
                    host_commands = tuple(AimCommand(op, profile.encode(0, blocks, col))
                                          for op, col in [('WR', 0), ('RD', 1), ('WR', 2), ('RD', 3)])
                    host = [ctx.graph.add(Task(host_name, operator=f'host{b}',
                             execution=ExecutionRequest(host_name, path, operation='stream', payload=host_commands),
                             meta={'label': 'Synthetic host traffic'}), prior + [host_held])]
                prior = [done, *host]
            sim.provenance.record(path, 'execution_profile', PROFILE, '1', SOURCE)
            sim.provenance.record(path, 'controller_clock', 1., 'ns', '[defined_effective_initialization] ' + SOURCE + '/src/dram/impl/GDDR6.cpp:602-603')
            sim.provenance.record(path, 'hardware_queue_capacity', 33, 'count',
                                  SOURCE + '/src/base/request.h:532-541; max_size=32 and <= enqueue')
            sim.provenance.record(path, 'program_slots', 33, 'count',
                                  '[synthetic] frontend bound equal to hardware queue capacity, not an IPU parameter')
            sim.provenance.record(path, 'frontend_issue', 1, 'command/client/cycle',
                                  '[synthetic] deterministic request generator; no host execution/cast claim')
            for (scope, before, after), cycles in timing_rules(profile).items():
                sim.provenance.record(path, f'timing.{scope}.{before}.{after}', cycles, 'cycle',
                                      '[third-party] defined effective V values and explicit caller nRFC; ' + SOURCE + '/src/dram/impl/GDDR6.cpp:775-999')
            sim.provenance.record(path, 'refresh_interval', profile.refi, 'cycle',
                                  '[defined_effective_initialization] JEDEC_rounding(7800ns,1000ps); GDDR6.cpp:677')
            sim.provenance.record(path, 'refresh_service', profile.refresh_service_ns, 'ns',
                                  '[caller_configuration] ' + profile.refresh_source)
            sim.provenance.record(path, 'refresh_service_cycles', profile.rfc, 'cycle',
                                  '[derived] ceil(caller refresh_service_ns / 1ns)')
            sim.provenance.record(path, 'refresh_policy', 'channel-local periodic all-bank', '1',
                                  '[independent_extension] no rank in GDDR6, no mean derate; not source refresh-manager equivalence')


register_template(AimShared)
