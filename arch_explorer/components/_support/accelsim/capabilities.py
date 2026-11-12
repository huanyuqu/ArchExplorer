# Concrete Accel-Sim implementation; owned by the component provider.
"""Input/resource preflight without loading an engine or querying a device.

The result separates executable software from physical device validation. It
never substitutes the H100 profile for a target's missing ISA or parameters.
"""
import argparse
import json
import math
from pathlib import Path
import struct

from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
from arch_explorer.components.accelsim.config import compile_adf
from arch_explorer.components.accelsim.profile import PROFILE
from arch_explorer.components.accelsim.trace import exact, load_trace
from arch_explorer.components.accelsim.trace_stream import file_digest
from arch_explorer.components.accelsim.native_isa import ISAError, isa_blocker, isa_facts


def assess(config, manifest, actual, *, purpose='software', target=None, engine=None):
    if purpose not in ('software', 'hardware-accuracy'):
        raise ValueError('unknown capability purpose')
    result = {'schema': 'arch-explorer.accelsim-capability.v1', 'purpose': purpose,
              'profile': PROFILE, 'trace_schema': manifest['schema'],
              'source_declaration': manifest['source'], 'status': 'passed',
              'blockers': [], 'hardware_accuracy': 'not validated',
              'assumptions': ['one complete kernel; stream 0; native fresh simulation state',
                              'no host execution, copies, application state or concurrency',
                              'ordinary SM90 semantics; non-R scoreboard dependencies are not modeled']}
    values = {row['param']: row['value'] for row in config.component_map if row['param'] != 'repeat'}
    if config.device_facts:
        result['device_facts'] = config.device_facts
        if math.prod(actual['block']) > config.device_facts['launch_facts']['max_threads_block']:
            result['blockers'].append({'kind': 'device_fact_resource', 'reason': 'block exceeds captured device per-block thread bound'})
    typed=manifest['schema']=='arch-explorer.accelsim-trace.v3'
    if typed:
        from arch_explorer.components.accelsim.runner import load_engine
        from arch_explorer.components.accelsim.variant import require_trace_engine,PROFILE as MECHANISM_PROFILE
        native,_,_,engine_hash=load_engine(engine)
        require_trace_engine(manifest,native)
        if values.get('sim28_typed_regs')!=1 or values.get('gpgpu_coalesce_arch')!=120:
            raise ValueError('typed SM120 requires its explicit synthetic bank/coalescing ADF')
        result.update(profile=MECHANISM_PROFILE,engine_manifest_sha256=engine_hash,
            assumptions=['one complete observed or explicitly synthetic SM120 kernel',
                'R/UR/P/UP logical dependencies; four synthetic bank spaces, uniform operand access one logical unit',
                'no validated Blackwell register-bank/uniform-pipeline/cache/network/DRAM/clock physical parameters'])
    elif values.get('sim28_typed_regs'):
        raise ValueError('SM120 mechanism ADF cannot silently consume a legacy SM90 trace')
    if 'nregs' in actual:
        # Same launch feasibility calculation as pinned shader.cc:4407-4436.
        # This is a resource check, never a Python latency/occupancy correction.
        padded = math.ceil(math.prod(actual['block']) / 32) * 32
        registers = padded * ((actual['nregs'] + 3) & ~3)
        thread_capacity = int(values['gpgpu_shader_core_pipeline'].split(':')[0])
        capacities = {'threads': thread_capacity // padded,
                      'registers': values['gpgpu_shader_registers'] // registers,
                      'ctas': values['gpgpu_shader_cta']}
        if actual['shmem']:
            capacities['shared'] = values['gpgpu_shmem_size'] // actual['shmem']
        result['resources'] = {'padded_threads': padded, 'allocated_registers': registers,
                               'shared_bytes': actual['shmem'], 'cta_capacity_by_limit': capacities,
                               'source': 'pinned GPGPU-Sim shader.cc:4407-4436; conservative per-block ADF bounds'}
        if (min(capacities.values()) < 1 or registers > values['gpgpu_registers_per_block']
                or actual['shmem'] > values['gpgpu_shmem_per_block']):
            result['blockers'].append({'kind': 'adapter_resource', 'reason': 'actual kernel resources exceed native/declared ADF block capacity'})
    if purpose == 'hardware-accuracy':
        if not isinstance(target, dict) or set(target) != {'gpu', 'isa', 'evidence_sha256'}:
            result['blockers'].append({'kind': 'target_identity', 'reason': 'target gpu/isa/evidence_sha256 required'})
        else:
            result['target_isa'] = isa_facts(target['isa'],engine=engine if typed else None)
            blocker = isa_blocker(target['isa'],engine=engine if typed else None)
            if blocker:
                result['blockers'].append(blocker)
        # No calibrated target profile/closed validation window has been supplied
        # for any device, including the trace source. Passing software is weaker.
        result['blockers'].append({'kind': 'physical_parameters', 'reason': 'no independently validated same-device native resource/cache/network/DRAM/clock parameter set'})
    result['status'] = 'blocked' if result['blockers'] else 'passed'
    return result


def preflight_request(payload, *, purpose='software', target=None):
    """Return a machine-readable refusal as well as a reachable software pass."""
    try:
        from arch_explorer.components.accelsim.result_contract import RESULT_SCHEMAS
        if payload.get('schema') in RESULT_SCHEMAS:
            payload = payload['request']
        exact(payload, {'schema', 'adf', 'workload', 'mapping', 'options'}, 'accelsim preflight request')
        if payload['schema'] != 'arch-explorer.accelsim-request.v1':
            raise ValueError('unsupported accelsim request schema')
        config = compile_adf(from_dict(payload['adf']))
        manifest, _, manifest_hash, actual = load_trace(WorkloadSpec(**payload['workload']), MappingSpec(**payload['mapping']))
        result = assess(config, manifest, actual, purpose=purpose, target=target,engine=payload['options'].get('engine'))
        result.update(trace_manifest_sha256=manifest_hash, config_sha256=config.sha256,
                      actual=actual, adf_sha256=config.adf_sha256)
        return result
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        reason = str(error)
        kind = (error.kind if isinstance(error, ISAError) else 'native_isa' if 'native_isa' in reason else 'adapter' if 'adapter:' in reason
                else 'input_contract')
        return {'schema': 'arch-explorer.accelsim-capability.v1', 'purpose': purpose,
                'status': 'blocked', 'blockers': [{'kind': kind, 'reason': reason}],
                'hardware_accuracy': 'not validated'}


def preflight_cubin(path, *, engine=None):
    """Inspect existing CUDA ELF bytes, without a device or a dynamic-trace claim.

    CUDA ELF64 little-endian e_flags bits 8..15 are the binary SM identifier;
    the delivery checks it against the fixed offline NVIDIA disassembler.
    Architecture feature flags remain recorded as raw bits, not guessed away.
    """
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(64)
    if len(header) != 64 or header[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', header, 18)[0] != 190:
        raise ValueError('requires CUDA ELF64 little-endian cubin, not PTX/text/trace')
    flags = struct.unpack_from('<I', header, 48)[0]
    sm = (flags >> 8) & 0xff
    blockers = [{'kind': 'dynamic_trace_missing', 'reason': 'static cubin is not an observed dynamic instruction/address/mask stream'}]
    blocker = isa_blocker('sm'+str(sm),engine=engine)
    if blocker:
        blockers.insert(0, blocker)
    return {'schema': 'arch-explorer.accelsim-capability.v1', 'status': 'blocked',
            'input_kind': 'static-cubin', 'sha256': file_digest(path), 'bytes': path.stat().st_size,
            'elf_flags': hex(flags), 'isa': 'sm'+str(sm), 'isa_capability': isa_facts('sm'+str(sm),engine=engine), 'blockers': blockers,
            'hardware_accuracy': 'not validated', 'device_queried': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path, nargs='?')
    parser.add_argument('--cubin', type=Path, help='inspect existing static CUDA bytes; no trace reconstruction')
    parser.add_argument('--purpose', choices=['software', 'hardware-accuracy'], default='software')
    parser.add_argument('--target', type=Path, help='existing evidence identity JSON; never queried from GPU')
    parser.add_argument('--engine', type=Path, help='optional exact native variant for static ISA assessment')
    args = parser.parse_args()
    if (args.request is None) == (args.cubin is None):
        parser.error('supply exactly one request or --cubin')
    result = preflight_cubin(args.cubin,engine=args.engine) if args.cubin else preflight_request(json.loads(args.request.read_text()), purpose=args.purpose,
                               target=json.loads(args.target.read_text()) if args.target else None)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
