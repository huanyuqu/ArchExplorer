# Concrete Accel-Sim implementation; owned by the component provider.
"""ADF -> native component configuration -> native cycles -> Evaluation."""
import json
from pathlib import Path

from arch_explorer.adf import Evaluation, MappingSpec, WorkloadSpec, from_dict
from arch_explorer.components.accelsim.config import compile_adf
from arch_explorer.components.accelsim.capabilities import assess
from arch_explorer.components.accelsim.profile import ACCELSIM_SHA, GPGPUSIM_SHA, PROFILE, ENGINE_DEFAULTS
from arch_explorer.components.accelsim.runner import run_native
from arch_explorer.components.accelsim.stats import parse_stats
from arch_explorer.components.accelsim.trace import digest, exact, load_trace
from arch_explorer.components.accelsim.result_contract import RESULT_SCHEMA, RESULT_SCHEMAS, DETAIL_SCHEMA, COMPONENT_MAP_SCHEMA

REQUEST_SCHEMA = 'arch-explorer.accelsim-request.v1'
CAPABILITIES = {
    'profile': PROFILE, 'isa': ['sm90'],
    'trace_contracts': {'v1': 'unchanged single full warp traceg v3',
                        'v2': 'bounded complete multiwarp/CTA ordinary traceg/tracez v3/v6; dynamic masks/PCs',
                        'v3': 'SM120 typed native v7; original SIM28NV1 or registered finite SIM30NV1/ELF binding for captured inputs'},
    'instructions': 'v1 fixed subset; v2 pinned Hopper base opcode catalog and ordinary operand grammar; explicit specialized-format refusal',
    'kernel_count': 1, 'streams': [0], 'copies': False, 'concurrency': False,
    'blackwell': False, 'blackwell_scope': 'no validated physical profile; ordinary native execution only with approved SM120 mechanism engine',
    'full_models': False, 'hardware_accuracy': 'not validated',
    'physical_metrics': 'unknown', 'defaults': ENGINE_DEFAULTS}


def evaluate_accelsim(adf, workload, mapping, *, engine=None, output_dir=None, timeout_s=120,
                      max_cycles=1_000_000, memory_limit_mb=4096):
    if workload.kind in ('vit', 'clip'):
        # Complete models have their own program contract; never route them
        # through a per-kernel cold-run sum or the events throughput model.
        import tempfile
        from arch_explorer.components.accelsim.program import prepare_request, evaluate_request as evaluate_program
        from arch_explorer.components.accelsim.runner import load_engine
        load_engine(engine)  # Missing native installation is an actionable error.
        parent = None if output_dir is None else Path(output_dir).resolve()
        if parent is not None: parent.mkdir(parents=True, exist_ok=True)
        root = tempfile.mkdtemp(prefix='program-', dir=parent)
        request = prepare_request(adf, workload, mapping, output_dir=root, engine=engine,
            options={'timeout_s':timeout_s,'max_cycles':max_cycles,'memory_limit_mb':memory_limit_mb})
        return Evaluation(**evaluate_program(request)['evaluation'])
    config = compile_adf(adf)
    manifest, trace, trace_hash, expected = load_trace(workload, mapping)
    from arch_explorer.components.accelsim.runner import load_engine
    from arch_explorer.components.accelsim.variant import require_trace_engine
    capability = assess(config, manifest, expected,engine=engine)
    if capability['status'] != 'passed':
        raise ValueError('accelsim preflight blocked: ' + json.dumps(capability['blockers']))
    if engine is not None:
        native_manifest,_,_,_=load_engine(engine)
        require_trace_engine(manifest,native_manifest)
    directory, run = run_native(config, trace, engine, output_dir=output_dir,
                                timeout_s=timeout_s, max_cycles=max_cycles, memory_limit_mb=memory_limit_mb)
    stdout = (directory / 'stdout.log').read_text(errors='replace')
    stderr = (directory / 'stderr.log').read_text(errors='replace')
    try:
        # Both engines print source build identities, checked separately from
        # the manifest's pinned executable and shared-library content hashes.
        if ACCELSIM_SHA[:7] not in stdout or GPGPUSIM_SHA[:7] not in stdout:
            raise ValueError('accelsim native build identity absent/mismatched')
        if manifest['schema']=='arch-explorer.accelsim-trace.v3' and 'ARCH_EXPLORER_SM120_TYPED_V1' not in stdout:
            raise ValueError('approved typed native dispatch marker absent')
        core_mhz = next(row['value'] for row in config.component_map if row['param'] == 'clock_mhz' and row['clock_domain'] == 'sm')
        stats = parse_stats(stdout, stderr, expected, core_mhz)
    except ValueError as error:
        run.update(status='invalid_result', error=str(error))
        (directory / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        raise ValueError(f'{error}; evidence: {directory}') from error
    run['status'] = 'complete'
    run['stats'] = stats
    run['trace_manifest_sha256'] = trace_hash
    (directory / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    provenance = [{k: row[k] for k in ['path', 'param', 'value', 'unit', 'source']} for row in config.component_map]
    result = Evaluation(stats['latency_ns'], backend='accelsim', provenance=provenance,
        warnings=['Native software integration only; no Blackwell, ViT/CLIP or hardware accuracy claim.'],
        details={'schema': DETAIL_SCHEMA, 'contract': REQUEST_SCHEMA, 'capabilities': CAPABILITIES, 'preflight': capability,
                 'trace_contract': manifest['schema'], 'trace_facts': expected,
                 'engine': run['engine'], 'engine_manifest_sha256': run['engine_manifest_sha256'],
                 'trace_manifest_sha256': trace_hash, 'trace_sha256': digest(trace) if isinstance(trace,bytes) else trace.sha256, 'trace_source': manifest['source'],
                 'config_sha256': config.sha256, 'adf_sha256': config.adf_sha256,
                 'component_map': config.component_map, 'component_map_schema': COMPONENT_MAP_SCHEMA,
                 'device_facts': config.device_facts, 'native': stats,
                 'evidence': str(directory), 'run': run})
    return result


def evaluate_request(payload):
    if payload.get('schema') in RESULT_SCHEMAS:
        from arch_explorer.components.accelsim.history import replay_result
        return replay_result(payload)
    exact(payload, {'schema', 'adf', 'workload', 'mapping', 'options'}, 'accelsim CLI request')
    if payload['schema'] != REQUEST_SCHEMA:
        raise ValueError('unsupported accelsim request version')
    result = evaluate_accelsim(from_dict(payload['adf']), WorkloadSpec(**payload['workload']),
                              MappingSpec(**payload['mapping']), **payload['options'])
    return {'schema': RESULT_SCHEMA, 'request': payload, 'evaluation': result.to_dict()}


