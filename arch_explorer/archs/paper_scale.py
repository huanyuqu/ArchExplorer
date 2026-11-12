"""Dimension-complete scale requests. No roots, weights or timing are allocated.

Model dimensions come from the registered workload catalogue. Resource estimates
and graphs must subsequently come from the authoritative workload preflight.
"""
import math
import json
from pathlib import Path


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be an explicit positive integer')


def llm_scale_cases(*, batch, seq_len, dtype, phases=('prefill', 'decode'), tp=1):
    """Full declared layer counts for the four registered paper LLM families.

    Explicit batch/context/dtype are experiment choices, not claimed paper
    settings. These are layer-level performance workloads without numeric
    weights, embeddings or an end-to-end generation accuracy claim.
    """
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.llm import model_config
    for name, value in [('batch', batch), ('seq_len', seq_len), ('tp', tp)]:
        _positive(name, value)
    if dtype not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('dtype must be fp32/fp16/bf16')
    if (not isinstance(phases, (list, tuple)) or not phases or len(set(phases)) != len(phases)
            or any(p not in ('prefill', 'decode') for p in phases)):
        raise ValueError('phases requires distinct prefill/decode values')
    result = []
    for name in ('llama2-7b', 'llama2-70b', 'llama3-70b', 'qwen-72b'):
        config = model_config(name)
        for phase in phases:
            params = {'model': config.to_dict(), 'phase': phase, 'batch': batch, 'seq_len': seq_len,
                      'layers': config.layers, 'dtype': dtype, 'iterations': 1}
            spec = WorkloadSpec(kind='transformer', name=f'{name}-{phase}', params=params)
            result.append({'id': f'{name}-{phase}-b{batch}-s{seq_len}-{dtype}-tp{tp}',
                'family': 'llm-performance', 'workload': spec.to_dict(), 'tp': tp,
                'dimensions': params, 'full_declared_layer_count': True,
                'model_source': config.source,
                'experiment_choices': {'batch': batch, 'seq_len': seq_len, 'dtype': dtype, 'tp': tp},
                'numeric_weights_present': False, 'stage': 'dimensions-only',
                'pending': ['authoritative preflight', 'graph', 'mapping/compile', 'service', 'numeric reference']})
    return result


def mla_scale_cases(*, width, query_rank, content_dim, output_width, dtype, epsilon=1e-6):
    """Fig16 B/S/H/latent/position dimensions with explicit remaining choices.

    Returns parameter requests only; it never allocates the compressed cache.
    A builder/preflight is responsible for shapes, memory estimates and roots.
    """
    for name, value in [('width', width), ('query_rank', query_rank),
                        ('content_dim', content_dim), ('output_width', output_width)]:
        _positive(name, value)
    if dtype not in ('fp32', 'fp16', 'bf16') or type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Explicit supported storage dtype and positive finite epsilon required')
    result = []
    for batch in (1, 4):
        for sequence in (2048, 8192):
            paper = {'batch': batch, 'sequence': sequence, 'heads': 128, 'latent_rank': 512, 'positional_dim': 64}
            choices = {'width': width, 'query_rank': query_rank, 'content_dim': content_dim,
                       'output_width': output_width, 'dtype': dtype, 'epsilon': epsilon}
            result.append({'id': f'mla-b{batch}-s{sequence}-h128-{dtype}', 'family': 'mla-decode',
                'parameters': {**paper, **choices}, 'paper_dimensions': paper,
                'experiment_choices': choices, 'stage': 'dimensions-only',
                'topology_candidates': [{'groups': 64, 'tiles_per_group': 1, 'shared': False},
                    {'groups': 4, 'tiles_per_group': 16, 'shared': True},
                    {'groups': 8, 'tiles_per_group': 8, 'shared': True},
                    {'groups': 1, 'tiles_per_group': 64, 'shared': True}],
                'private_sram_bytes_per_tile': 1024**2, 'matrix_shape': [16, 16],
                'shared_capacity_note': 'Supply explicit shared-memory bytes; paper MB and private MiB are distinct.',
                'pending': ['authoritative preflight', 'explicit roots', 'graph', 'compile', 'service', 'numeric reference']})
    return result


def export_scale_case(directory):
    """Describe an actual exported file bundle without loading roots or weights.

    File sizes and declared exporter node counts are observations, not a host
    memory estimate or source authentication. Authoritative importer preflight
    must still validate model/external/root bytes before graph allocation.
    """
    root = Path(directory).resolve()
    manifest = json.loads((root / 'EXPORT.json').read_text())
    if manifest.get('schema') != 'arch-explorer.framework-export.v1' or manifest.get('status') != 'exported-and-checked':
        raise ValueError('A completed actual framework export is required')
    shapes = json.loads((root / 'INPUT-SHAPES.json').read_text())
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError('Original concrete root shapes required')
    for shape in shapes.values():
        if not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError('Complete positive logical root shapes required')
    files = []
    for entry in manifest['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Export file lies outside its source directory or is missing')
        files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                      'exported_sha256': entry['sha256']})
    return {'id': root.name, 'family': 'framework-onnx', 'directory': str(root),
        'dimensions': {'input_shapes': shapes, 'exported_nodes': manifest['nodes']},
        'files': files, 'observed_source_bytes': sum(item['bytes'] for item in files),
        'exporter': manifest['framework'], 'stage': 'dimensions-and-file-sizes-only',
        'pending': ['authoritative source/budget preflight', 'import/graph', 'compile', 'service', 'numeric reference']}


def preflight_scale_case(case, *, resource_budget=None, build_budget=None,
                          cache_replicas=0, tiles=None, streaming=None):
    """Call the producer's exact estimates without loading roots or a second IR.

    The returned producer result is retained whole. Interface availability,
    admission, graph, compile and service remain distinct stage outcomes.
    """
    if type(case) is not dict or not isinstance(case.get('id'), str):
        raise ValueError('An explicit scale case with an id is required')
    family = case.get('family')
    if streaming is not None:
        if family != 'mla-decode':
            raise ValueError('streaming requires an explicit mla-decode scale case')
        if type(streaming) is not dict or set(streaming) != {'chunk', 'slots', 'capacity', 'cache_mode'}:
            raise ValueError('streaming requires exactly chunk, slots, capacity and cache_mode')
        if (resource_budget is not None or build_budget is not None or tiles is not None
                or type(cache_replicas) is not int or cache_replicas != 0):
            raise ValueError('Stream plan does not accept dense, host or replica budgets; supply graph resource_budget at construction')
        from arch_explorer.workload.api import mla_stream_plan
        params = dict(case['parameters'])
        if params.pop('dtype') != 'fp32':
            raise ValueError('The formal stream plan requires an explicit fp32 scale case')
        params.pop('epsilon')
        params.update(streaming)
        result = mla_stream_plan(**params)
        return {'schema': 'arch-explorer.scale-preflight.v1', 'case': case,
            'streaming': dict(streaming), 'status': 'plan-estimated',
            'producer': 'arch_explorer.workload.api.mla_stream_plan', 'result': result,
            'graph_executed': False, 'service_executed': False}
    if family == 'framework-onnx':
        from arch_explorer.workload.api import onnx_preflight
        root = Path(case['directory'])
        result = onnx_preflight(root / 'model.onnx', input_shapes=case['dimensions']['input_shapes'],
            external_root=root, resource_budget=resource_budget)
        producer = 'arch_explorer.workload.api.onnx_preflight'
    elif family == 'mla-decode':
        from arch_explorer.workload.mla_budget import mla_preflight
        params = dict(case['parameters'])
        params.pop('epsilon')
        result = mla_preflight(**params, budget=build_budget, cache_replicas=cache_replicas, tiles=tiles)
        producer = 'arch_explorer.workload.mla_budget.mla_preflight'
    elif family == 'llm-performance':
        from arch_explorer.workload.api import transformer_preflight
        if case['workload']['kind'] != 'transformer':
            raise ValueError('An LLM performance case requires the transformer workload')
        params = dict(case['workload']['params'])
        params['tp'] = case['tp']
        if resource_budget is not None:
            params['resource_budget'] = resource_budget
        result = transformer_preflight(**params)
        producer = 'arch_explorer.workload.api.transformer_preflight'
    else:
        raise ValueError('Unknown scale case family')
    return {'schema': 'arch-explorer.scale-preflight.v1', 'case': case,
        'status': 'admitted' if result['admitted'] else 'budget-refused',
        'producer': producer, 'result': result, 'graph_executed': False, 'service_executed': False}
