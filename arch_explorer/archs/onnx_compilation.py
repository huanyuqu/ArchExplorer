"""User configuration for imported workloads; parsing and lowering stay in workload.api."""
from copy import deepcopy
import json
from pathlib import Path


def onnx_workload(model, *, inputs, input_shapes=None, external_root=None, name='onnx', resource_budget=None):
    """Import one real file using SIM's authoritative optional ONNX frontend.

    ``input_shapes`` maps original root names to complete positive shapes.
    The returned source snapshot includes model/external bytes and root values;
    no original absolute path is needed for serialization or subsequent replay.
    """
    from arch_explorer.workload.onnx_import import import_onnx
    # The authoritative importer performs preflight before copying roots. Do
    # not duplicate a potentially large root JSON before it can reject budgets.
    options = {} if resource_budget is None else {'resource_budget': resource_budget}
    return import_onnx(model, inputs=inputs, input_shapes=input_shapes,
                       external_root=external_root, name=name, **options)


def onnx_workload_from_files(model, *, inputs_path, input_shapes_path=None,
                             external_root=None, name='onnx', resource_budget=None):
    """Preflight the real source before loading potentially large root JSON.

    The producer owns all estimates, normalization and lowering. Default-budget
    requests keep the historical four-field portable workload representation.
    """
    from arch_explorer.workload.api import onnx_preflight
    shapes = json.loads(Path(input_shapes_path).read_text()) if input_shapes_path is not None else None
    preflight = onnx_preflight(model, input_shapes=shapes, external_root=external_root,
                               resource_budget=resource_budget)
    if not preflight['admitted']:
        raise ValueError('ONNX preflight resource budget refused: ' + json.dumps(preflight['violations'], sort_keys=True))
    path = Path(inputs_path)
    if path.stat().st_size > preflight['budget']['max_host_bytes']:
        raise ValueError('Root JSON file exceeds the declared host byte budget before loading')
    return onnx_workload(model, inputs=json.loads(path.read_text()), input_shapes=shapes,
        external_root=external_root, name=name, resource_budget=resource_budget)


def onnx_scene(workload, *, candidate=0, layout='materialized', hardware=None):
    """Place one already imported public WorkloadSpec; return (ADF, spec, mapping).

    Operator IDs and registered kinds come from the public Graph. The function
    does not inspect a model name, protobuf node order or initializer contents.
    ``hardware`` is an optional factory with the existing scene triple contract.
    """
    if candidate is None or layout is None:
        raise ValueError('A single scene requires candidate 0/1 and an explicit layout')
    adf, spec, mappings = _problem(workload, hardware=hardware)
    return adf, spec, _selected(mappings, candidate=candidate, layout=layout)[0]


def onnx_search_config(workload, *, candidate=None, layout=None, hardware=None):
    """Return an existing Env configuration: two placements × two layouts, at most 12 units."""
    adf, spec, mappings = _problem(workload, hardware=hardware)
    return _configuration(adf, spec, _selected(mappings, candidate=candidate, layout=layout))


def onnx_agent_config(workload, *, candidate=0, layout='materialized', hardware=None):
    """Return a one-unit offline discover/evaluate/submit script for the existing Agent."""
    from .compiled_configs import scripted_agent_config
    if candidate is None or layout is None:
        raise ValueError('The offline Agent requires one candidate and layout')
    return scripted_agent_config(onnx_search_config(workload, candidate=candidate,
                                                   layout=layout, hardware=hardware))


def _problem(workload, *, hardware=None):
    from arch_explorer.workload.api import layout_candidates
    from .compilation import paired_board
    from .compiled_configs import per_operator_mappings
    adf, computes, memory = paired_board(hardware)
    mappings = []
    for mapping in per_operator_mappings(workload, computes, memory,
                                         matmul_tiles=((2, 3, 4), (3, 2, 3))):
        layouts = layout_candidates(workload, mapping)
        mappings.append({name: layouts[name] for name in ('materialized', 'views')})
    return adf, workload, mappings


def _selected(mappings, *, candidate, layout):
    if candidate is not None and (type(candidate) is not int or candidate not in (0, 1)):
        raise ValueError('candidate must be None, 0 or 1')
    if layout is not None and layout not in ('materialized', 'views'):
        raise ValueError('layout must be None, materialized or views')
    return [mappings[index][name]
            for index in ((0, 1) if candidate is None else (candidate,))
            for name in (('materialized', 'views') if layout is None else (layout,))]


def _configuration(adf, workload, mappings):
    from arch_explorer.adf import find
    if not mappings:
        raise ValueError('An imported search needs at least one complete mapping')
    memory = mappings[0].params['memory']
    point = find(adf, memory)
    capacity, bandwidth = point.value('capacity'), point.value('bandwidth')
    source = 'bounded synthetic ONNX example hardware proposals; not calibrated measurements'
    return {'seed_adf': adf.to_dict(), 'workloads': [workload.to_dict()],
        'compilation': {}, 'backend_options': {'trace': True},
        'mapping_templates': ['compile_graph_v3'],
        'mapping_selectors': {'compile_graph_v3': [deepcopy(item.params) for item in mappings]},
        'max_evaluations': 3 * len(mappings), 'parameter_samples': 1,
        'mapping_samples': len(mappings), 'random_seed': 7,
        'parameter_domains': [
            {'path': memory, 'param': 'capacity', 'values': [1, capacity], 'source': source},
            {'path': memory, 'param': 'bandwidth', 'values': [bandwidth, bandwidth * 2], 'source': source}],
        'search': {'max_steps': 3, 'max_proposals': 3, 'proposals': [
            {'tool': 'set_parameters', 'args': {'parameters': [
                {'path': memory, 'param': key, 'value': value, 'source': source}]}}
            for key, value in (('capacity', 1), ('bandwidth', bandwidth * 2))]}}
