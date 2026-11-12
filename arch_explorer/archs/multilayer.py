"""Explicit multi-layer teaching inputs; graph semantics live in workload.api."""
from copy import deepcopy

from .transformer import _values


def encoder_inputs(*, layers=2, batch=2, seq=3, hidden=8, heads=2, mlp=16,
                   input_case='regular'):
    """Return one X and independent parameter dictionaries for 2–4 blocks.

    Layer dictionaries contain the sixteen weights, biases and normalization
    parameters, never a second X or a precomputed activation. Values are finite
    synthetic data; the public workload API owns all precision and computation.
    """
    from arch_explorer.workload.api import transformer_block_input_shapes
    if type(layers) is not int or not 2 <= layers <= 4:
        raise ValueError('layers must be an integer from 2 through 4')
    if input_case not in ('regular', 'boundary', 'constant'):
        raise ValueError('input_case must be regular, boundary or constant')
    shapes = transformer_block_input_shapes(batch=batch, sequence=seq, hidden=hidden,
                                            heads=heads, mlp_hidden=mlp)
    parameters = []
    for layer in range(layers):
        values = {}
        for index, (name, shape) in enumerate(sorted(shapes.items())):
            if name == 'X':
                continue
            offset = (layer + 1) * 19 + index * 5
            if name in ('norm1_gamma', 'norm2_gamma'):
                values[name] = _values(shape, offset=offset, scale=0.125,
                                       shift=1.0 + layer / 64.0)
            elif name in ('norm1_beta', 'norm2_beta', 'bq', 'bk', 'bv', 'bo', 'b1', 'b2'):
                values[name] = _values(shape, offset=offset, scale=0.0625,
                                       shift=layer / 1024.0)
            else:
                values[name] = _values(shape, offset=offset, scale=0.25,
                                       shift=layer / 512.0)
        parameters.append(values)
    return {'X': _values(shapes['X'], pattern=input_case), 'layers': parameters}


def _hardware(hardware):
    """Two independently bound compute points in the same teaching-board domain."""
    from .compilation import paired_board
    return paired_board(hardware)


def encoder_workload(*, layers=2, dtype='fp32', batch=2, seq=3, hidden=8, heads=2,
                     mlp=16, epsilon=1e-5, input_case='regular', inputs=None):
    """Return the public composed encoder spec with one explicit model input."""
    from arch_explorer.workload.api import transformer_encoder
    if type(layers) is not int or not 2 <= layers <= 4:
        raise ValueError('layers must be an integer from 2 through 4')
    values = encoder_inputs(layers=layers, batch=batch, seq=seq, hidden=hidden,
                            heads=heads, mlp=mlp, input_case=input_case) if inputs is None else deepcopy(inputs)
    if isinstance(values, dict) and isinstance(values.get('layers'), list) and len(values['layers']) != layers:
        raise ValueError('Explicit inputs.layers count differs from the requested layers')
    return transformer_encoder(inputs=values, batch=batch, sequence=seq, hidden=hidden,
                               heads=heads, mlp_hidden=mlp, dtype=dtype, epsilon=epsilon)


def _mappings(workload, compute_paths, memory):
    """Read opaque operator identities from the public composition metadata."""
    from arch_explorer.adf import MappingSpec
    from arch_explorer.workload.api import workload_from_spec
    graph = workload_from_spec(workload)
    operators = {op.name: op for op in graph.operators}
    choices = []
    for variant in (0, 1):
        placement, overrides = {}, {}
        for index, instance in enumerate(graph.meta['composition']['instances']):
            selected = (index + variant) % 2
            tiles = ((2, 3, 5), (3, 5, 3))[selected]
            for name in instance['operator_names'].values():
                placement[name] = compute_paths[selected]
                if operators[name].op == 'matmul':
                    overrides[name] = {'id': 'matmul.tiled', 'version': '1', 'options': {
                        'tiles': dict(zip(('m', 'k', 'n'), tiles))}}
        if set(placement) != set(operators):
            raise ValueError('Public encoder instance metadata does not cover its operators')
        choices.append(MappingSpec('compile_graph_v3', {
            'compute': placement, 'memory': memory,
            'rules': {'matmul': {'id': 'matmul.tiled', 'version': '1', 'options': {
                'tiles': {'m': 2, 'k': 3, 'n': 5}}}}, 'operator_rules': overrides}))
    return choices


def _layout_mapping(workload, mapping, layout):
    """Consume the producer's complete candidate without resolving layouts."""
    from arch_explorer.workload.api import layout_candidates
    return layout_candidates(workload, mapping)[layout]


def _selection(candidate, layout):
    if type(candidate) is not int or candidate not in (0, 1):
        raise ValueError('candidate must be 0 or 1')
    if layout not in ('materialized', 'views'):
        raise ValueError('layout must be materialized or views')


def _problem(*, layers, dtype, batch, seq, hidden, heads, mlp, epsilon, input_case,
             inputs, hardware):
    adf, points, memory = _hardware(hardware)
    workload = encoder_workload(layers=layers, dtype=dtype, batch=batch, seq=seq,
        hidden=hidden, heads=heads, mlp=mlp, epsilon=epsilon, input_case=input_case, inputs=inputs)
    return adf, workload, _mappings(workload, points, memory)


def encoder_scene(*, layers=2, dtype='fp32', batch=2, seq=3, hidden=8, heads=2,
                  mlp=16, epsilon=1e-5, input_case='regular', inputs=None,
                  hardware=None, candidate=0, layout='materialized'):
    """Return ADF, public composed WorkloadSpec and one exact instance mapping.

    Two same-domain compute points alternate by explicit instance membership.
    Candidate 1 swaps both positions and nondivisible tile triples. The public
    layout candidate helper selects materialization or supported readonly views
    for this same workload. Frontends do not resolve layout or operator rules.
    """
    _selection(candidate, layout)
    adf, workload, mappings = _problem(layers=layers, dtype=dtype, batch=batch, seq=seq,
        hidden=hidden, heads=heads, mlp=mlp, epsilon=epsilon, input_case=input_case,
        inputs=inputs, hardware=hardware)
    return adf, workload, _layout_mapping(workload, mappings[candidate], layout)


def encoder_search_config(*, layers=2, dtype='fp32', batch=2, seq=3, hidden=8, heads=2,
                          mlp=16, epsilon=1e-5, input_case='regular', inputs=None,
                          hardware=None, candidate=0, layout='materialized',
                          compare_layouts=True):
    """Existing search over public layouts, instance tiles and hardware proposals.

    Both layouts use the same workload and placement choices. Four candidates
    across three hardware proposals reserve twelve units; a single-layout
    control retains the original six-unit teaching configuration.
    """
    if type(compare_layouts) is not bool:
        raise ValueError('compare_layouts must be a boolean')
    _selection(candidate, layout)
    adf, workload, choices = _problem(layers=layers, dtype=dtype, batch=batch, seq=seq,
        hidden=hidden, heads=heads, mlp=mlp, epsilon=epsilon, input_case=input_case,
        inputs=inputs, hardware=hardware)
    layouts = (layout, 'views' if layout == 'materialized' else 'materialized') if compare_layouts else (layout,)
    from arch_explorer.workload.api import layout_candidates
    requests = []
    for index in (candidate, 1 - candidate):
        products = layout_candidates(workload, choices[index])
        requests.extend(products[selected] for selected in layouts)
    first = requests[0]
    source = 'bounded multi-layer example; explicit synthetic hardware proposal'
    return {'seed_adf': adf.to_dict(), 'workloads': [workload.to_dict()],
            'compilation': {}, 'backend_options': {'trace': True},
            'mapping_templates': [first.template],
            'mapping_selectors': {first.template: [deepcopy(item.params) for item in requests]},
            'max_evaluations': 3 * len(requests), 'parameter_samples': 1, 'mapping_samples': len(requests),
            'parameter_domains': [
                {'path': first.params['memory'], 'param': 'capacity', 'values': [1, 65536], 'source': source},
                {'path': first.params['memory'], 'param': 'bandwidth', 'values': [16.0, 32.0], 'source': source}],
            'random_seed': 0, 'search': {'max_steps': 3, 'max_proposals': 3, 'proposals': [
                {'tool': 'set_parameters', 'args': {'parameters': [{
                    'path': first.params['memory'], 'param': key, 'value': value, 'source': source}]}}
                for key, value in (('capacity', 1), ('bandwidth', 32.0))]}}


def encoder_agent_config(*, layers=2, dtype='fp32', batch=2, seq=3, hidden=8, heads=2,
                         mlp=16, epsilon=1e-5, input_case='regular', inputs=None,
                         hardware=None, candidate=0, layout='materialized'):
    """Offline scripted Agent: discover, execute one exact mapping and submit."""
    config = encoder_search_config(layers=layers, dtype=dtype, batch=batch, seq=seq,
        hidden=hidden, heads=heads, mlp=mlp, epsilon=epsilon, input_case=input_case,
        inputs=inputs, hardware=hardware, candidate=candidate, layout=layout, compare_layouts=False)
    mapping = {'template': config['mapping_templates'][0],
               'params': deepcopy(next(iter(config['mapping_selectors'].values()))[0])}
    config.update(max_evaluations=1, agent={'max_steps': 3, 'max_tool_calls': 3, 'script': [
        {'tool': 'operator_mappings', 'args': {}},
        {'tool': 'evaluate', 'args': {'optimize': 'none', 'mappings': [mapping]}},
        {'tool': 'submit', 'args': {}}]})
    config.pop('search')
    return config
