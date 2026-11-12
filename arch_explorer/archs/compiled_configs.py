"""Bounded teaching configurations for the existing model-independent tools."""
from copy import deepcopy


def per_operator_mappings(workload, compute_paths, memory, *,
                          matmul_tiles=((2, 3, 5), (3, 5, 3))):
    """Two explicit placements/tile configurations, keyed by public operator ID.

    The scene selects registered matmul rules and never expands them. Operator
    ordering only makes this deterministic; it does not describe layer internals.
    Every original instance has its own placement and, where applicable, binding.
    Explicit tile proposals may be larger for graphs with many instances; the
    registered rule still validates and expands them. Existing defaults persist.
    """
    from arch_explorer.adf import MappingSpec
    from arch_explorer.workload.api import workload_from_spec
    if (not isinstance(matmul_tiles, (tuple, list)) or len(matmul_tiles) != 2
            or any(not isinstance(tile, (tuple, list)) or len(tile) != 3
                   or any(type(size) is not int or size <= 0 for size in tile)
                   for tile in matmul_tiles)):
        raise ValueError('matmul_tiles requires two positive integer (m, k, n) proposals')
    graph = workload_from_spec(workload)
    operators = graph.operators
    membership = {}
    instances = graph.meta.get('composition', {}).get('instances')
    if instances is not None:
        for index, instance in enumerate(instances):
            for name in instance['operator_names'].values():
                if name in membership:
                    raise ValueError('Composition instances overlap in original operator identity')
                membership[name] = index
        if set(membership) != {operator.name for operator in operators}:
            raise ValueError('Composition metadata must cover all original operators')
    choices = []
    for variant in (0, 1):
        placements, overrides = {}, {}
        for index, operator in enumerate(operators):
            selected = (membership.get(operator.name, index) + variant) % 2
            placements[operator.name] = compute_paths[selected]
            if operator.op == 'matmul':
                tiles = matmul_tiles[selected]
                overrides[operator.name] = {'id': 'matmul.tiled', 'version': '1',
                    'options': {'tiles': dict(zip(('m', 'k', 'n'), tiles))}}
        choices.append(MappingSpec('compile_graph_v3', {
            'compute': placements, 'memory': memory,
            'rules': {}, 'operator_rules': overrides}))
    return choices


def bounded_search_config(adf, workload, mappings, *, source):
    """Reserve one unit per mapping on each of three explicit hardware choices."""
    if not mappings:
        raise ValueError('At least one complete mapping is required')
    first = mappings[0]
    memory = first.params['memory']
    if any(item.template != first.template or item.params['memory'] != memory for item in mappings):
        raise ValueError('Teaching search requires one template and shared memory parameter domain')
    return {'seed_adf': adf.to_dict(), 'workloads': [workload.to_dict()],
            'compilation': {}, 'backend_options': {'trace': True},
            'mapping_templates': [first.template],
            'mapping_selectors': {first.template: [deepcopy(item.params) for item in mappings]},
            'max_evaluations': 3 * len(mappings), 'parameter_samples': 1,
            'mapping_samples': len(mappings), 'random_seed': 0,
            'parameter_domains': [
                {'path': memory, 'param': 'capacity', 'values': [1, 65536], 'source': source},
                {'path': memory, 'param': 'bandwidth', 'values': [16.0, 32.0], 'source': source}],
            'search': {'max_steps': 3, 'max_proposals': 3, 'proposals': [
                {'tool': 'set_parameters', 'args': {'parameters': [{
                    'path': memory, 'param': key, 'value': value, 'source': source}]}}
                for key, value in (('capacity', 1), ('bandwidth', 32.0))]}}


def scripted_agent_config(config):
    """Discover, evaluate the first explicit mapping, then submit; no network."""
    result = deepcopy(config)
    template = result['mapping_templates'][0]
    mapping = {'template': template, 'params': deepcopy(result['mapping_selectors'][template][0])}
    result.update(max_evaluations=1, agent={'max_steps': 3, 'max_tool_calls': 3, 'script': [
        {'tool': 'operator_mappings', 'args': {}},
        {'tool': 'evaluate', 'args': {'optimize': 'none', 'mappings': [mapping]}},
        {'tool': 'submit', 'args': {}}]})
    result.pop('search')
    return result
