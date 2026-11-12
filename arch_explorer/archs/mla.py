"""Small MLA consumers of public graph, topology, storage and layout contracts."""
from copy import deepcopy

from .transformer import _values


def mla_inputs(*, batch=1, sequence=5, width=6, query_rank=3, latent_rank=4,
               heads=4, content_dim=4, positional_dim=2, output_width=5):
    """Generate only explicit roots from public shapes; no cache/model answers."""
    from arch_explorer.workload.mla_decode import mla_input_shapes
    shapes = mla_input_shapes(batch=batch, sequence=sequence, width=width,
        query_rank=query_rank, latent_rank=latent_rank, heads=heads,
        content_dim=content_dim, positional_dim=positional_dim, output_width=output_width)
    roots = {}
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        if name == 'q_gamma':
            roots[name] = _values(shape, pattern='constant', scale=1.0)
        elif name in ('rope_cos', 'rope_sin'):
            roots[name] = _values(shape, pattern='constant', scale=0.6 if name == 'rope_cos' else 0.8)
        else:
            roots[name] = _values(shape, offset=11 + index * 17, scale=0.25, shift=index / 2048.0)
    return roots


def mla_workload(*, inputs=None, batch=1, sequence=5, width=6, query_rank=3,
                 latent_rank=4, heads=4, content_dim=4, positional_dim=2,
                 output_width=5, epsilon=1e-6, dtype='fp32'):
    """Return the registered absorbed-decode spec with every root retained.

    Supplied roots are passed intact to the builder. Its FP32, shape, input and
    bounded-scale checks remain authoritative. This is one decode operation
    over an explicit compressed cache, not a whole language-model layer.
    """
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import workload_from_spec
    roots = mla_inputs(batch=batch, sequence=sequence, width=width, query_rank=query_rank,
        latent_rank=latent_rank, heads=heads, content_dim=content_dim,
        positional_dim=positional_dim, output_width=output_width) if inputs is None else deepcopy(inputs)
    spec = WorkloadSpec('mla_decode', {'inputs': roots, 'batch': batch, 'sequence': sequence,
        'width': width, 'query_rank': query_rank, 'latent_rank': latent_rank, 'heads': heads,
        'content_dim': content_dim, 'positional_dim': positional_dim, 'output_width': output_width,
        'epsilon': epsilon, 'dtype': dtype})
    workload_from_spec(spec)
    return spec


def _problem(*, shared, hardware, **options):
    from arch_explorer.components.shared_memory import shared_memory_hardware
    from arch_explorer.workload.api import workload_from_spec
    if type(shared) is not bool:
        raise ValueError('shared must be bool')
    values = {'groups': 2, 'tiles_per_group': 2, 'capacity': 32768,
              'link_bandwidth': 16.0, 'port_bandwidth': 16.0, 'hop_latency': 1.0}
    if hardware is not None:
        if not isinstance(hardware, dict) or set(hardware) != set(values):
            raise ValueError('hardware must explicitly provide groups, tiles_per_group, capacity and all tree parameters')
        values = deepcopy(hardware)
    adf, topology = shared_memory_hardware(shared=shared, **values)
    spec = mla_workload(**options)
    graph = workload_from_spec(spec)
    return adf, topology, spec, graph


def _mapping(graph, spec, topology, *, multicast, candidate):
    from arch_explorer.adf import MappingSpec
    if type(multicast) is not bool or type(candidate) is not int or candidate not in (0, 1):
        raise ValueError('multicast must be bool and candidate must be 0 or 1')
    instances = graph.meta['mla']['instances']
    paths = [path for group in topology['groups'].values() for path in group['computes']]
    if not paths:
        raise ValueError('The public topology must supply compute paths')
    if candidate == 1:
        paths = list(reversed(paths))
    placement, ownership = {}, {}
    selected = 0
    for instance, record in instances.items():
        if instance in ('query', 'merge'):
            target = topology['root_compute']
        else:
            target = paths[selected % len(paths)]
            selected += 1
        for name in record['operators']:
            if name in ownership:
                raise ValueError('MLA instance metadata overlaps in original operator identity')
            ownership[name], placement[name] = instance, target
    if set(placement) != {op.name for op in graph.operators}:
        raise ValueError('MLA instance metadata must cover every original operator')
    local = {**topology['compute_storage'], topology['root_compute']: topology['root_memory']}
    outputs = {name: local[path] for name, path in placement.items()}
    storage = {name: topology['dram'] for name in spec.params['inputs']}
    replicas = []
    for source in ('latent_kv', 'pos_key'):
        destinations = {}
        for op in graph.operators:
            if source in op.inputs:
                destinations.setdefault(local[placement[op.name]], []).append(op.name)
        replicas.append({'id': source, 'source': source, 'multicast': multicast, 'destinations': [
            {'tensor': f'replica.{source}.{index}', 'memory': memory, 'consumers': consumers}
            for index, (memory, consumers) in enumerate(destinations.items())]})
    tiles = (2, 3, 4) if candidate == 0 else (3, 2, 3)
    return MappingSpec('compile_graph_v4', {'compute': placement, 'memory': outputs,
        'rules': {'matmul': {'id': 'matmul.tiled', 'version': '1',
                             'options': {'tiles': dict(zip(('m', 'k', 'n'), tiles))}}},
        'operator_rules': {}, 'storage': storage, 'replicas': replicas})


def mla_scene(*, shared=True, multicast=True, candidate=0, layout='materialized', hardware=None, **options):
    """Return one actual ADF/spec/v4 mapping; no compile or component execution."""
    from arch_explorer.workload.api import layout_candidates
    if layout not in ('materialized', 'views'):
        raise ValueError('layout must be materialized or views')
    adf, topology, spec, graph = _problem(shared=shared, hardware=hardware, **options)
    mapping = _mapping(graph, spec, topology, multicast=multicast, candidate=candidate)
    return adf, spec, layout_candidates(spec, mapping)[layout]


def mla_search_config(*, shared=True, multicast=True, hardware=None, candidate=None,
                      layout=None, **options):
    """Two placements × two layouts × three hardware choices, at most 12 units.

    Public mapping helpers own layout/rule validation. Existing Env owns all
    compilation attempts, prepare failures, winner identity and budget charges.
    """
    from arch_explorer.adf import find
    from arch_explorer.workload.api import layout_candidates
    if candidate is not None and (type(candidate) is not int or candidate not in (0, 1)):
        raise ValueError('candidate must be None, 0 or 1')
    if layout is not None and layout not in ('materialized', 'views'):
        raise ValueError('layout must be None, materialized or views')
    adf, topology, spec, graph = _problem(shared=shared, hardware=hardware, **options)
    mappings = []
    for index in (0, 1) if candidate is None else (candidate,):
        products = layout_candidates(spec, _mapping(graph, spec, topology,
                                                   multicast=multicast, candidate=index))
        mappings.extend(products[key] for key in (('materialized', 'views') if layout is None else (layout,)))
    memory = topology['root_memory']
    point = find(adf, memory)
    capacity, bandwidth = point.value('capacity'), point.value('bandwidth')
    source = 'explicit bounded synthetic shared-memory experiment; no paper timing calibration'
    return {'seed_adf': adf.to_dict(), 'workloads': [spec.to_dict()], 'compilation': {},
        'backend_options': {'trace': True}, 'mapping_templates': ['compile_graph_v4'],
        'mapping_selectors': {'compile_graph_v4': [deepcopy(mapping.params) for mapping in mappings]},
        'max_evaluations': 3 * len(mappings), 'parameter_samples': 1, 'mapping_samples': len(mappings),
        'random_seed': 7, 'parameter_domains': [
            {'path': memory, 'param': 'capacity', 'values': [1, capacity], 'source': source},
            {'path': memory, 'param': 'bandwidth', 'values': [bandwidth, bandwidth * 2], 'source': source}],
        'search': {'max_steps': 3, 'max_proposals': 3, 'proposals': [
            {'tool': 'set_parameters', 'args': {'parameters': [
                {'path': memory, 'param': key, 'value': value, 'source': source}]}}
            for key, value in (('capacity', 1), ('bandwidth', bandwidth * 2))]}}


def mla_agent_config(**options):
    """Offline discovery, one exact candidate evaluation and submit."""
    from .compiled_configs import scripted_agent_config
    return scripted_agent_config(mla_search_config(**options))
