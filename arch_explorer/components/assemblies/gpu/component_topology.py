"""GPU assembly from native component parameters and explicit dynamic wires."""
from copy import deepcopy

from arch_explorer.adf import Connection, Param, SpacePoint, walk
from arch_explorer.adf.dynamic import validate_connections

TOPOLOGY_SCHEMA = 'arch-explorer.accelsim-topology.v1'


def _layout(adf):
    from arch_explorer.components.native_bindings import native_adf
    adf = native_adf(adf)
    from arch_explorer.components.accelsim.config import compile_adf
    compile_adf(adf)
    cluster, partition = adf.elements
    if not any(point.component in ('accelsim_sm', 'accelsim_sm120_mechanism') for point in cluster.elements):
        cluster, partition = partition, cluster
    sm = next(p for p in cluster.elements if p.component.startswith('accelsim_sm'))
    l2 = next(p for p in partition.elements if p.component == 'accelsim_l2')
    dram = next(p for p in partition.elements if p.component == 'accelsim_dram')
    def path(group, index, point, sub=None):
        a = group.name+('['+str(index)+']' if group.repeat != 1 else '')
        b = point.name+('['+str(sub)+']' if point.repeat != 1 else '')
        return '/'.join((adf.name, a, b))
    nodes = [('icnt', adf.name+'/'+adf.networks[0].name, 'icnt', 0)]
    for i in range(cluster.repeat):
        nodes.append(('sm'+str(i), path(cluster, i, sm), 'sm', i))
    for i in range(partition.repeat):
        nodes.append(('dram'+str(i), path(partition, i, dram), 'dram', i))
        for j in range(l2.repeat):
            k = i*l2.repeat+j
            nodes.append(('l2'+str(k), path(partition, i, l2, j), 'l2', k))
    edges = []
    for i in range(cluster.repeat):
        edges.extend([('sm'+str(i), 'request', 'icnt', 'request'+str(i)),
                      ('icnt', 'reply'+str(i), 'sm'+str(i), 'reply')])
    for i in range(partition.repeat*l2.repeat):
        p, q = divmod(i, l2.repeat)
        edges.extend([('icnt', 'request_mem'+str(i), 'l2'+str(i), 'request'),
                      ('l2'+str(i), 'reply', 'icnt', 'reply_mem'+str(i)),
                      ('l2'+str(i), 'dram_request', 'dram'+str(p), 'request'+str(q)),
                      ('dram'+str(p), 'reply'+str(q), 'l2'+str(i), 'dram_reply')])
    return nodes, edges


def connect_gpu(adf):
    """Copy an ordinary native ADF and explicitly connect every physical port."""
    adf = deepcopy(adf)
    from arch_explorer.components.native_bindings import native_adf
    native = native_adf(adf)
    owners = {point.component: point for _, point in walk(native) if isinstance(point, SpacePoint)}
    if (owners.get('accelsim_dram') is None or owners.get('accelsim_l2') is None
            or owners['accelsim_dram'].value('gpgpu_n_chiplet_partition') != 1
            or owners['accelsim_l2'].value('gpgpu_lrc_enabled') != 0):
        raise ValueError('six-edge assembly supports one chiplet and LRC disabled; select these explicitly')
    if any(getattr(node, 'connections', []) for _, node in walk(adf)):
        raise ValueError('connect_gpu expects an unwired ADF; existing wires are never overwritten')
    nodes, edges = _layout(adf)
    paths = {name: path for name, path, _, _ in nodes}
    adf.connections = [Connection(paths[a], ap, paths[b], bp) for a, ap, b, bp in edges]
    return adf


def gpu_adf(*, sm_clusters=4, memory_partitions=8, l2_subpartitions=2,
            clocks_mhz=None, detailed_dram=False):
    """Construct an attributed native component assembly with synthetic topology.

    Defaults are a bounded software demonstration, not measured device facts.
    All native parameter values retain the single profile vocabulary authority.
    """
    from arch_explorer.components.accelsim.scenarios import sm90_adf
    for value in (sm_clusters, memory_partitions, l2_subpartitions):
        if type(value) is not int or value <= 0:
            raise ValueError('native instance counts must be positive integers')
    adf = sm90_adf()
    adf.name = 'component_gpu'
    adf.child('cluster').repeat = sm_clusters
    adf.child('partition').repeat = memory_partitions
    adf.child('partition').child('l2').repeat = l2_subpartitions
    if type(detailed_dram) is not bool:
        raise ValueError('detailed_dram must be boolean')
    for owner, key, value in (
            ('dram', 'gpgpu_n_chiplet_partition', 1),
            ('l2', 'gpgpu_lrc_enabled', 0),
            ('dram', 'gpgpu_simple_dram_model', int(not detailed_dram))):
        point = adf.child('partition').child(owner)
        point.params[key] = Param(value, point.params[key].unit,
            'explicit six-edge component assembly support: single chiplet, no LRC, selected DRAM mechanism')
    adf.networks[0].params['booksim_k'] = Param(
        sm_clusters+memory_partitions*l2_subpartitions, 'count',
        'explicit native component topology; single-stage fly endpoint count')
    if clocks_mhz is not None:
        if not isinstance(clocks_mhz, dict) or set(clocks_mhz) != {'sm', 'icnt', 'l2', 'dram'}:
            raise ValueError('all four native clock domains must be explicit')
        for _, point in walk(adf):
            if isinstance(point, SpacePoint) and point.name in clocks_mhz:
                value = clocks_mhz[point.name]
                if type(value) is not int or value <= 0:
                    raise ValueError('clock MHz must be positive integers')
                point.params['clock_mhz'] = Param(value, 'MHz', 'explicit software clock-domain scenario')
    return connect_gpu(adf)


def compile_topology(adf):
    """Validate complete native wiring and compile the frozen native TSV ABI."""
    validate_connections(adf)
    from arch_explorer.components.native_bindings import native_adf
    adf = native_adf(adf)
    owners = {point.component: point for _, point in walk(adf) if isinstance(point, SpacePoint)}
    if (owners['accelsim_dram'].value('gpgpu_n_chiplet_partition') != 1
            or owners['accelsim_l2'].value('gpgpu_lrc_enabled') != 0):
        raise ValueError('cross-L2 chiplet/LRC routing is outside the six-edge assembly')
    nodes, edges = _layout(adf)
    names = {path: name for name, path, _, _ in nodes}
    supplied = []
    for _, node in walk(adf):
        for edge in getattr(node, 'connections', []):
            if edge.protocol != 'transaction.v1':
                raise ValueError('unsupported dynamic wire protocol')
            try:
                supplied.append((names[edge.source], edge.source_port,
                                 names[edge.destination], edge.destination_port))
            except KeyError as error:
                raise ValueError('native assembly endpoint must be an executable physical component') from error
    if len(supplied) != len(edges) or set(supplied) != set(edges):
        raise ValueError('native assembly has missing/extra/misconnected dynamic ports')
    lines = [TOPOLOGY_SCHEMA]
    lines.extend('\t'.join(('node', name, kind, str(index))) for name, _, kind, index in nodes)
    lines.extend('\t'.join(('connect', *edge)) for edge in sorted(supplied))
    return '\n'.join(lines)+'\n'


__all__ = ['gpu_adf', 'connect_gpu', 'compile_topology', 'TOPOLOGY_SCHEMA']
