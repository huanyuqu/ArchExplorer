"""Compile the supported ADF topology, with every native setting attributable."""
from dataclasses import dataclass
import hashlib

from arch_explorer.adf import COMPONENTS, SpaceMatrix, SpacePoint, fingerprint, validate, walk
from arch_explorer.components.accelsim.profile import CLOCKS, OBJECTS, PROFILE
from arch_explorer.components.accelsim.constraints import topology
from arch_explorer.components.accelsim.device import audit_parameters


@dataclass
class CompiledConfig:
    text: str
    component_map: list
    adf_sha256: str
    interconnect_text: str
    device_facts: dict = None

    @property
    def artifacts(self):
        return {'gpgpusim.config': self.text, 'config_hopper_islip.icnt': self.interconnect_text}

    @property
    def sha256(self):
        import json
        return hashlib.sha256(json.dumps(self.artifacts, sort_keys=True).encode()).hexdigest()


def compile_adf(adf):
    import arch_explorer.components  # Register the declarative catalog, no native libraries.
    validate(adf, COMPONENTS)
    original_adf = adf
    from arch_explorer.components.native_bindings import native_adf
    adf = native_adf(adf)
    fail = lambda msg: ValueError(f'accelsim unsupported topology: {msg}')
    if not isinstance(adf, SpaceMatrix) or adf.repeat != 1 or len(adf.elements) != 2 or len(adf.networks) != 1:
        raise fail('one GPU root, two groups (SM/L1 and L2/DRAM), one BookSim fly network; root repeat=1')
    groups = {}
    for group in adf.elements:
        if not isinstance(group, SpaceMatrix) or group.networks or len(group.elements) != 2:
            raise fail('groups contain exactly two native points and no nested networks')
        if any(not isinstance(p, SpacePoint) for p in group.elements):
            raise fail('nested matrices are unsupported')
        kinds = frozenset(p.component for p in group.elements)
        groups[kinds] = group
    cluster = groups.get(frozenset({'accelsim_sm', 'accelsim_l1_shared'}))
    mechanism = groups.get(frozenset({'accelsim_sm120_mechanism', 'accelsim_l1_shared'}))
    if cluster is not None and mechanism is not None:
        raise fail('exactly one SM component family required')
    cluster = mechanism if mechanism is not None else cluster
    partition = groups.get(frozenset({'accelsim_l2', 'accelsim_dram'}))
    if cluster is None or partition is None or adf.networks[0].component != 'accelsim_icnt':
        raise fail('requires SM+L1 group, L2+DRAM group and accelsim_icnt network')
    nodes = {('sm' if p.component == 'accelsim_sm120_mechanism' else p.component.removeprefix('accelsim_')): (path, p)
             for path, p in walk(adf) if isinstance(p, SpacePoint)}
    if len(nodes) != 5:
        raise fail('exactly five component types required')
    if any(p.repeat != 1 for owner, (_, p) in nodes.items() if owner != 'l2'):
        raise fail('point repeat must be 1 except L2 subpartitions per DRAM channel')
    def value(owner, key):
        if key == 'gpgpu_shmem_sizeDefault' and key not in nodes[owner][1].params:
            key = 'gpgpu_shmem_size'
        return nodes[owner][1].value(key)
    if (value('l1_shared', 'gpgpu_coalesce_arch') == 120) != (mechanism is not None):
        raise fail('SM120 sector mechanism and typed SM component must be used together')
    subpartitions = nodes['l2'][1].repeat
    topology(cluster.repeat, partition.repeat, subpartitions, value)
    audit = audit_parameters(adf)
    facts = audit.facts
    options, booksim, rows = {}, {}, []
    for owner, (path, point) in nodes.items():
        copies = cluster.repeat if owner in {'sm', 'l1_shared'} else partition.repeat * point.repeat if owner == 'l2' else partition.repeat if owner == 'dram' else 1
        for key, param in sorted(point.params.items()):
            if key == 'device_evidence':
                native_key = 'metadata:device_evidence'
            elif key == 'clock_mhz':
                native_key = 'gpgpu_clock_domains'
            elif key.startswith('booksim_'):
                native_key = 'config_hopper_islip.icnt:' + key.removeprefix('booksim_')
                booksim[key.removeprefix('booksim_')] = param.value
            else:
                native_key = key
                if key in options:
                    raise ValueError(f'duplicate native config key {key}')
                options[key] = param.value
            rows.append(dict(path=path, param=key, value=param.value, unit=param.unit,
                             source=param.source, native_key=native_key, native_object=OBJECTS[owner],
                             instances=copies,
                             clock_domain=('sm' if key in {'dram_latency', 'gpgpu_l2_rop_latency'}
                                           else owner if owner in CLOCKS else 'sm'),
                             constraint=COMPONENTS.get(point.component).params[key].domain,
                             status=audit.status(owner, key)))
    if 'gpgpu_shmem_sizeDefault' not in options:
        options['gpgpu_shmem_sizeDefault'] = options['gpgpu_shmem_size']
        original = next(row for row in rows if row['param'] == 'gpgpu_shmem_size')
        rows.append(dict(original, native_key='gpgpu_shmem_sizeDefault',
                         derived_from='gpgpu_shmem_size; native initial allocation alias, no second ADF value'))
    domains = ':'.join(str(value(owner, 'clock_mhz')) for owner in ['sm', 'icnt', 'l2', 'dram'])
    options.update(gpgpu_n_clusters=cluster.repeat, gpgpu_n_mem=partition.repeat,
                   gpgpu_n_sub_partition_per_mchannel=subpartitions, gpgpu_clock_domains=domains)
    for group, key, copies in [(cluster, 'gpgpu_n_clusters', cluster.repeat),
                               (partition, 'gpgpu_n_mem', partition.repeat),
                               (nodes['l2'][1], 'gpgpu_n_sub_partition_per_mchannel', subpartitions)]:
        path = next(path for path, node in walk(adf) if node is group)
        rows.append(dict(path=path, param='repeat', value=copies, unit='count',
                         source=('device-fact:' + facts['original']['sha256'] + '#/limits/sm_count'
                                 if facts and group is cluster else f'ADF topology; explicit scenario; {PROFILE}'), native_key=key,
                         native_object='native instance count', instances=copies,
                         status='device_fact' if facts and group is cluster else 'scenario_topology',
                         constraint='positive native instances; chiplet divisibility; power-of-two L2 subpartitions; IPOLY degree; exact fly endpoints'))
    profile = 'SM120 mechanism synthetic; NOT a Blackwell physical profile' if mechanism is not None else PROFILE
    text = f'# Generated from ADF; profile {profile}\n' + ''.join(f'-{k} {v}\n' for k, v in sorted(options.items()))
    icnt_text = '// Generated from ADF; BookSim single-stage fly\n' + ''.join(f'{k} = {v};\n' for k, v in sorted(booksim.items()))
    return CompiledConfig(text, rows, fingerprint(original_adf), icnt_text, facts)
