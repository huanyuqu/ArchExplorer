# Concrete Accel-Sim implementation; owned by the component provider.
"""Versioned result annotations; native semantics stay in their original parsers.

Legacy projections are deliberately finite and source-bound. They reconstruct
only the metadata emitted by two known producer revisions, never execute an
archive, ignore arbitrary fields, or substitute a different native model.
"""
from arch_explorer.components._support.paths import package_root
from copy import deepcopy
import hashlib
from pathlib import Path

from arch_explorer.components.accelsim.profile import CHOICES, CLOCKS, FIELDS, OBJECTS, PROFILE

RESULT_SCHEMA = 'arch-explorer.accelsim-result.v1'
RESULT_SCHEMAS = (RESULT_SCHEMA,)
DETAIL_SCHEMA = 'arch-explorer.accelsim-result-details.v2'
COMPONENT_MAP_SCHEMA = 'arch-explorer.accelsim-component-map.v2'
DETAIL_FIELDS = {'contract', 'capabilities', 'preflight', 'trace_contract', 'trace_facts',
                 'engine', 'engine_manifest_sha256', 'trace_manifest_sha256', 'trace_sha256',
                 'trace_source', 'config_sha256', 'adf_sha256', 'component_map', 'native',
                 'evidence', 'run'}

# Byte identities of the producer semantics in c2d4bdc (SIM28) and 94a515c
# (SIM29). The complete archive must additionally match run.adapter_sources.
# A caller cannot opt into compatibility merely by editing a result label.
LEGACY_SOURCES = {
    'sim28': {
        'components/accelsim.py': '783d9c56d193abc9f76bdfaede623d4de94d90e06fa22f61e3b4e56a4f7ea29e',
        'simulator/accelsim/config.py': 'd24d5d0bcd83e809facd471408d16f75cebb3ef4a9d1320b80c1e1641028f95e',
        'simulator/accelsim/profile.py': '144f0e30818360ffc12f202a72a1554881b602465bf86afc78c16cec957d23d0',
        'simulator/accelsim/backend.py': 'b3db8cfc6d028a0e6c3520ddcdc89caeab9a353a2623f637e3a4d78a8946b878',
    },
    'sim29': {
        'components/accelsim.py': '8e552f6df657801b0b59abac1a6f749c2a47756a4e419ad0f0660fd8926bab97',
        'simulator/accelsim/config.py': '5018dd5f25520e1cffe4bff1721e6f2e5ce626d572c53d1e122e89a9aa6658b2',
        'simulator/accelsim/profile.py': '144f0e30818360ffc12f202a72a1554881b602465bf86afc78c16cec957d23d0',
        'simulator/accelsim/backend.py': '23eb7e02a0aa1a0466fd6bffc244f314bb5516b0ded936dabd08345343ebafcc',
        'simulator/accelsim/device.py': '2230b11303258a3b8673e140f8e1d1679b329dc7fb37b06f5d23f452b34a6dc7',
        'simulator/accelsim/constraints.py': '416803cbbfb35b32c2593336861a3031a661b2496655cf784bcfb654187eb38f',
    },
}


# Integrated SIM29A emits the same v2 annotation contract. Future producer
# changes do not erase read-only support for this fixed complete source set.
DETAIL_V2_SOURCES = {
    'components/accelsim.py': '8e552f6df657801b0b59abac1a6f749c2a47756a4e419ad0f0660fd8926bab97',
    'simulator/accelsim/backend.py': '7622908566c3ad5fb53ee6affebcd6e88bebed219136ce4d4d540d827285316d',
    'simulator/accelsim/config.py': 'ca1c57d6e2054654ce9f879b79e7ef1273b7b7663efb498fb34e60b3db44e80e',
    'simulator/accelsim/profile.py': '144f0e30818360ffc12f202a72a1554881b602465bf86afc78c16cec957d23d0',
    'simulator/accelsim/device.py': '420ff66423aeb2885e413b14c59eb7d8a1b51d3191723fe68c5b1937f094a8e3',
    'simulator/accelsim/constraints.py': '416803cbbfb35b32c2593336861a3031a661b2496655cf784bcfb654187eb38f',
    'simulator/accelsim/result_contract.py': '3fffbbc4ba9ebd120b5f76c827538eb8747796ae63b2ee576ebcb1bd0c48d227',
}

def producer_contract(schema, sources):
    """Select a result dialect only after archive/run source closure succeeds."""
    if schema != RESULT_SCHEMA:
        raise ValueError('unsupported result envelope')
    from arch_explorer.components.accelsim.source_compat import BASELINE_RESULT_V2_SOURCES
    if 'components/accelsim/result_contract.py' in sources:
        package = package_root()
        names = {'components/accelsim/declarations.py'} | {
            'components/accelsim/'+n+'.py'
            for n in ('backend', 'config', 'profile', 'device', 'constraints', 'result_contract')}
        current = {n: hashlib.sha256((package/n).read_bytes()).hexdigest() for n in names}
        if all(sources.get(n) == sha for n, sha in current.items()):
            return 'v2'
    elif 'simulator/accelsim/result_contract.py' in sources:
        for frozen in (DETAIL_V2_SOURCES, BASELINE_RESULT_V2_SOURCES):
            if all(sources.get(n) == sha for n, sha in frozen.items()):
                return 'v2'
    else:
        for version, expected in LEGACY_SOURCES.items():
            if all(sources.get(n) == sha for n, sha in expected.items()):
                return version
    raise ValueError('result version is not supported by the hash-bound producer semantics')


def detail_fields(version):
    if version == 'sim28':
        return DETAIL_FIELDS
    if version == 'sim29':
        return DETAIL_FIELDS | {'device_facts'}
    if version == 'v2':
        return DETAIL_FIELDS | {'schema', 'device_facts', 'component_map_schema'}
    raise ValueError('unknown result dialect')


def component_map(config, version):
    """Reconstruct exact versioned metadata around the unchanged native config.

    SIM28 used finite input domains and attributed timing to component clocks;
    its old annotations are checked as old annotations, never physical evidence.
    SIM29 added statuses/domains and corrected CORE ready-cycle annotations.
    V2 also classifies topology rows from the same audited facts.
    """
    rows = deepcopy(config.component_map)
    if version == 'v2':
        return rows
    for row in rows:
        if version == 'sim29':
            if row['param'] == 'repeat':
                row.pop('status')
            continue
        row.pop('status')
        if row['param'] == 'repeat':
            row['source'] = 'ADF topology; ' + PROFILE
            row['constraint'] = 'coupled SM clusters / chiplet memory partitions'
            continue
        # The profile and component declarations above are pinned by byte hash.
        # This is a metadata projection, not a second effective configuration.
        owner = next(o for o, name in OBJECTS.items() if name == row['native_object'])
        if row['param'] == 'clock_mhz':
            domain = {'sm': [990, 1980, 3960], 'icnt': [1980, 3960, 7920],
                      'l2': [1980, 3960, 7920], 'dram': [1553, 3106]}[owner]
        elif row['param'] == 'sim28_typed_regs':
            domain = [1]
        else:
            matches = [v for o, k, v, _, _ in FIELDS if (o, k) == (owner, row['param'])]
            if len(matches) != 1:
                raise ValueError('parameter did not exist in the SIM28 producer contract')
            domain = list(CHOICES.get(row['param'], matches))
            if row['param'] == 'gpgpu_coalesce_arch':
                domain += [120]
        if row['value'] not in domain or 'derived_from' in row:
            raise ValueError('value/derived alias could not be produced by the SIM28 contract')
        row['constraint'] = domain
        row['clock_domain'] = owner if owner in CLOCKS else 'sm'
    if version == 'sim28':
        if config.device_facts is not None:
            raise ValueError('SIM28 results cannot claim later device-fact metadata')
        counts = {r['native_key']: r['value'] for r in rows if r['param'] == 'repeat'}
        typed = any(r['param'] == 'sim28_typed_regs' for r in rows)
        if (counts['gpgpu_n_clusters'] not in ([2, 4, 8, 16] if typed else [2, 4, 8, 16, 32, 64, 128, 132])
                or counts['gpgpu_n_mem'] not in ([40] if typed else [20, 40])
                or counts['gpgpu_n_sub_partition_per_mchannel'] != 2):
            raise ValueError('topology could not be produced by the SIM28 contract')
        clocks = {r['clock_domain']: r['value'] for r in rows if r['param'] == 'clock_mhz'}
        if clocks['icnt'] != 2 * clocks['sm'] or clocks['l2'] != clocks['icnt']:
            raise ValueError('clocks violate the SIM28 producer contract')
    return rows
