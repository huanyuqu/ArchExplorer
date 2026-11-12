# Concrete Accel-Sim implementation; owned by the component provider.
"""Bind historical device facts to one explicitly assumed ADF scenario.

This module never queries a device. The original SHA-bound evidence is provenance,
not a second writable profile. ADF values are the sole effective configuration.
Unknown timing/cache/network fields stay explicit assumptions, never measurements.
"""
import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from arch_explorer.adf import Param, SpacePoint, from_dict, walk

EVIDENCE_SCHEMA = 'arch-explorer.accelsim-device-evidence.v1'
EVIDENCE_SOURCE = 'historical immutable device-fact provenance; not an execution lease or calibration'
# native owner, ADF key, original JSON pointer, unit. Initial shared allocation is
# derived by config.py, rather than mirrored as another editable device value.
FACTS = (
    ('sm', 'gpgpu_shader_registers', '/limits/registers_sm', 'register'),
    ('sm', 'gpgpu_registers_per_block', '/limits/registers_block', 'register'),
    ('sm', 'gpgpu_compute_capability_major', '/cc/0', '1'),
    ('sm', 'gpgpu_compute_capability_minor', '/cc/1', '1'),
    ('l1_shared', 'gpgpu_shmem_size', '/limits/shared_sm', 'B'),
    ('l1_shared', 'gpgpu_shmem_per_block', '/limits/shared_block', 'B'),
)
# An explicit static initial-allocation alias may claim exactly the same fact.
# An omitted alias is derived by the compiler from the validated capacity row.
ALIASES = (('l1_shared', 'gpgpu_shmem_sizeDefault', '/limits/shared_sm', 'B'),)


@dataclass
class ParameterAudit:
    facts: dict | None
    statuses: dict

    def status(self, owner, key):
        return self.statuses[owner, key]


def _nonfact_status(source):
    if source.lower().startswith('synthetic'):
        return 'synthetic_assumption'
    if source.startswith(('Accel-Sim', 'GPGPU-Sim')):
        return 'upstream_configuration'
    return 'unverified_source'


def _at(data, pointer):
    for part in pointer.strip('/').split('/'):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def _load(reference):
    if not isinstance(reference, dict) or set(reference) != {'schema', 'path', 'sha256'} or reference['schema'] != EVIDENCE_SCHEMA:
        raise ValueError('device evidence requires exactly schema/path/sha256')
    path = Path(reference['path'])
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 2**20:
        raise ValueError('device evidence needs a bounded existing absolute file')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('device evidence SHA256 mismatch')
    data = json.loads(raw)
    if data.get('schema') not in ('arch-explorer.sim28-live-preflight.v1', 'arch-explorer.sim30-live-preflight.v1') or data.get('status') != 'passed':
        raise ValueError('unsupported historical device evidence; not a live lease')
    if not data.get('gpu') or not data.get('uuid_hex') or len(data['cc']) != 2:
        raise ValueError('device evidence lacks device/UUID/CC identity')
    for pointer in ['/limits/sm_count', '/limits/warp_size', '/limits/max_threads_block'] + [x[2] for x in FACTS]:
        value = _at(data, pointer)
        if type(value) is not int or value < (0 if pointer == '/cc/1' else 1):
            raise ValueError('device fact must be an integer with correct physical range: ' + pointer)
    if data['limits']['warp_size'] != 32:
        raise ValueError('pinned native trace engine requires warp_size=32')
    return data


def _nodes(adf):
    return {('sm' if n.component == 'accelsim_sm120_mechanism' else n.component.removeprefix('accelsim_')): n
            for _, n in walk(adf) if isinstance(n, SpacePoint)}


def _source(reference, pointer):
    return 'device-fact:' + reference['sha256'] + '#' + pointer


def apply_device_facts(assumptions, evidence):
    """Return a fresh scenario; the caller must supply all synthetic assumptions.

    No default physical profile is selected. Only the explicitly enumerated
    factual fields and SM count are imported; clocks/topology/timings are not.
    """
    path = Path(evidence).resolve()
    reference = {'schema': EVIDENCE_SCHEMA, 'path': str(path),
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    data = _load(reference)
    adf = deepcopy(assumptions)
    nodes = _nodes(adf)
    if nodes['sm'].component != 'accelsim_sm120_mechanism' or data['cc'] != [12, 0]:
        raise ValueError('ordinary SM120 scenario requires the corresponding CC12.0 evidence')
    for node in nodes.values():
        for key, param in node.params.items():
            if not param.source.lower().startswith('synthetic'):
                raise ValueError('all unobserved scenario parameters must be explicitly synthetic assumptions: ' + key)
    for owner, key, pointer, unit in FACTS:
        nodes[owner].params[key] = Param(_at(data, pointer), unit, _source(reference, pointer))
    adf.child('cluster').repeat = data['limits']['sm_count']
    nodes['sm'].params['device_evidence'] = Param(json.dumps(reference, sort_keys=True), '1', EVIDENCE_SOURCE)
    nodes['l1_shared'].params.pop('gpgpu_shmem_sizeDefault', None)
    adf.name = 'device_fact_constrained_synthetic_scenario'
    # Caller must explicitly choose a consistent memory/chiplet/BookSim scenario.
    # In particular, this function does not derive unknown memory channels.
    return adf


def audit_parameters(adf, nodes=None):
    """Own both semantic validation and classification of every parameter.

    A fact is an owner/key/pointer/value/unit binding, never a source prefix.
    File hashes provide byte identity, not authentication of device provenance.
    """
    nodes = _nodes(adf) if nodes is None else nodes
    meta = nodes['sm'].params.get('device_evidence')
    if meta is None:
        # Reject a dangling physical-fact claim without the original bytes.
        if any(p.source.startswith('device-fact:') for n in nodes.values() for p in n.params.values()):
            raise ValueError('device-fact values require their original bound device_evidence')
        return ParameterAudit(None, {(o, k): _nonfact_status(p.source)
                                     for o, n in nodes.items() for k, p in n.params.items()})
    if meta.unit != '1' or meta.source != EVIDENCE_SOURCE:
        raise ValueError('device_evidence metadata unit/source conflicts with the evidence contract')
    reference = json.loads(meta.value)
    data = _load(reference)
    if nodes['sm'].component != 'accelsim_sm120_mechanism' or data['cc'] != [12, 0]:
        raise ValueError('ordinary SM120 scenario requires the corresponding CC12.0 evidence')
    if adf.child('cluster').repeat != data['limits']['sm_count']:
        raise ValueError('device SM count conflicts with ADF cluster repeat')
    bindings = list(FACTS)
    bindings += [(o, k, p, u) for o, k, p, u in ALIASES
                 if k in nodes[o].params and nodes[o].params[k].source.startswith('device-fact:')]
    for owner, key, pointer, unit in bindings:
        param = nodes[owner].params[key]
        if type(param.value) is not int or param.value != _at(data, pointer) or param.unit != unit or param.source != _source(reference, pointer):
            raise ValueError('device fact value/unit/source conflicts with ADF: ' + key)
    verified = {(o, k) for o, k, _, _ in bindings}
    statuses = {}
    unobserved = []
    for owner, node in nodes.items():
        for key, param in node.params.items():
            if (owner, key) in verified:
                statuses[owner, key] = 'device_fact'
                continue
            if (owner, key) == ('sm', 'device_evidence'):
                statuses[owner, key] = 'evidence_reference'
                continue
            if param.source.startswith('device-fact:'):
                raise ValueError('unsupported device-fact owner/key binding: ' + owner + '.' + key)
            if not param.source.lower().startswith('synthetic'):
                raise ValueError('unobserved native parameter lacks explicit synthetic assumption: ' + key)
            statuses[owner, key] = 'synthetic_assumption'
            unobserved.append(owner + '.' + key)
    if nodes['l1_shared'].value('gpgpu_adaptive_cache_config') != 0:
        raise ValueError('device-fact shared capacity requires explicit static allocation; adaptive cache split is unknown')
    facts = {'gpu': data['gpu'], 'uuid': data['uuid_hex'], 'cc': data['cc'],
            'original': reference, 'known': [{'owner': o, 'param': k, 'value': _at(data, p), 'unit': u, 'pointer': p}
                                             for o, k, p, u in bindings] +
                      [{'owner': 'cluster', 'param': 'repeat', 'value': data['limits']['sm_count'], 'unit': 'count', 'pointer': '/limits/sm_count'}],
            'launch_facts': {'warp_size': 32, 'max_threads_block': data['limits']['max_threads_block']},
            'synthetic_assumptions': sorted(unobserved),
            'unknown': ['SM thread/CTA capacity beyond captured bounds', 'physical clocks and data rates',
                        'register/uniform/predicate banks and pipeline timing', 'cache capacity/service/hash',
                        'memory channels/chiplets/DRAM geometry and timing', 'interconnect topology/service'],
            'hardware_accuracy': 'not validated', 'live_lease': False}
    return ParameterAudit(facts, statuses)


def audit_device(adf, nodes=None):
    """Return the fact report, using the same audit consumed by the compiler."""
    return audit_parameters(adf, nodes).facts


def counterfactual(adf):
    """Explicitly leave the device-fact claim for a synthetic sensitivity scene."""
    result = deepcopy(adf)
    result.name = 'synthetic_counterfactual'
    for node in _nodes(result).values():
        node.params.pop('device_evidence', None)
        for key, param in list(node.params.items()):
            node.params[key] = Param(param.value, param.unit, 'synthetic counterfactual assumption; ' + param.source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assumptions', help='complete explicitly synthetic ADF JSON')
    parser.add_argument('evidence', help='existing SHA-bound historical preflight JSON; no query')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = apply_device_facts(from_dict(json.loads(Path(args.assumptions).read_text())), args.evidence)
    from arch_explorer.components.accelsim.config import compile_adf
    compiled = compile_adf(result)  # Contradictions fail before producing a usable scenario.
    Path(args.output).write_text(json.dumps({'adf': result.to_dict(), 'device_facts': compiled.device_facts}, indent=2) + '\n')


if __name__ == '__main__':
    main()
