"""Fact/config boundaries only; fixture metadata is synthetic, never GPU proof."""
from copy import deepcopy
import json
import pytest

from arch_explorer.adf import Param
from arch_explorer.archs.accelsim import sm120_mechanism_adf, device_scenario
from arch_explorer.simulator.accelsim.config import compile_adf
from arch_explorer.simulator.accelsim.device import counterfactual


def setp(adf, owner, key, value):
    node = (adf.networks[0] if owner == 'icnt' else
            adf.child('cluster' if owner in ('sm', 'l1_shared') else 'partition').child(owner))
    original = node.params[key]
    node.params[key] = Param(value, original.unit, 'synthetic test assumption')


@pytest.fixture
def scenario(tmp_path):
    evidence = {'schema': 'arch-explorer.sim28-live-preflight.v1', 'status': 'passed',
                'gpu': 'SYNTHETIC evidence fixture', 'uuid_hex': 'not-real', 'cc': [12, 0],
                'limits': {'sm_count': 110, 'registers_sm': 65536, 'registers_block': 65536,
                           'shared_sm': 102400, 'shared_block': 49152,
                           'warp_size': 32, 'max_threads_block': 1024}}
    p = tmp_path / 'synthetic-evidence.json'
    p.write_text(json.dumps(evidence))
    a = sm120_mechanism_adf()
    setp(a, 'icnt', 'booksim_k', 190)
    setp(a, 'l1_shared', 'gpgpu_adaptive_cache_config', 0)
    return device_scenario(a, p), p


def test_fact_scenario_has_one_effective_capacity_and_explicit_unknowns(scenario):
    a, _ = scenario
    c = compile_adf(a)
    assert '-gpgpu_n_clusters 110\n' in c.text
    assert '-gpgpu_shmem_size 102400\n' in c.text
    assert '-gpgpu_shmem_sizeDefault 102400\n' in c.text
    assert 'gpgpu_shmem_sizeDefault' not in a.child('cluster').child('l1_shared').params
    assert 'k = 190;' in c.interconnect_text
    assert c.device_facts['live_lease'] is False
    assert c.device_facts['unknown'] and c.device_facts['synthetic_assumptions']
    assert all(r['clock_domain']=='sm' for r in c.component_map if r['param'] in ['dram_latency','gpgpu_l2_rop_latency'])
    assert 'device_evidence' not in '\n'.join(line for line in c.text.splitlines() if line.startswith('-'))


@pytest.mark.parametrize('change', ['value', 'source', 'unit', 'count', 'original', 'unmarked-assumption'])
def test_contradictory_provenance_is_rejected_before_native(scenario, change):
    a, p = scenario
    sm = a.child('cluster').child('sm')
    if change == 'count':
        a.child('cluster').repeat = 112
    elif change == 'original':
        p.write_text(p.read_text() + '\n')
    elif change == 'unmarked-assumption':
        old = sm.params['clock_mhz']
        sm.params['clock_mhz'] = Param(old.value, old.unit, 'a target fit presented as hardware')
    else:
        old = sm.params['gpgpu_shader_registers']
        sm.params['gpgpu_shader_registers'] = Param(
            old.value + (1024 if change == 'value' else 0),
            'B' if change == 'unit' else old.unit,
            'synthetic overwrite' if change == 'source' else old.source)
    with pytest.raises(ValueError):
        compile_adf(a)


def test_nonfixture_topology_and_four_independent_clock_domains(scenario):
    a = counterfactual(scenario[0])
    a.child('cluster').repeat = 114
    a.child('partition').repeat = 24
    setp(a, 'icnt', 'booksim_k', 162)
    for owner, mhz in [('sm', 1370), ('icnt', 2111), ('l2', 997), ('dram', 1733)]:
        setp(a, owner, 'clock_mhz', mhz)
    c = compile_adf(a)
    assert '-gpgpu_clock_domains 1370:2111:997:1733\n' in c.text
    assert c.device_facts is None
    assert '-gpgpu_n_clusters 114\n' in c.text


@pytest.mark.parametrize('channels,parts,ports', [(39, 2, 188), (8, 16, 238), (40, 3, 230), (40, 2, 191)])
def test_native_hash_subpartition_or_endpoint_constraints(scenario, channels, parts, ports):
    a = counterfactual(scenario[0])
    a.child('partition').repeat = channels
    a.child('partition').child('l2').repeat = parts
    setp(a, 'icnt', 'booksim_k', ports)
    with pytest.raises(ValueError):
        compile_adf(a)


def test_per_block_and_per_sm_caps_are_not_interchangeable(scenario):
    a = counterfactual(scenario[0])
    setp(a, 'sm', 'gpgpu_shader_registers', 32768)
    with pytest.raises(ValueError, match='registers_per_block'):
        compile_adf(a)
    a = counterfactual(scenario[0])
    setp(a, 'l1_shared', 'gpgpu_shmem_size', 32768)
    with pytest.raises(ValueError, match='shared_per_block'):
        compile_adf(a)


def test_metadata_cannot_outlive_its_original_fact_binding(scenario):
    a, _ = scenario
    del a.child('cluster').child('sm').params['device_evidence']
    with pytest.raises(ValueError, match='original bound'):
        compile_adf(a)


@pytest.mark.parametrize('owner,key,donor_owner,donor_key', [
    ('sm', 'clock_mhz', 'sm', 'gpgpu_compute_capability_major'),
    ('dram', 'dram_latency', 'sm', 'gpgpu_shader_registers'),
    # Equal values AND units are insufficient: per-block is not per-SM.
    ('sm', 'gpgpu_registers_per_block', 'sm', 'gpgpu_shader_registers'),
    ('l1_shared', 'gpgpu_shmem_per_block', 'l1_shared', 'gpgpu_shmem_size'),
])
def test_fact_sources_are_semantic_bindings(scenario, owner, key, donor_owner, donor_key):
    a, _ = scenario
    def node(o):
        return a.child('cluster' if o in ('sm', 'l1_shared') else 'partition').child(o)
    param = node(owner).params[key]
    node(owner).params[key] = Param(param.value, param.unit, node(donor_owner).params[donor_key].source)
    with pytest.raises(ValueError, match='device.fact'):
        compile_adf(a)


def test_only_declared_shared_alias_inherits_the_fact(scenario):
    a, _ = scenario
    shared = a.child('cluster').child('l1_shared')
    original = shared.params['gpgpu_shmem_size']
    shared.params['gpgpu_shmem_sizeDefault'] = original
    c = compile_adf(a)
    aliases = [r for r in c.component_map if r['native_key'] == 'gpgpu_shmem_sizeDefault']
    assert len(aliases) == 1 and aliases[0]['status'] == 'device_fact'
    assert any(k['param'] == 'gpgpu_shmem_sizeDefault' for k in c.device_facts['known'])
    shared.params['gpgpu_shmem_sizeDefault'] = Param(original.value, 'B',
        a.child('cluster').child('sm').params['gpgpu_shader_registers'].source)
    with pytest.raises(ValueError, match='device fact'):
        compile_adf(a)


def test_audited_topology_and_derived_alias_classification(scenario):
    c = compile_adf(scenario[0])
    cluster = next(r for r in c.component_map if r['native_key'] == 'gpgpu_n_clusters')
    assert cluster['status'] == 'device_fact'
    alias = next(r for r in c.component_map if 'derived_from' in r)
    assert alias['status'] == 'device_fact' and alias['unit'] == 'B'
    synthetic = compile_adf(counterfactual(scenario[0]))
    assert synthetic.device_facts is None
    assert not any(r['status'] == 'device_fact' for r in synthetic.component_map)
