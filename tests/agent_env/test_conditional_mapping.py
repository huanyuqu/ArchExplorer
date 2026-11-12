"""One public active-parameter contract for discovery, bounded sampling and validation."""
import random
from dataclasses import replace

import pytest
import arch_explorer.simulator.events  # register built-ins
from arch_explorer.adf import MappingSpec, WorkloadSpec, walk
from arch_explorer.adf.registry import (MAPPING_TEMPLATES, ConditionalMappingSpace, mapping_spaces,
                                       validate_mapping_params)
from arch_explorer.agent_env.sampling import mapping_candidates, check_mapping
from non_gpu_fixture import shared_adf
from arch_explorer.devices.pim import neupims, hsndi
from arch_explorer.mapping.pim import PimOffload


def workload(**overrides):
    model = dict(name='synthetic', hidden=128, heads=1, kv_heads=1, head_dim=128, intermediate=128,
                 layers=1, vocab=128, gated=False, norm='layernorm', activation='gelu', source='test')
    return WorkloadSpec('transformer', dict(model=model, phase='decode', batch=2, seq_len=32, layers=1, **overrides))


@pytest.fixture(scope='module')
def case():
    adf, spec = neupims(64), workload()
    return adf, spec, PimOffload().space(adf, spec)


def test_discovery_all_products_have_only_active_keys(case):
    adf, spec, contract = case
    assert isinstance(contract, ConditionalMappingSpace)
    assert contract.to_dict()['kind'] == 'conditional'
    modes = set()
    for branch in contract.branches:
        params = {k: v[0] for k, v in branch.items()}
        check_mapping(adf, spec, MappingSpec('pim_offload', params), MAPPING_TEMPLATES)
        mode = params['npu_execution'], params['memory_execution']
        modes.add(mode)
        if mode[0] == 'instructions':
            assert not any(k.startswith('inner.') for k in params)
        elif params['inner'] == 'gsm_tiled':
            assert 'inner.tile_m' in params and 'inner.weights' not in params
        else:
            assert 'inner.weights' in params and 'inner.tile_m' not in params
    assert modes == {('aggregate', 'aggregate'), ('instructions', 'aggregate'), ('instructions', 'transactions')}


@pytest.mark.parametrize('memory', ['aggregate', 'transactions'])
def test_sampling_explicit_repeatable_and_bounded(case, memory):
    adf, spec, _ = case
    selectors = {'pim_offload': {'npu_execution': 'instructions', 'memory_execution': memory}}
    first = mapping_candidates(adf, spec, MAPPING_TEMPLATES, 3, random.Random(17), ['pim_offload'], selectors)
    assert first == mapping_candidates(adf, spec, MAPPING_TEMPLATES, 3, random.Random(17), ['pim_offload'], selectors)
    assert 0 < len(first) <= 3
    for mapping in first:
        check_mapping(adf, spec, mapping, MAPPING_TEMPLATES)
        assert mapping.params['memory_execution'] == memory
        assert mapping.params['npu_execution'] == 'instructions'


def test_default_sampling_never_runs_explicit_program(case):
    adf, spec, _ = case
    samples = mapping_candidates(adf, spec, MAPPING_TEMPLATES, 6, random.Random(29), ['pim_offload'])
    assert len(samples) == 6
    for mapping in samples:
        assert mapping.params['npu_execution'] == mapping.params['memory_execution'] == 'aggregate'
        check_mapping(adf, spec, mapping, MAPPING_TEMPLATES)


@pytest.mark.parametrize('change', [
    {'inner.tile_m': 64}, {'unexpected': 1}, {'attention': 'inner'}, {'sub_batches': 2},
    {'inner': 'dmc_partition'}, {'npu_execution': 'aggregate'}, {'tp': True},
])
def test_explicit_bad_keys_and_conflicting_choices_rejected(case, change):
    _, _, space = case
    product = mapping_spaces(space, {'npu_execution': 'instructions', 'memory_execution': 'transactions'})[0]
    params = {k:v[0] for k,v in product.items()}
    params.update(change)
    with pytest.raises(ValueError, match='Invalid mapping'):
        validate_mapping_params(space, params, complete=True)
    with pytest.raises(ValueError, match='Invalid mapping'):
        validate_mapping_params(space, params)


def test_complete_requires_active_keys_but_direct_defaults_are_explicit(case):
    _, _, space = case
    short = {'npu_execution': 'instructions'}
    with pytest.raises(ValueError, match='missing active'):
        validate_mapping_params(space, short, complete=True)
    normalized = validate_mapping_params(space, short)
    assert normalized == dict(attention='pim', sub_batches=1, tp=1, inner='gsm_tiled',
                              npu_execution='instructions', memory_execution='aggregate')
    for branch in space.branches:
        if 'inner.tile_m' in branch:
            params = {k:v[0] for k,v in branch.items()}
            del params['inner.tile_m']
            with pytest.raises(ValueError, match='missing active'):
                validate_mapping_params(space, params, complete=True)
            break


@pytest.mark.parametrize('options', [{'dtype':'fp32'}, {'iterations':2}])
def test_unsupported_workload_not_advertised(options):
    space = PimOffload().space(neupims(64), workload(**options))
    assert all(b['npu_execution'] == ['aggregate'] for b in space.branches)
    with pytest.raises(ValueError, match='fp16, one iteration'):
        mapping_spaces(space, {'npu_execution':'instructions'})


@pytest.mark.parametrize('adf', [neupims(64, channels=16), hsndi(cores=2)])
def test_unsupported_hardware_not_sampled(adf):
    space = PimOffload().space(adf, workload())
    assert all(b['npu_execution'] == ['aggregate'] for b in space.branches)
    assert space.unavailable
    with pytest.raises(ValueError, match='No supported mapping branch'):
        mapping_spaces(space, {'npu_execution':'instructions'})


def test_clock_change_excludes_source_profile():
    adf = neupims(64)
    for _, node in walk(adf):
        if getattr(node, 'component', None) == 'systolic_array':
            node.params['clock_ghz'] = replace(node.params['clock_ghz'], value=2.0)
    space = PimOffload().space(adf, workload())
    assert any('1 GHz' in row['reason'] for row in space.unavailable)
    assert all(b['npu_execution'] == ['aggregate'] for b in space.branches)


def test_head_dimension_and_gqa_rejected_before_transactions():
    spec = workload()
    spec.params['model'].update(hidden=256, heads=2, kv_heads=1)
    space = PimOffload().space(neupims(64), spec)
    with pytest.raises(ValueError, match='grouped-query'):
        mapping_spaces(space, {'npu_execution':'instructions', 'memory_execution':'transactions'})
    spec.params['model'].update(hidden=24, heads=1, kv_heads=1, head_dim=24)
    space = PimOffload().space(neupims(64), spec)
    assert all(b['npu_execution'] == ['aggregate'] for b in space.branches)
    assert any('dimension' in row['reason'] for row in space.unavailable)


def test_non_pim_flat_space_and_huge_product():
    adf, spec = shared_adf(2), WorkloadSpec('operator', {'op':'matmul', 'm':16,'k':32,'n':32})
    samples = mapping_candidates(adf, spec, MAPPING_TEMPLATES, 4, random.Random(43), ['gsm_tiled'])
    assert len(samples) == 4
    for mapping in samples:
        check_mapping(adf, spec, mapping, MAPPING_TEMPLATES)
    # 10**12 logical combinations: only small lists and sampled indices exist.
    class Huge:
        def applicable(self, a, w): return True
        def space(self, a, w): return {str(i): list(range(1000)) for i in range(4)}
        def request_space(self, a, w, params=None): return self.space(a, w)
    class Registry:
        def names(self): return ['huge']
        def get(self, name): return Huge()
    values = mapping_candidates(None, None, Registry(), 3, random.Random(17))
    assert len(values) == 3
    assert len({tuple(m.params.items()) for m in values}) == 3


def test_unknown_selectors_do_not_silently_fall_back(case):
    adf, spec, space = case
    with pytest.raises(ValueError, match='No supported'):
        mapping_spaces(space, {'unknown':1})
    with pytest.raises(ValueError, match='No supported'):
        mapping_spaces(space, {'memory_execution':'transactions'})
    with pytest.raises(ValueError, match='object'):
        mapping_spaces(space, [])


@pytest.mark.parametrize('extra', [{'inner.weights':'stream'}, {'inner.split':'n'}])
def test_aggregate_rejects_parameters_of_another_inner(case, extra):
    _, _, space = case
    product = mapping_spaces(space, {'inner':'gsm_tiled'})[0]
    params = {k:v[0] for k,v in product.items()}
    params.update(extra)
    with pytest.raises(ValueError, match='inactive'):
        validate_mapping_params(space, params, complete=True)
    with pytest.raises(ValueError, match='inactive'):
        validate_mapping_params(space, params)
