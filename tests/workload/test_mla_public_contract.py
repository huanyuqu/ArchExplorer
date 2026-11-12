"""Deferred installed consumers of SIM50; collection never means acceptance."""
from copy import deepcopy
import importlib

import pytest
from arch_explorer.adf import WorkloadSpec
from arch_explorer.workload.mla_decode import mla_input_shapes,mla_decode
from mla_absorbed_expectations import GEOMETRY, root_shapes
from mla_contract_cases import (REJECT, compare_stages, frozen_inputs, independent_params, mla_api, scene)
from test_operator_mapping_attention_contract import no_backend


def test_explicit_root_shapes_metadata_and_full_CPU_oracle(mla_api,monkeypatch):
    w,s = mla_api
    params = deepcopy(frozen_inputs()['workload']['params'])
    before = deepcopy(params)
    with no_backend(monkeypatch):
        assert mla_input_shapes(**{k:params[k] for k in GEOMETRY}) == root_shapes(params)
        spec = WorkloadSpec('mla_decode',params)
        graph = w.workload_from_spec(spec)
        assert mla_decode(**params).to_dict() == graph.to_dict()
        assert spec.kind == 'mla_decode'
        assert set(graph.meta['inputs']) == set(root_shapes(params))
        assert graph.meta['outputs'] == ['Y']
        facts = compare_stages(w.execute_reference(spec),params)
        assert facts['compared_stages']
    assert params == before


@pytest.mark.parametrize('batch,heads,sequence',[(2,3,3),(1,1,1),(2,2,1)])
def test_unequal_dimensions_batch_head_cache_and_one_token_boundary(mla_api,monkeypatch,batch,heads,sequence):
    w,s = mla_api;params = independent_params(batch=batch,heads=heads,sequence=sequence)
    with no_backend(monkeypatch):
        spec = WorkloadSpec('mla_decode',params)
        compare_stages(w.execute_reference(spec),params)


def test_builder_does_not_compute_intermediate_answers(mla_api,monkeypatch):
    w,s = mla_api
    def forbidden(*args,**kwargs):
        raise AssertionError('MLA builder computed numerical answers')
    with no_backend(monkeypatch),monkeypatch.context() as patch:
        for module,name in [('precision','matmul_fp32'),('reference_api','execute_reference')]:
            patch.setattr(importlib.import_module('arch_explorer.workload.'+module),name,forbidden)
        spec = WorkloadSpec('mla_decode',deepcopy(frozen_inputs()['workload']['params']))
        assert w.workload_from_spec(spec).meta['outputs'] == ['Y']


@pytest.mark.parametrize('failure',['fp16','bf16','missing','extra','nonfinite','short',
                                     'odd-position','zero-heads','epsilon'])
def test_production_boundary_refusal_has_nonexecuting_control(mla_api,monkeypatch,failure):
    w,s = mla_api;params = deepcopy(frozen_inputs()['workload']['params'])
    with no_backend(monkeypatch):
        assert w.workload_from_spec(WorkloadSpec('mla_decode',params))
        if failure in ('fp16','bf16'):params['dtype'] = failure
        elif failure == 'missing':del params['inputs']['head0.WVO']
        elif failure == 'extra':params['inputs']['precomputed.Y'] = [0]
        elif failure == 'nonfinite':params['inputs']['X'][0] = float('inf')
        elif failure == 'short':params['inputs']['latent_kv'].pop()
        elif failure == 'odd-position':params['positional_dim'] = 3
        elif failure == 'zero-heads':params['heads'] = 0
        else:params['epsilon'] = -1
        if failure == 'short':
            # Structure-only construction records roots; their values are
            # checked at the documented compile/reference boundary.
            adf,control,mapping=scene('grouped_multicast')
            assert w.compile_workload(adf,control,mapping)
            with pytest.raises(REJECT):w.compile_workload(adf,WorkloadSpec('mla_decode',params),mapping)
        else:
            with pytest.raises(REJECT):w.workload_from_spec(WorkloadSpec('mla_decode',params))


def test_paper_scale_is_shape_only_and_explicit_execution_limit(mla_api,monkeypatch):
    w,s = mla_api
    with no_backend(monkeypatch):
        shapes = mla_input_shapes(heads=128,latent_rank=512,positional_dim=64,sequence=8192)
        assert shapes['latent_kv'] == (1,8192,512)
        assert shapes['pos_key'] == (1,8192,64)
        assert len([n for n in shapes if n.endswith('.WQK')]) == 128
        # More heads need no numerical values to show a builder-boundary refusal.
        params = deepcopy(frozen_inputs()['workload']['params']);params['heads'] = 17
        with pytest.raises(REJECT):w.workload_from_spec(WorkloadSpec('mla_decode',params))


@pytest.mark.parametrize('failure',['unknown-storage','duplicate-consumer'])
def test_v4_mapping_failure_has_passing_compile_control(mla_api,monkeypatch,failure):
    from arch_explorer.adf import MappingSpec
    w,s = mla_api;adf,spec,mapping = scene('grouped_multicast')
    with no_backend(monkeypatch):
        assert w.compile_workload(adf,spec,mapping)
        params = deepcopy(mapping.params)
        if failure == 'unknown-storage':params['storage']['not-an-external-root'] = params['storage']['latent_kv']
        elif failure == 'duplicate-consumer':params['replicas'].append(deepcopy(params['replicas'][0]))
        with pytest.raises(REJECT):w.compile_workload(adf,spec,MappingSpec('compile_graph_v4',params))


def test_storage_plan_tamper_is_rejected_without_backend_execution(mla_api,monkeypatch):
    w,s = mla_api;adf,spec,mapping = scene('grouped_multicast')
    with no_backend(monkeypatch):
        compiled = w.compile_workload(adf,spec,mapping)
        serialized = compiled.to_dict() if hasattr(compiled,'to_dict') else deepcopy(compiled)
        w.validate_compilation(serialized)
        bad = deepcopy(serialized)
        bad['program']['meta']['storage_plan']['locations']['latent_kv'] = 'missing/physical/root'
        with pytest.raises(REJECT):w.validate_compilation(bad)
