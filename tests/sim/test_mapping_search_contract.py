"""Public mapping helpers: real request branches, bounded search and replay."""
from copy import deepcopy
import itertools
import json

import pytest

import arch_explorer.simulator.events  # register real components/templates/backend
from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate, from_dict
from arch_explorer.adf.evaluation import Evaluation
from arch_explorer.adf.registry import (
    ConditionalMappingSpace, MAPPING_TEMPLATES, MappingTemplate, Registry,
    validate_mapping_params,
)
from non_gpu_fixture import shared_adf
from arch_explorer.devices.pim import neupims
from arch_explorer.mapping import best_mapping, mapping_space
from arch_explorer.mapping import search
from arch_explorer.mapping.multi_device import MultiDevice
from tools.generate_multi_device_execution import configuration


@pytest.fixture(scope="module")
def pim_case():
    model = dict(name="synthetic", hidden=128, heads=1, kv_heads=1, head_dim=128,
                 intermediate=128, layers=1, vocab=128, gated=False, norm="layernorm",
                 activation="gelu", source="small public-helper test")
    workload = WorkloadSpec("transformer", dict(model=model, phase="decode", batch=2,
                                                seq_len=32, layers=1))
    return neupims(64), workload


@pytest.fixture
def matrix_case():
    return shared_adf(1), WorkloadSpec("operator", dict(op="matmul", batch=1, m=8, k=5, n=8))


@pytest.fixture
def multi_case():
    config = configuration("operator", tp=True)
    return (from_dict(config["seed_adf"]), WorkloadSpec(**config["workloads"][0]),
            config["mapping_selectors"]["multi_device"])


def test_real_pim_sampling_defaults_and_inactive_keys(pim_case, monkeypatch):
    adf, workload = pim_case
    def no_evaluation(*args, **kwargs):
        pytest.fail("discovery must not evaluate candidates")
    monkeypatch.setattr(search, "evaluate", no_evaluation)
    candidates = mapping_space(adf, workload, "pim_offload")
    assert candidates
    space = MAPPING_TEMPLATES.get("pim_offload").request_space(adf, workload, None)
    for spec in candidates:
        assert spec.params["npu_execution"] == spec.params["memory_execution"] == "aggregate"
        validate_mapping_params(space, spec.params, complete=True)
        assert not ("inner.tile_m" in spec.params and "inner.weights" in spec.params)


@pytest.mark.parametrize("memory", ["aggregate", "transactions"])
def test_real_pim_explicit_instruction_branch(pim_case, memory):
    adf, workload = pim_case
    fixed = dict(npu_execution="instructions", memory_execution=memory, tp=1)
    candidates = mapping_space(adf, workload, "pim_offload", fixed)
    assert len(candidates) == 1
    spec = candidates[0]
    assert spec.params.items() >= fixed.items()
    assert not any(key.startswith("inner.") for key in spec.params)
    validate_mapping_params(MAPPING_TEMPLATES.get("pim_offload").request_space(adf, workload, fixed),
                            spec.params, complete=True)


@pytest.mark.parametrize("fixed", [
    {"typo": 1}, {"tp": True}, {"tp": 3},
    {"npu_execution": "instructions", "inner.tile_m": 64},
    {"memory_execution": "transactions"}, {"npu_execution": "instructions", "sub_batches": 2},
    [], False,
])
def test_real_pim_invalid_selectors_are_not_dropped(pim_case, fixed):
    with pytest.raises(ValueError):
        mapping_space(*pim_case, "pim_offload", fixed)


def test_real_pim_best_and_complete_json_winner_replay(pim_case):
    adf, workload = pim_case
    space = MAPPING_TEMPLATES.get("pim_offload").request_space(adf, workload, None)
    branch = next(b for b in space.branches if "inner.tile_n" in b)
    fixed = {k: values[0] for k, values in branch.items() if k != "inner.tile_n"}
    original = deepcopy(fixed)
    winner = best_mapping(adf, workload, ["pim_offload"], {"pim_offload": fixed},
                          limit=2, cost_model=False)
    assert fixed == original
    assert len(winner.evaluated) == 2
    assert winner.skipped == len(branch["inner.tile_n"]) - 2
    assert all(isinstance(latency, (int, float)) for _, latency in winner.evaluated)
    replay = evaluate(from_dict(json.loads(json.dumps(adf.to_dict()))),
                      WorkloadSpec(**json.loads(json.dumps(workload.to_dict()))),
                      MappingSpec(**json.loads(json.dumps(winner.best.to_dict()))), cost_model=False)
    assert replay.to_dict() == winner.evaluation.to_dict()


def test_flat_order_matches_original_product_and_does_not_mutate(matrix_case):
    adf, workload = matrix_case
    space = MAPPING_TEMPLATES.get("gsm_tiled").space(adf, workload)
    original = deepcopy(space)
    fixed = {"rows_per_cta": 4}
    expected = deepcopy(space)
    expected["rows_per_cta"] = [4]
    keys = sorted(expected)
    mappings = [MappingSpec("gsm_tiled", dict(zip(keys, values)))
                for values in itertools.product(*(expected[k] for k in keys))]
    assert mapping_space(adf, workload, "gsm_tiled", fixed) == mappings
    assert space == original and fixed == {"rows_per_cta": 4}
    with pytest.raises(ValueError):
        mapping_space(adf, workload, "gsm_tiled", {"rows_per_cta": True})
    assert mapping_space(adf, workload, "pim_offload", {"typo": 1}) == []


def test_implicit_discovery_does_not_request_or_disable_multi_device(matrix_case, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("implicit discovery invented a multi-device request")
    monkeypatch.setattr(MultiDevice, "request_space", unexpected)
    result = best_mapping(*matrix_case, limit=2, cost_model=False)
    assert len(result.evaluated) == 2
    expected = [spec for name in MAPPING_TEMPLATES.names() if name != "multi_device"
                for spec in mapping_space(*matrix_case, name)]
    assert [spec for spec, _ in result.evaluated] == expected[:2]
    assert result.skipped == len(expected) - 2
    assert result.best in expected[:2]
    assert "multi_device" in MAPPING_TEMPLATES.names()


@pytest.mark.parametrize("fixed", [None, {}, {"multi_device": None}, {"multi_device": {}},
                                   {"multi_device": {"typo": 1}}])
def test_explicit_multi_device_missing_or_invalid_request_rejected(matrix_case, fixed):
    with pytest.raises(ValueError):
        best_mapping(*matrix_case, templates=["gsm_tiled", "multi_device"], fixed=fixed,
                     limit=1, cost_model=False)


@pytest.mark.parametrize("params", [None, {}, {"typo": 1}])
def test_multi_device_supplied_bad_request_rejected_during_implicit_discovery(matrix_case, params):
    with pytest.raises(ValueError):
        best_mapping(*matrix_case, fixed={"multi_device": params}, limit=1, cost_model=False)
    with pytest.raises(ValueError):
        mapping_space(*matrix_case, "multi_device", params)


@pytest.mark.parametrize("templates", [None, ["multi_device"]])
def test_complete_multi_device_request_preserved_and_executed(multi_case, templates):
    adf, workload, request = multi_case
    original = deepcopy(request)
    assert mapping_space(adf, workload, "multi_device", request) == [MappingSpec("multi_device", request)]
    result = best_mapping(adf, workload, templates, {"multi_device": request}, limit=1, cost_model=False)
    assert result.best.params == request == original
    assert result.best.template == "multi_device"
    assert len(result.evaluated) == 1
    assert result.evaluation.to_dict() == evaluate(adf, workload, result.best, cost_model=False).to_dict()


@pytest.mark.parametrize("bad", ["route", "shape", "local", "rank_bool"])
def test_multi_device_formal_request_rejections(multi_case, bad):
    adf, workload, request = multi_case
    if bad == "route":
        request["topology_edges"] = []
    elif bad == "shape":
        request["shards"][0]["size"] = 99
    elif bad == "local":
        request["placements"]["matmul"]["mapping"]["params"]["tile_m"] = 3
    else:
        request["placements"]["matmul"]["ranks"][0] = False
    with pytest.raises(ValueError):
        mapping_space(adf, workload, "multi_device", request)


@pytest.fixture
def controlled_contract(monkeypatch):
    """Isolated protocol fixture only for deterministic budget/error/tie accounting."""
    registry = Registry("test mapping", MappingTemplate)
    class Branches(MappingTemplate):
        name = "branches"
        def applicable(self, adf, workload):
            return True
        def space(self, adf, workload):
            raise AssertionError("must use request_space, not space")
        def request_space(self, adf, workload, params=None):
            return ConditionalMappingSpace([
                {"mode": ["first"], "z": [2, 1]},
                {"mode": ["second"], "a": [5, 4]},
            ], {}, {}, [])
        def apply(self, simulator, workload, params):
            raise AssertionError("accounting-only fixture")
    class Flat(Branches):
        name = "flat"
        def request_space(self, adf, workload, params=None):
            return {"only": [1]}
    registry.register(Branches)
    registry.register(Flat)
    monkeypatch.setattr(search, "MAPPING_TEMPLATES", registry)
    calls = []
    def evaluator(adf, workload, spec, **options):
        calls.append(spec)
        if spec.params.get("z") == 2:
            raise ValueError("invalid execution")
        if spec.params.get("a") == 5:
            raise NotImplementedError("unavailable execution")
        return Evaluation(latency_ns=7)
    monkeypatch.setattr(search, "evaluate", evaluator)
    return calls


def test_request_only_contract_order_budget_errors_and_first_tie(controlled_contract):
    candidates = mapping_space(None, None, "branches")
    assert [m.params for m in candidates] == [
        {"mode": "first", "z": 2}, {"mode": "first", "z": 1},
        {"a": 5, "mode": "second"}, {"a": 4, "mode": "second"}]
    assert [list(m.params) for m in candidates] == [["mode", "z"], ["mode", "z"],
                                                    ["a", "mode"], ["a", "mode"]]
    assert controlled_contract == []
    result = best_mapping(None, None, ["branches", "flat"], limit=4)
    assert controlled_contract == candidates
    assert result.skipped == 1
    assert [row[1] for row in result.evaluated] == [
        "error: invalid execution", 7, "error: unavailable execution", 7]
    assert result.best == candidates[1]


def test_failed_budget_is_not_refunded_and_no_valid_mapping_raises(controlled_contract):
    with pytest.raises(ValueError, match="No valid mapping among 0 candidates"):
        best_mapping(None, None, ["branches", "flat"], limit=0)
    assert controlled_contract == []
    with pytest.raises(ValueError, match="No valid mapping among 1 candidates"):
        best_mapping(None, None, ["branches", "flat"], limit=1)
    assert len(controlled_contract) == 1


def test_template_order_and_unlimited_budget(controlled_contract):
    result = best_mapping(None, None, ["flat", "branches"])
    assert result.best.template == "flat" and result.skipped == 0
    assert len(result.evaluated) == len(controlled_contract) == 5


def test_no_matching_branch_and_unexpected_failure_not_swallowed(controlled_contract, monkeypatch):
    with pytest.raises(ValueError, match="No supported mapping branch"):
        mapping_space(None, None, "branches", {"mode": "first", "a": 5})
    with pytest.raises(ValueError, match="selected mapping templates"):
        best_mapping(None, None, ["branches"], {"unknown": {}})
    assert not controlled_contract
    def broken(*args, **kwargs):
        raise RuntimeError("backend bug")
    monkeypatch.setattr(search, "evaluate", broken)
    with pytest.raises(RuntimeError, match="backend bug"):
        best_mapping(None, None, ["flat"], limit=1)
