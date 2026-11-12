from copy import deepcopy
import json
import random

import pytest

from arch_explorer.adf import COMPONENTS, ParamSpec, find, from_dict, register_backend
from arch_explorer.agent_env import EnvError, ExplorationEnv, TOOL_SCHEMAS, replay_result, replay_trace
from arch_explorer.agent_env.sampling import parameter_variant
from arch_explorer.archs.adf_seeds import memory
from non_gpu_fixture import shared_adf
from arch_explorer.search.nested import run_search
from arch_explorer.agent_env.driver import run_agent


def unwrap(result):
    assert "error" not in result, result
    return result["result"]


def test_all_tools_atomic_edits_domains_and_state_control(config):
    env = ExplorationEnv(config)
    description = unwrap(env.describe_environment())
    assert len(description["components"]) == 8
    assert {t["name"] for t in description["tools"]} == {t["name"] for t in TOOL_SCHEMAS}
    initial = unwrap(env.get_state())
    bad = env.apply_edits(edits=[{"op": "delete", "component": "chip/l2"}, {"op": "delete", "component": "chip/missing"}])
    assert bad["error"]["code"] == "path_not_found"
    assert unwrap(env.get_state())["adf"] == initial["adf"]
    assert env.used == 0
    unwrap(env.checkpoint(name="initial"))
    unwrap(env.set_parameters(parameters=[{"path": "chip/core/vector", "param": "clock_ghz", "value": 2.0, "source": "test choice"}]))
    assert env.used == 0
    evaluated = unwrap(env.evaluate(optimize="mapping"))
    used = env.used
    assert evaluated["score"]["feasible"] and used == 2
    assert evaluated["workloads"][0]["evaluation"]["provenance"]
    unwrap(env.restore(name="initial"))
    assert env.used == used and unwrap(env.get_state())["latest_evaluation"] is None
    unwrap(env.undo())
    assert unwrap(env.get_state())["latest_evaluation"]["score"] == evaluated["score"]
    assert env.used == used
    assert unwrap(env.best())["candidate"]["score"]["objective"] == evaluated["score"]["objective"]
    assert unwrap(env.history())["evaluation_count"] == used
    for value in [-1, 1e9, True, "fast", float("nan")]:
        before = env.adf.to_dict()
        assert "error" in env.set_parameters(parameters=[{"path": "chip/core/vector", "param": "clock_ghz", "value": value}])
        assert env.adf.to_dict() == before


def test_schema_errors_and_grouped_high_transform(config):
    env = ExplorationEnv(config)
    assert env.call("missing")["error"]["code"] == "unknown_tool"
    assert env.call("evaluate", {"optimize": "other"})["error"]["code"] == "invalid_arguments"
    assert env.call("checkpoint", {"unexpected": 1})["error"]["code"] == "invalid_arguments"
    assert env.apply_edits(edits=[])["error"]["code"] == "invalid_arguments"
    assert env.apply_edits(edits=[{"op": "delete", "component": "chip/l2"}] * 4)["error"]["code"] == "invalid_arguments"
    high = {"op": "bond_memory", "core": "chip/core", "memory_spec": memory("bonded", technology="bonded").to_dict()}
    assert env.apply_edits(edits=[high, {"op": "delete", "component": "chip/l2"}])["error"]["code"] == "invalid_edits"
    unwrap(env.apply_edits(edits=[high]))
    assert env.used == 0


def test_budget_counts_backend_attempts_and_never_refunds(config):
    config.update(max_evaluations=7, parameter_samples=4, mapping_samples=3)
    config["workloads"].append({"kind": "operator", "params": {"operations": 8192}, "weight": 2.0})
    env = ExplorationEnv(config)
    result = unwrap(env.evaluate(optimize="parameters_and_mapping"))
    assert env.used == 6 and env.remaining == 1  # no partial two-workload suite
    expected = sum(row["workload"]["weight"] * row["evaluation"]["latency_ns"] for row in result["workloads"])
    assert result["score"]["weighted_latency_ns"] == pytest.approx(expected)
    unwrap(env.undo())
    assert env.used == 6
    assert env.evaluate()["error"]["code"] == "evaluation_budget_exhausted"
    assert env.used == 6


def test_failed_backend_counts_and_partial_suite_never_becomes_best(config):
    config.update(max_evaluations=3, backend_options={"fail": True})
    env = ExplorationEnv(config)
    assert env.evaluate(optimize="parameters_and_mapping")["error"]["code"] == "evaluation_failed"
    assert env.used == 3
    assert all(not row["valid"] for row in env.attempts)
    assert env.best_candidate is None and env.latest is None


def test_missing_cost_area_are_unavailable_not_zero(config):
    config["backend_options"] = {"missing_metrics": True}
    # An unconsumed physical-model override makes the cost incomplete, even
    # though the latency backend can still execute the candidate.
    config["cost_model"] = {"overrides": {"missing/path": {"capacity": {
        "value": 1, "unit": "B", "source": "test unavailable model input"}}}}
    env = ExplorationEnv(config)
    score = unwrap(env.evaluate())["score"]
    assert score["objective"] is None and not score["feasible"]
    assert score["metrics"] == {"cost_usd": None, "area_mm2": None}
    assert set(score["unavailable_constraints"]) == {"cost_usd", "area_mm2"}
    with pytest.raises(EnvError, match="unavailable"):
        run_search(config)
    config.update(cost_budget_usd=None, area_budget_mm2=None)
    score = unwrap(ExplorationEnv(config).evaluate())["score"]
    assert score["feasible"] and score["objective"] == score["weighted_latency_ns"]


def test_objective_penalties_are_from_cost_model(config):
    from arch_explorer.cost import estimate
    config.update(cost_budget_usd=1.0, area_budget_mm2=1.0, cost_penalty=3.0, area_penalty=5.0)
    env = ExplorationEnv(config)
    cost = estimate(env.adf)
    score = unwrap(env.evaluate())["score"]
    assert score["metrics"] == {"cost_usd": cost.cost_usd, "area_mm2": cost.area_mm2}
    assert score["objective"] == pytest.approx(score["weighted_latency_ns"] +
        3 * (score["metrics"]["cost_usd"] - 1) + 5 * (score["metrics"]["area_mm2"] - 1))
    assert not score["feasible"]


def test_trace_and_submission_round_trip(config, tmp_path):
    config["trace_path"] = str(tmp_path / "tools.jsonl")
    env = ExplorationEnv(config)
    unwrap(env.describe_environment())
    unwrap(env.checkpoint(name="seed"))
    unwrap(env.apply_edits(edits=[{"op": "delete", "component": "chip/l2"}]))
    unwrap(env.evaluate(optimize="parameters_and_mapping"))
    unwrap(env.restore(name="seed"))
    unwrap(env.evaluate())
    result = unwrap(env.submit(path=str(tmp_path / "best.json")))
    rows = [json.loads(line) for line in env.trace_path.read_text().splitlines()]
    again = replay_trace(rows)
    assert again.used == env.used
    assert again.best_candidate == env.best_candidate
    replayed = replay_result(result)
    assert replayed["best"] == result["best"]


@pytest.mark.parametrize("policy", ["sa", "greedy", "random"])
@pytest.mark.parametrize("basic_only", [True, False])
def test_search_uses_environment_and_replays(config, policy, basic_only):
    env = ExplorationEnv(config)
    result = run_search(config, env=env, policy=policy, basic_only=basic_only)
    assert 0 < env.used <= config["max_evaluations"]
    assert result["evaluations_used"] == env.used
    assert result["best"]["score"]["objective"] is not None
    assert any(row["tool"] == "apply_edits" for row in env.calls)
    if basic_only:
        assert all(edit["op"] in {"split", "merge", "insert", "remove", "create", "delete"}
                   for row in result["search"]["trajectory"] for edit in row["edits"])
    replayed = replay_trace(env.trace, config=config)
    assert replayed.best_candidate == env.best_candidate
    assert replayed.used == env.used


def test_scripted_agent_uses_identical_metering(config):
    env = ExplorationEnv(config)
    result = run_agent(config, env=env, policy="scripted")
    assert env.used == config["max_evaluations"]
    assert result["agent"]["stop_reason"] == "evaluation_budget"
    again = replay_trace(env.trace, config=config)
    assert again.best_candidate == result["best"]


def test_local_sampler_stays_near_candidate_and_biases_growth():
    # Narrow to one tunable parameter to examine the sampling distribution itself.
    for cls in COMPONENTS._items.values():
        cls.params = {k: ParamSpec(s.unit, s.kind, s.extensive, None, s.required) for k, s in cls.params.items()}
    COMPONENTS._items["vector_unit"].params["clock_ghz"] = ParamSpec("GHz", domain=(0.1, 10.0))
    tree = shared_adf()
    values = [find(parameter_variant(tree, COMPONENTS, random.Random(seed), 0.1, grow=True),
                   "chip/core/vector").value("clock_ghz") for seed in range(20)]
    assert all(1.0 <= value <= 1.1 for value in values)


def test_shared_budget_for_every_policy(config):
    actual = {}
    for policy in ("sa", "greedy", "random"):
        actual[policy] = run_search(config, policy=policy)["evaluations_used"]
    actual["scripted"] = run_agent(config)["evaluations_used"]
    assert actual == {policy: 24 for policy in ("sa", "greedy", "random", "scripted")}


@pytest.mark.parametrize("policy,accepted", [("greedy", False), ("random", True), ("sa", True)])
def test_outer_acceptance_rules(config, policy, accepted):
    from core_fakes import FakeMapping
    from arch_explorer.adf import MAPPING_TEMPLATES

    class SingleMapping(FakeMapping):
        def space(self, adf, workload):
            return {"partitions": [1], "storage": ["shared"]}

    MAPPING_TEMPLATES._items[SingleMapping.name] = SingleMapping
    config.update(max_evaluations=2, parameter_samples=1, mapping_samples=1)
    config["search"].update(max_steps=1, temperature=1e12, max_relative_degradation=10.0,
                            proposals=[{"op": "delete", "component": "chip/l2"}])
    env = ExplorationEnv(config)
    result = run_search(config, env=env, policy=policy)
    assert result["search"]["trajectory"][0]["accepted"] is accepted
    assert env.used == 2
    assert any(n.name == "l2" for n in env.adf.elements) is not accepted


def test_nonfinite_and_nonjson_inputs_are_traced(config):
    env = ExplorationEnv(config)
    bad = [{"parameters": [{"path": "chip/core/vector", "param": "clock_ghz", "value": float("nan")}]},
           {"parameters": object()}]
    for args in bad:
        assert env.call("set_parameters", args)["error"]["code"] == "invalid_arguments"
    rows = [json.loads(line) for line in env.trace_path.read_text().splitlines()]
    restored = replay_trace(rows)
    assert restored.used == 0 and len(restored.calls) == 2


def test_backend_cost_metrics_do_not_override_cost_model(config):
    from core_fakes import fake_backend

    def inconsistent(adf, workload, mapping):
        result = fake_backend(adf, workload, mapping)
        result.cost_usd = workload.params["operations"]
        return result

    register_backend("core-test-inconsistent", inconsistent)
    config["backend"] = "core-test-inconsistent"
    config["workloads"].append({"kind": "operator", "params": {"operations": 8192}})
    env = ExplorationEnv(config)
    result = unwrap(env.evaluate())
    assert result["score"]["metrics"]["cost_usd"] == result["cost_estimate"]["cost_usd"]
    assert result["score"]["metrics"]["cost_usd"] != 4096
    assert env.used == 2 and env.best_candidate is not None


def test_fixed_parameter_rejected_and_provenance_bottleneck_summary(config):
    COMPONENTS._items["vector_unit"].params["clock_ghz"] = ParamSpec("GHz")
    env = ExplorationEnv(config)
    result = env.set_parameters(parameters=[{"path": "chip/core/vector", "param": "clock_ghz", "value": 2.0}])
    assert result["error"]["code"] == "fixed_parameter"
    evaluation = unwrap(env.evaluate())["workloads"][0]
    assert evaluation["diagnostics"]["provenance"]["missing_sources"] == 0
    assert evaluation["diagnostics"]["operators_by_accumulated_task_ns"][0]["operator"] == "synthetic"
