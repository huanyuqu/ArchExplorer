"""TP request branches agree with real inner mappings; CPU-only synthetic cases."""
from copy import deepcopy
import json
import random
import subprocess
import sys

import pytest

from arch_explorer.adf import MappingSpec, WorkloadSpec, SpaceMatrix, evaluate
from arch_explorer.adf.registry import MAPPING_TEMPLATES, mapping_spaces
from arch_explorer.agent_env import ExplorationEnv, replay_trace
from arch_explorer.agent_env.environment import load_registrations
from arch_explorer.agent_env.errors import EnvError
from arch_explorer.agent_env.sampling import check_mapping, mapping_candidates
from arch_explorer.archs.adf_seeds import dmc_adf, point, memory
from non_gpu_fixture import shared_adf
from non_gpu_fixture import shared_configuration
from arch_explorer.mapping.tp import TensorParallel, _rank_views


def workload(kind="operator"):
    if kind == "operator":
        return WorkloadSpec(kind, dict(op="matmul", batch=1, m=33, k=23, n=67, dtype="fp16"))
    return WorkloadSpec(kind, dict(model=dict(hidden=32, heads=4, kv_heads=2, head_dim=8,
        intermediate=64, layers=1, vocab=64, gated=True, norm="rmsnorm", activation="silu"),
        phase="prefill", batch=1, seq_len=4, layers=1, dtype="fp16"))


def server(*chips):
    return SpaceMatrix("system", list(chips), [point("link", "crossbar",
        dict(port_bandwidth=(50., "B/ns"), latency=(2., "ns")))])


def fixture(inner="gsm_tiled", ranks=1):
    chip = (shared_adf if inner == "gsm_tiled" else dmc_adf)(1)
    if ranks == 1:
        return chip
    chip.repeat = ranks
    return server(chip)


def sample(adf, w, selectors=None, count=12):
    return mapping_candidates(adf, w, MAPPING_TEMPLATES, count, random.Random(19),
        names=["tensor_parallel"], selectors=None if selectors is None else {"tensor_parallel": selectors})


def ok(response):
    assert "error" not in response, response
    return response["result"]


@pytest.mark.parametrize("kind", ["operator", "transformer"])
@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("inner", [None, "gsm_tiled", "dmc_partition"])
def test_sampling_and_discovery_use_only_active_rank_domains(kind, ranks, inner, monkeypatch):
    load_registrations({})
    adf, w = fixture(inner or 'gsm_tiled', ranks=ranks), workload(kind)
    # Discovery must not evaluate candidates to remove invalid combinations.
    def forbidden(*args, **kwargs):
        raise AssertionError("discovery must not execute the simulator")
    monkeypatch.setattr("arch_explorer.simulator.events.backend.run_mapping", forbidden)
    selectors = None if inner is None else {"inner": inner}
    space = TensorParallel().request_space(adf, w, selectors)
    assert space.branches
    for branch in mapping_spaces(space):
        name = branch["inner"][0]
        expected = set(MAPPING_TEMPLATES.get(name).space(_rank_views(adf)[0], w))
        assert {k[6:] for k in branch if k.startswith("inner.")} == expected
        assert branch["tp"] in ([[1], [2]] if kind == "transformer" and ranks == 2 else [[1]])
    for m in sample(adf, w, selectors):
        check_mapping(adf, w, m, MAPPING_TEMPLATES)
        if inner:
            assert m.params["inner"] == inner


@pytest.mark.parametrize("bad", [
    {"inner": "gsm_tiled", "inner.weights": "local"},
    {"inner": "gsm_tiled", "inner.activations": "dram"},
    {"inner": "gsm_tiled", "inner.split": "n"},
    {"inner": "dmc_partition", "inner.tile_n": 64},
    {"inner": "dmc_partition", "inner.rows_per_cta": 4},
    {"inner": "missing"}, {"inner.typo": 1}, {"tp": True}, {"tp": 2},
    {"collective": "missing"}, {"ignored": 1},
])
def test_illegal_unknown_or_inactive_selectors_rejected_at_every_boundary(bad):
    adf, w = fixture(), workload()
    before = deepcopy(bad)
    with pytest.raises(ValueError):
        TensorParallel().request_space(adf, w, bad)
    with pytest.raises(EnvError):
        sample(adf, w, bad)
    with pytest.raises(EnvError):
        check_mapping(adf, w, MappingSpec("tensor_parallel", bad), MAPPING_TEMPLATES)
    with pytest.raises(ValueError):
        evaluate(adf, w, MappingSpec("tensor_parallel", bad))
    assert bad == before


@pytest.mark.parametrize("kind", ["vit", "clip", "pi05"])
@pytest.mark.parametrize("ranks", [1, 2])
def test_wrapper_cannot_bypass_complete_model_scope(kind, ranks):
    adf, w = fixture(ranks=ranks), WorkloadSpec(kind, {})
    template = TensorParallel()
    assert not template.applicable(adf, w)
    space = template.request_space(adf, w)
    assert not space.branches and "full models" in space.unavailable[0]["reason"]
    with pytest.raises(EnvError):
        sample(adf, w)
    with pytest.raises(EnvError):
        check_mapping(adf, w, MappingSpec(template.name, {}), MAPPING_TEMPLATES)
    with pytest.raises(ValueError):
        evaluate(adf, w, MappingSpec(template.name, {}))
    assert template.name in MAPPING_TEMPLATES.names()


def test_rank_applicability_and_bonded_choices_do_not_leak_from_other_devices():
    a, b = dmc_adf(1), dmc_adf(1)
    a.name, b.name = "a", "b"
    a.elements[0].elements.append(memory("bonded", technology="bonded"))
    adf, w = server(a, b), workload("transformer")
    one = TensorParallel().request_space(adf, w, {"tp": 1, "inner": "dmc_partition"})
    two = TensorParallel().request_space(adf, w, {"tp": 2, "inner": "dmc_partition"})
    assert "bonded" in one.branches[0]["inner.weights"]
    assert "bonded" not in two.branches[0]["inner.weights"]
    # Remove B's per-core SRAM: TP=1 still uses A, TP=2 must reject DMC.
    b.elements[0].elements = [p for p in b.elements[0].elements if p.component != "sram"]
    assert TensorParallel().request_space(adf, w, {"tp": 1, "inner": "dmc_partition"}).branches
    with pytest.raises(ValueError, match="rank subtrees"):
        TensorParallel().request_space(adf, w, {"tp": 2, "inner": "dmc_partition"})
    assert TensorParallel().request_space(adf, w, {"tp": 2, "inner": "gsm_tiled"}).branches


def test_non_gpu_rank_order_and_removed_gpu_knobs_are_explicit():
    a, b = shared_adf(1), shared_adf(1)
    a.name, b.name = "a", "b"
    group = SpaceMatrix("group", [a, b], [point("sw", "crossbar",
        dict(port_bandwidth=(50., "B/ns"), latency=(2., "ns")))], repeat=2)
    adf, w = server(group), workload("transformer")
    original = deepcopy(adf.to_dict())
    assert [rank.name for rank in _rank_views(adf)] == ["a", "b", "a", "b"]
    one = TensorParallel().request_space(adf, w, {"tp": 1, "inner": "gsm_tiled"})
    two = TensorParallel().request_space(adf, w, {"tp": 2, "inner": "gsm_tiled"})
    for space in (one, two):
        assert space.branches
        assert all("inner.tile_k" not in branch and "inner.warp_tile" not in branch for branch in space.branches)
    with pytest.raises(ValueError):
        TensorParallel().request_space(adf, w, {"tp": 2, "inner.tile_k": 32})
    assert adf.to_dict() == original


def test_small_finite_union_enumeration_and_defaults():
    adf, w = fixture('dmc_partition'), workload()
    candidates = sample(adf, w, {"tp": 1, "collective": "ring"}, count=1000)
    # GSM 4*4*4, DMC 2*2*2: disjoint union, not their product.
    assert len(candidates) == 64 + 8
    assert len({json.dumps(m.to_dict(), sort_keys=True) for m in candidates}) == len(candidates)
    for m in candidates:
        check_mapping(adf, w, m, MAPPING_TEMPLATES)
    for inner in ("gsm_tiled", "dmc_partition"):
        adf, w = fixture(inner, 2), workload("transformer")
        p = {"inner": inner}
        a = evaluate(adf, w, MappingSpec("tensor_parallel", p), trace=True)
        b = evaluate(adf, w, MappingSpec("tensor_parallel", dict(tp=2, collective="ring", **p)), trace=True)
        assert a.to_dict() == b.to_dict()
        assert any(row["label"] == "AllReduce" for row in a.breakdown)


@pytest.mark.parametrize("inner", ["gsm_tiled", "dmc_partition"])
def test_real_env_discovery_evaluation_state_and_winner_replay(inner, tmp_path):
    adf, w = fixture(inner, 2), workload("transformer")
    cfg = dict(seed_adf=adf.to_dict(), workloads=[w.to_dict()], backend="events",
        mapping_templates=["tensor_parallel"], mapping_selectors={"tensor_parallel": {"inner": inner, "tp": 2}},
        mapping_samples=2, max_evaluations=8, random_seed=19, trace_path=str(tmp_path/"trace.jsonl"))
    env = ExplorationEnv(cfg)
    desc = ok(env.describe_environment())["mapping_spaces"][0]["tensor_parallel"]
    assert all(b["inner"] == [inner] for b in desc["sampling_spaces"])
    m = sample(adf, w, {"inner": inner, "tp": 2}, count=1)[0]
    ok(env.set_mappings(mappings=[m.to_dict()]))
    result = ok(env.evaluate())
    assert result["workloads"][0]["evaluation"]["latency_ns"] > 0
    winner = tmp_path/"winner.json"
    ok(env.submit(path=str(winner)))
    replay = replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines())
    assert replay.get_state() == env.get_state()
    target = tmp_path/"replayed.json"
    process = subprocess.run([sys.executable, "-m", "arch_explorer", "evaluate", str(winner),
        "--output", str(target)], text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert json.loads(winner.read_text())["best"] == json.loads(target.read_text())["best"]

@pytest.mark.parametrize("preset", ["shared", "dmc"])
def test_default_domain_changes_keep_accounting_and_complete_replay(preset, tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    cfg = (shared_configuration() if preset == 'shared' else
           json.loads((root / 'examples/adf-dmc.json').read_text()))
    cfg["trace_path"] = str(tmp_path/"default-trace.jsonl")
    env = ExplorationEnv(cfg)
    result = ok(env.evaluate(optimize="parameters_and_mapping"))
    # Branch-union sampling changes RNG consumption and hence proposed hardware.
    # Rejected preflight variants are not backend calls; actual attempts still count.
    assert 0 < env.used <= cfg["parameter_samples"] * cfg["mapping_samples"]
    assert len(env.attempts) == env.used
    assert [a["index"] for a in env.attempts] == list(range(env.used))
    assert all(not a.get("error", "").startswith("inner.weights=") for a in env.attempts)
    assert any("score" in t for t in result["trials"])
    winner = tmp_path/"default-winner.json"
    ok(env.submit(path=str(winner)))
    restored = replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines())
    assert restored.get_state() == env.get_state()
    target = tmp_path/"default-replayed.json"
    p = subprocess.run([sys.executable, "-m", "arch_explorer", "evaluate", str(winner),
        "--output", str(target)], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert json.loads(target.read_text())["best"] == json.loads(winner.read_text())["best"]
