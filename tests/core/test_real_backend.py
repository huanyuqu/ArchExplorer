"""Agent entry points against built-in models in fresh processes (no fake registry)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from non_gpu_fixture import shared_adf


ROOT = Path(__file__).resolve().parents[2]


def cli(*args, wire=None):
    result = subprocess.run([sys.executable, "-m", "arch_explorer", *args], input=wire,
                            text=True, capture_output=True, cwd=ROOT, timeout=60)
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize("preset", ["shared", "dmc"])
def test_real_env_edit_failure_restore_and_submit(preset, tmp_path):
    config = configuration(preset, tmp_path)
    output = tmp_path / "env-best.json"
    edit = ({"op": "delete", "component": "chip/core"} if preset == "shared" else
            {"op": "remove", "level": "chip/core"})
    calls = [("describe_environment", {}), ("checkpoint", {"name": "seed"}),
             ("evaluate", {}), ("apply_edits", {"edits": [edit]}), ("evaluate", {}),
             ("history", {}), ("restore", {"name": "seed"}), ("evaluate", {}),
             ("submit", {"path": str(output)})]
    wire = '{"id":NaN,"tool":"get_state","args":{}}\n'
    wire += "\n".join(json.dumps({"id": i, "tool": tool, "args": args})
                      for i, (tool, args) in enumerate(calls)) + "\n"
    result = cli("env", "serve", "--config", str(config), wire=wire)
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(frames) == len(calls) + 1
    assert frames.pop(0)["error"]["code"] == "invalid_request"
    for i, frame in enumerate(frames):
        assert frame["id"] == i
        assert ("error" in frame) == (i == 4), frame
    error = frames[4]["error"]
    assert error["code"] == "evaluation_failed"
    assert "docs/components.md#adf-requirements" in error["details"]["recovery"]
    cause = error["details"]["trials"][0]["cause"]
    if preset == "shared":
        assert cause["code"] == "no_mapping"
    else:
        attempts = cause["details"]["attempts"]
        assert "chip/noc: 9 endpoints do not fit a 1x5 mesh" in attempts[0]["error"]
        assert attempts[0]["index"] == 1
        assert "template" in attempts[0]["mapping"]
    payload = json.loads(output.read_text())
    assert payload["backend"] == "events" and payload["registration_modules"] == []
    assert payload["best"]["score"]["objective"] > 0


@pytest.mark.parametrize("preset", ["shared", "dmc"])
def test_real_scripted_example_and_independent_replay(preset, tmp_path):
    output, replayed = tmp_path / "scripted.json", tmp_path / "replayed.json"
    cli("agent", "run", "--policy", "scripted", "--config", str(configuration(preset, tmp_path)),
        "--output", str(output))
    cli("evaluate", str(output), "--output", str(replayed))
    result = json.loads(output.read_text())
    assert result["backend"] == "events" and result["best"]["score"]["objective"] > 0
    assert result["best"] == json.loads(replayed.read_text())["best"]


def test_real_budget_pressure_samples_explicit_domain_and_replays(tmp_path):
    output, replayed = tmp_path / "budget.json", tmp_path / "replayed.json"
    wire = '\n'.join(json.dumps({"id":i, "tool":tool, "args":args}) for i, (tool,args) in enumerate([
        ("evaluate", {"optimize":"parameters_and_mapping"}), ("submit", {"path":str(output)})])) + '\n'
    cfg = json.loads((ROOT / 'examples/adf-budget.json').read_text())
    cfg.update(growth_probability=0.0, shrink_probability=1.0, cost_budget_usd=0.1, cost_penalty=1e6,
        parameter_domains=[{'path': 'chip/core/vector', 'param': 'lanes', 'values': [8, 16],
                            'source': 'explicit shrinking lane test domain'}])
    cfg_path = tmp_path / 'budget-config.json'
    cfg_path.write_text(json.dumps(cfg))
    result = cli("env", "serve", "--config", str(cfg_path), wire=wire)
    evaluation = json.loads(result.stdout.splitlines()[0])["result"]
    costs = [trial["score"]["metrics"]["cost_usd"] for trial in evaluation["trials"]]
    known = [cost for cost in costs if cost is not None]
    assert costs[0] is not None and min(known) < costs[0]
    assert evaluation["score"]["metrics"]["cost_usd"] == min(known)
    # Non-GPU DRAM categorical alternatives can have unknown prices. They
    # remain null and ineligible rather than being treated as cheap winners.
    for trial in evaluation['trials']:
        if trial['score']['metrics']['cost_usd'] is None:
            assert trial['score']['objective'] is None and not trial['score']['feasible']
            assert 'cost_usd' in trial['score']['unavailable_constraints']
    cli("evaluate", str(output), "--output", str(replayed))
    assert json.loads(output.read_text())["best"] == json.loads(replayed.read_text())["best"]
    cli("search", "--policy", "sa", "--config", str(cfg_path), "--output", str(output))
    cli("evaluate", str(output), "--output", str(replayed))
    result = json.loads(output.read_text())
    assert result["best"] == json.loads(replayed.read_text())["best"]
    assert result["best"]["score"]["metrics"]["cost_usd"] == result["best"]["cost_estimate"]["cost_usd"]


def configuration(preset, tmp_path):
    if preset == 'dmc':
        return ROOT / 'examples/adf-dmc.json'
    cfg = {'seed_adf': shared_adf().to_dict(), 'workloads': [{'kind': 'operator',
        'params': {'op': 'matmul', 'm': 16, 'k': 32, 'n': 32}}], 'backend': 'events',
        'mapping_templates': ['gsm_tiled'], 'max_evaluations': 8,
        'mapping_samples': 1, 'parameter_samples': 1, 'random_seed': 17}
    path = tmp_path / 'shared-vector.json'
    path.write_text(json.dumps(cfg))
    return path
