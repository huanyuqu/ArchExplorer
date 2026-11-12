"""Real input/transport checks. No pi05 backend is substituted."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from arch_explorer.adf import WorkloadSpec, MappingSpec, evaluate
from arch_explorer.adf.evaluation import supports_model_spec
from arch_explorer.agent_env import ExplorationEnv, replay_trace
from arch_explorer.agent_env.environment import load_registrations
from arch_explorer.model_specs import expand, normalize
from tools.generate_pi05 import cases, configuration

ROOT = Path(__file__).resolve().parents[2]
DRAFT = {"template": "gsm_tiled", "params": {"tile_m": 64, "tile_n": 64, "rows_per_cta": 1}}

@pytest.fixture(autouse=True)
def fake_registry():
    load_registrations({})  # override directory fixture with real production registrations

@pytest.mark.parametrize("name", list(cases()))
def test_roundtrip_execution_and_lifecycle(name, tmp_path):
    cfg = cases()[name]
    cfg["trace_path"] = str(tmp_path/"trace.jsonl")
    w = WorkloadSpec(**cfg["workloads"][0])
    assert WorkloadSpec(**json.loads(json.dumps(w.to_dict()))) == w
    env = ExplorationEnv(cfg)
    assert env.config["workloads"][0]["params"] == cfg["workloads"][0]["params"]
    assert env.describe_environment()["result"]["model_specs"][0]["executor_registered"] is True
    assert "error" not in env.set_mappings(mappings=[DRAFT])
    assert "error" not in env.checkpoint(name="draft")
    before = env.get_state()["result"]
    assert "error" not in env.set_parameters(parameters=[{"path":"chip/dram", "param":"technology", "value":"gddr"}])
    # Supported mapping drafts are invalidated by a structural parameter edit.
    assert env.get_state()["result"]["mappings"] is None
    assert "error" not in env.undo()
    assert env.get_state()["result"] == before
    assert "error" not in env.restore(name="draft")
    for optimize in ("none", "mapping", "parameters_and_mapping"):
        result = env.evaluate(optimize=optimize)
        assert "error" not in result, result
    assert env.used == 3 and len(env.attempts) == 3 and env.best_candidate is not None
    assert "error" not in env.submit()
    actual = evaluate(env.adf, w, MappingSpec(**DRAFT))
    assert actual.latency_ns > 0
    graph = actual.details["full_model"]
    for output in graph["outputs"]:
        assert graph["tensors"][output]["shape"] == [w.params["batch"], w.params["action_horizon"], w.params["action_dim"]]
        assert graph["tensors"][output]["ready_ns"] <= actual.latency_ns
    restored = replay_trace([json.loads(x) for x in env.trace_path.read_text().splitlines()])
    assert restored.get_state() == env.get_state()

@pytest.mark.parametrize("change", [
    {"pi05":True}, {"num_steps":True}, {"num_steps":0}, {"action_horizon":-1},
    {"batch":1.0}, {"max_token_len":201}, {"action_dim":False}, {"iterations":0},
    {"profile":"pi05_droid"}, {"dtype":"fp16"}, {"noise_seed":-1},
    {"image_masks":[[True]]}, {"image_masks":[[1],[True],[True]]},
    {"token_mask":[[True]]}, {"noise_seed":True}, {"camera_count":2},
])
def test_bad_request_rejected(change):
    with pytest.raises(ValueError):
        WorkloadSpec("pi05", change)

def test_shape_and_capacity_mask_does_not_shrink_compute():
    all_on = expand({})
    masked = expand({"image_masks":[[False],[False],[False]], "token_mask":[[False]*200]})
    assert all_on["shapes"] == masked["shapes"]
    assert all_on["storage"] == masked["storage"]
    assert masked["valid_prefix_tokens"] == [0]
    assert all_on["prefix_tokens"] == 968
    assert all_on["storage"]["prefix_kv_bytes"] == 17842176
    assert all_on["counts_per_iteration"]["prefix_passes"] == 1
    assert supports_model_spec("events", "pi05")
    assert supports_model_spec("events", "vit")
    p = normalize({})
    p["image_masks"][0][0] = False
    assert normalize({})["image_masks"][0][0] is True

def calls():
    return [("describe_environment", {}), ("set_mappings", {"mappings":[DRAFT]}),
            ("checkpoint", {"name":"saved"}), ("restore", {"name":"saved"}),
            ("evaluate", {}), ("submit", {}), ("get_state", {})]

def test_capability_requires_exact_registered_contract(monkeypatch):
    from arch_explorer.adf.evaluation import _BACKENDS
    backend = _BACKENDS["events"]
    monkeypatch.setattr(backend, "model_spec_contracts", {"pi05":"wrong-version"}, raising=False)
    assert not supports_model_spec("events", "pi05")
    env = ExplorationEnv(configuration(profile="openpi_pi05_dummy", num_steps=1, action_horizon=1, max_token_len=1))
    rejected = env.evaluate()
    assert rejected["error"]["code"] == "unsupported_workload"
    assert rejected["error"]["details"]["backend_attempted"] is False
    assert env.used == 0 and env.attempts == [] and env.best_candidate is None
    monkeypatch.setattr(backend, "model_spec_contracts", {"pi05":"openpi.pi05.v1"})
    assert supports_model_spec("events", "pi05")
    # Declaration check only: no evaluation or claim that this backend can run pi05.

@pytest.mark.parametrize("draft", [
    {"template":"missing", "params":{}},
    {"template":"gsm_tiled", "params":{"v":float("nan")}},
    {"template":"gsm_tiled", "params":[]},
])
def test_malformed_draft_has_no_budget(draft):
    env = ExplorationEnv(configuration())
    assert "error" in env.set_mappings(mappings=[draft])
    assert env.used == 0 and env.attempts == []

def test_real_jsonl_cli_replay(tmp_path):
    cfg = cases()["two-cameras-tail.json"]
    cfg["trace_path"] = str(tmp_path/"trace.jsonl")
    path = tmp_path/"config.json"
    path.write_text(json.dumps(cfg))
    request = "\n".join(json.dumps({"id":i, "tool":name, "args":args}) for i,(name,args) in enumerate(calls()))+"\n"
    run = subprocess.run([sys.executable,"-m","arch_explorer","env","serve","--config",str(path)],
                         input=request,text=True,capture_output=True,cwd=tmp_path,timeout=30)
    assert run.returncode == 0, run.stderr
    rows = list(map(json.loads,run.stdout.splitlines()))
    assert "error" not in rows[4] and "error" not in rows[5]
    assert rows[-1]["result"]["evaluations_used"] == 1
    output = tmp_path/"state.json"
    replay = subprocess.run([sys.executable,"-m","arch_explorer","env","replay",cfg["trace_path"],
                             "--output",str(output)],text=True,capture_output=True,cwd=tmp_path,timeout=30)
    assert replay.returncode == 0, replay.stderr
    assert json.loads(output.read_text()) == rows[-1]["result"]
    cfg.pop("trace_path")
    path.write_text(json.dumps(cfg))
    search = subprocess.run([sys.executable,"-m","arch_explorer","search","--config",str(path),
                             "--output",str(tmp_path/"winner.json")],text=True,capture_output=True,cwd=tmp_path,timeout=30)
    assert search.returncode == 0 and (tmp_path/"winner.json").exists(), search.stderr
    rerun = subprocess.run([sys.executable,"-m","arch_explorer","evaluate",str(tmp_path/"winner.json"),
        "--output",str(tmp_path/"winner-replay.json")],text=True,capture_output=True,cwd=tmp_path,timeout=60)
    assert rerun.returncode == 0, rerun.stderr
    assert json.loads((tmp_path/"winner.json").read_text())["best"] == json.loads((tmp_path/"winner-replay.json").read_text())["best"]

def test_real_mcp(tmp_path):
    pytest.importorskip('mcp', reason='Optional MCP dependency absent; real transport check not executed')
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    path = tmp_path/"config.json"
    path.write_text(json.dumps(cases()["boundary.json"]))
    async def client():
        environment = dict(os.environ)
        params = StdioServerParameters(command=sys.executable,
            args=["-m","arch_explorer","mcp","--config",str(path)],env=environment)
        async with stdio_client(params) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                await session.initialize()
                assert "evaluate" in {t.name for t in (await session.list_tools()).tools}
                for name,args in calls():
                    response = await session.call_tool(name,args)
                    data = json.loads(response.content[0].text)
                    assert not response.isError, data
                state = json.loads((await session.call_tool("get_state",{})).content[0].text)["result"]
                assert state["evaluations_used"] == 1 and state["latest_evaluation"] is not None
    asyncio.run(asyncio.wait_for(client(),timeout=30))
