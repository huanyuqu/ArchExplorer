import asyncio
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arch_explorer.agent_env import ExplorationEnv, TOOL_SCHEMAS
from arch_explorer.agent_env.driver import run_agent
from arch_explorer.agent_env.transport import serve_json_lines


@pytest.fixture
def config(config):
    """Transport contracts use a non-GPU seed after GSM removal."""
    config["seed_adf"] = {"preset": "dmc", "parameters": {"num_cores": 4}}
    return config


def requests():
    return [
        ("describe_environment", {}), ("get_state", {}), ("checkpoint", {"name": "seed"}),
        ("apply_edits", {"edits": [{"op": "delete", "component": "chip/core/sram"}]}),
        ("set_parameters", {"parameters": [{"path": "chip/core/vector", "param": "clock_ghz", "value": 1.2}]}),
        ("set_mappings", {"mappings": [{"template": "core-test-map", "params": {"partitions": 1, "storage": "shared"}}]}),
        ("evaluate", {"optimize": "mapping"}), ("best", {}), ("history", {}),
        ("undo", {}), ("restore", {"name": "seed"}), ("submit", {}),
    ]


def subprocess_config(config, tmp_path):
    config = {**config, "registration_modules": ["register_core_fakes"]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [
        str(Path(__file__).parent), os.environ.get("PYTHONPATH")]))}
    return path, environment


def test_json_lines_all_tools_in_real_subprocess(config, tmp_path):
    path, environment = subprocess_config(config, tmp_path)
    lines = [json.dumps({"id": i, "tool": name, "args": args}) for i, (name, args) in enumerate(requests())]
    lines.extend(["not json", json.dumps({"id": 99, "tool": "evaluate", "args": {"optimize": "bad"}})])
    result = subprocess.run([sys.executable, "-m", "arch_explorer", "env", "serve", "--config", str(path)],
                            input="\n".join(lines) + "\n", text=True, capture_output=True, env=environment, timeout=15)
    assert result.returncode == 0, result.stderr
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(outputs) == len(lines)
    assert all("result" in row for row in outputs[:len(requests())]), outputs
    assert outputs[-2]["error"]["code"] == "invalid_request"
    assert outputs[-1]["error"]["code"] == "invalid_arguments"
    assert {n for n, _ in requests()} == {t["name"] for t in TOOL_SCHEMAS}
    assert [row["id"] for row in outputs[:len(requests())]] == list(range(len(requests())))


def test_json_lines_null_arguments_rejected(config):
    output = io.StringIO()
    serve_json_lines(ExplorationEnv(config), io.StringIO('{"id":1,"tool":"evaluate","args":null}\n'), output)
    assert json.loads(output.getvalue())["error"]["code"] == "invalid_arguments"


def test_json_lines_recovers_after_bad_frames_in_subprocess(config, tmp_path):
    path, environment = subprocess_config(config, tmp_path)
    bad = [f'{{"id":{value},"tool":"get_state","args":{{}}}}' for value in
           ("NaN", "Infinity", "-Infinity", "1e309", "true", "false", "1.5", "1.0", "[]", "{}")]
    bad += ['{"id":1,"tool":"checkpoint","args":{"name":NaN}}',
            '{"id":1,"tool":"get_state","args":{"nested":[1e309]}}',
            '{"id":1,"tool":null}', '{"id":1}', '[]', '{broken']
    lines = []
    for index, frame in enumerate(bad):
        lines += [frame, json.dumps({"id": index, "tool": "get_state", "args": {}})]
    result = subprocess.run([sys.executable, "-m", "arch_explorer", "env", "serve", "--config", str(path)],
                            input="\n".join(lines) + "\n", text=True, capture_output=True,
                            env=environment, timeout=15)
    assert result.returncode == 0, result.stderr
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(frames) == len(lines)
    for index in range(len(bad)):
        assert frames[2 * index]["error"]["code"] == "invalid_request"
        assert frames[2 * index + 1]["id"] == index
        assert "result" in frames[2 * index + 1]
    assert frames[0]["id"] is None  # The acceptance team's exact NaN-id case.


def test_json_lines_invalid_frame_never_calls_env_and_valid_ids_roundtrip(config):
    env, output = ExplorationEnv(config), io.StringIO()
    lines = ['{"id":false,"tool":"checkpoint","args":{"name":"bad"}}']
    ids = [None, "session-1", 0, -42, 2 ** 64]
    lines += [json.dumps({"id": value, "tool": "get_state"}) for value in ids]
    serve_json_lines(env, io.StringIO("\n".join(lines)), output)
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    assert frames[0]["id"] is None
    assert [frame["id"] for frame in frames[1:]] == ids
    assert not env.checkpoints and len(env.calls) == len(ids)


@pytest.mark.parametrize("bad_result", [float("nan"), float("inf"), object()])
def test_json_lines_response_serialization_failure_recovers(bad_result):
    class BrokenResultEnv:
        def call(self, tool, args):
            return {"result": bad_result if tool == "bad" else "ok"}

    output = io.StringIO()
    serve_json_lines(BrokenResultEnv(), io.StringIO(
        '{"id":"bad-id","tool":"bad"}\n{"id":2,"tool":"good"}\n'), output)
    first, second = map(json.loads, output.getvalue().splitlines())
    assert first["id"] == "bad-id" and first["error"]["code"] == "invalid_response"
    assert second == {"id": 2, "result": "ok"}


def test_mcp_bad_numbers_do_not_discard_next_request(config, tmp_path):
    pytest.importorskip("mcp")
    path, environment = subprocess_config(config, tmp_path)

    async def client():
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "arch_explorer", "mcp", "--config", str(path),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=environment)

        async def send(wire):
            process.stdin.write((wire + "\n").encode())
            await process.stdin.drain()

        async def receive():
            return json.loads(await asyncio.wait_for(process.stdout.readline(), 5))

        try:
            await send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-11-25", "capabilities": {},
                "clientInfo": {"name": "bad-frame-regression", "version": "1"}}}))
            assert (await receive())["id"] == 1
            await send('{"jsonrpc":"2.0","method":"notifications/initialized"}')
            for value in ("NaN", "Infinity", "-Infinity", "1e309"):
                await send('{"jsonrpc":"2.0","id":' + value + ',"method":"tools/list"}')
            # SDK drops the invalid-id frames; nonfinite tool arguments reach
            # the environment and return its finite invalid_arguments envelope.
            await send('{"jsonrpc":"2.0","id":2,"method":"tools/call",'
                       '"params":{"name":"checkpoint","arguments":{"name":NaN}}}')
            invalid = await receive()
            assert invalid["id"] == 2 and invalid["result"]["isError"]
            assert json.loads(invalid["result"]["content"][0]["text"])["error"]["code"] == "invalid_arguments"
            await send('{"jsonrpc":"2.0","id":3,"method":"tools/call",'
                       '"params":{"name":"get_state","arguments":{}}}')
            valid = await receive()
            assert valid["id"] == 3 and not valid["result"]["isError"]
            process.stdin.close()
            assert await asyncio.wait_for(process.wait(), 5) == 0
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    asyncio.run(asyncio.wait_for(client(), 20))


def test_mcp_handshake_schema_and_all_tools(config, tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    path, environment = subprocess_config(config, tmp_path)

    async def client():
        params = StdioServerParameters(command=sys.executable, args=["-m", "arch_explorer", "mcp", "--config", str(path)], env=environment)
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert {t.name for t in listed.tools} == {t["name"] for t in TOOL_SCHEMAS}
                for tool in listed.tools:
                    expected = next(t for t in TOOL_SCHEMAS if t["name"] == tool.name)
                    assert tool.inputSchema == expected["inputSchema"]
                for name, args in requests():
                    response = await session.call_tool(name, args)
                    assert not response.isError, (name, response)
                    assert "result" in json.loads(response.content[0].text)
                invalid = await session.call_tool("evaluate", {"optimize": "bad"})
                assert invalid.isError
                assert json.loads(invalid.content[0].text)["error"]["code"] == "invalid_arguments"

    asyncio.run(asyncio.wait_for(client(), timeout=20))


@pytest.mark.parametrize("command", ["search", "agent"])
def test_cli_submission_independent_replay(config, tmp_path, command):
    config["max_evaluations"] = 8
    path, environment = subprocess_config(config, tmp_path)
    output, replayed = tmp_path / "best.json", tmp_path / "replayed.json"
    args = ["search", "--policy", "greedy"] if command == "search" else ["agent", "run", "--policy", "scripted"]
    result = subprocess.run([sys.executable, "-m", "arch_explorer", *args, "--config", str(path), "--output", str(output)],
                            capture_output=True, text=True, env=environment, timeout=15)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "arch_explorer", "evaluate", str(output), "--output", str(replayed)],
                            capture_output=True, text=True, env=environment, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["best"] == json.loads(replayed.read_text())["best"]


def test_offline_driver_performs_no_network_io(config, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline policy attempted network I/O")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    result = run_agent(config, policy="scripted")
    assert result["agent"]["policy"] == "scripted"
    assert result["evaluations_used"] == config["max_evaluations"]
    with pytest.raises(ValueError, match="offline"):
        run_agent(config, policy="remote")


def test_scripted_step_limit_and_explicit_actions(config):
    config["agent"] = {"max_steps": 3, "script": [
        {"tool": "apply_edits", "args": {"edits": [{"op": "delete", "component": "chip/l2"}]}},
        {"tool": "evaluate", "args": {}}, {"tool": "checkpoint", "args": {}},
        {"tool": "evaluate", "args": {}},
    ]}
    result = run_agent(config)
    assert result["agent"]["steps"] == 3
    assert result["evaluations_used"] == 1
    assert result["agent"]["stop_reason"] == "step_limit"
