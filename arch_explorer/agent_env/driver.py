"""Offline scripted tool client for CI and demonstration; no model API integration."""
from copy import deepcopy
import json
from pathlib import Path

from .environment import ExplorationEnv
from .errors import EnvError
from .config import validate_config


def run_agent(config=None, *, env=None, policy="scripted", max_steps=None):
    if policy != "scripted":
        raise ValueError("Only the offline scripted policy is supported")
    config = env.config if config is None and env is not None else config
    validate_config(config)
    settings = config.get("agent", {})
    if set(settings) - {"script", "max_steps", "max_tool_calls", "log_path"}:
        raise ValueError("agent accepts only script, max_steps, max_tool_calls and log_path")
    steps = max_steps if max_steps is not None else settings.get("max_steps", 30)
    max_tools = settings.get("max_tool_calls", steps)
    if type(steps) is not int or steps < 1 or type(max_tools) is not int or max_tools < 1:
        raise ValueError("Agent step and tool limits must be positive integers")
    script = settings.get("script")
    if script is not None and (not isinstance(script, list) or any(
            not isinstance(action, dict) or not isinstance(action.get("tool"), str)
            or set(action) - {"tool", "args"} for action in script)):
        raise ValueError("agent.script must be a list of {tool, args} objects")
    env = env or ExplorationEnv(config)
    log_path = Path(settings.get("log_path", str(env.trace_path) + ".agent.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"policy": policy, "script": script}, allow_nan=False) + "\n")

    def record(row):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    used_tools, stop = 0, "step_limit"
    for i in range(steps):
        if used_tools >= max_tools:
            stop = "tool_limit"
            break
        if script is None:
            if env.remaining < len(env.workloads):
                stop = "evaluation_budget"
                break
            action = {"tool": "evaluate", "args": {"optimize": "parameters_and_mapping"}}
        elif i < len(script):
            action = deepcopy(script[i])
        else:
            stop = "script_complete"
            break
        used_tools += 1
        result = env.call(action["tool"], action.get("args", {}))
        record({"action": action, **result})
        if action["tool"] == "submit" and "result" in result:
            stop = "submitted"
            break
    submitted = env.submit()
    if "error" in submitted:
        raise EnvError("no_submission", "Script finished without an evaluated candidate", stop_reason=stop)
    result = submitted["result"]
    result["agent"] = {"policy": policy, "steps": used_tools, "tool_calls": used_tools,
                       "stop_reason": stop, "log_path": str(log_path), "trace_path": str(env.trace_path)}
    from .compiled import RESULT_SCHEMA as COMPILED_SEARCH_SCHEMA, seal
    if result["schema"] == COMPILED_SEARCH_SCHEMA:
        result = seal(result)
    return result
