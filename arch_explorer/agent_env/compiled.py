"""Public compilation consumption for the existing metered exploration loop.

There are no operator rules, adapter names, provider selection or service
formulas here. Every candidate uses the public compiler and saved results keep
the simulator's own source-bound result, rather than a latency-only cache.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .errors import EnvError
from .objective import score

RESULT_SCHEMA = "archexplorer.compilation-search-result.v1"
COMPILED_RESULT_SCHEMA = "arch-explorer.compiled-result.v3"


def settings(value):
    if type(value) is not dict or set(value) - {"bindings", "implementations"}:
        raise EnvError("invalid_config", "compilation accepts optional bindings and implementations objects")
    if any(type(item) is not dict for item in value.values()):
        raise EnvError("invalid_config", "compilation bindings and implementations must be objects")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise EnvError("invalid_config", "compilation settings must be finite JSON") from None
    return deepcopy(value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def seal(value):
    value = {key: item for key, item in value.items() if key != "binding"}
    return dict(value, binding={"sha256": hashlib.sha256(canonical(value)).hexdigest()})


def effective_mapping(result):
    """Read the producer's exact normalized mapping; never infer rule choices."""
    from arch_explorer.adf import MappingSpec
    if type(result) is not dict or result.get("schema") != COMPILED_RESULT_SCHEMA:
        raise ValueError("Effective mapping requires a public compiled result")
    return MappingSpec(**deepcopy(result["request"]["compilation"]["source"]["mapping"]))


def evaluate_mapping(env, adf, workload, mapping, attempt):
    from arch_explorer.workload.api import compile_workload
    from arch_explorer.simulator.api import prepare_compiled, evaluate_request
    attempt.update(phase="compile", execution_attempted=False, requested_mapping=deepcopy(mapping.to_dict()))
    compilation = compile_workload(adf, workload, mapping, registry=env.components, **env.compilation)
    attempt["compilation_binding"] = deepcopy(compilation["binding"])
    attempt["effective_mapping"] = deepcopy(compilation["source"]["mapping"])
    attempt["phase"] = "prepare"
    directory = env.trace_path.parent / (env.trace_path.name + ".compiled") / f"attempt-{attempt['index']:06d}"
    request = prepare_compiled(compilation, output_dir=directory,
                               options=env.config.get("backend_options", {}), registry=env.components)
    attempt.update(phase="evaluate", request_binding=deepcopy(request["binding"]), execution_attempted=True)
    result = evaluate_request(request)
    if result.get("schema") != COMPILED_RESULT_SCHEMA:
        raise ValueError("compiler evaluation must return its public compiled-result contract")
    path = directory / "EXPLORATION-COMPILED-RESULT.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    attempt.update(phase="completed", result_binding=deepcopy(result["binding"]),
                   public_result={"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result["evaluation"], result


def export(env, result):
    result.pop("backend")
    result.update(schema=RESULT_SCHEMA, compilation=deepcopy(env.compilation),
                  max_evaluations=env.budget, cache_enabled=False,
                  evaluation_attempts=[{key: deepcopy(value) for key, value in attempt.items() if key != "evaluation"}
                                       for attempt in env.attempts])
    return seal(result)


def replay(payload):
    """Validate the fixed winner without preparing a request or evaluating anew."""
    from arch_explorer.adf import WorkloadSpec, fingerprint, from_dict
    from arch_explorer.cost import estimate
    from arch_explorer.structure import normalize
    from arch_explorer.simulator.api import evaluate_request
    from .environment import ExplorationEnv, _positive

    fields = {"schema", "resource_constraints", "cost_model", "parameter_domains", "backend_options",
              "registration_modules", "objective_config", "best", "evaluations_used", "compilation",
              "max_evaluations", "cache_enabled", "evaluation_attempts", "binding"}
    if (type(payload) is not dict or payload.get("schema") != RESULT_SCHEMA
            or not fields <= set(payload) or set(payload) - fields - {"search", "agent"}):
        raise EnvError("invalid_result", "Unsupported compiled exploration result fields")
    if payload["binding"] != seal(payload)["binding"]:
        raise EnvError("invalid_result", "Compiled exploration content binding differs")
    if "agent" in payload:
        agent = payload["agent"]
        if (type(agent) is not dict or set(agent) != {
                "policy", "steps", "tool_calls", "stop_reason", "log_path", "trace_path"}
                or agent["policy"] != "scripted"
                or any(type(agent[key]) is not int or agent[key] < 0 for key in ("steps", "tool_calls"))
                or agent["steps"] != agent["tool_calls"]
                or type(agent["stop_reason"]) is not str or agent["stop_reason"] not in {
                    "step_limit", "tool_limit", "evaluation_budget", "script_complete", "submitted"}
                or any(type(agent[key]) is not str or not agent[key] for key in ("log_path", "trace_path"))):
            raise EnvError("invalid_result", "Invalid compiled exploration Agent metadata")
    chosen = settings(payload["compilation"])
    best = payload["best"]
    if type(best) is not dict or set(best) != {"adf", "fingerprint", "mappings", "cost_estimate", "score", "workloads"}:
        raise EnvError("invalid_result", "Compiled exploration needs a complete fixed winner")
    rows = best["workloads"]
    if type(rows) is not list or not rows or type(best["mappings"]) is not list or len(rows) != len(best["mappings"]):
        raise EnvError("invalid_result", "Winner workloads and mappings differ")
    adf = from_dict(best["adf"])
    if fingerprint(normalize(adf)) != best["fingerprint"]:
        raise EnvError("invalid_result", "Winner ADF fingerprint differs")
    cost = estimate(adf, options=payload["cost_model"]).to_dict()
    if cost != best["cost_estimate"]:
        raise EnvError("invalid_result", "Winner cost estimate differs")
    evaluations, weights = [], []
    for row, mapping in zip(rows, best["mappings"]):
        if type(row) is not dict or set(row) != {"workload", "mapping", "evaluation", "diagnostics", "compiled_result"}:
            raise EnvError("invalid_result", "Winner needs each public compiled-result")
        saved = row["compiled_result"]
        # Check the result role before dispatch: passing a request would execute.
        if type(saved) is not dict or saved.get("schema") != COMPILED_RESULT_SCHEMA:
            raise EnvError("invalid_result", "Read-only replay accepts saved compiled results, never requests")
        verified = evaluate_request(saved)
        if verified != saved or verified["evaluation"] != row["evaluation"]:
            raise EnvError("invalid_result", "Winner evaluation differs from its bound result")
        compilation = verified["request"]["compilation"]
        if verified["request"]["request"]["options"] != payload["backend_options"]:
            raise EnvError("invalid_result", "Winner provider options differ from its bound request")
        original = compilation["source"]
        if original != {"adf": best["adf"], "workload": row["workload"], "mapping": mapping} or row["mapping"] != mapping:
            raise EnvError("invalid_result", "Winner ADF, workload or mapping differs from its bound compilation")
        for path, binding in chosen.get("bindings", {}).items():
            if compilation["bindings"].get(path) != binding:
                raise EnvError("invalid_result", "Winner adapter binding differs from exploration settings")
        if compilation["implementations"] != chosen.get("implementations", {}):
            raise EnvError("invalid_result", "Winner implementation selection differs")
        if row["diagnostics"] != ExplorationEnv._diagnostics(row["evaluation"]):
            raise EnvError("invalid_result", "Winner diagnostics differ")
        evaluations.append(row["evaluation"])
        weights.append(WorkloadSpec(**row["workload"]).weight)
    objective = payload["objective_config"]
    if type(objective) is not dict or set(objective) != {"cost_budget_usd", "area_budget_mm2", "cost_penalty", "area_penalty"}:
        raise EnvError("invalid_result", "Unsupported saved objective")
    for key in ("cost_budget_usd", "area_budget_mm2"):
        if objective[key] is not None:
            _positive(objective[key], key)
    for key in ("cost_penalty", "area_penalty"):
        _positive(objective[key], key, zero=True)
    expected = score(evaluations, cost, weights,
        {"cost_usd": objective["cost_budget_usd"], "area_mm2": objective["area_budget_mm2"]},
        {"cost_usd": objective["cost_penalty"], "area_mm2": objective["area_penalty"]})
    if best["score"] != expected:
        raise EnvError("invalid_result", "Winner objective differs from verified evaluations")
    attempts, used, budget = payload["evaluation_attempts"], payload["evaluations_used"], payload["max_evaluations"]
    if (type(used) is not int or type(budget) is not int or not 0 < used <= budget
            or type(attempts) is not list or len(attempts) != used
            or any(type(row) is not dict or type(row.get("index")) is not int or row["index"] != index
                   or type(row.get("valid")) is not bool for index, row in enumerate(attempts))
            or payload["cache_enabled"] is not False):
        raise EnvError("invalid_result", "Compiled exploration budget ledger differs")
    return deepcopy(payload)


def rerun(payload, *, output_dir):
    """Explicit new execution of the fixed winner with a fresh metered Env."""
    verified = replay(payload)
    from .environment import ExplorationEnv
    directory = Path(output_dir)
    if directory.exists():
        raise ValueError("Explicit rerun requires a new output directory")
    best = verified["best"]
    config = {"seed_adf": best["adf"], "workloads": [row["workload"] for row in best["workloads"]],
              "max_evaluations": len(best["workloads"]), "trace_path": str(directory / "tools.jsonl"),
              "compilation": verified["compilation"], "backend_options": verified["backend_options"],
              "registration_modules": verified["registration_modules"], "cost_model": verified["cost_model"],
              "mapping_templates": sorted({mapping["template"] for mapping in best["mappings"]}),
              **verified["objective_config"]}
    if verified["resource_constraints"] is not None:
        config["resource_constraints"] = verified["resource_constraints"]
    env = ExplorationEnv(config)
    response = env.evaluate(optimize="none", mappings=best["mappings"])
    if "error" in response:
        error = response["error"]
        raise EnvError(error["code"], error["message"], **error.get("details", {}))
    return env.submit(selection="current")["result"]
