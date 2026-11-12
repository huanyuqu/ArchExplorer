"""SA, greedy and random clients of the public exploration tool protocol."""
from copy import deepcopy
import math
import random

from .candidates import proposals
from arch_explorer.agent_env import EnvError, ExplorationEnv
from arch_explorer.structure.transforms import TRANSFORMS


def _unwrap(envelope):
    if "error" in envelope:
        error = envelope["error"]
        raise EnvError(error["code"], error["message"], **error.get("details", {}))
    return envelope["result"]


def run_search(config=None, *, env=None, policy="sa", basic_only=False):
    if policy not in {"sa", "greedy", "random"}:
        raise ValueError("policy must be sa, greedy or random")
    env = env or ExplorationEnv(config)
    config = config or env.config
    settings = config.get("search", {})
    max_steps = settings.get("max_steps", config.get("max_evaluations", 100) * 20)
    temperature = settings.get("temperature", 1.0)
    cooling = settings.get("cooling", 0.95)
    degradation = settings.get("max_relative_degradation", 0.25)
    if (type(max_steps) is not int or max_steps < 1 or type(temperature) not in (int, float)
            or not math.isfinite(temperature) or temperature <= 0 or type(cooling) not in (int, float)
            or not 0 < cooling <= 1 or type(degradation) not in (int, float)
            or not math.isfinite(degradation) or degradation < 0):
        raise ValueError("Invalid search max_steps/temperature/cooling/max_relative_degradation")
    max_proposals = settings.get("max_proposals", 100)
    if type(max_proposals) is not int or max_proposals < 1:
        raise ValueError("max_proposals must be a positive integer")
    rng = random.Random(config.get("random_seed", 0))
    description = _unwrap(env.describe_environment())
    initial = _unwrap(env.evaluate(optimize="parameters_and_mapping"))
    if initial["score"]["objective"] is None:
        raise EnvError("unavailable_constraint", "Search objective unavailable", constraints=initial["score"]["unavailable_constraints"])
    current_score = initial["score"]["objective"]
    initial_state = _unwrap(env.get_state())
    seen = {initial_state["fingerprint"]}
    proposal_audit, termination = [], "max_steps"
    pool = []
    trajectory = []
    for step in range(max_steps):
        state = _unwrap(env.get_state())
        if state["evaluations_remaining"] < len(description["workloads"]):
            termination = "backend_budget_exhausted"
            break
        _unwrap(env.checkpoint(name="search-current"))
        if len(proposal_audit) >= max_proposals and (not pool or "proposals" in settings or state.get("resource_constraints") is None):
            termination = "proposal_budget_exhausted"
            break
        choices = pool or (deepcopy(settings.get("proposals")) if "proposals" in settings else proposals(state, description["components"], basic_only, max_attempts=max_proposals-len(proposal_audit), rng=rng, audit=proposal_audit, registry=env.components))
        if state.get("resource_constraints") is not None:
            pool = choices
        if basic_only:
            choices = [p for p in choices if (p.get("op") not in TRANSFORMS if isinstance(p, dict)
                       else all(e.get("op") not in TRANSFORMS for e in p))]
        if not choices:
            termination = "no_candidate"
            break
        chosen = rng.choice(choices)
        if state.get("resource_constraints") is None or "proposals" in settings:
            key = "proposal" if isinstance(chosen, dict) and "tool" in chosen else "edits"
            proposal_audit.append({key: deepcopy(chosen), "status": "selected"})
        if pool:
            pool.remove(chosen)
        row = {"step": step, "accepted": False, "temperature": temperature}
        if isinstance(chosen, dict) and "tool" in chosen:
            # Validated public parameter proposals use the same tool as an Agent.
            mutation = env.call(chosen["tool"], chosen["args"])
            row.update(tool=chosen["tool"], args=deepcopy(chosen["args"]))
        else:
            if isinstance(chosen, list):
                edits = chosen
            elif state.get("resource_constraints") is not None or chosen["op"] in TRANSFORMS or "proposals" in settings:
                edits = [chosen]
            else:
                basics = [p for p in choices if isinstance(p, dict) and p.get("op") not in TRANSFORMS]
                edits = [chosen] + [rng.choice(basics) for _ in range(rng.randrange(3))]
            mutation = env.apply_edits(edits=edits)
            row["edits"] = edits
        if "error" in mutation:
            row["error"] = mutation["error"]
        elif mutation["result"]["fingerprint"] in seen:
            row["duplicate"] = True
            _unwrap(env.restore(name="search-current"))
        else:
            seen.add(mutation["result"]["fingerprint"])
            evaluation = env.evaluate(optimize="parameters_and_mapping")
            if "error" in evaluation:
                row["error"] = evaluation["error"]
                _unwrap(env.restore(name="search-current"))
            else:
                candidate = evaluation["result"]
                value = candidate["score"]["objective"]
                delta = value - current_score if value is not None else float("inf")
                accept = value is not None and (policy == "random" or delta <= 0 or
                    (policy == "sa" and delta <= degradation * max(abs(current_score), 1e-12)
                     and rng.random() < math.exp(-delta / max(temperature, 1e-12))))
                row.update(objective=value, accepted=accept, evaluations_used=candidate["evaluations_used"])
                seen.add(candidate["fingerprint"])
                if accept:
                    pool = []  # Accepted state invalidates the preflight pool.
                    current_score = value
                else:
                    _unwrap(env.restore(name="search-current"))
        trajectory.append(row)
        temperature *= cooling
    result = _unwrap(env.submit())
    result["search"] = {"policy": policy, "basic_only": basic_only, "proposal_audit": proposal_audit,
                        "termination": termination, "max_proposals": max_proposals, "trajectory": trajectory,
                        "steps": len(trajectory), "max_steps": max_steps,
                        "trace_path": str(env.trace_path),
                        "evaluations_remaining": _unwrap(env.get_state())["evaluations_remaining"]}
    from arch_explorer.agent_env.compiled import RESULT_SCHEMA as COMPILED_SEARCH_SCHEMA, seal
    if result["schema"] == COMPILED_SEARCH_SCHEMA:
        result = seal(result)
    return result
