"""The configuration contract shared by all ADF exploration entry points."""
from arch_explorer.adf import SpaceMatrix, SpacePoint
from .errors import EnvError
from .schemas import TOOL_SCHEMAS, validate_arguments

FIELDS = {
    "seed_adf", "workloads", "backend", "backend_options", "registration_modules",
    "max_evaluations", "parameter_samples", "mapping_samples", "parameter_radius",
    "growth_probability", "shrink_probability", "parameter_domains", "mapping_templates", "compilation",
    "mapping_selectors", "cost_model", "cost_budget_usd", "area_budget_mm2", "cost_penalty",
    "area_penalty", "resource_constraints", "random_seed", "trace_path", "search", "agent",
}


def _object(value, name, allowed=None, required=()):
    if not isinstance(value, dict):
        raise EnvError("invalid_config", f"{name} must be an object")
    if allowed is not None and set(value) - allowed:
        raise EnvError("invalid_config", f"Unknown {name} fields: {sorted(set(value) - allowed)}")
    if set(required) - set(value):
        raise EnvError("invalid_config", f"Missing {name} fields: {sorted(set(required) - set(value))}")


def validate_config(config):
    _object(config, "config", FIELDS, ("seed_adf", "workloads"))
    if "compilation" in config:
        from .compiled import settings
        settings(config["compilation"])
        if "backend" in config:
            raise EnvError("invalid_config", "Compiled exploration routes through component bindings; omit backend")
    if "resource_constraints" in config:
        from arch_explorer.structure.resources import validate_constraints
        try:
            validate_constraints(config["resource_constraints"])
        except ValueError as error:
            raise EnvError("invalid_config", str(error)) from error
    seed = config["seed_adf"]
    if not isinstance(seed, (SpaceMatrix, SpacePoint)):
        _object(seed, "seed_adf")
        if "preset" in seed:
            _object(seed, "seed_adf", {"preset", "parameters"})
            if not isinstance(seed["preset"], str):
                raise EnvError("invalid_config", "preset must be a string")
            _object(seed.get("parameters", {}), "seed parameters")
    if not isinstance(config["workloads"], list) or not config["workloads"]:
        raise EnvError("invalid_config", "workloads must be a nonempty list")
    for key in ("backend_options", "mapping_selectors", "cost_model"):
        _object(config.get(key, {}), key)
    for key in ("parameter_domains", "registration_modules"):
        if not isinstance(config.get(key, []), list):
            raise EnvError("invalid_config", f"{key} must be a list")
    if "mapping_templates" in config:
        names = config["mapping_templates"]
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            raise EnvError("invalid_config", "mapping_templates must be a nonempty list of names")
    _object(config.get("search", {}), "search", {
        "max_steps", "temperature", "cooling", "max_relative_degradation", "max_proposals", "proposals"})
    proposals = config.get("search", {}).get("proposals", [])
    if not isinstance(proposals, list) or any(not isinstance(p, (dict, list)) for p in proposals):
        raise EnvError("invalid_config", "search.proposals must be a list of edits or edit groups")
    edit_schema = next(s["inputSchema"] for s in TOOL_SCHEMAS if s["name"] == "apply_edits")
    for proposal in proposals:
        if isinstance(proposal, dict) and "tool" in proposal:
            _object(proposal, "search proposal", {"tool", "args"}, ("tool", "args"))
            if proposal["tool"] != "set_parameters":
                raise EnvError("invalid_config", "Explicit tool proposals support set_parameters; structure proposals use edits")
            schema = next(s["inputSchema"] for s in TOOL_SCHEMAS if s["name"] == proposal["tool"])
            validate_arguments(proposal["args"], schema)
        else:
            validate_arguments({"edits": proposal if isinstance(proposal, list) else [proposal]}, edit_schema)
    _object(config.get("agent", {}), "agent", {"script", "max_steps", "max_tool_calls", "log_path"})
