"""One schema catalog for Python, JSON-lines, MCP and model tool calls."""
from copy import deepcopy
from .errors import EnvError


def obj(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}


STRING = {"type": "string", "minLength": 1}
POSINT = {"type": "integer", "minimum": 1}
NODE = {"type": "object"}
PATHS = {"type": "array", "items": STRING, "minItems": 1, "uniqueItems": True}


def edit(op, properties, required):
    return obj({"op": {"const": op}, **properties}, ["op", *required])


BASIC_SCHEMAS = [
    edit("split", {"point": STRING, "num": {"type": "integer", "minimum": 2}, "algo": STRING}, ["point", "num"]),
    edit("merge", {"points": {**PATHS, "minItems": 2}, "algo": STRING, "name": STRING}, ["points"]),
    edit("insert", {"parent": STRING, "children": PATHS, "name": STRING, "group_size": POSINT}, ["parent", "children"]),
    edit("remove", {"level": STRING}, ["level"]),
    edit("create", {"container": STRING, "spec": NODE, "component": STRING}, ["container"]),
    edit("delete", {"component": STRING}, ["component"]),
]
HIGH_SCHEMAS = [
    edit("chiplet_transform", {"chip": STRING, "cores_per_chiplet": POSINT, "noc_spec": NODE, "nop_spec": NODE},
         ["chip", "cores_per_chiplet", "noc_spec", "nop_spec"]),
    edit("gsm_to_dmc", {"chip": STRING, "shared_memory": STRING, "parts": POSINT, "noc_spec": NODE, "bus_spec": NODE},
         ["chip", "shared_memory", "parts"]),
    edit("pim_transform", {"memory": STRING, "level": {"oneOf": [STRING, {"type": "array", "items": STRING, "minItems": 1}]},
                           "nums": {"oneOf": [POSINT, {"type": "array", "items": POSINT, "minItems": 1}]}, "spec": NODE},
         ["memory", "level", "nums", "spec"]),
    edit("bond_memory", {"core": STRING, "memory_spec": NODE}, ["core", "memory_spec"]),
    edit("share_private_memory", {"chip": STRING, "cores_per_group": POSINT, "memory_name": STRING,
                                  "group_network": NODE, "root_network": NODE, "pool_name": STRING},
         ["chip", "cores_per_group", "memory_name", "group_network", "root_network"]),
]
_TOOLS = {
    "describe_environment": ("Describe objective, budgets, registries and tools.", obj()),
    "task_adapters": ("Describe registered task adapter identities, component compatibility, parameters and group boundaries without execution.", obj()),
    "operator_mappings": ("Describe registered semantic mapping rules and their actual callbacks; optional semantic kind filter.",
                          obj({"operator": STRING})),
    "get_state": ("Read current ADF, fingerprint, resource costs and latest evaluation.", obj()),
    "apply_edits": ("Atomically apply 1–3 basic edits or exactly one high-level transform, without evaluation.",
                    obj({"edits": {"type": "array", "minItems": 1, "maxItems": 3,
                                   "items": {"oneOf": BASIC_SCHEMAS + HIGH_SCHEMAS}}}, ["edits"])),
    "set_parameters": ("Set values in registered domains, without evaluation.", obj({"parameters": {
        "type": "array", "minItems": 1, "items": obj({"path": STRING, "param": STRING,
             "value": {"type": ["number", "string", "boolean"]}, "source": STRING}, ["path", "param", "value"]) }}, ["parameters"])),
    "set_mappings": ("Save one concrete MappingSpec per workload; validate entry syntax without evaluating. An unavailable executor remains unsupported.",
                     obj({"mappings": {"type": "array", "minItems": 1,
                         "items": obj({"template": STRING, "params": NODE}, ["template", "params"])}}, ["mappings"])),
    "compile_workload": ("Compile the current ADF and one workload with public placement/tiles and optional component-path adapter bindings. No evaluation or search.",
                         obj({"workload_index": {"type": "integer", "minimum": 0},
                              "mapping": NODE, "bindings": NODE, "implementations": NODE}, ["mapping"])),
    "migrate_compilation_v1": ("Explicitly recompile a v1 compilation as a new current artifact, without migrating execution receipts or evaluating.",
                               obj({"compilation": NODE}, ["compilation"])),
    "migrate_compilation_v2": ("Explicitly recompile a v2 input with registered mapping rules; keeps historical execution identities.",
                               obj({"compilation": NODE}, ["compilation"])),
    "prepare_compiled": ("Prepare a public compilation artifact in a new output directory without evaluation.",
                         obj({"compilation": NODE, "output_dir": STRING, "options": NODE},
                             ["compilation", "output_dir"])),
    "evaluate": ("Evaluate current ADF; compiled mode counts each selected mapping attempt, including compile/prepare failures; other modes count attempted backend calls.", obj({
        "optimize": {"enum": ["none", "mapping", "parameters_and_mapping"]},
        "mappings": {"type": "array", "items": obj({"template": STRING, "params": NODE}, ["template"])}})),
    "evaluate_request": ("Consume a public prepared request, saved result or compiled exploration winner. A request attempt uses one evaluation; saved-result validation is read-only and uses none. Does not replace the current exploration candidate.",
                         obj({"payload": NODE}, ["payload"])),
    "checkpoint": ("Save current architecture and mapping state; evaluation budget is never restored.", obj({"name": STRING})),
    "restore": ("Restore a checkpoint without refunding evaluations or rewinding randomness.", obj({"name": STRING}, ["name"])),
    "undo": ("Undo the last state mutation; evaluation history and budget remain monotonic.", obj()),
    "history": ("Read tool-call summaries and evaluation attempts.", obj({"offset": {"type": "integer", "minimum": 0}, "limit": POSINT})),
    "best": ("Read the evaluated candidate minimizing the objective; feasibility is reported explicitly.", obj()),
    "submit": ("Export best or current evaluated candidate for standalone evaluate replay.", obj({
        "path": STRING, "selection": {"enum": ["best", "current"]}})),
}
TOOL_SCHEMAS = [{"name": name, "description": desc, "inputSchema": schema} for name, (desc, schema) in _TOOLS.items()]


def validate_arguments(value, schema, path="args"):
    """Validate the JSON Schema subset used by this closed catalog, without a dependency."""
    def fail(message):
        raise EnvError("invalid_arguments", f"{path}: {message}")
    if "oneOf" in schema:
        matches = 0
        for choice in schema["oneOf"]:
            try:
                validate_arguments(value, choice, path)
                matches += 1
            except EnvError:
                pass
        if matches != 1:
            fail("must match exactly one allowed schema")
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool}
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if kinds and not any(isinstance(value, types[k]) and (k not in {"integer", "number"} or type(value) is not bool) for k in kinds):
        fail(f"expected {kind}")
    if "const" in schema and value != schema["const"]:
        fail("wrong constant")
    if "enum" in schema and value not in schema["enum"]:
        fail(f"expected one of {schema['enum']}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys():
            fail("missing required properties")
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            fail(f"unknown properties {sorted(value.keys() - properties.keys())}")
        for key in value.keys() & properties.keys():
            validate_arguments(value[key], properties[key], f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            fail("invalid array length")
        if schema.get("uniqueItems") and any(item in value[:i] for i, item in enumerate(value)):
            fail("duplicate items")
        for i, item in enumerate(value):
            validate_arguments(item, schema.get("items", {}), f"{path}[{i}]")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        fail("string is too short")
    if type(value) in (int, float):
        import math
        if not math.isfinite(value) or value < schema.get("minimum", -float("inf")):
            fail("invalid numeric value")
    return deepcopy(value)
