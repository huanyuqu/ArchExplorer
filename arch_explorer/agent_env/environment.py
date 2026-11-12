"""Shared, metered exploration substrate for humans, agents and search policies."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path
import random
import tempfile

from arch_explorer.adf import (COMPONENTS, MAPPING_TEMPLATES, MappingSpec, SpaceMatrix, SpacePoint,
                              WorkloadSpec, evaluate as backend_evaluate, find, fingerprint, from_dict,
                              walk)
from arch_explorer.archs.adf_seeds import dmc_adf
from arch_explorer.cost import estimate as estimate_cost
from arch_explorer.structure import (StructureError, apply_edit, apply_transform, check, normalize)
from arch_explorer.structure.transforms import TRANSFORMS
from arch_explorer.adf.evaluation import supports_model_spec
from arch_explorer.model_specs import expand as expand_model_spec
from .errors import EnvError
from .config import validate_config
from arch_explorer.structure.resources import resource_report, validate_constraints
from .sampling import check_mapping, mapping_candidates, mapping_contract, parameter_variant
from .schemas import BASIC_SCHEMAS, HIGH_SCHEMAS, TOOL_SCHEMAS, validate_arguments
from . import multi_device
from arch_explorer.adf.multi_device import TEMPLATE as MULTI_DEVICE, UNSUPPORTED as MULTI_UNSUPPORTED

RESULT_SCHEMA = "archexplorer.adf-result.v1"
_NO_ARGS = object()


def _trace_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"$archexplorer_nonfinite": repr(value)}
    if isinstance(value, dict):
        return {key: _trace_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"$archexplorer_unsupported": type(value).__name__}


def _trace_arguments(value):
    if isinstance(value, dict):
        if set(value) == {"$archexplorer_nonfinite"}:
            return float(value["$archexplorer_nonfinite"])
        if set(value) == {"$archexplorer_unsupported"}:
            return object()
        return {key: _trace_arguments(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_trace_arguments(item) for item in value]
    return value


def load_registrations(config):
    # Resolve built-ins and installed extensions through the public author API.
    # Catalogs are needed before seed validation and mapping discovery.
    from arch_explorer.components.api import discover_components
    discover_components()
    for name in config.get("registration_modules", []):
        importlib.import_module(name)


def _positive(value, name, integer=False, zero=False):
    if (type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise EnvError("invalid_config", f"{name} must be a {'nonnegative' if zero else 'positive'} {'integer' if integer else 'finite number'}")
    return value


def _metric(value, name):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise EnvError("invalid_evaluation", f"{name} must be finite and nonnegative, or None")
    return value


class ExplorationEnv:
    def __init__(self, config, *, components=COMPONENTS, templates=MAPPING_TEMPLATES):
        validate_config(config)
        self.config = deepcopy(config)
        load_registrations(config)
        self.components, self.templates = components, templates
        selectors = config.get("mapping_selectors", {})
        selected_names = config.get("mapping_templates") or templates.names()
        if (not isinstance(selectors, dict) or any(name not in selected_names or not (isinstance(values, dict)
                                                 or isinstance(values, list) and values and all(isinstance(v, dict) for v in values))
                                                 for name, values in selectors.items())):
            raise EnvError("invalid_config", "mapping_selectors values must be a request object or a nonempty union of request objects")
        seed = config["seed_adf"]
        if isinstance(seed, (SpaceMatrix, SpacePoint)):
            self.adf = deepcopy(seed)
            self.config["seed_adf"] = seed.to_dict()
        elif "preset" in seed:
            if seed["preset"] == "gsm":
                raise EnvError("invalid_config", "The GSM GPU predictor was removed; use the native Accel-Sim demo (archexplorer demo --help)")
            builders = {"dmc": dmc_adf}
            if seed["preset"] in builders:
                self.adf = builders[seed["preset"]](**seed.get("parameters", {}))
            else:
                from arch_explorer.devices import get
                self.adf = get(seed["preset"], **seed.get("parameters", {}))
        else:
            self.adf = from_dict(seed)
        self.adf = from_dict(self.adf.to_dict())
        check(self.adf, components)
        self.workloads = [w if isinstance(w, WorkloadSpec) else WorkloadSpec(**w) for w in config["workloads"]]
        if not self.workloads:
            raise EnvError("invalid_config", "workloads cannot be empty")
        for workload in self.workloads:
            _positive(workload.weight, "workload.weight")
        self.config["workloads"] = [w.to_dict() for w in self.workloads]
        self.budget = _positive(config.get("max_evaluations", 100), "max_evaluations", integer=True)
        self.parameter_samples = _positive(config.get("parameter_samples", 4), "parameter_samples", integer=True)
        self.mapping_samples = _positive(config.get("mapping_samples", 4), "mapping_samples", integer=True)
        self.radius = _positive(config.get("parameter_radius", 0.25), "parameter_radius")
        self.growth_probability = config.get("growth_probability", 0.75)
        _positive(self.growth_probability, "growth_probability", zero=True)
        if self.growth_probability > 1:
            raise EnvError("invalid_config", "growth_probability must be <= 1")
        self.shrink_probability = config.get("shrink_probability", 0.75)
        _positive(self.shrink_probability, "shrink_probability", zero=True)
        if self.shrink_probability > 1:
            raise EnvError("invalid_config", "shrink_probability must be <= 1")
        self.resource_constraints = deepcopy(config.get("resource_constraints"))
        if self.resource_constraints is not None:
            try:
                validate_constraints(self.resource_constraints, components)
            except ValueError as error:
                raise EnvError("invalid_config", str(error)) from error
        self.cost_model = deepcopy(config.get("cost_model", {}))
        estimate_cost(self.adf, options=self.cost_model)  # Validate model options at startup.
        self.parameter_domains = {}
        for domain in config.get("parameter_domains", []):
            if set(domain) != {"path", "param", "values", "source"} or not isinstance(domain["source"], str) or not domain["source"].strip():
                raise EnvError("invalid_config", "parameter_domains entries require path, param, values and source")
            path, key, values = domain["path"], domain["param"], domain["values"]
            target = find(self.adf, path)
            if not isinstance(target, SpacePoint) or key not in target.params or (path, key) in self.parameter_domains:
                raise EnvError("invalid_config", "Parameter domains must name unique existing point parameters")
            if not isinstance(values, list) or not values or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                raise EnvError("invalid_config", "Explicit parameter domain values must be a nonempty list of finite positive numbers")
            # Existing component domains remain authoritative. Trial every choice
            # against them; an explicit search list can narrow but cannot expand one.
            for value in values:
                candidate = deepcopy(self.adf)
                find(candidate, path).params[key] = replace(target.params[key], value=value)
                check(candidate, self.components)
            if type(target.params[key].value) not in (int, float):
                raise EnvError("invalid_config", "Explicit domains currently support numeric parameters only")
            if target.params[key].value not in values:
                raise EnvError("invalid_config", "Explicit domain must include the seed parameter value")
            self.parameter_domains[(path, key)] = deepcopy(values)
        self.penalties = {"cost_usd": _positive(config.get("cost_penalty", 1.0), "cost_penalty", zero=True),
                          "area_mm2": _positive(config.get("area_penalty", 1.0), "area_penalty", zero=True)}
        self.limits = {"cost_usd": config.get("cost_budget_usd"), "area_mm2": config.get("area_budget_mm2")}
        for key, value in self.limits.items():
            if value is not None:
                _positive(value, key + " budget")
        if type(config.get("random_seed", 0)) is not int:
            raise EnvError("invalid_config", "random_seed must be an integer")
        self.rng = random.Random(config.get("random_seed", 0))
        self.backend = config.get("backend", "events")
        self.compilation = deepcopy(config.get("compilation"))
        self.used = 0
        self.mappings = None
        self.latest = None
        self.best_candidate = None
        self.calls, self.attempts, self.undo_stack, self.checkpoints = [], [], [], {}
        self.edit_records = []
        self.trace_path = Path(config["trace_path"]) if config.get("trace_path") else Path(tempfile.mkdtemp(prefix="archexplorer-")) / "tools.jsonl"
        self.trace = []
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            # A run owns its trace; fail instead of appending an unrelated run.
            with self.trace_path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps({"type": "config", "config": self.config}, allow_nan=False) + "\n")

    @property
    def remaining(self):
        return self.budget - self.used

    def call(self, tool, args=_NO_ARGS):
        """All transports use this method and receive the same JSON result/error envelope."""
        args = {} if args is _NO_ARGS else args
        before = self.used
        try:
            try:
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError):
                raise EnvError("invalid_arguments", "Tool arguments must be finite JSON values") from None
            schema = next((s for s in TOOL_SCHEMAS if s["name"] == tool), None)
            if schema is None:
                raise EnvError("unknown_tool", f"Unknown tool {tool!r}")
            validate_arguments(args, schema["inputSchema"])
            value = getattr(self, "_" + tool)(**deepcopy(args))
            result = {"result": value}
        except (EnvError, StructureError) as error:
            result = {"error": error.to_dict()}
        except (ValueError, TypeError, KeyError, NotImplementedError, OSError, RuntimeError) as error:
            result = {"error": {"code": "operation_failed", "message": str(error), "details": {}}}
        record = {"type": "tool", "index": len(self.calls), "tool": tool, "args": deepcopy(args),
                  **deepcopy(result), "evaluations_before": before, "evaluations_after": self.used}
        record = _trace_safe(record)
        self.calls.append({"index": record["index"], "tool": tool, "ok": "result" in result,
                           "evaluations_before": before, "evaluations_after": self.used})
        self.trace.append(record)
        if self.trace_path:
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return deepcopy(result)

    def _describe_environment(self):
        from arch_explorer.components.api import component_catalog
        description = component_catalog(registry=self.components)
        # Keep the existing Agent/search wire projection, with all identities and
        # implementation contracts supplied by the same public catalog as CLI.
        catalog = deepcopy(description["components"])
        for component in catalog:
            for spec in component["parameters"].values():
                domain = spec["domain"]
                if isinstance(domain, dict):
                    spec["domain"] = [domain["min"], domain["max"]]
                    spec["domain_kind"] = "range"
                else:
                    spec["domain_kind"] = ("range" if isinstance(domain, tuple) else
                                           "choices" if isinstance(domain, list) else "fixed")
        mapping_contracts = []
        selectors = self.config.get("mapping_selectors", {})
        for wi, workload in enumerate(self.workloads):
            contracts = {}
            if not supports_model_spec(self.backend, workload.kind):
                mapping_contracts.append({})
                continue
            draft = self.mappings[wi] if self.mappings else None
            names = list(self.config.get("mapping_templates") or self.templates.names())
            if draft and draft.template not in names:
                names.append(draft.template)
            for name in names:
                # Resolve once for both validation/discovery and sampling. A
                # concrete draft belongs to this workload and this template.
                has_draft = draft is not None and draft.template == name
                params = draft.params if has_draft else selectors.get(name)
                if name == MULTI_DEVICE and not has_draft and name not in selectors:
                    # Registration is not an executable empty topology. An
                    # explicitly supplied empty/invalid request still validates.
                    continue
                if name == MULTI_DEVICE and name not in self.templates.names():
                    contracts[name] = multi_device.description(self.adf, params, self.components)
                    continue
                template = self.templates.get(name)
                if template.applicable(self.adf, workload):
                    contracts[name] = mapping_contract(template, self.adf, workload, params)
            mapping_contracts.append(contracts)
        return {**self._model_spec_state(), "mapping_spaces": mapping_contracts, "objective": "sum(weight * latency_ns) + cost_penalty * max(0,cost-cost_budget) + area_penalty * max(0,area-area_budget)",
                "resource_constraints": self.resource_constraints, "limits": self.limits, "penalties": self.penalties, "max_evaluations": self.budget,
                "evaluations_used": self.used, "evaluations_remaining": self.remaining,
                "evaluation_unit": ("one mapping candidate attempt, including compiler/prepare failures; no evaluator cache; saved-result validation uses zero"
                                    if self.compilation is not None else "one attempted backend call for one workload and mapping; failures count; evaluate_request uses one unit for a prepared request and zero for saved-result validation"),
                "compilation": {"tools": ["task_adapters", "compile_workload", "prepare_compiled", "evaluate_request", "migrate_compilation_v1", "migrate_compilation_v2"],
                                "adapter_selection": "component-path bindings; automatic selection only for one declared compatible adapter",
                                "adapter_options": "bindings[path].options; execution-provider options are separate",
                                "compile_and_prepare_evaluations": 0,
                                "search_enabled": self.compilation is not None,
                                "search_settings": deepcopy(self.compilation),
                                "search_replay": "source-validated and read-only; explicit rerun creates new requests",
                                "prepared_results_replace_exploration_candidate": False},
                "task_adapters": self._task_adapters(),
                "operator_mappings": self._operator_mappings(),
                "workloads": [w.to_dict() for w in self.workloads], "backend": self.backend,
                "sampling": {"parameter_samples": self.parameter_samples, "mapping_samples": self.mapping_samples,
                             "parameter_radius": self.radius, "growth_probability": self.growth_probability,
                             "shrink_probability": self.shrink_probability, "cost_model": self.cost_model,
                             "parameter_domains": self.config.get("parameter_domains", []),
                             "random_seed": self.config.get("random_seed", 0)},
                "component_catalog": description, "components": catalog,
                "basic_edits": BASIC_SCHEMAS, "transforms": HIGH_SCHEMAS,
                "mapping_templates": self.templates.names(), "tools": TOOL_SCHEMAS,
                "search_space": {"kind": "component_graph", "candidate_source": "catalog_and_current_tree",
                                 "validity": "component domains, graph validation, then mapping/backend requirements"}}


    def _task_adapters(self):
        from arch_explorer.workload.api import task_adapters
        return task_adapters()

    def _operator_mappings(self, operator=None):
        from arch_explorer.workload.api import operator_mappings
        return operator_mappings(operator)

    def _compile_workload(self, mapping, workload_index=0, implementations=None, bindings=None):
        from arch_explorer.workload.api import compile_workload
        if workload_index >= len(self.workloads):
            raise EnvError("invalid_arguments", "workload_index is outside the current workload list")
        return compile_workload(self.adf, self.workloads[workload_index], MappingSpec(**mapping),
                                bindings=bindings, implementations=implementations, registry=self.components)

    def _migrate_compilation_v1(self, compilation):
        from arch_explorer.components.api import COMPONENTS
        if self.components is not COMPONENTS:
            raise EnvError("operation_failed", "Compilation migration requires the installed component registry; process-local custom Registry cannot be saved or silently replaced")
        from arch_explorer.workload.api import migrate_compilation_v1
        return migrate_compilation_v1(compilation)

    def _migrate_compilation_v2(self, compilation):
        if self.components is not COMPONENTS:
            raise EnvError("operation_failed", "Compilation migration does not support a process-local custom registry")
        from arch_explorer.workload.api import migrate_compilation_v2
        return migrate_compilation_v2(compilation)

    def _prepare_compiled(self, compilation, output_dir, options=None):
        from arch_explorer.simulator.api import prepare_compiled
        return prepare_compiled(compilation, output_dir=output_dir, options=options, registry=self.components)

    def _evaluate_request(self, payload):
        from arch_explorer.simulator.api import REQUEST_SCHEMA, RESULT_SCHEMA, evaluate_request
        schema = payload.get("schema")
        from .compiled import RESULT_SCHEMA as COMPILED_SEARCH_SCHEMA, replay as replay_compiled
        if schema == COMPILED_SEARCH_SCHEMA:
            return replay_compiled(payload)
        if schema in (RESULT_SCHEMA, "arch-explorer.compiled-result.v3", "arch-explorer.compiled-result.v2", "arch-explorer.compiled-result.v1"):
            return evaluate_request(payload)
        if schema not in (REQUEST_SCHEMA, "arch-explorer.compiled-request.v3", "arch-explorer.compiled-request.v2", "arch-explorer.compiled-request.v1"):
            raise EnvError("invalid_arguments", "Expected a public simulator/compiled request or result envelope")
        if self.remaining < 1:
            raise EnvError("evaluation_budget_exhausted", "No evaluations remain for a prepared request")
        attempt = {"index": self.used, "kind": "prepared_request", "request_schema": schema}
        self.used += 1
        try:
            result = evaluate_request(payload)
            attempt.update(valid=True, result_binding=deepcopy(result.get("binding")))
            return result
        except (EnvError, ValueError, TypeError, KeyError, NotImplementedError, RuntimeError, ImportError, OSError) as error:
            attempt.update(valid=False, error=str(error))
            raise
        finally:
            if self.compilation is not None:
                self.attempts.append(attempt)

    def _cost_breakdown(self):
        return estimate_cost(self.adf, options=self.cost_model).to_dict()

    def _get_state(self):
        lines = []
        for path, node in walk(self.adf):
            detail = node.component if isinstance(node, SpacePoint) else "container"
            lines.append("  " * path.count("/") + f"{node.name} x{node.repeat} [{detail}]")
        return {**self._model_spec_state(), "adf": self.adf.to_dict(), "tree": "\n".join(lines), "fingerprint": fingerprint(normalize(self.adf, registry=self.components)),
                "costs": self._cost_breakdown(), "latest_evaluation": self.latest,
                "resource_constraints": self.resource_constraints, "constraint_context": {"cost_model": self.cost_model, "limits": self.limits},
                "mappings": [m.to_dict() for m in self._entry_mappings()] if self._entry_mappings() else None,
                "multi_device": self._multi_device_state(),
                "evaluations_used": self.used, "evaluations_remaining": self.remaining}

    def _model_spec_state(self):
        rows = [{"workload_index": i, "spec": expand_model_spec(w.params),
                 "executor_registered": supports_model_spec(self.backend, w.kind)}
                for i, w in enumerate(self.workloads) if w.kind == "pi05"]
        return {"model_specs": rows} if rows else {}

    def _pending_draft(self, mapping):
        # Syntax only: physical mapping legality remains the future executor's job.
        if not isinstance(mapping.template, str) or mapping.template not in self.templates.names():
            raise EnvError("invalid_mapping", "Draft must name a registered mapping template")
        if not isinstance(mapping.params, dict):
            raise EnvError("invalid_mapping", "Draft params must be an object")
        def valid_json(v):
            if isinstance(v, dict):
                return all(isinstance(k, str) and valid_json(x) for k, x in v.items())
            if isinstance(v, list):
                return all(valid_json(x) for x in v)
            return v is None or type(v) in (str, bool, int) or type(v) is float and math.isfinite(v)
        if not valid_json(mapping.params):
            raise EnvError("invalid_mapping", "Draft params must be finite JSON")

    def _entry_mappings(self):
        return self.mappings or multi_device.configured(self.config, len(self.workloads))

    def _multi_device_state(self):
        mappings = self._entry_mappings() or []
        if not any(m.template == MULTI_DEVICE for m in mappings):
            return None
        from arch_explorer.devices.topology import discover
        result = {"requests": [], "executor_registered": MULTI_DEVICE in self.templates.names()}
        try:
            result["inventory"] = discover(self.adf, registry=self.components)
        except ValueError as error:
            result["inventory_error"] = str(error)
        for mapping in mappings:
            if mapping.template != MULTI_DEVICE:
                continue
            try:
                multi_device.validate(self.adf, mapping.params, self.components)
                status = {"valid_entry": True}
            except EnvError as error:
                status = {"valid_entry": False, "error": error.to_dict()}
            result["requests"].append(status)
        return result

    def _set_mappings(self, mappings):
        if len(mappings) != len(self.workloads):
            raise EnvError("invalid_mapping", "Provide one concrete mapping per workload")
        proposed = [MappingSpec(**m) for m in mappings]
        for workload, mapping in zip(self.workloads, proposed):
            if not supports_model_spec(self.backend, workload.kind):
                self._pending_draft(mapping)
                continue
            if mapping.template == MULTI_DEVICE:
                multi_device.validate(self.adf, mapping.params, self.components)
            else:
                if MULTI_DEVICE in self.config.get("mapping_templates", []):
                    raise EnvError("invalid_mapping", "Explicit multi-device configuration requires multi_device mappings")
                check_mapping(self.adf, workload, mapping, self.templates)
        self._adopt(self.adf, mappings=proposed)
        return self._get_state()

    def _snapshot(self):
        return deepcopy((self.adf, self.mappings, self.latest, self.edit_records))

    def _preflight(self, adf, cost=None):
        if self.resource_constraints is not None:
            cost = cost or estimate_cost(adf, options=self.cost_model)
            report = resource_report(adf, self.resource_constraints, self.components,
                                     cost=cost.to_dict(), limits=self.limits)
            if not report["valid"]:
                raise EnvError("resource_constraint", "Candidate violates resource/cost constraints",
                               report=report, backend_attempted=False)

    def _adopt(self, adf, mappings=None, latest=None, records=None):
        self._preflight(adf)
        self.undo_stack.append(self._snapshot())
        # A directed topology must never disappear after an edit and fall back
        # to an implicit complete graph. Keep the draft; stale paths are visible
        # in get_state and rejected before evaluation. Checkpoint/undo preserve it.
        if mappings is None and self.mappings and (any(m.template == MULTI_DEVICE for m in self.mappings) or
                any(not supports_model_spec(self.backend, w.kind) for w in self.workloads)):
            mappings = self.mappings
        self.adf, self.mappings, self.latest = deepcopy(adf), deepcopy(mappings), deepcopy(latest)
        if records is not None:
            self.edit_records.extend(deepcopy(records))

    def _apply_edits(self, edits):
        high = [e for e in edits if e["op"] in TRANSFORMS]
        if high and len(edits) != 1:
            raise EnvError("invalid_edits", "Use 1–3 basic edits or one high-level transform")
        candidate, records = deepcopy(self.adf), []
        for edit in edits:
            if high:
                result = apply_transform(candidate, edit, registry=self.components)
                records.extend(result.records)
            else:
                result = apply_edit(candidate, edit, registry=self.components)
                records.append(result.record)
            candidate = result.adf
        self._adopt(candidate, records=records)
        return {**self._get_state(), "edits": records}

    def _set_parameters(self, parameters):
        candidate = deepcopy(self.adf)
        seen = set()
        for item in parameters:
            path, key = item["path"], item["param"]
            if (path, key) in seen:
                raise EnvError("invalid_parameters", "Duplicate parameter assignment")
            seen.add((path, key))
            node = find(candidate, path)
            if not isinstance(node, SpacePoint) or key not in node.params:
                raise EnvError("invalid_parameters", f"{path}.{key} is not an existing point parameter")
            spec = self.components.get(node.component).params[key]
            explicit_domain = self.parameter_domains.get((path, key))
            if explicit_domain is not None and item["value"] not in explicit_domain:
                raise EnvError("invalid_parameters", f"{path}.{key} is outside the explicit search domain")
            if spec.domain is None and explicit_domain is None and item["value"] != node.params[key].value:
                raise EnvError("fixed_parameter", f"{path}.{key} has no tunable domain")
            node.params[key] = replace(node.params[key], value=item["value"], source=item.get("source", "explicit environment parameter"))
        check(candidate, self.components)
        self._adopt(candidate)
        return self._get_state()

    def _checkpoint(self, name=None):
        name = name or f"checkpoint-{len(self.checkpoints)}"
        self.checkpoints[name] = self._snapshot()
        return {"name": name, "fingerprint": fingerprint(self.adf), "evaluations_used": self.used}

    def _restore(self, name):
        if name not in self.checkpoints:
            raise EnvError("checkpoint_not_found", name)
        self.undo_stack.append(self._snapshot())
        self.adf, self.mappings, self.latest, self.edit_records = deepcopy(self.checkpoints[name])
        return self._get_state()

    def _undo(self):
        if not self.undo_stack:
            raise EnvError("nothing_to_undo", "No prior state")
        self.adf, self.mappings, self.latest, self.edit_records = self.undo_stack.pop()
        return self._get_state()

    def _history(self, offset=0, limit=100):
        return {"calls": self.calls[offset:offset + limit], "evaluations": self.attempts[offset:offset + limit],
                "call_count": len(self.calls), "evaluation_count": self.used}

    def _best(self):
        return {"candidate": self.best_candidate, "evaluations_used": self.used}

    def _score(self, evaluations, cost):
        from .objective import score
        return score(evaluations, cost.to_dict(), [w.weight for w in self.workloads], self.limits, self.penalties)

    @staticmethod
    def _rank(candidate):
        score = candidate["score"]
        return (bool(score["unavailable_constraints"]),
                score["objective"] if score["objective"] is not None else score["weighted_latency_ns"])

    @staticmethod
    def _diagnostics(evaluation):
        operators = {}
        for row in evaluation["breakdown"]:
            operator = row.get("operator", "unknown")
            duration = max(0.0, row.get("end_ns", 0.0) - row.get("start_ns", 0.0))
            operators[operator] = operators.get(operator, 0.0) + duration
        return {"operators_by_accumulated_task_ns": sorted(
                    ({"operator": key, "task_ns": value} for key, value in operators.items()),
                    key=lambda row: row["task_ns"], reverse=True)[:10],
                "resources_by_utilization": sorted(
                    ({"resource": key, "utilization": value} for key, value in evaluation["utilization"].items()),
                    key=lambda row: row["utilization"], reverse=True)[:10],
                "provenance": {"parameter_count": len(evaluation["provenance"]),
                    "sources": sorted({row.get("source", "") for row in evaluation["provenance"]}),
                    "missing_sources": sum(not row.get("source") for row in evaluation["provenance"])},
                "note": "Accumulated task durations may overlap; resource busy fractions are backend-defined."}

    def _evaluate(self, optimize="none", mappings=None):
        self._preflight(self.adf)
        requested = [MappingSpec(**m) for m in mappings] if mappings is not None else self._entry_mappings()
        if requested and len(requested) != len(self.workloads):
            raise EnvError("invalid_mapping", "Provide one concrete mapping per workload")
        for i, workload in enumerate(self.workloads):
            if workload.kind == "pi05":
                expand_model_spec(workload.params)
            if not supports_model_spec(self.backend, workload.kind):
                if requested:
                    self._pending_draft(requested[i])
                raise EnvError("unsupported_workload", "pi05 full-model executor pending SIM-17",
                               backend=self.backend, kind=workload.kind, backend_attempted=False)
        for mapping in requested or []:
            if mapping.template != MULTI_DEVICE:
                if MULTI_DEVICE in self.config.get("mapping_templates", []):
                    raise EnvError("invalid_mapping", "Explicit multi-device configuration requires multi_device mappings")
                continue
            multi_device.validate(self.adf, mapping.params, self.components)
            if MULTI_DEVICE not in self.templates.names():
                raise EnvError("unsupported_mapping", MULTI_UNSUPPORTED, backend=self.backend,
                               mapping=mapping.to_dict(), backend_attempted=False)
        if self.remaining < len(self.workloads):
            raise EnvError("evaluation_budget_exhausted", "Not enough evaluations for one complete workload suite")
        if mappings is not None and len(mappings) != len(self.workloads):
            raise EnvError("invalid_mapping", "Provide one concrete mapping per workload")
        provided = [MappingSpec(**m) for m in mappings] if mappings is not None else self.mappings
        if mappings is not None and optimize != "none":
            raise EnvError("invalid_mapping", "Explicit mappings require optimize='none'")
        base, winner, trials = deepcopy(self.adf), None, []
        base_cost = estimate_cost(base, options=self.cost_model)
        base_metrics = {"cost_usd": base_cost.cost_usd, "area_mm2": base_cost.area_mm2}
        pressure = any(limit is not None and base_metrics[k] is not None and base_metrics[k] > limit
                       for k, limit in self.limits.items())
        grow = not pressure and all(limit is None or base_metrics[k] is not None for k, limit in self.limits.items()) and any(
            limit is not None and base_metrics[k] < limit for k, limit in self.limits.items())
        count = self.parameter_samples if optimize == "parameters_and_mapping" else 1
        for variant in range(count):
            if self.remaining < len(self.workloads):
                break
            try:
                shrink = variant > 0 and pressure and self.rng.random() < self.shrink_probability
                adf = base if variant == 0 else parameter_variant(base, self.components, self.rng, self.radius,
                    grow=grow and self.rng.random() < self.growth_probability, shrink=shrink, domains=self.parameter_domains)
                adf = from_dict(adf.to_dict())
                cost = estimate_cost(adf, options=self.cost_model)
                self._preflight(adf, cost)
                selected, evaluations, compiled_results = [], [], []
                for wi, workload in enumerate(self.workloads):
                    attempt_start = len(self.attempts)
                    if optimize == "none" and provided:
                        options = [provided[wi]]
                    else:
                        names = self.config.get("mapping_templates")
                        selectors = self.config.get("mapping_selectors")
                        if requested and requested[wi].template == MULTI_DEVICE:
                            # Structured drafts are correlated constraints, not a
                            # hint that may be lost when re-sampling mappings.
                            names = [MULTI_DEVICE]
                            selectors = {MULTI_DEVICE: requested[wi].params}
                        options = mapping_candidates(adf, workload, self.templates,
                            self.mapping_samples if optimize != "none" else 1, self.rng,
                            names, selectors)
                    best_mapping, best_evaluation, best_compiled = None, None, None
                    for mapping in options:
                        # Reserve one backend call for each remaining workload.
                        if self.remaining <= len(self.workloads) - wi - 1:
                            break
                        attempt = {"index": self.used, "fingerprint": fingerprint(adf), "workload": wi,
                                   "mapping": mapping.to_dict()}
                        if self.compilation is not None:
                            attempt.update(phase="mapping_validation", execution_attempted=False)
                        self.used += 1
                        try:
                            check_mapping(adf, workload, mapping, self.templates)
                            compiled_result = None
                            candidate_mapping = mapping
                            if self.compilation is None:
                                evaluation = backend_evaluate(adf, workload, mapping, backend=self.backend,
                                                              cost_model=self.cost_model,
                                                              **self.config.get("backend_options", {})).to_dict()
                            else:
                                from .compiled import evaluate_mapping, effective_mapping
                                evaluation, compiled_result = evaluate_mapping(self, adf, workload, mapping, attempt)
                                candidate_mapping = effective_mapping(compiled_result)
                            json.dumps(evaluation, allow_nan=False)
                            for key in ("latency_ns", "area_mm2", "cost_usd"):
                                _metric(evaluation[key], key)
                            if evaluation["latency_ns"] is None:
                                raise EnvError("invalid_evaluation", "latency_ns is required")
                            attempt.update(valid=True, evaluation=evaluation)
                            if best_evaluation is None or evaluation["latency_ns"] < best_evaluation["latency_ns"]:
                                best_mapping, best_evaluation, best_compiled = candidate_mapping, evaluation, compiled_result
                        except (EnvError, ValueError, TypeError, KeyError, NotImplementedError, RuntimeError, ImportError, OSError) as error:
                            attempt.update(valid=False, error=str(error))
                            if isinstance(error, EnvError):
                                attempt["cause"] = error.to_dict()
                        self.attempts.append(attempt)
                        if optimize == "none":
                            break
                    if best_evaluation is None:
                        raise EnvError("evaluation_failed", f"No valid mapping for workload {wi}",
                            attempts=[{k: attempt[k] for k in ("index", "mapping", "error")}
                                      for attempt in self.attempts[attempt_start:] if not attempt["valid"]])
                    selected.append(best_mapping)
                    evaluations.append(best_evaluation)
                    compiled_results.append(best_compiled)
                score = self._score(evaluations, cost)
                summary = {"score": score, "workloads": [{"workload": w.to_dict(), "mapping": m.to_dict(),
                            "evaluation": e, "diagnostics": self._diagnostics(e)} for w, m, e in zip(self.workloads, selected, evaluations)]}
                if self.compilation is not None:
                    for row, compiled_result in zip(summary["workloads"], compiled_results):
                        row["compiled_result"] = compiled_result
                candidate = {"adf": adf.to_dict(), "fingerprint": fingerprint(normalize(adf, registry=self.components)),
                             "mappings": [m.to_dict() for m in selected], "cost_estimate": cost.to_dict(), **summary}
                trials.append({"fingerprint": candidate["fingerprint"], "score": score})
                if winner is None or self._rank(candidate) < self._rank(winner):
                    winner = candidate
                if self.best_candidate is None or self._rank(candidate) < self._rank(self.best_candidate):
                    self.best_candidate = deepcopy(candidate)
            except (EnvError, StructureError, ValueError, TypeError, KeyError, NotImplementedError) as error:
                trial = {"error": str(error)}
                if isinstance(error, (EnvError, StructureError)):
                    trial["cause"] = error.to_dict()
                trials.append(trial)
        if winner is None:
            raise EnvError("evaluation_failed", "No complete workload suite could be evaluated", trials=trials,
                recovery="Inspect get_state and these trial causes; correct the edit or restore a checkpoint. "
                         "See docs/components.md#adf-requirements-of-the-builder-and-the-templates")
        self._adopt(from_dict(winner["adf"]), [MappingSpec(**m) for m in winner["mappings"]], winner)
        return {**winner, "trials": trials, "evaluations_used": self.used, "evaluations_remaining": self.remaining}

    def _submit(self, path=None, selection="best"):
        candidate = self.best_candidate if selection == "best" else self.latest
        if candidate is None:
            raise EnvError("no_evaluation", "Evaluate a complete candidate before submit")
        result = {"schema": RESULT_SCHEMA, "backend": self.backend,
                  "resource_constraints": self.resource_constraints,
                  "cost_model": self.cost_model,
                  "parameter_domains": self.config.get("parameter_domains", []),
                  "backend_options": self.config.get("backend_options", {}),
                  "registration_modules": self.config.get("registration_modules", []),
                  "objective_config": {"cost_budget_usd": self.limits["cost_usd"], "area_budget_mm2": self.limits["area_mm2"],
                                       "cost_penalty": self.penalties["cost_usd"], "area_penalty": self.penalties["area_mm2"]},
                  "best": deepcopy(candidate), "evaluations_used": self.used}
        if self.compilation is not None:
            from .compiled import export
            result = export(self, result)
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result


def _public_tool(name):
    def invoke(self, **args):
        return self.call(name, args)
    invoke.__name__ = name
    return invoke


for _tool in TOOL_SCHEMAS:
    setattr(ExplorationEnv, _tool["name"], _public_tool(_tool["name"]))


def replay_result(payload):
    from .compiled import RESULT_SCHEMA as COMPILED_SEARCH_SCHEMA, replay as replay_compiled
    if isinstance(payload, dict) and payload.get("schema") == COMPILED_SEARCH_SCHEMA:
        return replay_compiled(payload)
    if not isinstance(payload, dict) or payload.get("schema") != RESULT_SCHEMA:
        raise EnvError("invalid_result", "Unsupported ADF result schema")
    best = payload.get("best")
    if not isinstance(best, dict) or not isinstance(best.get("workloads"), list) or not best["workloads"]:
        raise EnvError("invalid_result", "Result needs a complete best candidate and workload list")
    config = {"seed_adf": best["adf"], "workloads": [row["workload"] for row in best["workloads"]],
              "max_evaluations": len(best["workloads"]), "backend": payload["backend"],
              "backend_options": payload.get("backend_options", {}),
              "registration_modules": payload.get("registration_modules", []), "cost_model": payload.get("cost_model", {}),
              **payload["objective_config"]}
    if payload.get("resource_constraints") is not None:
        config["resource_constraints"] = deepcopy(payload["resource_constraints"])
    env = ExplorationEnv(config)
    # Replaying a fixed winner needs no search domains: a structure edit may
    # have deleted their seed paths. Preserve them as run metadata on export.
    env.config["parameter_domains"] = deepcopy(payload.get("parameter_domains", []))
    result = env.evaluate(optimize="none", mappings=best["mappings"])
    if "error" in result:
        error = result["error"]
        raise EnvError(error["code"], error["message"], **error.get("details", {}))
    return env.submit(selection="current")["result"]


def replay_trace(records, *, config=None, components=COMPONENTS, templates=MAPPING_TEMPLATES):
    """Reexecute recorded calls, verifying results (submit file writes are suppressed)."""
    rows = list(records)
    if config is None:
        config = rows.pop(0)["config"]
    config = deepcopy(config)
    config.pop("trace_path", None)
    env = ExplorationEnv(config, components=components, templates=templates)
    for row in rows:
        if row.get("type") != "tool":
            continue
        args = _trace_arguments(deepcopy(row["args"]))
        if row["tool"] == "submit":
            args.pop("path", None)
        actual = env.call(row["tool"], args)
        expected = {k: row[k] for k in ("result", "error") if k in row}
        # JSON normalization makes tuples in registry domains equivalent to wire arrays.
        if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True) or env.used != row["evaluations_after"]:
            raise EnvError("trace_mismatch", f"Trace diverged at call {row['index']}")
    return env
