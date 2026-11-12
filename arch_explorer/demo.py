"""Organize standard workload, ADF and mapping objects for registered providers.

The selected provider owns lowering, capability checks and execution.
This module only persists its public requests, assessments and evaluations.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path

from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict

DEMO_SCHEMA = "arch-explorer.demo.v1"
COMPARISON_SCHEMA = "arch-explorer.demo-comparison.v1"
PREPARATION_SCHEMA = "arch-explorer.demo-preparation.v1"
REQUEST_SCHEMA = "arch-explorer.simulator-request.v1"
RESULT_SCHEMA = "arch-explorer.simulator-result.v1"


def program_api():
    """Load the public generation API without importing a concrete provider."""
    name = "arch_explorer.simulator.api"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise ImportError("The public simulator.api generation interface is unavailable; "
                          "install a build containing that interface and the selected provider. "
                          "No simulation was performed.") from error


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _canonical(value):
    """JSON identity independent of formatting, preserving exact JSON types."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _request_identity(request):
    return hashlib.sha256(_canonical(request).encode("utf-8")).hexdigest()


def _envelope(value, name):
    value = _object(value, name)
    if not isinstance(value.get("schema"), str) or not value["schema"].strip():
        raise ValueError(f"{name} requires a nonempty public schema")
    return value


def _assessment(value):
    value = _object(value, "assessment")
    if type(value.get("supported")) is not bool or not isinstance(value.get("gaps"), list):
        raise ValueError("Assessment requires supported: bool and gaps: list")
    return value


def _scene_request(value, specs, mapping):
    """Check only standard public fields; provider payload stays opaque."""
    request = _envelope(value, "provider request")
    provider = _object(request.get("provider"), "request provider")
    if (request["schema"] != REQUEST_SCHEMA or provider.get("id") != specs["provider"]
            or not isinstance(provider.get("version"), str) or not provider["version"].strip()):
        raise ValueError("Expected a generic request for the selected provider with an exact version")
    for key, expected in (("adf", specs["adf"]), ("workload", specs["workload"]),
                          ("mapping", mapping), ("options", specs["options"])):
        if _canonical(request.get(key)) != _canonical(expected):
            raise ValueError(f"Prepared request {key} differs from its scene candidate")
    return request


def prepare_demo(output_dir, *, model=None, specs=None, provider=None, engine=None, options=None):
    """Prepare every candidate without evaluating it. Use a new output directory.

    ``specs`` has the same public shape as ``archs.accelsim.program_demo_specs(model)``. Editing
    workload shapes, mappings or ADF parameters needs no runner/model-name branch.
    Provider selection belongs to the scene, with an optional explicit override.
    ``engine`` is a convenience for options['engine']; options carry execution
    limits and dependencies, never workload shapes or hardware parameters.
    """
    if (model is None) == (specs is None):
        raise ValueError("Supply exactly one of model or specs")
    api = program_api()
    if specs is None:
        from arch_explorer.archs.accelsim import program_demo_specs
        specs = program_demo_specs(model)
    specs = deepcopy(_object(specs, "specs"))
    if not {"adf", "workload", "mappings", "scope"} <= specs.keys():
        raise ValueError("specs requires adf, workload, mappings and scope")
    adf = from_dict(_object(specs["adf"], "adf"))
    workload = WorkloadSpec(**_object(specs["workload"], "workload"))
    candidates = specs["mappings"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("Provide at least two distinct mapping candidates")
    mappings = [MappingSpec(**_object(value, "mapping")) for value in candidates]
    if len({_canonical(value.to_dict()) for value in mappings}) != len(mappings):
        raise ValueError("Mapping candidates must be distinct")
    provider = specs.get("provider") if provider is None else provider
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Select a nonempty provider id in specs or with --provider")
    execution_options = deepcopy(_object(specs.get("options", {}), "specs.options"))
    if options is not None:
        execution_options.update(deepcopy(_object(options, "options")))
    if engine is not None:
        engine = str(Path(engine).resolve())
        if "engine" in execution_options and execution_options["engine"] != engine:
            raise ValueError("Conflicting engine paths in --engine and options")
        execution_options["engine"] = engine
    _canonical(execution_options)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    specs.update(adf=adf.to_dict(), workload=workload.to_dict(), mappings=[m.to_dict() for m in mappings],
                 provider=provider, options=execution_options)
    _write(directory / "specs.json", specs)
    _write(directory / "adf.json", specs["adf"])
    _write(directory / "workload.json", specs["workload"])
    _write(directory / "mappings.json", specs["mappings"])
    rows = []
    manifest = {"schema": DEMO_SCHEMA, "status": "preparing", "scope": specs["scope"],
                "provider": provider, "specs": "specs.json", "candidates": rows}
    preparation = {"schema": PREPARATION_SCHEMA, "specs": deepcopy(specs), "bindings": []}
    request_ids = set()
    try:
        for index, mapping in enumerate(mappings):
            prefix = f"mapping-{index:03d}"
            (directory / prefix).mkdir()
            request = api.prepare_request(deepcopy(adf), deepcopy(workload), deepcopy(mapping),
                                          output_dir=str(directory / prefix / "program"),
                                          provider=provider, options=deepcopy(execution_options))
            request = deepcopy(_scene_request(request, specs, mapping.to_dict()))
            identity = _request_identity(request)
            if identity in request_ids:
                raise ValueError("Provider returned a duplicate prepared request for distinct mappings")
            request_ids.add(identity)
            _write(directory / prefix / "request.json", request)
            assessment = _assessment(api.assess_request(deepcopy(request)))
            _write(directory / prefix / "assessment.json", assessment)
            rows.append({"mapping": mapping.to_dict(), "request": f"{prefix}/request.json",
                         "assessment": f"{prefix}/assessment.json",
                         "status": "supported" if assessment["supported"] else "unsupported"})
            preparation["bindings"].append({"candidate": deepcopy(rows[-1]), "request_identity": identity})
        manifest["status"] = "prepared" if all(row["status"] == "supported" for row in rows) else "unsupported"
        _write(directory / "preparation.json", preparation)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        _write(directory / "demo.json", manifest)
    return {"manifest": str(directory / "demo.json"), **manifest}


def _local_file(directory, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Demo file references must be relative paths")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Demo file reference leaves its output directory")
    return path


def _prepared_requests(directory, manifest):
    """Validate the complete local preparation before any provider assessment.

    Requests are opaque public JSON. This record checks association with the
    original prepare calls, not the provider's lowering or arbitrary rewrites
    of the entire bundle. Program semantics remain the provider's authority.
    """
    try:
        preparation = _object(json.loads(_local_file(directory, "preparation.json").read_text()),
                              "preparation record")
    except FileNotFoundError as error:
        raise ValueError("Missing preparation record; prepare a new output directory") from error
    if preparation.get("schema") != PREPARATION_SCHEMA:
        raise ValueError("Expected a completed demo preparation record")
    specs = _object(preparation.get("specs"), "prepared specs")
    mappings = specs.get("mappings")
    bindings = preparation.get("bindings")
    candidates = manifest.get("candidates")
    if (not isinstance(mappings, list) or len(mappings) < 2
            or not isinstance(bindings, list) or len(bindings) != len(mappings)
            or not isinstance(candidates, list) or len(candidates) != len(bindings)):
        raise ValueError("Demo candidate set differs from preparation; prepare a new output directory")
    if len({_canonical(mapping) for mapping in mappings}) != len(mappings):
        raise ValueError("Duplicate prepared mapping candidates")
    if (manifest.get("specs") != "specs.json"
            or _canonical(manifest.get("provider")) != _canonical(specs.get("provider"))
            or _canonical(manifest.get("scope")) != _canonical(specs["scope"])):
        raise ValueError("Demo specs reference or scope differs from preparation")
    for name, expected in (("specs.json", specs), ("adf.json", specs["adf"]),
                           ("workload.json", specs["workload"]), ("mappings.json", mappings)):
        current = json.loads(_local_file(directory, name).read_text())
        if _canonical(current) != _canonical(expected):
            raise ValueError(f"{name} differs from preparation; prepare a new output directory")
    requests, paths, identities = [], set(), set()
    for index, (candidate, binding) in enumerate(zip(candidates, bindings)):
        candidate = _object(candidate, "demo candidate")
        binding = _object(binding, "prepared binding")
        prepared = _object(binding.get("candidate"), "prepared candidate")
        # Resolve before comparison so aliases cannot conceal duplicate or
        # out-of-directory references. Candidate order and declarations freeze.
        path = _local_file(directory, candidate["request"])
        assessment_path = _local_file(directory, candidate["assessment"])
        if path in paths:
            raise ValueError("Duplicate candidate request reference")
        paths.add(path)
        if (_canonical(candidate) != _canonical(prepared)
                or _canonical(prepared.get("mapping")) != _canonical(mappings[index])
                or path != _local_file(directory, prepared["request"])
                or assessment_path != _local_file(directory, prepared["assessment"])):
            raise ValueError("Candidate mapping or reference differs from preparation; prepare a new output directory")
        request = _scene_request(json.loads(path.read_text()), specs, mappings[index])
        identity = _request_identity(request)
        if identity in identities:
            raise ValueError("Duplicate prepared request content")
        identities.add(identity)
        if identity != binding.get("request_identity"):
            raise ValueError("Prepared request changed; prepare a new output directory")
        requests.append((deepcopy(prepared), request))
    return requests


def run_demo(directory):
    """Reassess and execute prepared requests; preserve the full Evaluation.

    A failed or partial run never produces a successful comparison. Provider
    counters, output/reference data and provenance remain in backend details;
    no host-side timing, speedup or cost is substituted for unknown metrics.
    """
    api = program_api()
    directory = Path(directory).resolve()
    manifest = _object(json.loads((directory / "demo.json").read_text()), "demo manifest")
    if manifest.get("schema") != DEMO_SCHEMA or manifest.get("status") not in {"prepared", "unsupported"}:
        raise ValueError("Expected a prepared demo manifest")
    if (directory / "comparison.json").exists() or (directory / "run.json").exists():
        raise ValueError("This demo already has a run; prepare a new output directory")
    comparison = {"schema": COMPARISON_SCHEMA, "status": "running", "scope": manifest["scope"],
                  "provider": manifest.get("provider"), "candidates": []}
    try:
        candidates = manifest.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError("Demo requires at least two candidates")
        requests = _prepared_requests(directory, manifest)
        for row, request in requests:
            assessment = _assessment(api.assess_request(deepcopy(request)))
            _write(_local_file(directory, row["assessment"]), assessment)
            if not assessment["supported"]:
                raise ValueError(f"Unsupported provider request: {json.dumps(assessment, sort_keys=True)}")
        for index, (row, request) in enumerate(requests):
            result = _envelope(api.evaluate_request(deepcopy(request)), "provider result")
            if result["schema"] != RESULT_SCHEMA or _canonical(result.get("provider")) != _canonical(request["provider"]):
                raise ValueError("Expected a generic result from the prepared provider version")
            if _canonical(result.get("request")) != _canonical(request):
                raise ValueError("Provider result does not preserve its prepared request")
            evaluation = _object(result.get("evaluation"), "evaluation")
            latency = evaluation.get("latency_ns")
            if (not isinstance(evaluation.get("backend"), str) or not evaluation["backend"].strip()
                    or type(latency) not in (int, float)
                    or not math.isfinite(latency) or latency <= 0):
                raise ValueError("Provider did not return a positive finite Evaluation with a backend id")
            result_path = f"mapping-{index:03d}/result.json"
            _write(_local_file(directory, result_path), result)
            comparison["candidates"].append({"mapping": row["mapping"], "request": row["request"],
                                              "result": result_path, "evaluation": evaluation})
        comparison["status"] = "completed"
        _write(directory / "comparison.json", comparison)
    except Exception as error:
        comparison.update(status="failed", error=str(error))
        raise
    finally:
        _write(directory / "run.json", comparison)
    return comparison


if __name__ == "__main__":
    import sys
    from arch_explorer.cli import main
    raise SystemExit(main(["demo", *sys.argv[1:]]))
