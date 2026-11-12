"""Organize two public scenes; native execution and comparison stay provider-owned.

The installed component package owns topology and comparison. This module
handles public scene construction, fresh outputs, preparation and presentation.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

from arch_explorer.archs.gpu_instantiation import CASE_SCHEMA

RUN_SCHEMA = "arch-explorer.gpu-instantiation-run.v1"
COMPARISON_SCHEMA = "arch-explorer.accelsim-component-comparison.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_case(path):
    case = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(case, dict) or set(case) != {"schema", "scope", "candidate", "oracle"}
            or case["schema"] != CASE_SCHEMA or not isinstance(case["scope"], str) or not case["scope"].strip()):
        raise ValueError("Expected a GPU instantiation case with scope, candidate and oracle scenes")
    canonical(case)  # Reject non-finite input before creating any output.
    for side, provider in (("candidate", "accelsim_components"), ("oracle", "accelsim")):
        scene = case[side]
        if (not isinstance(scene, dict) or scene.get("provider") != provider
                or not {"adf", "workload", "mappings", "scope"} <= scene.keys()
                or not isinstance(scene.get("scope"), str) or not scene["scope"].strip()
                or not isinstance(scene.get("options", {}), dict)):
            raise ValueError(f"{side} requires an explicit {provider} scene and object options")
        mappings = scene["mappings"]
        if (not isinstance(mappings, list) or len(mappings) < 2
                or len({canonical(value) for value in mappings}) != len(mappings)):
            raise ValueError(f"{side} requires at least two distinct complete mappings")
        from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
        scene["adf"] = from_dict(scene["adf"]).to_dict()
        scene["workload"] = WorkloadSpec(**scene["workload"]).to_dict()
        scene["mappings"] = [MappingSpec(**value).to_dict() for value in mappings]
        if len({canonical(value) for value in scene["mappings"]}) != len(mappings):
            raise ValueError(f"{side} mappings normalize to duplicate candidates")
        scene.setdefault("options", {})
    for key in ("adf", "workload", "mappings", "options"):
        if canonical(case["candidate"][key]) != canonical(case["oracle"][key]):
            raise ValueError(f"Both paths require the same complete {key}")
    # Source/file/trace equivalence also belongs to the public comparator.
    # Matching frontend objects alone cannot establish native identity.
    return case


def local_file(directory, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Result references must be relative")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Result reference leaves its run directory")
    return path


def validate_result(result, scene, index):
    if not isinstance(result, dict) or result.get("schema") != "arch-explorer.simulator-result.v1":
        raise ValueError("Expected a complete public simulator result")
    identity = {"id": scene["provider"], "version": "1"}
    request = result.get("request")
    if (not isinstance(request, dict) or result.get("provider") != identity
            or request.get("provider") != identity):
        raise ValueError("Result provider identity differs from its case")
    for key, expected in (("adf", scene["adf"]), ("workload", scene["workload"]),
                          ("mapping", scene["mappings"][index]), ("options", scene["options"])):
        if canonical(request.get(key)) != canonical(expected):
            raise ValueError(f"Result {key} differs from its prepared case")


def compare_pair(component, oracle):
    from arch_explorer.components.assemblies.gpu.componentization import compare_results
    report = compare_results(component, oracle)
    if not isinstance(report, dict) or report.get("schema") != COMPARISON_SCHEMA:
        raise ValueError("Expected the complete public component comparison report")
    return report


def timing_summary(result):
    """Present verified public timings without deriving native cycles or ns."""
    evaluation = result["evaluation"]
    native = evaluation["details"]["native"]
    return {"latency_semantics": "physical_system_drain", "latency_ns": evaluation["latency_ns"],
            "foreground": {"cycles": native["gpu_tot_sim_cycle"], "latency_ns": native["latency_ns"],
                           "latency_ns_exact": deepcopy(native["latency_ns_exact"])},
            "system_drained": {"cycles": native["system_drained_cycles"],
                               "latency_ns": native["system_drained_latency_ns"],
                               "latency_ns_exact": deepcopy(native["system_drained_latency_ns_exact"])},
            "drain_ticks": native["drain_ticks"], "drain_core_cycles": native["drain_core_cycles"],
            "physical_system_drain": [deepcopy(row) for row in evaluation["breakdown"]
                                      if row.get("phase") == "physical_system_drain"]}


def execute_case(case_path, output_dir):
    """Execute components before the oracle, then call the public authority.

    The public component comparator validates exact identity, scope, metrics
    and drain conditions. Its complete report is preserved even on mismatch.
    No counters, clocks or success tolerances are implemented here.
    """
    case = load_case(case_path)
    from arch_explorer.demo import prepare_demo, run_demo
    from arch_explorer.simulator.api import assess_request
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "case.json", case)
    record = {"schema": RUN_SCHEMA, "scope": case["scope"], "status": "preparing",
              "paths": {"candidate": "candidate", "oracle": "oracle"}, "comparisons": []}
    try:
        for side in ("candidate", "oracle"):
            result = prepare_demo(directory / side, specs=deepcopy(case[side]))
            if result["status"] != "prepared":
                raise ValueError(f"{side} preparation is unsupported; inspect its capability reports")
        # Check both complete sets before allowing either side to execute.
        for side in ("candidate", "oracle"):
            path = directory / side
            manifest = json.loads((path / "demo.json").read_text(encoding="utf-8"))
            for row in manifest["candidates"]:
                request = json.loads(local_file(path, row["request"]).read_text(encoding="utf-8"))
                assessment = assess_request(request)
                if (type(assessment.get("supported")) is not bool or not isinstance(assessment.get("gaps"), list)
                        or not assessment["supported"]):
                    raise ValueError(f"{side} is not supported: {assessment}")
        runs = {}
        for side in ("candidate", "oracle"):
            record["status"] = "running_" + side
            write(directory / "run.json", record)
            runs[side] = run_demo(directory / side)
        record["status"] = "comparing"
        write(directory / "run.json", record)
        candidate_rows = runs["candidate"]["candidates"]
        oracle_rows = runs["oracle"]["candidates"]
        expected = len(case["candidate"]["mappings"])
        if len(candidate_rows) != expected or len(oracle_rows) != expected:
            raise ValueError("Both paths must preserve every prepared mapping candidate")
        for index, (candidate, oracle) in enumerate(zip(candidate_rows, oracle_rows)):
            left = local_file(directory / "candidate", candidate["result"])
            right = local_file(directory / "oracle", oracle["result"])
            component = json.loads(left.read_text(encoding="utf-8"))
            oracle_result = json.loads(right.read_text(encoding="utf-8"))
            validate_result(component, case["candidate"], index)
            validate_result(oracle_result, case["oracle"], index)
            report = compare_pair(component, oracle_result)
            name = f"comparison-{index:03d}.json"
            write(directory / name, report)
            record["comparisons"].append({"candidate": left.relative_to(directory).as_posix(),
                                           "oracle": right.relative_to(directory).as_posix(), "report": name})
            if report.get("status") != "passed":
                raise ValueError(f"Component/oracle mismatch; inspect {name}")
            record["comparisons"][-1]["timing"] = {"candidate": timing_summary(component),
                                                   "oracle": timing_summary(oracle_result)}
        # This status means the authority accepted every comparison. It is
        # never reached by merely running both engines or checking total time.
        record["status"] = "compared"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(directory / "run.json", record)
    return record


def replay_case(directory):
    """Use the public reader again without changing the original run tree."""
    directory = Path(directory).resolve()
    case = load_case(directory / "case.json")
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if record.get("schema") != RUN_SCHEMA or record.get("status") != "compared":
        raise ValueError("Only complete compared cases can be replayed as a comparison")
    rows = record.get("comparisons")
    if not isinstance(rows, list) or len(rows) != len(case["candidate"]["mappings"]):
        raise ValueError("Comparison set differs from its case")
    reports, timings, paths = [], [], set()
    for index, row in enumerate(rows):
        component_path = local_file(directory, row["candidate"])
        oracle_path = local_file(directory, row["oracle"])
        if component_path in paths or oracle_path in paths:
            raise ValueError("Duplicate comparison result reference")
        paths.update((component_path, oracle_path))
        component = json.loads(component_path.read_text(encoding="utf-8"))
        oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        validate_result(component, case["candidate"], index)
        validate_result(oracle, case["oracle"], index)
        report = compare_pair(component, oracle)
        if report.get("status") != "passed":
            raise ValueError("Saved component/oracle comparison no longer passes")
        reports.append(report)
        timings.append({"candidate": timing_summary(component), "oracle": timing_summary(oracle)})
    return {"schema": RUN_SCHEMA, "status": "compared", "scope": case["scope"],
            "comparisons": reports, "timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Generate paired scenes without simulating")
    create.add_argument("--model", choices=("vit", "clip"), required=True)
    create.add_argument("--engine", type=Path, required=True)
    create.add_argument("--hardware", type=Path, help="gpu_adf keyword JSON")
    create.add_argument("--options", type=Path, help="Bounded native options JSON")
    create.add_argument("--output", type=Path, required=True)
    topology = commands.add_parser("topology", help="Validate and render all physical connections")
    topology.add_argument("case", type=Path)
    topology.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Execute components then the independent oracle and compare")
    run.add_argument("case", type=Path)
    run.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="Read-only comparison of saved original results")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            from arch_explorer.archs.gpu_instantiation import case_specs
            value = case_specs(args.model, engine=args.engine,
                               hardware=json.loads(args.hardware.read_text()) if args.hardware else None,
                               options=json.loads(args.options.read_text()) if args.options else None)
        elif args.command == "topology":
            from arch_explorer.adf import from_dict
            from arch_explorer.components.assemblies.gpu import compile_topology
            case = load_case(args.case)
            value = compile_topology(from_dict(case["candidate"]["adf"]))
        elif args.command == "run":
            execute_case(args.case, args.output)
            print(f"Compared: {args.output / 'run.json'}; latency_semantics=physical_system_drain")
            return 0
        else:
            if args.output.resolve().is_relative_to(args.directory.resolve()):
                raise ValueError("Replay output must be outside the original run tree")
            value = replay_case(args.directory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"Wrote: {args.output}")
        return 0
    except (ImportError, KeyError, NotImplementedError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(2, f"gpu-instantiation: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
