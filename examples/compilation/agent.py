"""Use the public Agent tools for compile/prepare and optional execution."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", type=Path, help="Same source specs used by archexplorer compile")
    parser.add_argument("--bindings", type=Path, help="Component-path adapter bindings JSON; overrides specs.bindings")
    parser.add_argument("--output", type=Path, required=True, help="New preparation directory")
    parser.add_argument("--options", type=Path)
    parser.add_argument("--execute", action="store_true", help="Explicitly execute and read-only replay the prepared request")
    args = parser.parse_args(argv)
    from arch_explorer.agent_env import ExplorationEnv
    source = json.loads(args.specs.read_text())
    if "target" in source:
        raise ValueError("Saved whole-graph target is retired; provide explicit component-path adapter bindings")
    config = {"seed_adf": source["adf"], "workloads": [source["workload"]],
              "mapping_templates": ["execution_graph"], "max_evaluations": 1}
    env = ExplorationEnv(config)

    def call(tool, **parameters):
        response = env.call(tool, parameters)
        if "error" in response:
            raise ValueError(json.dumps(response["error"], sort_keys=True))
        return response["result"]

    compile_args = {"mapping": source["mapping"]}
    if args.bindings or "bindings" in source:
        compile_args["bindings"] = (json.loads(args.bindings.read_text()) if args.bindings
                                     else source["bindings"])
    if "implementations" in source:
        compile_args["implementations"] = source["implementations"]
    compiled = call("compile_workload", **compile_args)
    options = json.loads(args.options.read_text()) if args.options else {}
    request = call("prepare_compiled", compilation=compiled,
                   output_dir=str(args.output), options=options)
    assert env.used == 0
    with (args.output / "EXAMPLE-AGENT-REQUEST.json").open("x") as stream:
        json.dump(request, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if args.execute:
        result = call("evaluate_request", payload=request)
        assert env.used == 1
        replayed = call("evaluate_request", payload=result)
        assert replayed == result and env.used == 1
        with (args.output / "EXAMPLE-AGENT-RESULT.json").open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"request": request, "evaluations_used": env.used,
                      "trace": str(env.trace_path)}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"compilation Agent example: {error}") from None
