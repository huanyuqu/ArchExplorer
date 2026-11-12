"""Command-line entry points for exploration and candidate replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(prog="archexplorer")
    parser.add_argument("--version", action="version", version="ArchExplorer 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)
    components = commands.add_parser("components", help="Describe hardware components and their execution implementations")
    components.add_argument("--output", type=Path, help="Write the public component catalog as JSON")
    adapters = commands.add_parser("task-adapters", help="Describe registered task adapters and their execution boundaries")
    adapters.add_argument("--output", type=Path)
    rules = commands.add_parser("operator-mappings", help="Describe registered semantic mapping rules and their callback provenance")
    rules.add_argument("--operator", help="Optional semantic kind filter")
    rules.add_argument("--output", type=Path)
    old_targets = commands.add_parser("compiler-targets", help="Retired: use task-adapters")
    old_targets.add_argument("--output", type=Path)
    import_onnx = commands.add_parser("import-onnx", help="Import a standard ONNX file into a portable public WorkloadSpec")
    import_onnx.add_argument("model", type=Path)
    import_onnx.add_argument("--inputs", type=Path, required=True, help="Complete runtime root values as a JSON object")
    import_onnx.add_argument("--input-shapes", type=Path, help="Original root names to concrete positive logical shapes as JSON")
    import_onnx.add_argument("--external-root", type=Path, help="Explicit allowed root for external tensor files")
    import_onnx.add_argument("--resource-budget", type=Path, help="Explicit authoritative source/graph/reference resource budget JSON")
    import_onnx.add_argument("--name", default="onnx", help="Public imported workload display name")
    import_onnx.add_argument("--output", type=Path, required=True, help="Write the self-contained workload spec as JSON")
    export_torch = commands.add_parser("export-torch", help="Export a caller-supplied CPU TorchScript file using explicit legacy opset17")
    export_torch.add_argument("module", type=Path)
    export_torch.add_argument("--inputs", type=Path, required=True, help="Name -> dtype/shape/values JSON")
    export_torch.add_argument("--input-names", nargs='+', required=True, help="Every root in module argument order, independent of JSON key order")
    export_torch.add_argument("--output-names", nargs='+', required=True)
    export_torch.add_argument("--dynamic-axes", type=Path, help="Declared input/output names -> axis numbers -> symbols")
    export_torch.add_argument("--external-data", action="store_true")
    export_torch.add_argument("--output-dir", type=Path, required=True, help="New export directory")
    compare = commands.add_parser("compare-hardware", help="Compare complete provenance-bound measured/predicted timing pairs")
    compare.add_argument("samples", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    scale = commands.add_parser("scale-run", help="Run one explicit, preflighted stage with a host lock and process-group supervision")
    scale.add_argument("plan", type=Path, help="One stage: argv/cwd/wall_seconds/memory_bytes/stage/dimensions/estimate")
    scale.add_argument("--output-dir", type=Path, required=True)
    compile_command = commands.add_parser("compile", help="Compile explicit ADF/workload/mapping specs without execution")
    compile_command.add_argument("specs", type=Path, help="JSON with adf/workload/mapping and optional bindings/implementations")
    compile_command.add_argument("--bindings", type=Path, help="JSON object of compute paths to adapter id/version/options; overrides specs.bindings")
    compile_command.add_argument("--target", help=argparse.SUPPRESS)
    compile_command.add_argument("--output", type=Path, help="Write the public compilation artifact as JSON")
    prepare_compilation = commands.add_parser("prepare", help="Prepare a public compilation artifact without execution")
    prepare_compilation.add_argument("compilation", type=Path)
    prepare_compilation.add_argument("--options", type=Path, help="Execution-provider options JSON; adapter options belong in compilation bindings")
    prepare_compilation.add_argument("--output", type=Path, required=True, help="New request directory")
    migrate = commands.add_parser("migrate-compilation-v1", help="Explicitly recompile a v1 input; never migrates execution results")
    migrate.add_argument("compilation", type=Path)
    migrate.add_argument("--output", type=Path)
    migrate_v2 = commands.add_parser("migrate-compilation-v2", help="Explicitly recompile a v2 input using registered mapping rules")
    migrate_v2.add_argument("compilation", type=Path)
    migrate_v2.add_argument("--output", type=Path)
    demo = commands.add_parser("demo", help="Prepare and compare mappings using a registered provider")
    demo_commands = demo.add_subparsers(dest="demo_command", required=True)
    prepare = demo_commands.add_parser("prepare", help="Generate requests and capability reports; does not simulate")
    scene = prepare.add_mutually_exclusive_group(required=True)
    scene.add_argument("--model", help="Built-in workload scene, e.g. vit or clip")
    scene.add_argument("--specs", type=Path, help="ADF/workload/mappings/provider/scope JSON")
    prepare.add_argument("--provider", help="Registered provider id; overrides the scene selection")
    prepare.add_argument("--engine", type=Path, help="Convenience for options.engine (external engine manifest)")
    prepare.add_argument("--options", type=Path, help="Provider options as a JSON object")
    prepare.add_argument("--output", type=Path, required=True, help="New output directory")
    demo_run = demo_commands.add_parser("run", help="Execute all prepared mappings with their selected provider")
    demo_run.add_argument("directory", type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("--output", type=Path)
    rerun = commands.add_parser("rerun", help="Explicitly execute a fixed compiled exploration winner in a new directory")
    rerun.add_argument("config", type=Path)
    rerun.add_argument("--work-dir", type=Path, required=True)
    rerun.add_argument("--output", type=Path)
    search = commands.add_parser("search")
    search.add_argument("--config", type=Path, required=True)
    search.add_argument("--policy", choices=("sa", "greedy", "random"), default="sa")
    search.add_argument("--basic-only", action="store_true")
    search.add_argument("--output", type=Path)
    environment = commands.add_parser("env").add_subparsers(dest="env_command", required=True)
    serve = environment.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    trace_replay = environment.add_parser("replay", help="Reexecute a tool trace, including rejected requests")
    trace_replay.add_argument("trace", type=Path)
    trace_replay.add_argument("--output", type=Path)
    mcp = commands.add_parser("mcp")
    mcp.add_argument("--config", type=Path, required=True)
    agent = commands.add_parser("agent").add_subparsers(dest="agent_command", required=True)
    run = agent.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--policy", choices=("scripted",), default="scripted")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "compare-hardware":
            from arch_explorer.archs.hardware_comparison import compare_hardware
            result = compare_hardware(json.loads(args.samples.read_text()))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
            return 0 if result['all_samples_accepted'] else 2
        if args.command == "scale-run":
            from arch_explorer.archs.scale_runner import run_scale_stage
            plan = json.loads(args.plan.read_text())
            required = {'argv', 'cwd', 'wall_seconds', 'memory_bytes', 'stage', 'dimensions', 'estimate'}
            if type(plan) is not dict or set(plan) != required:
                raise ValueError('scale-run requires exact stage argv/cwd/budgets/dimensions/estimate fields')
            result = run_scale_stage(**plan, output_dir=args.output_dir)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0 if result['status'] == 'success' else 2
        if args.command == "export-torch":
            from arch_explorer.archs.framework_export import export_torch
            axes = json.loads(args.dynamic_axes.read_text()) if args.dynamic_axes else None
            if axes is not None:
                if (not isinstance(axes, dict) or any(not isinstance(a, dict) or any(
                        not k.isdecimal() or str(int(k)) != k for k in a) for a in axes.values())):
                    raise ValueError('dynamic axes JSON requires canonical nonnegative integer keys')
                axes = {name: {int(axis): symbol for axis, symbol in a.items()} for name, a in axes.items()}
            result = export_torch(args.module, inputs=json.loads(args.inputs.read_text()),
                output_dir=args.output_dir, output_names=args.output_names, input_names=args.input_names,
                dynamic_axes=axes, external_data=args.external_data)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "import-onnx":
            from arch_explorer.archs.onnx_compilation import onnx_workload_from_files
            spec = onnx_workload_from_files(args.model, inputs_path=args.inputs,
                                 input_shapes_path=args.input_shapes,
                                 external_root=args.external_root, name=args.name,
                                 resource_budget=json.loads(args.resource_budget.read_text()) if args.resource_budget else None)
            text = json.dumps(spec.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            return 0
        if args.command == "components":
            from arch_explorer.components.api import component_catalog
            text = json.dumps(component_catalog(), indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text)
                print(f"Saved {args.output}")
            else:
                print(text, end="")
            return 0
        if args.command == "compile":
            from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
            from arch_explorer.workload.api import compile_workload
            specs = json.loads(args.specs.read_text())
            if args.target is not None or (isinstance(specs, dict) and "target" in specs):
                raise ValueError("Whole-graph target selection is retired; use component-path bindings with adapter id/version/options")
            required = {"adf", "workload", "mapping"}
            if (not isinstance(specs, dict) or not required <= specs.keys()
                    or set(specs) - required - {"implementations", "bindings"}):
                raise ValueError("Compilation specs require adf, workload, mapping and optional implementations/bindings")
            bindings = json.loads(args.bindings.read_text()) if args.bindings else specs.get("bindings")
            result = compile_workload(from_dict(specs["adf"]), WorkloadSpec(**specs["workload"]),
                                      MappingSpec(**specs["mapping"]), bindings=bindings,
                                      implementations=specs.get("implementations"))
            text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text)
            else:
                print(text, end="")
            return 0
        if args.command == "compiler-targets":
            raise ValueError("compiler-targets is retired; use task-adapters for component-bound conversion capabilities")
        if args.command in {"task-adapters", "operator-mappings", "migrate-compilation-v1", "migrate-compilation-v2"}:
            if args.command == "task-adapters":
                from arch_explorer.workload.api import task_adapters
                result = task_adapters()
            elif args.command == "operator-mappings":
                from arch_explorer.workload.api import operator_mappings
                result = operator_mappings(args.operator)
            elif args.command == "migrate-compilation-v1":
                from arch_explorer.workload.api import migrate_compilation_v1
                result = migrate_compilation_v1(json.loads(args.compilation.read_text()))
            else:
                from arch_explorer.workload.api import migrate_compilation_v2
                result = migrate_compilation_v2(json.loads(args.compilation.read_text()))
            text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text)
            else:
                print(text, end="")
            return 0
        if args.command == "prepare":
            from arch_explorer.simulator.api import prepare_compiled
            compilation = json.loads(args.compilation.read_text())
            options = json.loads(args.options.read_text()) if args.options else None
            result = prepare_compiled(compilation, output_dir=args.output, options=options)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "demo":
            from arch_explorer.demo import prepare_demo, run_demo
            if args.demo_command == "prepare":
                result = prepare_demo(args.output, model=args.model,
                                      specs=json.loads(args.specs.read_text()) if args.specs else None,
                                      provider=args.provider,
                                      engine=str(args.engine.resolve()) if args.engine else None,
                                      options=json.loads(args.options.read_text()) if args.options else None)
            else:
                result = run_demo(args.directory)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        is_trace_replay = args.command == "env" and args.env_command == "replay"
        if is_trace_replay:
            from arch_explorer.agent_env import replay_trace
            rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
            if any(not isinstance(row, dict) for row in rows):
                raise ValueError("Tool trace records must be JSON objects")
            if not rows or not isinstance(rows[0], dict) or rows[0].get("type") != "config":
                raise ValueError("Tool trace must start with a config record")
            result = replay_trace(rows).get_state()["result"]
        else:
            config = json.loads(args.config.read_text())
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a JSON object")
        if not is_trace_replay and args.command in {"env", "mcp"}:
            from arch_explorer.agent_env import ExplorationEnv
            from arch_explorer.agent_env.transport import serve_json_lines, serve_mcp
            env = ExplorationEnv(config)
            print(f"Tool trace: {env.trace_path}", file=sys.stderr)
            (serve_mcp if args.command == "mcp" else serve_json_lines)(env)
            return 0
        if args.command == "rerun":
            from arch_explorer.exploration import rerun
            result = rerun(config, output_dir=args.work_dir)
        elif args.command == "search":
            from arch_explorer.search.nested import run_search
            result = run_search(config, policy=args.policy, basic_only=args.basic_only)
        elif args.command == "agent":
            from arch_explorer.agent_env.driver import run_agent
            result = run_agent(config, policy=args.policy, max_steps=args.max_steps)
        elif not is_trace_replay:
            from arch_explorer.agent_env.environment import RESULT_SCHEMA
            from arch_explorer.agent_env.compiled import RESULT_SCHEMA as COMPILED_SEARCH_SCHEMA
            if config.get("schema") in (RESULT_SCHEMA, COMPILED_SEARCH_SCHEMA):
                from arch_explorer.exploration import replay
                result = replay(config)
            else:
                from arch_explorer.demo import program_api
                result = program_api().evaluate_request(config)
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            print(f"Saved {args.output}")
        else:
            print(text, end="")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        print(f"archexplorer: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
