"""Generate explicit user inputs; use the generic CLI to simulate them."""
import argparse
import json
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create ADF, branching graph, input/reference arrays and event scenes")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--rows", type=int, default=8)
    generate.add_argument("--cols", type=int, default=16)
    generate.add_argument("--name", default="external-square-fork")
    generate.add_argument("--memory-bytes", type=int, default=32768)
    hardware = commands.add_parser("hardware", help="Create another ADF using the existing vector/memory components")
    hardware.add_argument("--output", type=Path, required=True)
    hardware.add_argument("--memory-bytes", type=int, default=32768)
    hardware.add_argument("--lanes", type=int, default=4)
    hardware.add_argument("--clock-ghz", type=float, default=1.0)
    scene = commands.add_parser("scene", help="Read explicit event ADF, graph and mapping files")
    for key in ("adf", "graph", "mappings", "output"):
        scene.add_argument("--" + key, type=Path, required=True)
    reference = commands.add_parser("check-reference", help="Check public graph reference against the independent sample oracle")
    reference.add_argument("--graph", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    gpu = commands.add_parser("gpu-case", help="Read a user graph/optional GPU ADF and create a standard comparison case")
    gpu.add_argument("--graph", type=Path, required=True)
    gpu.add_argument("--engine", type=Path, required=True)
    gpu.add_argument("--adf", type=Path)
    gpu.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from arch_explorer.components.api import discover_components
    discover_components()
    if args.command == "hardware":
        from .hardware import custom_adf
        write(args.output, custom_adf(memory_bytes=args.memory_bytes, lanes=args.lanes,
                                     clock_ghz=args.clock_ghz).to_dict())
    elif args.command == "generate":
        from .graphs import branch_graph, exact_expected, sample_inputs
        from .hardware import custom_adf
        from .scenes import event_mappings, event_scene
        graph = branch_graph(args.rows, args.cols, name=args.name)
        inputs = sample_inputs(args.rows, args.cols)
        graph.meta["inputs"] = {key: value.tolist() for key, value in inputs.items()}
        expected = exact_expected(inputs)
        adf = custom_adf(memory_bytes=args.memory_bytes)
        mappings = event_mappings(memory=adf.name + "/shared")
        payloads = {"adf.json": adf.to_dict(), "graph.json": graph.to_dict(),
                    "mappings.json": [mapping.to_dict() for mapping in mappings],
                    "inputs.json": {key: value.tolist() for key, value in inputs.items()},
                    "expected.json": {key: value.tolist() for key, value in expected.items()},
                    "events-scene.json": event_scene(adf, graph, mappings)}
        args.output.mkdir(parents=True, exist_ok=False)
        for filename, value in payloads.items():
            write(args.output / filename, value)
    elif args.command == "scene":
        from arch_explorer.adf import MappingSpec, from_dict
        from arch_explorer.workload.api import Workload
        from .scenes import event_scene
        adf = from_dict(json.loads(args.adf.read_text(encoding="utf-8")))
        graph = Workload.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
        mappings = [MappingSpec(**value) for value in json.loads(args.mappings.read_text(encoding="utf-8"))]
        write(args.output, event_scene(adf, graph, mappings))
    elif args.command == "check-reference":
        import numpy as np
        from arch_explorer.adf import WorkloadSpec
        from arch_explorer.workload.api import Workload
        from arch_explorer.workload.reference_api import execute_reference
        from .graphs import exact_expected
        graph = Workload.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
        expected = exact_expected({"x": np.asarray(graph.meta["inputs"]["x"], dtype=np.float32)})
        reference = execute_reference(WorkloadSpec("graph", graph.to_dict()))
        comparisons = {}
        for name, value in expected.items():
            actual = reference["tensors"][name]
            if not isinstance(actual, np.ndarray) or actual.dtype != value.dtype or not np.array_equal(actual, value):
                raise ValueError(f"public reference tensor {name!r} differs from the exact sample oracle")
            comparisons[name] = {"shape": list(value.shape), "exact_match": True}
        write(args.output, {"status": "passed", "comparisons": comparisons,
                            "scope": "public semantic graph reference vs sample rational oracle; not timing or hardware accuracy"})
    else:
        from arch_explorer.adf import from_dict
        from arch_explorer.workload.api import Workload
        from arch_explorer.components.accelsim.componentization import gpu_adf
        from .scenes import gpu_case
        graph = Workload.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
        adf = from_dict(json.loads(args.adf.read_text(encoding="utf-8"))) if args.adf else gpu_adf()
        write(args.output, gpu_case(adf, graph, args.engine))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, KeyError, NotImplementedError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"ae_operator_implementations: {error}") from None
