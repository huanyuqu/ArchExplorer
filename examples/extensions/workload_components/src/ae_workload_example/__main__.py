"""Generate ordinary input files; simulation uses the generic ArchExplorer CLI."""
import argparse
import json
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create explicit ADF, two graphs and mappings")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--rows", type=int, default=32)
    generate.add_argument("--cols", type=int, default=16)
    hardware = commands.add_parser("hardware", help="Create another explicit hardware description")
    hardware.add_argument("--lanes", type=int, required=True)
    hardware.add_argument("--clock-ghz", type=float, required=True)
    hardware.add_argument("--output", type=Path, required=True)
    scene = commands.add_parser("scene", help="Read ADF/graph/mapping files and create a generic scene")
    for name in ("adf", "graph", "mappings", "output"):
        scene.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from arch_explorer.components.api import discover_components
    discover_components()
    if args.command == "hardware":
        from .hardware import custom_adf
        write(args.output, custom_adf(lanes=args.lanes, clock_ghz=args.clock_ghz).to_dict())
    elif args.command == "generate":
        from arch_explorer.adf import MappingSpec
        from .graphs import affine_graph, builtin_graph
        from .hardware import custom_adf
        adf = custom_adf().to_dict()
        affine = affine_graph(args.rows, args.cols).to_dict()
        builtin = builtin_graph(args.rows, args.cols).to_dict()
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "adf.json", adf)
        write(args.output / "affine-graph.json", affine)
        write(args.output / "builtin-graph.json", builtin)
        write(args.output / "mappings.json", [MappingSpec("tutorial_rows", {"tile_rows": n}).to_dict() for n in (4, 16)])
    else:
        from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
        from arch_explorer.workload.api import Workload
        adf = from_dict(json.loads(args.adf.read_text(encoding="utf-8")))
        graph = Workload.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
        mappings = [MappingSpec(**value).to_dict() for value in json.loads(args.mappings.read_text(encoding="utf-8"))]
        value = {"provider": "events", "adf": adf.to_dict(), "workload": WorkloadSpec("graph", graph.to_dict()).to_dict(),
                 "mappings": mappings, "options": {"trace": True},
                 "scope": "synthetic branching graph on custom arrays/shared SRAM; no hardware accuracy claim"}
        write(args.output, value)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, KeyError, NotImplementedError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"ae_workload_example: {error}") from None
