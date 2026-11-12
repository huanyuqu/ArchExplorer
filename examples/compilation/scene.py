"""Write explicit source specs; compilation belongs to the public compiler."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument("--adf", type=Path, help="Explicit hardware and implementation bindings JSON")
    hardware.add_argument("--example", choices=("events", "roofline", "mixed", "native"),
                          help="Explicit teaching scene; default events")
    parser.add_argument("--compute", help="Selected compute point path, required with --adf")
    parser.add_argument("--memory", help="Selected storage point path, required with --adf")
    parser.add_argument("--bindings", type=Path, help="Explicit compute-path adapter bindings JSON, including adapter options")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--tile-m", type=int, default=2)
    parser.add_argument("--tile-k", type=int, default=3)
    parser.add_argument("--tile-n", type=int, default=4)
    parser.add_argument("--fanout", action="store_true")
    args = parser.parse_args(argv)
    from arch_explorer.adf import from_dict
    from arch_explorer.archs.compilation import (compile_mapping, matmul_workload,
                                               events_scene, roofline_scene, heterogeneous_scene, native_scene)
    mapping_options = {"tile_m": args.tile_m, "tile_k": args.tile_k, "tile_n": args.tile_n}
    if args.adf:
        if not args.compute or not args.memory:
            raise ValueError("--adf requires explicit --compute and --memory paths")
        adf = from_dict(json.loads(args.adf.read_text()))
        workload = matmul_workload(m=args.m, k=args.k, n=args.n, fanout=args.fanout)
        mapping = compile_mapping(compute=args.compute, memory=args.memory, **mapping_options)
    else:
        if args.compute or args.memory:
            raise ValueError("--compute/--memory require --adf; teaching scenes use their documented placement")
        constructor = {"events": events_scene, "roofline": roofline_scene,
                       "mixed": heterogeneous_scene, "native": native_scene}[args.example or "events"]
        adf, workload, mapping = constructor(m=args.m, k=args.k, n=args.n,
                                             fanout=args.fanout or args.example == "mixed", **mapping_options)
    specs = {"adf": adf.to_dict(), "workload": workload.to_dict(), "mapping": mapping.to_dict()}
    if args.bindings:
        specs["bindings"] = json.loads(args.bindings.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"compilation scene: {error}") from None
