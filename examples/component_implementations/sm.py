"""Explicit SM implementation inputs: roofline work or native program preparation."""
import argparse
import json
from pathlib import Path


def roofline_case(name):
    from arch_explorer.adf import Param, SpaceMatrix, SpacePoint
    from arch_explorer.components.api import ImplementationBinding
    from arch_explorer.components.sm.implementations.roofline import RooflineWork, evaluate_roofline
    # These are declared work/service examples, not fitted device constants.
    operations, boundary_bytes = {
        "compute": (4096, 256), "memory": (256, 4096), "balanced": (1024, 512),
    }[name]
    source = "explicit synthetic roofline example; ideal overlap, no hardware calibration"
    point = SpacePoint("compute", "sm", {"precision": Param("fp32", "1", source)},
                       implementation=ImplementationBinding("sm.roofline", "1", {
                           "peak_ops_per_ns": Param(128.0, "op/ns", source),
                           "bandwidth_bytes_per_ns": Param(64.0, "B/ns", source),
                           "boundary": Param("component_io", "1", source),
                       }))
    work = RooflineWork(operations, boundary_bytes, "component_io")
    result = evaluate_roofline(point, work)
    return {"schema": "arch-explorer.sm-implementation-example.v1",
            "adf": SpaceMatrix("component_demo", [point]).to_dict(),
            "work": {"operations": work.operations, "boundary_bytes": work.boundary_bytes,
                     "boundary": work.boundary},
            "evaluation": result.to_dict(),
            "scope": "single-component analytical estimate; component_io bytes charged here only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    roofline = commands.add_parser("roofline", help="Execute explicit single-component analytical work")
    roofline.add_argument("--case", choices=("compute", "memory", "balanced"), required=True)
    roofline.add_argument("--output", type=Path)
    native = commands.add_parser("native-prepare", help="Prepare a native component GPU program; never execute it")
    native.add_argument("--model", choices=("vit", "clip"), required=True)
    native.add_argument("--engine", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True, help="New preparation directory")
    args = parser.parse_args(argv)
    if args.command == "roofline":
        result = roofline_case(args.case)
        text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end="")
    else:
        from arch_explorer.archs.gpu_instantiation import case_specs
        from arch_explorer.demo import prepare_demo
        specs = case_specs(args.model, engine=args.engine)["candidate"]
        result = prepare_demo(args.output, specs=specs)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, NotImplementedError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"sm example: {error}") from None
