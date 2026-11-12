"""Compile and prepare the external SM through the same public compilation API."""
import argparse
import json
from pathlib import Path


def compilation_scene(*, ops_per_ns=4.0, startup_ns=3.0, m=3, k=5, n=7,
                      fanout=False, tile_m=2, tile_k=3, tile_n=4,
                      assembly_elements_per_ns=16.0, assembly_startup_ns=1.0):
    from arch_explorer.adf import Param, find
    from arch_explorer.archs.compilation import compile_mapping, matmul_workload
    from arch_explorer.components.api import ImplementationBinding
    from .scenes import serial_scene
    adf, _, _ = serial_scene(ops_per_ns=ops_per_ns, startup_ns=startup_ns, m=m, k=k, n=n)
    point = find(adf, "device/core/compute")
    source = "explicit CORE24 synthetic assembly issue service; no internal byte charge or calibration"
    point.implementation = ImplementationBinding("example.sm.serial_assembly", "1", {
        **point.implementation.params,
        "assembly_elements_per_ns": Param(assembly_elements_per_ns, "element/ns", source),
        "assembly_startup_ns": Param(assembly_startup_ns, "ns", source),
    })
    workload = matmul_workload(m=m, k=k, n=n, fanout=fanout)
    mapping = compile_mapping(compute="device/core/compute", memory="device/sram",
                              tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)
    return adf, workload, mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New preparation directory")
    parser.add_argument("--ops-per-ns", type=float, default=4.0)
    parser.add_argument("--startup-ns", type=float, default=3.0)
    parser.add_argument("--assembly-elements-per-ns", type=float, default=16.0)
    parser.add_argument("--assembly-startup-ns", type=float, default=1.0)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--tile-m", type=int, default=2)
    parser.add_argument("--tile-k", type=int, default=3)
    parser.add_argument("--tile-n", type=int, default=4)
    parser.add_argument("--fanout", action="store_true")
    args = parser.parse_args(argv)
    from arch_explorer.workload.api import compile_workload
    from arch_explorer.simulator.api import assess_request, prepare_compiled
    adf, workload, mapping = compilation_scene(
        ops_per_ns=args.ops_per_ns, startup_ns=args.startup_ns, m=args.m, k=args.k, n=args.n,
        fanout=args.fanout, tile_m=args.tile_m, tile_k=args.tile_k, tile_n=args.tile_n,
        assembly_elements_per_ns=args.assembly_elements_per_ns,
        assembly_startup_ns=args.assembly_startup_ns)
    compilation = compile_workload(adf, workload, mapping)
    request = prepare_compiled(compilation, output_dir=args.output)
    with (args.output / "EXAMPLE-PUBLIC-REQUEST.json").open("x") as stream:
        json.dump(request, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (args.output / "EXAMPLE-COMPILATION.json").open("x") as stream:
        json.dump(compilation, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    specs = {"adf": adf.to_dict(), "workload": workload.to_dict(), "mapping": mapping.to_dict()}
    with (args.output / "EXAMPLE-SOURCE-SPECS.json").open("x") as stream:
        json.dump(specs, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"request": request, "capability": assess_request(request)},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, NotImplementedError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"external SM compilation: {error}") from None
