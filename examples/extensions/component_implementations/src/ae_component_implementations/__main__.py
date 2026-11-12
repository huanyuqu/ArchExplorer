"""Prepare a public events request. Execution stays in the standard CLI/API."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New request directory")
    parser.add_argument("--ops-per-ns", type=float, default=4.0)
    parser.add_argument("--startup-ns", type=float, default=3.0)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n", type=int, default=2)
    args = parser.parse_args(argv)
    from arch_explorer.simulator.api import assess_request, prepare_request
    from . import scene
    adf, workload, mapping = scene(ops_per_ns=args.ops_per_ns, startup_ns=args.startup_ns,
                                   m=args.m, k=args.k, n=args.n)
    request = prepare_request(adf, workload, mapping, provider="events",
                              options={"trace": True}, output_dir=args.output)
    print(json.dumps({"request": request, "capability": assess_request(request)},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, NotImplementedError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"ae_component_implementations: {error}") from None
