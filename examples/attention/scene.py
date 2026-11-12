"""Save public attention source specs and an optional bounded search config."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New source-specs file")
    parser.add_argument("--search-config", type=Path, help="Also save input for the existing search command")
    parser.add_argument("--roofline", action="store_true", help="Use the synthetic Roofline teaching hardware")
    parser.add_argument("--input-case", choices=("regular", "large", "constant"), default="regular")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--keys", type=int, default=5)
    parser.add_argument("--head-dim", type=int, default=4)
    parser.add_argument("--value-dim", type=int, default=3)
    parser.add_argument("--tile-m", type=int, default=2)
    parser.add_argument("--tile-k", type=int, default=3)
    parser.add_argument("--tile-n", type=int, default=3)
    args = parser.parse_args(argv)
    from arch_explorer.archs.attention import attention_scene, attention_search_config
    from arch_explorer.archs.compilation import roofline_scene
    hardware = roofline_scene if args.roofline else None
    if args.search_config and (args.batch, args.queries, args.keys, args.head_dim, args.value_dim,
                               args.tile_m, args.tile_k, args.tile_n) != (1, 3, 5, 4, 3, 2, 3, 3):
        raise ValueError("The bounded search tutorial uses its explicit default dimensions and two documented mappings")
    adf, workload, mapping = attention_scene(hardware=hardware, batch=args.batch, queries=args.queries,
        keys=args.keys, head_dim=args.head_dim, value_dim=args.value_dim, input_case=args.input_case,
        tile_m=args.tile_m, tile_k=args.tile_k, tile_n=args.tile_n)
    files = [(args.output, {"adf": adf.to_dict(), "workload": workload.to_dict(), "mapping": mapping.to_dict()})]
    if args.search_config:
        files.append((args.search_config, attention_search_config(hardware=hardware, input_case=args.input_case)))
    if len({path.resolve() for path, _ in files}) != len(files) or any(path.exists() for path, _ in files):
        raise ValueError("Attention source and search files must have distinct new paths")
    for path, document in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"attention scene: {error}") from None
