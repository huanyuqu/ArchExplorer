"""Prepare and optionally execute the extension through public APIs."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New preparation directory")
    parser.add_argument("--roofline", action="store_true", help="Use the public synthetic Roofline teaching board")
    parser.add_argument("--input-case", choices=("regular", "large", "constant"), default="regular")
    parser.add_argument("--execute", action="store_true", help="Explicitly simulate, then validate the saved result read-only")
    parser.add_argument("--search-config", type=Path, help="Also save input for the existing bounded search engine")
    args = parser.parse_args(argv)
    from arch_explorer.archs.compilation import roofline_scene
    from arch_explorer.archs.attention import attention_search_config
    from arch_explorer.workload.api import compile_workload
    from arch_explorer.simulator.api import prepare_compiled, assess_request, evaluate_request
    from .graphs import scene
    adf, workload, mapping = scene(hardware=roofline_scene if args.roofline else None, input_case=args.input_case)
    compilation = compile_workload(adf, workload, mapping)
    request = prepare_compiled(compilation, output_dir=args.output, options={"trace": True})
    capability = assess_request(request)
    if not capability["supported"]:
        raise ValueError("Prepared extension request is unsupported: " + str(capability))
    documents = {"EXAMPLE-SOURCE-SPECS.json": {"adf": adf.to_dict(), "workload": workload.to_dict(), "mapping": mapping.to_dict()},
                 "EXAMPLE-COMPILATION.json": compilation, "EXAMPLE-PUBLIC-REQUEST.json": request}
    if args.execute:
        result = evaluate_request(request)
        if evaluate_request(result) != result:
            raise ValueError("Saved extension result did not validate read-only")
        documents["EXAMPLE-PUBLIC-RESULT.json"] = result
    for name, document in documents.items():
        with (args.output / name).open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    if args.search_config:
        config = attention_search_config(hardware=roofline_scene if args.roofline else None,
                                         workload=workload, input_case=args.input_case)
        args.search_config.parent.mkdir(parents=True, exist_ok=True)
        with args.search_config.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"request": request, "capability": capability, "executed": args.execute}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, KeyError, NotImplementedError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"external operator mapping: {error}") from None
