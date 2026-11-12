"""Use the installed native SDK to compile and run an external component caller.

The component package owns SDK validation, linking, constructor audit and
bounded CPU execution. This script preserves the caller's assertions and the
public receipts; it does not implement a simulator provider or comparator.
"""
import argparse
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path

PREFIX = "ARCH_EXPLORER_STANDALONE_RESULT "


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def package_files(root):
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_example(args):
    if not 1 <= args.timeout <= 600 or not 1 <= args.memory_mib <= 8192:
        raise ValueError("CPU execution is bounded to at most 600s / 8192 MiB")
    sdk_root = Path(str(resources.files("arch_explorer"))).resolve()
    distribution = metadata.distribution("arch-explorer")
    if Path(distribution.locate_file("arch_explorer")).resolve() != sdk_root:
        raise ValueError("Use a non-editable installed wheel, not another source checkout")
    from arch_explorer.components.accelsim import native_sdk

    source = args.source.resolve(strict=True)
    sdk = args.sdk.resolve(strict=True)
    document = native_sdk.load_sdk(sdk)
    targets = tuple(args.target or ("memory", "icnt"))
    source_sha, sdk_sha = sha(source), sha(sdk)
    before = package_files(sdk_root)
    output = args.output.resolve()
    if output.is_relative_to(sdk_root):
        raise ValueError("Keep caller outputs outside the installed SDK")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "arch-explorer.native-caller-run.v1", "status": "building",
               "scope": "External caller assertions and public native SDK receipts; not a provider Evaluation",
               "package": {"path": str(sdk_root), "version": metadata.version("arch-explorer"), "files_before": before},
               "source": {"path": str(source), "sha256": source_sha},
               "sdk": {"path": str(sdk), "sha256": sdk_sha, "archives": document["archives"]},
               "targets": targets, "limits": {"timeout_s": args.timeout,
                   "run_memory_mib": args.memory_mib, "build_memory_mib": 8192}}
    write(output / "run.json", receipt)
    try:
        receipt["public_build"] = native_sdk.compile_consumer(
            source, sdk, output / "build", targets=targets, timeout_s=args.timeout)
        receipt["status"] = "running"
        write(output / "run.json", receipt)
        process = native_sdk.run_consumer(receipt["public_build"], output / "execute",
            timeout_s=args.timeout, memory_limit_mb=args.memory_mib)
        receipt["public_run"] = process
        stdout = Path(process["stdout"]["path"])
        raw_output = stdout.read_bytes()
        if hashlib.sha256(raw_output).hexdigest() != process["stdout"]["sha256"]:
            raise ValueError("Caller stdout differs from the public execution receipt")
        values = [line[len(PREFIX):] for line in raw_output.decode().splitlines() if line.startswith(PREFIX)]
        if len(values) != 1:
            raise ValueError("The caller must emit one structured standalone result")

        def reject_constant(value):
            raise ValueError("Non-finite caller JSON constant: " + value)

        result = json.loads(values[0], parse_constant=reject_constant)
        if not isinstance(result, dict) or result.get("status") != "passed":
            raise ValueError("Caller assertions did not pass")
        if sha(source) != source_sha or sha(sdk) != sdk_sha:
            raise ValueError("Caller or SDK manifest changed during execution")
        native_sdk.load_sdk(sdk)
        receipt.update(result=result, status="executed")
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        after = package_files(sdk_root)
        receipt["package"].update(files_after=after, unchanged=before == after)
        if before != after:
            receipt.update(status="failed", error="Installed SDK files changed")
        write(output / "run.json", receipt)
    if receipt["status"] != "executed":
        raise RuntimeError(receipt["error"])
    return output / "run.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True,
                        help="Matching standalone SDK manifest, not a whole-engine manifest")
    parser.add_argument("--target", action="append", help="Archive target; repeat to select more than one")
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("standalone_icnt.cpp"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--memory-mib", type=int, default=8192, help="Runtime memory bound; compile uses 8192 MiB")
    print(run_example(parser.parse_args()))


if __name__ == "__main__":
    main()
