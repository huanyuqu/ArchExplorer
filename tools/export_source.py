"""Build a clean source tree with hashes, without exporting Git history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

FILES = ("README.md", "LICENSE", "NOTICE", "pyproject.toml", "requirements.txt", "MANIFEST.in", ".gitignore")
TREES = {"arch_explorer": {".py"}, "docs": {".md"}, "examples": {".json"},
         "tests": {".py"}, "tools": {".py"}, ".github": {".yml"}, "skills": {".md"}}
EXAMPLE_TREES = {
    "examples/extensions/synthetic_components": {".py", ".md", ".toml"},
    "examples/extensions/workload_components": {".py", ".md", ".toml"},
    "examples/extensions/operator_implementations": {".py", ".md", ".toml"},
    "examples/gpu_instantiation": {".py", ".md"},
    "examples/extensions/component_implementations": {".py", ".md", ".toml"},
    "examples/component_implementations": {".py", ".md"},
}
# Reviewed resources from SIM41 RESOURCES.json and CORE24 RESOURCE-REQUEST.json.
# Exact members only: no recursive native/JSON allowances. Missing members fail export.
SOURCE_FILES = (
    "tests/fixtures/adapter_boundary_extension/pyproject.toml",
    "tests/fixtures/adapter_boundary_extension/V1-COMPILATION.json",
    "tests/fixtures/operator_attention/INPUTS-v3.json",
    "tests/fixtures/operator_attention/V2-COMPILATION.json",
    "tests/fixtures/precision_transformer/INPUTS.json",
    "tests/fixtures/operator_mapping_extension/pyproject.toml",
    "tests/fixtures/composition_builder_extension/pyproject.toml",
    "tests/fixtures/multilayer_composition/INPUTS.json",
    "tests/fixtures/view_storage/INPUTS.json",
    "tests/fixtures/vit_compilation/INPUTS.json",
    "tests/fixtures/clip_compilation/INPUTS.json",
    "tests/fixtures/workload_scenarios/INPUTS.json",
    "tests/fixtures/onnx_import/INPUTS.json",
    "tests/fixtures/onnx_continuation/INPUTS.json",
    "tests/fixtures/onnx_import/attention/INPUTS.json",
    "tests/fixtures/onnx_import/attention/model.onnx",
    "tests/fixtures/onnx_import/mlp/INPUTS.json",
    "tests/fixtures/onnx_import/mlp/model.onnx",
    "tests/fixtures/onnx_import/mlp_external/INPUTS.json",
    "tests/fixtures/onnx_import/mlp_external/model.onnx",
    "tests/fixtures/onnx_import/mlp_external/weights/data.bin",
    "tests/fixtures/onnx_import/sim51/INPUTS.json",
    "tests/fixtures/onnx_import/sim51/models/attention.onnx",
    "tests/fixtures/onnx_import/sim51/models/mlp.onnx",
    "tests/fixtures/onnx_import/sim51/models/mlp.weights",
    "tests/workload/ONNX-COVERAGE.md",
    "examples/compilation/README.md",
    "examples/compilation/agent.py",
    "examples/compilation/scene.py",
    "examples/compilation/workloads.py",
    "examples/attention/README.md",
    "examples/attention/scene.py",
    "examples/transformer_block/README.md",
    "examples/transformer_block/scene.py",
    "examples/transformer_block/provider-options.json",
    "examples/multilayer_encoder/README.md",
    "examples/multilayer_encoder/scene.py",
    "examples/multilayer_encoder/provider-options.json",
    "examples/vit_compilation/README.md",
    "examples/vit_compilation/scene.py",
    "examples/vit_compilation/provider-options.json",
    "examples/clip_compilation/README.md",
    "examples/clip_compilation/scene.py",
    "examples/clip_compilation/provider-options.json",
    "examples/workload_scenarios/README.md",
    "examples/workload_scenarios/scene.py",
    "examples/workload_scenarios/mla.py",
    "examples/workload_scenarios/structure.py",
    "examples/workload_scenarios/joint_search.py",
    "examples/workload_scenarios/llm.json",
    "examples/onnx_import/README.md",
    "examples/onnx_import/models.py",
    "examples/onnx_import/emit.py",
    "examples/framework_export/README.md",
    "examples/framework_export/models.py",
    "examples/framework_export/emit.py",
    "examples/framework_export/vision_language.py",
    "examples/paper_scale/README.md",
    "examples/paper_scale/cases.py",
    "examples/paper_scale/preflight.py",
    "examples/paper_scale/block.py",
    "tests/validation/HARDWARE-SAMPLE.md",
    "tests/validation/SCALE-LOCK.md",
    "tests/validation/CONTINUATION.md",
    "tests/validation/CONTINUATION-INSTALLED.md",
    "tests/validation/SIM53-CONSUMER.md",
    "tests/workload/FRAMEWORK-CONSUMERS.md",
    "tests/workload/TRANSFORMER-METADATA.md",
    "tests/workload/TYPED-CONSUMER-ORDER.md",
    "tests/workload/TYPED-ONNX-ORACLE.md",
    "examples/extensions/operator_mappings/README.md",
    "examples/extensions/operator_mappings/pyproject.toml",
    "examples/extensions/operator_mappings/src/ae_operator_mappings/__init__.py",
    "examples/extensions/operator_mappings/src/ae_operator_mappings/__main__.py",
    "examples/extensions/operator_mappings/src/ae_operator_mappings/graphs.py",
    "examples/extensions/operator_mappings/src/ae_operator_mappings/mapping.py",
    "examples/extensions/operator_mappings/src/ae_operator_mappings/semantics.py",
    "arch_explorer/components/_support/accelsim/catalog_harness.cpp",
    "arch_explorer/components/_support/accelsim/catalog_policy.hpp",
    "arch_explorer/components/_support/accelsim/layout_v40.json",
    "arch_explorer/components/_support/accelsim/native/NOTICE",
    "arch_explorer/components/_support/accelsim/native/README.md",
    "arch_explorer/components/_support/accelsim/native/configuration_api.hpp",
    "arch_explorer/components/_support/accelsim/native/execution.cc",
    "arch_explorer/components/_support/accelsim/native/execution.hpp",
    "arch_explorer/components/_support/accelsim/native/instruction_config.hpp",
    "arch_explorer/components/_support/accelsim/native/memory_api.hpp",
    "arch_explorer/components/_support/accelsim/native/memory_links.cc",
    "arch_explorer/components/_support/accelsim/native/memory_links.hpp",
    "arch_explorer/components/_support/accelsim/native/memory_services.hpp",
    "arch_explorer/components/_support/accelsim/native/memory_statistics.cc",
    "arch_explorer/components/_support/accelsim/native/memory_system.cc",
    "arch_explorer/components/_support/accelsim/native/memory_system.hpp",
    "arch_explorer/components/_support/accelsim/native/partition_random.hpp",
    "arch_explorer/components/_support/accelsim/native/services.hpp",
    "arch_explorer/components/_support/accelsim/native/transactions.cc",
    "arch_explorer/components/_support/accelsim/native/transactions.hpp",
    "arch_explorer/components/_support/accelsim/native-baseline-3a6e614.json",
    "arch_explorer/components/_support/accelsim/state_cpu.cpp",
    "arch_explorer/components/_support/accelsim/state_executor.hpp",
    "arch_explorer/components/assemblies/gpu/native/README.md",
    "arch_explorer/components/assemblies/gpu/native/gpu_assembly.cc",
    "arch_explorer/components/assemblies/gpu/native/gpu_assembly.hpp",
    "arch_explorer/components/assemblies/gpu/native/gpu_entry.cc",
    "arch_explorer/components/assemblies/gpu/native/gpu_transport.cc",
    "arch_explorer/components/assemblies/gpu/native/gpu_transport.hpp",
    "arch_explorer/components/bus/README.md",
    "arch_explorer/components/cgra_pe/README.md",
    "arch_explorer/components/crossbar/README.md",
    "arch_explorer/components/dram/README.md",
    "arch_explorer/components/dram/implementations/accelsim/native/README.md",
    "arch_explorer/components/dram/implementations/accelsim/native/dram.cc",
    "arch_explorer/components/fifo_link/README.md",
    "arch_explorer/components/icnt/README.md",
    "arch_explorer/components/icnt/implementations/accelsim/native/README.md",
    "arch_explorer/components/icnt/implementations/accelsim/native/icnt.cc",
    "arch_explorer/components/icnt/implementations/accelsim/native/icnt_api.hpp",
    "arch_explorer/components/ipu_tile/README.md",
    "arch_explorer/components/l1_shared/README.md",
    "arch_explorer/components/l2/README.md",
    "arch_explorer/components/l2/implementations/accelsim/native/README.md",
    "arch_explorer/components/l2/implementations/accelsim/native/l2.cc",
    "arch_explorer/components/mesh2d/README.md",
    "arch_explorer/components/multicast_tree/README.md",
    "arch_explorer/components/p2p_switch/README.md",
    "arch_explorer/components/pim_unit/README.md",
    "arch_explorer/components/serial_service/README.md",
    "arch_explorer/components/service_ceiling/README.md",
    "arch_explorer/components/sm/README.md",
    "arch_explorer/components/sm/implementations/accelsim/native/README.md",
    "arch_explorer/components/sm/implementations/accelsim/native/shader.cc",
    "arch_explorer/components/sm/implementations/accelsim/native/shader_api.hpp",
    "arch_explorer/components/sm/implementations/accelsim/native/shader_services.hpp",
    "arch_explorer/components/sram/README.md",
    "arch_explorer/components/systolic_array/README.md",
    "arch_explorer/components/tianjic_core/README.md",
    "arch_explorer/components/tile_exchange/README.md",
    "arch_explorer/components/vector_unit/README.md",
)
SOURCE_GLOBS = (
    "examples/gpu_instantiation/standalone_icnt.cpp",
    "examples/gpu_instantiation/standalone_memory.cpp",
    "arch_explorer/components/accelsim/catalog_harness.cpp",
    "arch_explorer/components/accelsim/catalog_policy.hpp",
    "arch_explorer/components/accelsim/state_cpu.cpp",
    "arch_explorer/components/accelsim/state_executor.hpp",
    "arch_explorer/simulator/dynamic/*.hpp",
    "arch_explorer/components/synthetic_dynamic/*.cpp",
    "arch_explorer/components/accelsim/component_native/*.cc",
    "arch_explorer/components/accelsim/component_native/*.hpp",
    "arch_explorer/components/accelsim/component_native/NOTICE",
    "arch_explorer/components/accelsim/component_native/*.md",
    "tools/accelsim_native/patches/accelsim-sm120-typed-v1.patch",
    "tools/accelsim_native/patches/gpgpusim-sm120-typed-v1.patch",
    "tools/accelsim_native/componentization/native/*.py",
    "tools/accelsim_native/componentization/native/*.md",
)
FORBIDDEN = {".git", ".venv", "__pycache__", "experiments", "visualization", "results"}


def _allowed_path(path, root):
    parts = path.relative_to(root).parts
    return (not FORBIDDEN.intersection(parts)
            and not any(part in {"build", "dist"} or part.endswith(".egg-info")
                        for part in parts))


def export(destination, root=None):
    root = (Path(root) if root else Path(__file__).resolve().parents[1]).resolve()
    destination = Path(destination).resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("Export destination must be outside the integration checkout")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Export destination must be absent or empty")
    sources = [root/name for name in (*FILES, *SOURCE_FILES)]
    for name, suffixes in TREES.items():
        sources.extend(p for p in sorted((root/name).rglob("*"))
                       if p.is_file() and p.suffix in suffixes and _allowed_path(p, root))
    # Reviewed tutorials may include code/docs; only installable extensions
    # include metadata. Keep these allowances scoped to their exact directories.
    for tree, suffixes in EXAMPLE_TREES.items():
        sources.extend(p for p in sorted((root/tree).rglob("*"))
                       if p.is_file() and p.suffix in suffixes
                       and _allowed_path(p, root))
    # Native sources and patches are reviewed individually by directory/type.
    # These globs are non-recursive; never export arbitrary native build trees.
    for pattern in SOURCE_GLOBS:
        sources.extend(p for p in sorted(root.glob(pattern))
                       if p.is_file() and _allowed_path(p, root))
    sources = list(dict.fromkeys(sources))
    for p in sources:
        if not p.is_file() or p.is_symlink() or root not in p.resolve().parents:
            raise ValueError(f"Not a regular source file: {p.relative_to(root)}")
        if p.name == "api_key.json":
            raise ValueError("Credential files cannot be exported")
    manifest = {}
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        relative = source.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o644)
        manifest[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination/"EXPORT_MANIFEST.json").write_text(json.dumps({"schema_version": 1, "files": manifest}, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        files = export(args.destination)
    except ValueError as error:
        parser.error(str(error))
    print(f"Exported {len(files)} source files to {args.destination}")


if __name__ == "__main__":
    main()
