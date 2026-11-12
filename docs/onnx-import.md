# Standard ONNX File Import and Exploration

Users can hand real ONNX ModelProto files, complete runtime inputs, and explicit hardware configurations to ArchExplorer to obtain ordinary WorkloadSpec, continuing to use the common Graph, compiler, Env/search, offline Agent, and save/restore. Nodes in the file are mechanically converted to Graph operators; filenames, model names, and display names do not select handwritten network templates.

The optional dependency is ONNX 1.16.2; without installing that extra, existing workloads and tools can still be imported. The installation entry is `pip install 'arch_explorer[onnx]'`. Supports standard empty domain, opset17, actual IR8, inference DAG without control flow, real FLOAT/FLOAT16/BFLOAT16 storage, and FP32 computation. Supports bounded MatMul/Gemm/Add/Mul/Div/Erf/Cast/LayerNormalization/Relu/Softmax/Transpose/Reshape/Identity/Constant. Symbolic roots must explicitly provide complete shapes by original name; complete ONNX, training, quantization, custom domains, sparsity, dynamic control, and non-literal shape operations still have explicit gaps. [Actual framework export](framework-export.md) and dtype references, components, and installation acceptance are accounted for separately.

## Getting Common Workloads from Files

```python
import json
from pathlib import Path
from arch_explorer.archs.onnx_compilation import (
    onnx_workload, onnx_scene, onnx_search_config, onnx_agent_config,
)
from arch_explorer.workload.onnx_import import describe_onnx

root = Path('input-models/mlp')
spec = onnx_workload(
    root / 'model.onnx',
    inputs=json.loads((root / 'INPUTS.json').read_text()),
    input_shapes=json.loads((root / 'INPUT-SHAPES.json').read_text()),
    external_root=root,
    name='user-example',
)
identities = describe_onnx(spec)
adf, same_spec, mapping = onnx_scene(spec, candidate=0, layout='materialized')
config = onnx_search_config(spec)
agent = onnx_agent_config(spec)
```

`onnx_workload(model, *, inputs, input_shapes=None, external_root=None, name='onnx', resource_budget=None)` directly calls the official `workload.onnx_import.import_onnx`. inputs is a dictionary from original graph.input names to complete finite JSON numeric values; there is no default root or weights. input_shapes is a dictionary from root names to complete positive-integer shapes, used to specialize symbolic dimensions, not a mapping from symbolic names to integers. If an initializer is also listed in graph.input, the importer rejects it; it cannot implicitly override or ignore user values. CORE does not deepcopy huge numeric roots before the authoritative preflight.

The file entry point `onnx_workload_from_files(model, *, inputs_path, input_shapes_path=None, external_root=None, name='onnx', resource_budget=None)` first calls the public onnx_preflight to check the source and shape/budget, then loads the root JSON according to the admitted result; if the root file itself exceeds max_host_bytes, it is also rejected before reading. The CLI import-onnx and the framework import_export share this entry point. `--resource-budget BUDGET.json` is optional; the fully normalized budget enters the WorkloadSpec/Graph/compilation source. Omitting the budget preserves the historical four-field small-package representation and does not modify global limits.

When external tensors are present, external_root must be provided as the allowed root boundary; external locations are resolved relative to the directory containing the model, and the resolved file must lie within that allowed root. offset/length, actual bytes, and shape/dtype must all be consistent. Missing files, out-of-bounds access, directory escape, symlink escape, and unsupported attributes are all rejected by the importer before prepare, without making network requests. Valid ONNX may still fall outside the supported domain of this product; a successful official checker run does not substitute for product executability.

`kind='onnx'` is an automatically registered built-in workload; configuration can explicitly use `registration_modules=[]`; omitting that field uses the same empty-list default. No additional registrar or provider is needed; compilation and execution after import continue to use existing component bindings.

The returned `WorkloadSpec(kind='onnx')` has default parameters name/source/inputs/input_shapes; an explicit budget adds a normalized resource_budget. source stores the complete bytes and digest of the model and external files, with a default source budget of 16 MiB; limits such as max_source_bytes can be explicitly adjusted within the public max_host_bytes constraint, but protections cannot be removed entirely. The original absolute path is not a deployment prerequisite. After saving, deleting the original files still allows reading that snapshot; re-importing modified files produces a new source. After valid re-sealing, modifying model/external bytes, digest, roots, shapes, or budget still requires passing the original complete source and product validation.

`onnx_preflight(path, *, input_shapes=None, external_root=None, resource_budget=None)` does not accept root numeric values and does not construct the public graph/plan or execute components. Raw shapes, dtype, and root/weight/temp/host byte estimates, admitted/violations are retained in the result; additional temporaries introduced by lowering must still pass subsequent compilation budgets. The default reference_mode=execute includes the CPU reference; explicit inputs only validate all roots and storage rounding, execute zero numeric operators, and cannot be counted as a complete numeric pass. Real storage rounding for FP16/BF16, Cast traffic, and LayerNorm stash ordering are defined by the public workload/runtime.

`describe_onnx(spec)` provides a stable mapping from original nodes/values to Graph. Empty or duplicate display names do not affect identity; Gemm/non-final-axis Softmax may correspond to multiple Graph operators. A logical scalar [] is stored as Graph [1], with the original rank preserved. int64 shape literals are only for compile-time Reshape use; FP32 initializer/Constant are stored and read as real weights. Only literal Constants are folded; all other dynamic nodes, including dead nodes, retain their costs.

## Two Placements, Two Layouts, and Bounded Search

`onnx_scene(workload, *, candidate=0, layout='materialized', hardware=None)` returns the ADF, the original spec, and a MappingSpec. candidate is 0/1; layout is materialized/views. An existing hardware factory can be passed, following the `(adf, workload, mapping)` triple convention; mapping gives a formal path for compute and memory. The default synthetic Events board exposes two compute units and shared SRAM; all hardware parameters can be inspected in the serialized ADF.

`onnx_search_config(workload, *, candidate=None, layout=None, hardware=None)` by default selects two placement/tile × two layout candidates. The Graph's public operator IDs and registration rules determine the mapping; identity is not guessed from original display names. Tile candidates are (m2,k3,n4) and (m3,k2,n3); layout_candidates determine whether aliases are legal; reshapes that do not satisfy conditions fall back to materialization according to formal rules. Packing, broadcast, transpose, and real storage reads are billed separately; CORE does not copy their formulas.

Configuration uses the common Env, with seed 7 and an upper bound of 12 evaluation units. Capacity candidates are 1B/current capacity, bandwidth is current value/twice; search evaluates proposals, capacity compilation failures are also billed, and the ledger gives phase/valid/execution_attempted/error. Fixing candidate or layout reduces the candidate count, and the budget becomes three times the candidate count. `onnx_agent_config(workload, *, candidate=0, layout='materialized', hardware=None)` returns offline operator_mappings/evaluate/submit scripts, one evaluation unit, at most three steps and three tool calls.

## Console and Export Examples

`examples/onnx_import/models.py` generates real synthetic MLP and attention, complete root values, INPUT-SHAPES.json, and digests, with no precomputed intermediates. `--external-data` writes real MLP tensor files; `--symbolic-batch` writes complete specialization for symbolic roots; `--variant 0..1000` only changes explicit synthetic roots/initial values, not the model implementation selection.

```sh
python models.py --output input-models --external-data --symbolic-batch
archexplorer import-onnx input-models/mlp/model.onnx \
  --inputs input-models/mlp/INPUTS.json \
  --input-shapes input-models/mlp/INPUT-SHAPES.json \
  --external-root input-models/mlp --output WORKLOAD.json
python emit.py input-models/mlp/model.onnx \
  --inputs input-models/mlp/INPUTS.json \
  --input-shapes input-models/mlp/INPUT-SHAPES.json \
  --external-root input-models/mlp --output run-inputs
archexplorer compile run-inputs/SCENE.json --output COMPILED.json
archexplorer prepare COMPILED.json --output prepared
archexplorer evaluate prepared/COMPILATION-REQUEST.json --output RESULT.json
archexplorer search --config run-inputs/CONFIG.json --policy greedy --output BEST.json
archexplorer agent run --config run-inputs/AGENT-CONFIG.json --policy scripted --max-steps 3 --output AGENT.json
archexplorer evaluate BEST.json --output READONLY.json
archexplorer rerun BEST.json --work-dir rerun --output RERUN.json
```

For import-onnx, model, --inputs, and --output are required; --input-shapes/--external-root may be omitted, and --name defaults to onnx. emit.py takes the same parameters and, after a single import, outputs WORKLOAD.json, SCENE.json (adf/workload/mapping), CONFIG.json, AGENT-CONFIG.json, SCOPE.json. attention uses its own model.onnx and INPUTS.json; other user files use the same flow.

## Save and Restore

Exploration saves use `archexplorer.compilation-search-result.v1`; the winning row saves the actual effective mapping, compiled_result, evaluation, and diagnostics; the original request also retains evaluation_attempts[].requested_mapping. The complete model/external data in the compilation source, roots and input_shapes, complete instance rules, adapter/component bindings, and options all participate in the saved identity.

CLI evaluate BEST and `exploration.replay(saved)` strictly re-verify the source and content, allowing pure compile/plan reconstruction, and forbid actual prepare/engine/solver/writes or network access. The CLI writing the final JSON to --output is an explicit user output and is recorded separately. Env's read-only tool for saved results is `evaluate_request({'payload': saved})`, which does not consume evaluation units. An explicit rerun uses a new directory and actually bills execution.

This feature provides a bounded software frontend; the synthetic Events service has no hardware latency calibration, and the IPU required-incomplete and paper-scale gaps remain. For the official version specification and serialization entry points, see [ONNX 1.16.2 source](https://github.com/onnx/onnx/tree/v1.16.2) and [External Data specification](https://onnx.ai/onnx/repo-docs/ExternalData.html).
