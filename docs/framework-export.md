# Real PyTorch Module Export and Ordinary ONNX User Flow

Framework export is an optional entry point. Validated versions are CPU PyTorch 2.5.1+cpu and ONNX 1.16.2, using the legacy exporter (dynamo=False), opset 17, inference mode, parameter export, and constant folding. It actually generates IR8; other IRs are explicitly rejected without changing the protobuf version number. Without PyTorch/ONNX, the core library, CLI, and comparator can still be imported. It is recommended to install exact versions in a separate environment according to the [official CPU installation table](https://pytorch.org/get-started/previous-versions/) and not to download pretrained weights.

The Python entry point `arch_explorer.archs.framework_export.export_torch(module, *, inputs, output_dir, output_names, input_names=None, dynamic_axes=None, external_data=False)` accepts an actual CPU `torch.nn.Module` or a user's TorchScript file. The object is copied first and then evaluated; original weights and training flags are preserved. Only all roots explicitly provided by the user are read; graphs are not replaced based on model names.

The value for each name in `inputs` is exactly `{dtype, shape, values}`. dtype is fp32/fp16/bf16/int32/int64/bool, shape is a complete list of positive integers with [] for scalars; values are complete finite values. Inputs are actually converted to the declared storage precision before export; low-precision roots are not FP32 metadata labels. Integer roots reject decimals, bools, and out-of-range values before tensor allocation; boolean roots require JSON bool; valid int64 is constructed directly from integers and saved as integers, including values greater than 2^53.

`input_names` is the module positional argument order and must exactly cover all roots. Python callers may omit it and use their own dictionary insertion order; serialized JSON flows should pass this list explicitly. The CLI must use `--input-names` and not rely on JSON object key order. `output_names` must provide all distinct outputs; after generation, original root names and output names are verified item by item, and roots cannot be silently deleted or outputs omitted.

```sh
archexplorer export-torch module.pt --inputs TORCH-INPUTS.json \
  --input-names X R --output-names Y hidden --external-data \
  --dynamic-axes AXES.json --output-dir exported
```

AXES.json has the form `{"X":{"0":"batch"},"R":{"0":"batch"},"Y":{"0":"batch"},"hidden":{"0":"batch"}}`. Dynamic axes must be within the actual port rank; input axes are checked first, and after export it is verified that the symbols of all declared ports are indeed preserved. INPUT-SHAPES.json saves the complete actual shape of each original root and specializes by original root name on import. Symbols cannot lift the model's position table, vocabulary, or spatial size constraints. external_data uses the official ONNX save_model to place real initialization weights into weights.bin, with location relative to model.onnx; Constant nodes retain the exporter's original representation.

The output directory must be newly created and contains model.onnx, optional weights.bin, INPUTS.json, INPUT-SHAPES.json, TORCH-INPUTS.json, OPERATORS.json, and EXPORT.json. OPERATORS lists actual node/domain/opset/attributes/original names and initializer dtype/shape. EXPORT saves fixed export options, dependency versions, and file digests; after an export failure, it also saves the original type and diagnostics in a finally block. It records inputs and does not contain numerical answers.

Call boundaries such as root types, shapes, and input order are checked before creating the output directory; rejections at this point are saved by the caller as the original exception or CLI log, and no EXPORT file exists yet. Failures that have entered the export phase are recorded by EXPORT.json.

`import_export(directory, *, name='onnx', **import_options)` first runs the common onnx_preflight, then loads numerical roots and hands them to the sole official ONNX importer; a complete strategy can be passed via `resource_budget`. Omitting the strategy retains the old small-package representation. The specific supported domains, estimated resources, storage rounding, and lowering are determined by the workload's interface. Afterwards, `onnx_scene`, `onnx_search_config`, `onnx_agent_config`, `layout_candidates`, and the ordinary Env/CLI are reused:

```sh
python emit.py exported --output inputs --max-evaluations 12
archexplorer compile inputs/SCENE.json --output COMPILED.json
archexplorer prepare COMPILED.json --output request
archexplorer evaluate request/COMPILATION-REQUEST.json --output RESULT.json
archexplorer search --config inputs/CONFIG.json --policy greedy --output BEST.json
archexplorer agent run --config inputs/AGENT-CONFIG.json --policy scripted --output AGENT.json
archexplorer evaluate BEST.json --output READONLY.json
archexplorer rerun BEST.json --work-dir new-run --output RERUN.json
```

The two-layer vision/text dual tower exceeds the importer's default 256-node limit. For the small models in this directory, first save the following explicit strategy as SMALL-BUDGET.json; it declares preflight and execution resource limits and is saved with the self-contained source. Other models should choose according to their own resource constraints.

```json
{
  "schema": "arch-explorer.resource-budget.v1",
  "max_dimension": 8192,
  "max_host_bytes": 8589934592,
  "max_matrix_fmas": 100000000,
  "max_nodes": 4096,
  "max_rank": 4,
  "max_root_bytes": 33554432,
  "max_source_bytes": 16777216,
  "max_steps": 32768,
  "max_tensor_elements": 4000000,
  "max_tiles": 4096,
  "max_total_tensor_bytes": 268435456,
  "reference_mode": "execute"
}
```

Run `python emit.py exported --output inputs --candidate 1 --layout views --max-evaluations 3 --resource-budget SMALL-BUDGET.json` to restrict a set of placements and layouts. The defaults `candidate=None, layout=None` cover two placement groups and two layouts; restrictions only filter public configurations and do not change the Graph, values, or cost. SCENE and a single Agent use the selected combination, defaulting to 0/materialized when unspecified. max_evaluations is always an upper bound and is also limited by the original budget of the selected configuration; 3 units allow one baseline, one capacity rejection, and one bandwidth proposal; whether they actually execute and succeed is determined by the ledger. Omitting resource_budget retains the original default limit; rejection text must be preserved, and over-limit calls cannot be recorded as successes.

The budget is an upper limit on evaluation units and does not represent the number of different candidates. The default configuration includes two per-original-operator placement/tiling groups and two layouts, and explores explicit capacity and bandwidth; failures are also accounted for according to the original Env contract. The saved winner carries all valid rules, sources, inputs, shapes, budget, adapter/implementation, and compilation identity. Fresh reads perform pure parsing/compilation validation and separately verify artifact preparation, engine, solver, and writing; explicit rerun actually executes in a new directory and is billed separately.

Framework forward, official ONNX evaluator, independent CPU reference, mapped CPU, component timing, and installation acceptance each retain evidence. Completion of BF16 official evaluator execution does not mean numerical pass; original differences or unsupported diagnostics should be preserved, and incorrect results cannot be treated as an oracle. Original private answers, paper measurements, and hardware calibrations do not enter examples or public exports.

`vision_language.py` provides fixed-seed actual vision classifiers and image-text dual towers. The vision input is [2,3,4,6], Conv patch is [2,3], class token/position and two complete layers followed by a 3-class head; the dual tower additionally has int64 tokens[3,4], two causal text layers, row selection of the first maximum token, two projections/L2 normalization, and FP32 exp scaling, with outputs [2,3] and [3,2]. The same semantics are provided in FP32/FP16/BF16; exact dimensions and synthetic parameters are governed by the MODEL/original files. The small complete structure does not represent pretrained accuracy or paper scale.

The BF16 patch explicitly converts to FP32 in the real framework module for Conv, then converts back to BF16; ONNX opset17's [Conv type constraints](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11) do not allow direct BF16 Conv. Export constant folding may make patch weights FP32 initializers, and consumers are billed according to the actual file; protobuf dtypes cannot be rewritten and necessary Casts cannot be removed. Diagnostics of native BF16 export rejected by the official checker should be preserved and listed separately from the success of the explicit path.
