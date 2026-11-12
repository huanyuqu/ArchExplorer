# Standard ONNX File Examples

`models.py` uses official ONNX helpers to generate a small MLP and attention. The generator only provides the model, complete run roots, dimension specialization, and file summaries; it neither imports the ArchExplorer parser nor computes activation answers. The models are standard domain, opset 17, IR 8, FP32; both graphs have multiple outputs, the MLP shares initial weights, and attention uses batched matrix multiplication, non-default transpose, scaling, finite mask, softmax, int64 reshape constant, and identity.

After installing optional dependencies, use the export example outside the repository:

```sh
python models.py --output input-models --external-data
python emit.py input-models/mlp/model.onnx \
  --inputs input-models/mlp/INPUTS.json \
  --external-root input-models/mlp --output mlp-run-inputs
```

`--symbolic-batch` generates an explicit `INPUT-SHAPES.json` for each model; when consuming, add `--input-shapes input-models/mlp/INPUT-SHAPES.json`. The attention model and roots are located in `input-models/attention/`. User-owned models are imported the same way through model, complete roots, and allowed external-root; the consumption entry point does not select templates by model name or file name.

`emit.py` imports once and writes the same ordinary WorkloadSpec into `WORKLOAD.json`, `SCENE.json`, `CONFIG.json`, and `AGENT-CONFIG.json`. Then use the existing general commands:

```sh
archexplorer compile mlp-run-inputs/SCENE.json --output compilation.json
archexplorer prepare compilation.json --output prepared-request
archexplorer evaluate prepared-request/COMPILATION-REQUEST.json --output result.json
archexplorer search --config mlp-run-inputs/CONFIG.json --policy greedy --output best.json
archexplorer agent run --config mlp-run-inputs/AGENT-CONFIG.json \
  --policy scripted --max-steps 3 --output agent-best.json
archexplorer evaluate best.json --output readonly-best.json
archexplorer rerun best.json --work-dir rerun --output rerun-best.json
```

The search's two placements/tiles and two layouts share a 12-unit budget and explicit capacity/bandwidth candidates; the offline Agent spends one evaluation unit. The example hardware parameters are all visible synthetic configurations and do not represent measured latency accuracy.

Pure reading and explicit rerun are separate: the former verifies provenance and allows pure compile/plan reconstruction, while the latter actually executes in a new directory and is billed. The original model, external tensors, root values, dimension specialization, and importer identity are preserved by the formal provenance protocol; detailed behavior is in `docs/onnx-import.md`. The supported scope and rejection domain are determined by the formal import contract of the corresponding candidate.
