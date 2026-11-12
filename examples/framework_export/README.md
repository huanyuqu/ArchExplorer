# Reproducible Framework Inputs

`python models.py --output models [--reference-output PRIVATE_DIR]` uses fixed seed34 and deterministic synthetic weights to actually export MLP and decoder, each with FP32/FP16/BF16 and embedded/external variants. Symbolic batch is paired with complete original root shapes, and the directories must not exist. The decoder includes projection, two LayerNorms, residuals, exact GELU, and explicit storage casts for FP32 softmax.

The input package contains only the model, weights, root values and shapes, actual node list, and export identity. Optional framework forward outputs are written to a separate PRIVATE_DIR and do not enter the answer-free inputs. The example calls the public export_torch and does not construct ArchExplorer graphs.

`vision_language.py` actually exports a complete small vision classifier and an image-text dual tower. Fixed seed35, synthetic weights, CPU batch of 2 images/3 text groups; Conv patch, two-layer attention blocks, class/position tokens, and classification head are complete. Text uses int64 lookup, causal mask, and first max-token row selection; dual-tower projection and normalization produce 2×3 and 3×2 logits. FP16/BF16 are actual inputs and module storage; FP32 boundaries for LayerNorm/softmax, final L2, and logit_scale are explicitly preserved.

```sh
python vision_language.py --family vision --dtype fp32 --output vision-fp32
python vision_language.py --family dual --dtype fp16 --external-data --output dual-fp16
python vision_language.py --family dual --dtype bf16 --external-data --output dual-bf16
```

The BF16 patch is explicitly converted by the module to FP32 Conv and then cast back to storage, because the standard Conv in opset17 does not include BF16 types. The actual dtype and Cast of folded initializers are determined by the file. The official checker rejection of native BF16 Conv is documented separately from the supplementary path; the IR or dtype labels are not modified. `--reference-output PRIVATE_DIR` saves the real torch answers separately, and the model directory remains answer-free.

`emit.py DIRECTORY --output NEW_DIR [--max-evaluations 12] [--candidate 0|1] [--layout materialized|views] [--resource-budget SMALL-BUDGET.json]` reads the actual export directory and emits ordinary SCENE/CONFIG/AGENT-CONFIG JSON files; then use the existing compile/prepare/evaluate/search/agent/rerun commands. By default it explores two placement groups and two layouts; specifying candidate/layout can limit a complex model to one bounded consumption. max_evaluations is a budget cap; failures are still accounted for. The supported domain is determined by the public importer; a successful export does not bypass diagnostics for unsupported nodes.

The two-layer dual-tower export exceeds the default 256-node limit. First save an explicit SMALL-BUDGET.json according to the [framework entry](../../docs/framework-export.md), then run `python emit.py dual-fp16 --output inputs --candidate 1 --layout views --max-evaluations 3 --resource-budget SMALL-BUDGET.json`; this strategy preflights before root allocation and saves it along with the WorkloadSpec and winner. Then pass inputs/CONFIG.json to the general search and inputs/AGENT-CONFIG.json to the offline scripted Agent. The winner saves the full rules and self-contained model/external sources; evaluate BEST.json is read-only validation, and rerun BEST.json --work-dir NEW is a new execution. See the links below for complete commands and stage boundaries.

For user modules and the TorchScript CLI, complete fields, and replay flow, see [framework entry](../../docs/framework-export.md).
