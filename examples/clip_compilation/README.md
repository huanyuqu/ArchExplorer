# Complete synthetic CLIP through the public compiler

This example supplies preprocessed NCHW images, integer token IDs and explicit
parameters to one public two-tower graph. The graph includes image patches,
CLS/position encoding, every vision and causal text block, EOT selection,
feature projections, L2 normalization and scaled rectangular similarities.
Both `logits_per_image` and `logits_per_text` are returned. Inputs contain no
precomputed patch, embedding, activation or feature answers.

Generate source and ordinary configuration documents for the existing tools:

```sh
python examples/clip_compilation/scene.py --dtype bf16 --layout views \
  --output /absolute/clip/source.json \
  --search-config /absolute/clip/search.json \
  --agent-config /absolute/clip/agent.json
```

Defaults are image batch 2, text batch 3, image size 4, patch 2, width 4,
two vision blocks and two text blocks, context 4, vocabulary 13 and projection
width 3. `--layers 4 --text-layers 4` selects the four-layer control; both layer
counts are explicit and independent. `--inputs roots.json`
provides complete explicit root values; public validation rejects missing,
extra or malformed parameters. Token IDs remain integers, with no silent
truncation or conversion to the model's floating storage dtype. Synthetic
inputs include vocabulary endpoints, repeated tokens and maximum-ID ties.
EOT follows the first maximum token ID, not the last sequence position.
The source document saves all values, not just a random seed.

`--layout materialized` and `--layout views` consume the public layout helper
for the same semantic graph. `--candidate 0` and `--candidate 1` choose explicit
per-instance placement and nondivisible matmul tiles. `--roofline` selects the
existing synthetic Roofline board; Events is the default. No model-specific
CLI, provider, searcher or Agent tool is added.

Dimension flags are `--image-batch`, `--text-batch`, `--image-size`, `--patch`,
`--width`, `--heads`, `--mlp`, `--context`, `--vocab`, `--text-width`,
`--text-heads` and `--projection`. `--input-case boundary` supplies unrounded
image values around low-precision ties. Explicit inputs override generated data
as one complete dictionary; they are not supplemented with missing parameters.

```sh
archexplorer compile /absolute/clip/source.json --output /absolute/clip/compiled.json
archexplorer prepare /absolute/clip/compiled.json \
  --options examples/clip_compilation/provider-options.json \
  --output /absolute/clip/new-request > /absolute/clip/request.json
archexplorer evaluate /absolute/clip/request.json --output /absolute/clip/result.json
archexplorer evaluate /absolute/clip/result.json --output /absolute/clip/readonly.json
```

Compilation and preparation do not execute an engine. The last call validates
the saved execution without preparation or rescheduling. Deterministic source
and CPU reference validation may run during readonly checks. Preserve the
complete documents, their bound artifacts and the matching installed extension.

The bounded search compares both layouts, two instance configurations and
synthetic memory-capacity/bandwidth proposals. The default budget is twelve
units, including deliberate capacity failures. `--single-layout-search` keeps
two mappings and six units. The offline Agent executes one selected candidate
within one unit. All attempted candidates consume budget, including failures.

```sh
archexplorer search --config /absolute/clip/search.json --policy sa \
  --output /absolute/clip/best.json
archexplorer evaluate /absolute/clip/best.json --output /absolute/clip/readonly-best.json
archexplorer agent run --config /absolute/clip/agent.json \
  --output /absolute/clip/agent-result.json
archexplorer evaluate /absolute/clip/agent-result.json \
  --output /absolute/clip/readonly-agent.json
archexplorer rerun /absolute/clip/best.json --work-dir /absolute/clip/new-rerun \
  --output /absolute/clip/rerun.json
```

The saved winner was actually evaluated and contains the producer's full
effective mapping. Requested mappings and failure phases stay in the attempt
ledger. Fresh readonly preserves the old execution identity. Explicit `rerun`
uses a new request, directory and budget.

The [API/inspection guide](../../docs/clip-compilation.md) distinguishes CLIP's
QuickGELU and epsilon 1e-5 from the public ViT's erf/1e-12 and the earlier tanh
encoder. Floating storage supports FP32/FP16/BF16; token/EOT storage is int32
and logit scale remains FP32. Synthetic hardware still declares FP32 service,
with explicit conversion costs. BF16 is a declared new storage contract, not
an official mixed-precision or legacy-CLIP equivalence claim.

Lookup pays for actual indexed reads, while the complete embedding table still
occupies capacity. Repeated IDs do not imply cache reuse. A view removes its
copy but consumers still pay for reads, and shared returned outputs can retain
their whole root allocation. Views need not improve timing or peak memory.
CPU numerical values and Events/Roofline timing are separate observations.
Tokenizer/image preprocessing, pretrained weights, native/GPU execution,
standard-scale capacity, calibrated hardware accuracy, training and dropout
are outside this example.
