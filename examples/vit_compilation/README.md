# Complete synthetic ViT through the public compiler

This example supplies explicit NCHW images and separate model parameters to
the public workload builder. The graph includes patch packing, projection and
bias, a leading CLS token, position encoding, every encoder block, final norm,
CLS selection and the classifier logits. The default uses batch 2, image size
4, patch size 2, width 4, two layers, two heads, MLP width 7 and three classes.
It generates deterministic synthetic data, with no pretrained weights.

From an installed source distribution or checkout, generate three documents:

```sh
python examples/vit_compilation/scene.py --layers 2 --dtype bf16 --layout views \
  --output /absolute/vit/source.json \
  --search-config /absolute/vit/search.json \
  --agent-config /absolute/vit/agent.json
```

`--layers 4` selects four independent layers. `--input-case boundary` selects
signed storage-rounding boundary values; `constant` gives constant image rows.
All parameters remain explicit and nontrivial. `--inputs roots.json` supplies
the complete root data instead: missing, extra or misshaped parameters are
rejected by the public builder. No activation is supplied between layers.
The source document records actual values, not merely a seed.

`--candidate 0` and `--candidate 1` select distinct per-instance placement and
nondivisible tiles on two compute points in the same domain. `--roofline`
uses the existing synthetic Roofline board; the default is Events. Both
keep FP32 arithmetic service for `fp32`, `fp16` and `bf16` storage. Conversion
is charged separately; storage dtype does not imply a throughput multiplier.

`--layout materialized` and `--layout views` ask the public helper for complete
mapping candidates of the same semantic graph. Unsupported sharing stays
materialized. Patch packing, concatenation and any necessary copies remain
explicit. A view removes its copy, while its consumers still pay for actual
reads and an exported view can keep its complete root allocation alive.

Compile, prepare and execute with the existing commands:

```sh
archexplorer compile /absolute/vit/source.json --output /absolute/vit/compiled.json
archexplorer prepare /absolute/vit/compiled.json \
  --options examples/vit_compilation/provider-options.json \
  --output /absolute/vit/new-request > /absolute/vit/request.json
archexplorer evaluate /absolute/vit/request.json --output /absolute/vit/result.json
archexplorer evaluate /absolute/vit/result.json --output /absolute/vit/readonly.json
```

Use fresh paths. Compilation and preparation do not execute an engine. The
last command validates saved results without scheduling new work; deterministic
source and numerical-reference validation may still run on the CPU.

The search compares both layouts and both placement/tile candidates under seed
0 and twelve evaluation units, including explicit capacity and bandwidth
proposals. A one-byte capacity proposal deliberately fails. Every attempted
candidate is charged, including compile/prepare failures. `--single-layout-search`
keeps only the selected layout, with two mappings and six units. The offline
Agent executes the selected layout and candidate with one evaluation unit.

```sh
archexplorer search --config /absolute/vit/search.json --policy sa \
  --output /absolute/vit/best.json
archexplorer evaluate /absolute/vit/best.json --output /absolute/vit/readonly-best.json
archexplorer agent run --config /absolute/vit/agent.json \
  --output /absolute/vit/agent-result.json
archexplorer evaluate /absolute/vit/agent-result.json \
  --output /absolute/vit/readonly-agent.json
archexplorer rerun /absolute/vit/best.json --work-dir /absolute/vit/new-rerun \
  --output /absolute/vit/rerun.json
```

The saved winner is a candidate that was actually evaluated; it keeps the
producer's full effective per-operator mapping and source identity. The attempt
ledger also retains requested choices. Readonly preserves that execution
identity. Explicit `rerun` creates a new request, directory and budget.

The [API and inspection guide](../../docs/vit-compilation.md) explains the
semantic boundary. This ViT uses exact erf GELU and epsilon 1e-12; the earlier
encoder example retains its tanh GELU and its own precision contract. CPU
reference agreement and synthetic Events/Roofline timing are separate evidence.
Views need not reduce peak memory or improve timing. Standard model scale,
official mixed-precision kernels, native/GPU execution, device accuracy, CLIP,
training and dropout are outside this example.
