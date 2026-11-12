# Explicit pre-LN encoder block

This example provides a complete synthetic evaluation block with batch 2,
sequence 3, hidden width 8, two heads and MLP width 16. Input values, weights,
biases and both normalization parameter sets are explicit in the generated
workload. They are teaching data, with no pretrained weights or calibration.
The block is noncausal, has no dropout and uses evaluation semantics.

The public workload builder owns the graph and precision boundaries. The scene
uses that builder and the existing component-bound compiler, task adapters,
mapping registry and search engine. The frontend adds no operator dispatch or
timing formula. See [the precision and block contract](../../docs/precision-transformer.md)
for the exact normalization, GELU, accumulation and rounding definitions.

Create source inputs and two independent client configurations:

```sh
python examples/transformer_block/scene.py --dtype bf16 \
  --output /absolute/path/to/run/block-specs.json \
  --search-config /absolute/path/to/run/search.json \
  --agent-config /absolute/path/to/run/agent.json
```

Use `--dtype fp32` or `--dtype fp16` for the other declared precisions.
`--input-case boundary` supplies values around precision boundaries;
`--input-case constant` supplies constant activation rows. `--inputs data.json`
accepts an explicit complete input dictionary instead of generated values.
The public builder checks its keys, dimensions and supported contract. All
generated files require distinct new paths. `--roofline` selects the existing
synthetic Roofline hardware rather than the Events teaching hardware.

Compile and prepare without execution, then execute explicitly:

```sh
archexplorer compile /absolute/path/to/run/block-specs.json \
  --output /absolute/path/to/run/compilation.json
archexplorer prepare /absolute/path/to/run/compilation.json \
  --options examples/transformer_block/provider-options.json \
  --output /absolute/path/to/run/new-request > /absolute/path/to/run/request.json
archexplorer evaluate /absolute/path/to/run/request.json \
  --output /absolute/path/to/run/result.json
archexplorer evaluate /absolute/path/to/run/result.json \
  --output /absolute/path/to/run/readonly-result.json
```

The final command checks the saved execution read-only. CPU graph reference
values and component scheduling results have separate evidence. The synthetic
event/Roofline times are not device measurements or native numerical output.

Run the existing bounded search or the offline Agent:

```sh
archexplorer search --config /absolute/path/to/run/search.json --policy sa \
  --output /absolute/path/to/run/best.json
archexplorer evaluate /absolute/path/to/run/best.json \
  --output /absolute/path/to/run/readonly-best.json
archexplorer rerun /absolute/path/to/run/best.json \
  --work-dir /absolute/path/to/run/new-winner-execution \
  --output /absolute/path/to/run/rerun-best.json
archexplorer agent run --config /absolute/path/to/run/agent.json \
  --output /absolute/path/to/run/agent-result.json
archexplorer evaluate /absolute/path/to/run/agent-result.json \
  --output /absolute/path/to/run/readonly-agent.json
```

The search has a six-unit evaluation budget, seed 0, two concrete mappings and
explicit SRAM capacity/bandwidth proposals. Compile or prepare rejections stay
in its attempt ledger and consume their reserved units. The one-byte capacity
proposal deliberately produces a rejected candidate. The winner saves the
producer's complete effective mapping, including every resolved rule, while
the requested mapping remains in the audit ledger. `rerun` starts a distinct
new execution and budget in a new directory.

Python clients can use `arch_explorer.archs.transformer.block_scene`,
`block_search_config` and `block_agent_config` directly. These return standard
ADF/workload/mapping objects or the same JSON configurations used above.
An unknown installed operator/mapping extension continues through its own
declared precision and layout contract; installing this example does not
silently expand an extension's capabilities.
