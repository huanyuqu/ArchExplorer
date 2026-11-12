# Explicit multi-layer encoder

This teaching scene supplies one model input and separate, nontrivial weights,
biases and normalization parameters for each block. It uses 2–4 pre-LN blocks;
the default is two blocks with batch 2, sequence 3, hidden width 8, two heads
and MLP width 16. Every layer receives the previous layer's actual output
through the public subgraph composition contract. Intermediate answers are
not supplied as external data.

Generate a complete source document and configurations for the existing search
engine and offline scripted Agent:

```sh
python examples/multilayer_encoder/scene.py --layers 2 --dtype bf16 \
  --layout views \
  --output /absolute/run/model.json \
  --search-config /absolute/run/search.json \
  --agent-config /absolute/run/agent.json
```

`--layers 4` selects the four-layer scene. Storage choices are `fp32`, `fp16`
and `bf16`; arithmetic service remains FP32. `--input-case boundary` includes
rounding boundaries, while `constant` uses constant activation rows. Explicit
`--inputs values.json` has exactly `X` and `layers`; the latter is a list of
sixteen-key parameter dictionaries with no `X`. Public builders validate their
shapes, values and precision contracts. The source file records all numerical
inputs, not just a seed. No pretrained model or hardware calibration is used.

`--candidate 0` and `--candidate 1` select two complete mapping configurations.
Each has different, nondivisible tiles and explicit compute placement by stable
operator instance. `--roofline` uses the existing synthetic Roofline component
in place of the Events component. These selectors do not change operator
semantics, precision boundaries or workload inputs.

`--layout materialized` (the source and Agent default) and `--layout views`
consume the producer's two layout candidates for this same workload. Views
share supported transpose/slice/reshape storage. An unshareable reshape stays
explicitly materialized. The example never computes strides or inserts an
unreported packing copy. To compare identical hardware and tiling directly,
generate two source files with different `--layout` and the same other options,
or use public `layout_candidates(workload, mapping)` as shown in the
[inspection example](../../docs/multilayer-encoder.md#compare-views-with-materialization).

Compile, prepare and execute explicitly:

```sh
archexplorer compile /absolute/run/model.json --output /absolute/run/compiled.json
archexplorer prepare /absolute/run/compiled.json \
  --options examples/multilayer_encoder/provider-options.json \
  --output /absolute/run/new-request > /absolute/run/request.json
archexplorer evaluate /absolute/run/request.json --output /absolute/run/result.json
archexplorer evaluate /absolute/run/result.json --output /absolute/run/readonly.json
```

Compilation and preparation do not execute a simulator. The final command
validates the saved result read-only. CPU reference values and component timing
are separate observations; synthetic component time is not device measurement.

Use the same source through bounded search and the offline Agent:

```sh
archexplorer search --config /absolute/run/search.json --policy sa \
  --output /absolute/run/best.json
archexplorer evaluate /absolute/run/best.json --output /absolute/run/readonly-best.json
archexplorer rerun /absolute/run/best.json --work-dir /absolute/run/new-rerun \
  --output /absolute/run/rerun.json
archexplorer agent run --config /absolute/run/agent.json \
  --output /absolute/run/agent-result.json
archexplorer evaluate /absolute/run/agent-result.json \
  --output /absolute/run/readonly-agent.json
```

The bounded search uses seed 0, four explicit mappings (two layout policies
times two tile/placement configurations), twelve evaluation units, and public
SRAM capacity and bandwidth proposals. `--single-layout-search` keeps only
the selected layout, two mappings and six units. The selected `--layout` and
`--candidate` come first in the search configuration; the Agent executes that
single choice within one evaluation unit. Rejected compile/prepare candidates still consume
their reserved budget. A one-byte capacity proposal intentionally cannot hold
this graph. The saved winner retains the producer's full effective mapping,
including instance overrides, while its attempt ledger retains the request.
Read-only validation keeps the execution identity; `rerun` creates a new request
and fresh budget in a new directory.

The public graph builder owns composition, operator semantics, rounding and
cross-layer dependencies. The mapping registry owns expansion, traffic and
rule selection. This example adds data and configuration only. The existing
single-block API and an independently installed FP32 extension retain their
declared contracts. Older independent-only external adapters explicitly reject
alias tasks; use a supported adapter or a materialized control. Sharing a view
removes a copy but still charges actual consumer reads; a small view can retain
an entire root allocation. Runtime peaks come from allocation/free events,
not a sum of logical tensor sizes. Views need not be faster or smaller.
Native/GPU execution, arbitrary striding or alias writes, dropout,
KV cache and hardware calibration are outside this example's scope.
