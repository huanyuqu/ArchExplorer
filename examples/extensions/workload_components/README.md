# External branching workload example

This is an independent Python distribution, not part of `arch_explorer` package
discovery. Its `arch_explorer.components` entry point registers a new affine
operator, an optional workload builder, a synthetic array component and a row
mapping. It supplies **no provider**: the real `events` provider builds and
executes its resource/task graph.

Install a framework build exposing `arch_explorer.workload.api`, then install
this directory. From the main source distribution or checkout:

```sh
python -m pip install .
python -m pip install ./examples/extensions/workload_components
cd /tmp
python -m ae_workload_example generate --output branch-inputs
python -m ae_workload_example scene --adf branch-inputs/adf.json \
  --graph branch-inputs/affine-graph.json --mappings branch-inputs/mappings.json \
  --output branch-scene.json
archexplorer demo prepare --specs branch-scene.json --output branch-run
archexplorer demo run branch-run
archexplorer evaluate branch-run/mapping-000/result.json --output branch-replay.json
```

Use new output directories. No `PYTHONPATH`, main-source changes, model weights,
GPU queries or native engine are required. `generate` creates independently
editable ADF, graph and mapping JSON files. `scene` **reads those files** and
normalizes them through the public types. A JSON ADF is a description; it
becomes component instances and resources only when the framework prepares it.

`affine-graph.json` has two independent affine branches from `x`, an add join,
and a final affine. `tutorial_affine` implements `y = scale*x + bias` with an
optional ReLU. Negative/fractional coefficients and string activation attributes
are real operator semantics. It lowers to separate multiply/add SIMD
instructions, plus a compare/select for ReLU; it is not renamed matmul or a
constant latency callback. Graphs are shape/resource simulations, not numerical
tensor execution or model accuracy evidence.

`builtin-graph.json` is a new DAG using only existing scale/add definitions.
Select that file with the same `scene` command; no new model kind is needed.
The optional `tutorial_branches` builder produces the affine graph from
`rows`/`cols` and explicitly rejects `tp != 1`.

Two arrays share one explicit SRAM and bus. The synthetic array consumes
`launch_cycles + ceil(SIMD instructions / lanes)` cycles per tile; GHz converts
cycles to ns. Change the hardware separately:

```sh
python -m ae_workload_example hardware --lanes 8 --clock-ghz 2 --output faster-adf.json
python -m ae_workload_example scene --adf faster-adf.json \
  --graph branch-inputs/affine-graph.json --mappings branch-inputs/mappings.json \
  --output faster-scene.json
```

Both mappings use the same graph and ADF: `tile_rows=4` and `16` generate
different compute tasks and launch overheads. Inspect
`evaluation.details.tutorial_rows.placements`, task counts, memory peak and
provenance. Inputs start resident in shared SRAM; off-chip loading, SFUs,
matrix kernels, mixed dtype, aliasing, multiple iterations and TP are outside
this mapping's scope. Unsupported input fails rather than falling back.
Costs and hardware accuracy remain unknown.

For explicit registration in embedded Python, call `ae_workload_example.register()`.
The same registration objects may be reused; competing definitions must fail.
Registration never executes the workload. Replay validates the original receipt
and implementation sources without simulating again; retain the artifact paths.
The full guide is `docs/workload-extensions.md` in the main distribution.
