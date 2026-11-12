# External operator implementations

This independent package defines `example_shifted_square`, with FP32 semantics
`y = (x + shift) * (x + shift)`. Its graph has two independent branches, a merge,
and a final application of that operator. Inputs are deterministic small arrays;
exact rational arithmetic supplies an independent expected result.

The two event implementations use the existing `vector_unit` capability. One
returns a single vector kernel with two operations per element. The other
declares two dependent steps and an intermediate tensor. The ordinary public
`execution_graph` mapping handles placement, transfers and storage lifetimes.
There is no replacement provider or copied device state machine.

The same semantic has two `accelsim` implementations. They use the public
evaluation-local emitter to issue FADD/FMUL/load/store, with one kernel or two
kernels and a real global temporary. No model name is sent as an admission key.

## Install after the framework

This requires the matching open-execution API documented in
[the tutorial](../../../docs/operator-implementations.md). The earlier
single-lower API alone does not provide it. Build/install the framework first,
then install this separate package without changing the framework:

```sh
python -m pip install /ABS/FRAMEWORK/arch_explorer-0.2.0-py3-none-any.whl
python -m pip install /ABS/SOURCE/examples/extensions/operator_implementations
cd /tmp
python -m ae_operator_implementations generate --output square-inputs
python -m ae_operator_implementations check-reference --graph square-inputs/graph.json --output numerical.json
archexplorer demo prepare --specs square-inputs/events-scene.json --output square-events
archexplorer demo run square-events
archexplorer evaluate square-events/mapping-000/result.json --output square-events-replay.json
```

`graph.json` has explicit finite FP32 inputs in `meta.inputs`. The independent
rational oracle is limited to this sample graph and quarter-integer input range;
it is not a generic reference for an edited graph. Reference correctness and
event timing/resource correctness are separate evidence.

Both event mappings use round_robin placement and the same SRAM. They explicitly
choose one implementation for each shifted-square node. The two branches have
separate namespaced temporary lifetimes in the steps variant. Removing a choice
when both implementations are compatible should fail as ambiguous.

## Run on the existing component GPU

Use the unchanged, separately built component-engine manifest; no GPU capture
or new hardware model is needed. From outside the checkout:

```sh
python -m ae_operator_implementations gpu-case --graph square-inputs/graph.json --engine /ABS/ENGINE/engine.json --output square-gpu.json
python /ABS/SOURCE/examples/gpu_instantiation/workflow.py topology square-gpu.json --output square-topology.tsv
python /ABS/SOURCE/examples/gpu_instantiation/workflow.py run square-gpu.json --output square-native
python /ABS/SOURCE/examples/gpu_instantiation/workflow.py replay square-native --output square-native-replay.json
```

The optional `--adf user-gpu.json` reads an explicit component GPU ADF. Otherwise
the component package's public `gpu_adf` constructs it. No native parameter
rules are copied here. The case uses the same graph/input/configuration on both
independent processes, calls the existing public comparison authority, and
preserves its full reports. Mapping 0 selects one kernel per custom operator
and 32 threads/block; mapping 1 selects two kernels per custom operator with a
temporary and 64 threads/block. Both implementation and geometry change, so a
latency difference is not attributed to only one of them.

Targets are explicit: event vector kernels and SM90 scalar FP32/global-memory
emission on the existing supported single-device component GPU. Materialized
layouts only; no arbitrary dtype/ISA/topology promise. Primary native latency
is physical final completion; foreground counters and a real drain tail remain
separate. A tail is not a kernel. Each CPU native process is bounded by 600 s,
8192 MiB and 10 million core cycles including finish, with serial execution.

Hardware parameters are synthetic assumptions, not measurements. Program inputs
and traces are reference-generated; no real weights, framework CUDA capture,
hardware timing accuracy or full ONNX import is implied.
