# External Operators: Semantics, Target Implementations, and Graph Mapping

First read the user's ADF, graph, and inputs, then select target capabilities and execution implementations. The mathematical rules for new operators live in the user package; the execution implementation is responsible for turning them into artifacts supported by a target; mapping selects implementations, placement, movement, and storage; components and the simulator handle real mechanisms and scheduling. Adding a new operator does not require adding name branches to GPU providers, schedulers, or the CLI.

The [standalone package](../examples/extensions/operator_implementations/README.md) in this document defines `example_shifted_square`, unknown to the main library: FP32 `y=(x+shift)*(x+shift)`. The two branches have shifts of 1.25 and -0.5 respectively, merge at the add, and then compute the final output with shift=0.25. The graph name is an ordinary user name and does not select a built-in model. Synthetic inputs of quarter integers allow exact expected values to be generated using independent rational arithmetic, without depending on the device's timing model.

## Installation and Minimal Event Flow

The main package must provide the public `ImplementationDefinition`, `ExecutionProgram`, and `execution_graph` simultaneously. The old single-lower interface alone does not provide this capability. First install the matching main package, then install the external package separately:

```sh
python -m pip install /ABS/FRAMEWORK/arch_explorer-0.2.0-py3-none-any.whl
python -m pip install ./examples/extensions/operator_implementations
cd /tmp
python -m ae_operator_implementations generate --output square-inputs
python -m ae_operator_implementations check-reference --graph square-inputs/graph.json --output numerical.json
archexplorer demo prepare --specs square-inputs/events-scene.json --output square-events
archexplorer demo run square-events
archexplorer evaluate square-events/mapping-000/result.json --output square-replay.json
```

The framework is built first, and the external package is installed afterward; the main library, providers, and scheduler do not need to change because of this operator. The entry point is `ae_operator_implementations:register` in `arch_explorer.components`, which only registers and does not execute. Throughout the process, do not set PYTHONPATH, do not insert source directories, and do not monkeypatch. The generated inputs are ordinary JSON and can be edited and re-prepared.

`adf.json` uses the existing vector_unit, SRAM, DRAM, and bus. All parameters carry units and synthetic sources; no custom provider is created. `graph.json` is a strict topological graph with branches/merges; `inputs.json` and `expected.json` are small arrays and independent numeric expectations. `mappings.json` contains two explicit implementation selections; `events-scene.json` is a complete scene for the generic demo. After changing the ADF, re-read it first, then prepare the request; event performance results themselves do not compute tensor values.

## Registering Semantics and Implementations Separately

```python
from arch_explorer.workload.api import register_operator
from ae_operator_implementations import SEMANTIC

register_operator(SEMANTIC)  # reuses the same stable object as auto-discovery; identity is idempotent
```

The external package's module-level object is constructed via `OperatorDefinition(OPERATOR, validate, version='1', reference=reference)`. Semantic checks validate input/output shapes, FP32 dtype, and finite shift. reference receives read-only ndarrays in input order and returns an output ndarray; it does not schedule or select components. Numerical reference correctness, timing/resource correctness, and native software consistency are proven separately and cannot substitute for one another.

Execution implementations have stable IDs/versions, targets, dtypes, requirements, and supports. The two events implementations in this example:

- `example.shifted_square.single`: one vector Kernel, two SIMD operations per element; it does not disguise two operations as a single FMA.
- `example.shifted_square.steps`: two dependent steps, shift and square, declaring one internal temporary Tensor.

Both require the existing `kernel:vector` capability and support FP32 materialized layout. Real components must still accept the actual Kernel; capability tokens do not replace shape/mechanism validation. Internal temporary names are isolated per node by the parent mapping; branches do not share the same buffer. Temporary capacity, last consumer, and release go through the normal storage plan; the user package does not duplicate the device state machine or add a fixed delay.

The old `OperatorDefinition(name, validate, lower)` remains compatible as an events `legacy:<op>` implementation. The new package separates semantics from target implementations; the same semantics can have multiple implementations, and registration order cannot determine the result.

## Public Policy Selection and Error Diagnostics

```python
from arch_explorer.adf import MappingSpec

mapping = MappingSpec('execution_graph', {
    'implementations': {
        'left_square': 'example.shifted_square.steps',
        'right_square': 'example.shifted_square.steps',
        'final_square': 'example.shifted_square.steps',
    },
    'placement': 'round_robin',
    'serialize': False,
    'memory': 'operator_board/shared',
})
```

`implementations` are selected per real graph node, not by an operator-name whitelist. When two compatible implementations exist, omitting the selection must reject ambiguity; a nonexistent explicit ID, mismatched target, missing capability, or unsupported dtype/shape/layout must also be rejected. Diagnostics should retain the node, implementation, and target; after correcting inputs or installing the correct implementation, re-prepare in a new directory.

`placement` is optional first/round_robin; `serialize=False` preserves DAG parallelism, and True is an explicit serial policy. `memory` is optional, defaulting to that device's DRAM. In this example, both implementations use the same round_robin/non-serial/SRAM, so differences come from actual steps, temporaries, and storage/transfers, not from changing report labels. Specific times and peaks are based on the actual plan. Special policies can still be extended, but ordinary single-device graphs do not need to rewrite run_device.

## GPU, Sources, and Boundaries

GPU likewise selects a public target implementation with target=`accelsim`; the component provider consumes the executable program and source. The external lower obtains the ProgramEmitter for this evaluation from `context.options['emitter']`; it does not enter the request JSON. This package first loads inputs and the constant shift, then executes FADD/FMUL and finish. The multi-step version connects via `temporary('shifted', shape)`, two kernels, and real store/load; it is not a pseudo-kernel or estimated time. reference remains the same ordinary semantic callback. The example uses R8/R9, avoiding the emitter's index and address scratch registers.

`graph.meta['inputs']` explicitly provides finite numeric arrays for all external inputs/weights; there is no implicit random filling. The public `workload.reference_api.execute_reference` executes the complete graph; check-reference compares against independent rational expectations for both branches, the merge, and the output. After modifying the graph or inputs, a matching reference must be provided; the sample oracle cannot be reused.

```sh
python -m ae_operator_implementations gpu-case --graph square-inputs/graph.json --engine /ABS/ENGINE/engine.json --output square-gpu.json
python /ABS/SOURCE/examples/gpu_instantiation/workflow.py run square-gpu.json --output square-native
python /ABS/SOURCE/examples/gpu_instantiation/workflow.py replay square-native --output square-native-replay.json
```

Here `/ABS/SOURCE` is the matching public source-export/sdist; the workflow reuses the original CORE component GPU comparison entry. The two mappings are single-core/32 threads and multi-core temporaries/64 threads respectively; both factors change, so no single-factor performance attribution is made. Input graphs can be named arbitrarily; new semantics and implementations for both targets remain in the external package, without changing kind to pretend to be ViT/CLIP, and without writing a second provider to bypass. Existing ViT/CLIP use their own graph/reference adaptations through the same public mechanism.

Requests and results should bind the actual semantic validate/reference, execution lower/support/version, mapping, and input bytes. After the installed package source changes, old results must be rejected on read-only replay; a new plugin cannot be silently selected. Duplicate registrations and failed plugins must follow discovery's conflict/rollback rules. Preparation failures should not leave provider run outputs; missing sources cannot be zero-filled or ignored.

Native component/oracle comparison requires a separate process and output; foreground per-kernel and physical final/tail are checked separately. The new component-engine main latency is physical completion; retain foreground_final/drain_ticks/drain_core_cycles, and do not treat the tail as an extra kernel. The CPU native budget includes finish, with at most 600 seconds/8 GiB per run; GPU is not collected. Synthetic reference-generated traces are not framework CUDA real captures, pretrained weights, or hardware accuracy validation.
