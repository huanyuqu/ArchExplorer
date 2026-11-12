# Public workload compilation and task adapters

The public compiler turns operator semantics and an explicit mapping into a
source-bound task graph. An adapter converts a declared group for its bound
component. The component owns resource state and timing; the runtime coordinates
dependencies and completion.

| Owner | Facts and behavior |
| --- | --- |
| Public workload and mapping | Tensor shapes and values, regions, reduction blocks, placement, movement, dependencies and last-reader lifetime |
| Task adapter | Compatible semantics and components, group input conversion, dedicated options and input restoration |
| Component backend | Compute, memory and network mechanisms, arbitration and service resources |
| Runtime | A single supported execution-domain graph, dependency ordering, capacity and completion |

The common graph retains materialized `compiled.matmul` and
`compiled.assemble` nodes, indices and explicit K blocks. Compilation is not a
FLOP/byte summary, an evaluation or a mapping search. Semantic validation does
not require a native SDK or an instruction emitter.

## Compile and prepare

Use the same public API from Python, CLI or Agent:

```python
from arch_explorer.archs.compilation import events_scene
from arch_explorer.workload.api import (
    AdapterBinding, compile_workload, validate_compilation, task_adapters,
)
from arch_explorer.simulator.api import (
    prepare_compiled, assess_request, evaluate_request,
)

adf, workload, mapping = events_scene(fanout=True)
compiled = compile_workload(adf, workload, mapping)
validate_compilation(compiled)
request = prepare_compiled(compiled, output_dir="new-request", options={})
assert assess_request(request)["supported"]
result = evaluate_request(request)
assert evaluate_request(result) == result
```

`compile_graph_v2` parameters are exactly `compute`, `memory` and `rules`.
Compute and memory may each be a path or a dictionary covering every original
operator name. Rules bind each semantic kind to an exact ID/version/options.
The teaching scene selects `matmul.tiled@1` with `options.tiles={m:2,k:3,n:4}`,
explicit numerical inputs and optional two-reader fanout. Region bounds,
accesses and inspection belong to that registered rule, not a compiler branch.
`workload.api.operator_mappings()` and CLI `operator-mappings` describe the
registered callbacks. The [attention tutorial](attention.md) uses additional
materialized elementwise, reduction, transpose and broadcast rules. New
compilation supports contiguous FP32/FP16/BF16 storage with FP32 accumulation
and declared output rounding. The [complete pre-LN block](precision-transformer.md)
adds materialized reshape/cast, affine normalization and tanh-GELU through
registered rules. One iteration is supported; other layouts, fusion and
cross-domain simulation require separate contracts.

Discover registered adapter metadata using
`workload.api.task_adapters()` or `archexplorer task-adapters`.
A component implementation explicitly declares its supported adapter IDs.
The compiler can select automatically only when exactly one declared adapter
is compatible. Missing, ambiguous, wrong-version, wrong-domain or unsupported
bindings reject before execution. Neither frontend maintains adapter names.

Explicit bindings map compute paths to `AdapterBinding(id, version, options)`
or its serialized dictionary:

```python
compiled = compile_workload(
    adf, workload, mapping,
    bindings={"board/core/sm": AdapterBinding("events", "1", {})},
)
```

The events and Roofline adapters accept empty options. For the component-owned
native teaching assembly, put the explicit thread count in its adapter binding:

```python
from arch_explorer.archs.compilation import native_scene
adf, workload, mapping = native_scene()
compiled = compile_workload(adf, workload, mapping, bindings={
    "component_gpu/cluster/sm": AdapterBinding(
        "accelsim", "1", {"threads_per_block": 32}),
})
```

The native adapter validates its supported cluster and complete striped DRAM
domain. Thread count has no default and is not a public mapping parameter.
Provider options passed to `prepare_compiled` are separate: engine/SDK,
scheduler and execution limits belong there. Preparation does not execute
native code, install dependencies or search mappings.

## CLI and Agent

`specs.json` contains `adf`, `workload`, `mapping` and optional
`bindings` / `implementations`. The latter selects exact registered operator
lowering identities; the public compiler and adapter validate them.

```sh
archexplorer task-adapters --output adapters.json
archexplorer compile specs.json --output program.json
archexplorer prepare program.json --output new-request > public-request.json
archexplorer evaluate public-request.json --output result.json
archexplorer evaluate result.json --output replay.json
```

`compile --bindings bindings.json` explicitly replaces the saved bindings
object. `prepare --options provider-options.json` forwards provider options.
Save prepare stdout as the public envelope and retain its preparation directory;
a provider's private filenames are not a frontend contract.

Agent `task_adapters` and `describe_environment.task_adapters` use the same
public metadata. `compile_workload` accepts `mapping`, optional `bindings`,
`implementations` and `workload_index`. `prepare_compiled` accepts the
artifact, a new `output_dir` and optional provider `options`. Neither uses
evaluation budget. `evaluate_request` charges one evaluation per attempted
request, including failures; saved-result validation is read-only and costs
zero. This separate workflow does not replace the current search candidate.

A process-local custom component Registry still cannot be persisted in a
saved compilation. The Agent passes its actual registry, and public compile
and prepare reject unsupported custom registries before writing output.
Publicly installed registered extensions are supported.

## Groups, memory and numerical evidence

The plan exposes each task's original operator, component, adapter and group.
A group shares compute, memory, adapter and options. Same-domain events and
Roofline groups execute in one runtime graph with their real dependencies.
The current events-domain composition requires one device and one actual
memory. Unsupported cross-domain combinations reject; separate simulations
cannot be summed and presented as common simulation.

An adapter marks component traffic as external or included. Events compute-only
components use actual mapped memory/network tasks. Roofline `component_io`
already includes its stated traffic boundary, so the runtime retains capacity
and lifetime without charging those same bytes again. Adapters contain no
service-time formula.

The fanout scene keeps `product` live through both `left` and `right`.
Inspect its actual readers, regions, selected memory and last-consumer release.
`archs.compilation.heterogeneous_scene()` places the producer and right reader
on the events SM and the left reader on the Roofline SM, sharing the same SRAM.
Its mapping uses original operator names; groups retain all mapped tile tasks.
Use `workload.api.execute_reference` to compare original and mapped numerical
outputs separately from timing. Native timing-only output does not establish
native tensor-value equivalence or hardware accuracy.

## Installed components and adapters

Install the main package first, then the independently installable
[external SM example](../examples/extensions/component_implementations/README.md).
It provides a compute-only component factory and an unknown task adapter,
registering through the public extension transaction. It reuses the public
events execution domain without adding a provider or editing a main-package list.

`TaskAdapter(id, version, execution_domain, semantics, lower, restore=None,
parameters={}, grouping="component")` owns group conversion.
`lower(AdapterContext)` receives the common program, selected task names,
placement, resolved component, options and operator implementation selections.
It returns finite JSON in `AdaptedGroup(payload, scope)`. It must not mutate
the common program or insert computed latency. An events-domain payload covers
all group tasks with `work` and `traffic`; `restore(work)` reconstructs only
the component's saved input. Component factories own the actual timing model.

`AdapterDomain(id, compose)` is needed only for a new execution domain.
Composition consumes explicit groups and returns `TargetInput` with exactly
the common program. Reuse a compatible installed adapter when available; users
of an existing adapter do not need to write instruction generation.

## Versioning and provenance

The current schemas are `arch-explorer.compilation.v3`,
`arch-explorer.compiled-request.v3` and `arch-explorer.compiled-result.v3`.
Saved identity covers semantics, mapping rules and their actual callbacks/options, exact component and adapter versions,
options, conversion/restoration callbacks, generated products and extension
sources. Changing a hash alone cannot authorize modified options or source.

Whole-graph `target`, CLI `--target`, `compiler-targets` and old
`CompilerTarget` registration are retired with explicit errors. They do not
form another conversion implementation. To migrate a saved v1 compilation:

```sh
archexplorer migrate-compilation-v1 old-program.json --output new-program.json
```

Python and Agent expose `migrate_compilation_v1` as the same explicit
recompilation. A saved v2 input uses `migrate_compilation_v2` or CLI
`migrate-compilation-v2`. Both produce a new current v3 artifact; old executed
results retain their original schema and producer identity. Old `compile_graph`
requests are not silently converted to registered rule requests.
Keep successful and failed historical receipts.

The separately tracked IPU EXP/RCP sourced mechanism costs remain
required-incomplete. That evidence gap does not block this adapter boundary
migration. Private papers, experiments, weights and calibration are excluded
from the public examples.
