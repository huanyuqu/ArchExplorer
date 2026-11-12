---
name: arch-explorer-component
description: For ArchExplorer, integrate or reuse compute, storage, interconnect, stateful components, and external simulator providers, verifying creation, composition, and execution capabilities through public factories or independent extension packages.
---

# Component and Executor Integration

Read the matching version's `docs/component-extensions.md`; for ordinary components, refer to `examples/extensions/synthetic_components/`; for state models, also read `docs/unified-execution.md`. Import external types from `arch_explorer.components.api`, use public mapping interfaces, and do not directly call private scheduler implementations.

ADF is a description of component instances, parameters, and connections; ComponentType is a type declaration; build returns the runtime model for each build. Components should only know their own requests and capabilities, and should not depend on complete model names like ViT/CLIP.

First run `archexplorer components` and read `docs/component-implementations.md`, find components and existing implementations by hardware responsibility. ComponentType owns hardware identity, common parameters, and structural rules; to add a new execution method for existing hardware, use
`components.api.ComponentImplementation/register_component_implementation`, without creating a new hardware identity or provider.
ADF's ImplementationBinding precisely records the implementation ID, version, and implementation parameters with units/sources, separate from common parameters.
Catalogs, CLI, and Agent all consume the public component_catalog; do not maintain implementation whitelists or branch scheduling by source.

SM's roofline requires explicit F/B/P/BW and component_io boundaries; ordinary Kernels cannot infer B; native SM is currently
a cluster containing L1/shared, and should not therefore claim an independent L1 factory. Check each implementation's own required parameters, execution domain, and inputs.
New compute-only implementations can reuse external memory/network transfers; IO already accounted for in the roofline cannot be charged twice.
Independently installable examples are in `examples/extensions/component_implementations/`, and component capabilities and limitations are in
`docs/component-inventory.md`. Implementation parameters/versions must be preserved in serialization, fingerprints, and source replay.

Public compilation examples select adapters through `compile_graph` and explicit component relationships, without copying models/compilers or adding new providers.
Implementation adapters declare reusable adapters; only uniquely compatible items can be auto-selected, and explicit selection uses a dictionary from component paths to AdapterBinding.
New transformations publicly register `TaskAdapter`; lower consumes in-group tasks and returns `AdaptedGroup`; restore only restores component inputs.
Reuse existing execution domains; register `AdapterDomain` only for new domains; compose preserves the same public graph. Component models exclusively own latency/resource formulas.
Read the public mapped Workload; do not call built-in adapters' private lower; the old CompilerTarget entry is explicitly retired.
Component input/billing boundaries must be compatible;
traffic already accounted for in roofline component_io cannot be charged again as external DRAM/ICNT at the same boundary.
See `docs/compilation.md` for details; do not claim automatic bridging for arbitrary hardware/targets.

## Choosing Execution Boundaries

If a new operation only needs existing component capabilities, first use the semantic/target execution registration in `docs/operator-implementations.md`.
OperatorDefinition owns mathematical rules and reference; ImplementationDefinition owns lower/support;
ordinary multi-step/temporary quantities are handled by public graph policies. Do not add operator/model name branches to components, providers, or schedulers for this reason.

1. **Resource demand model**: ComponentType declares ParamSpec, Conventions, and capabilities; build returns ComputeModel/MemoryModel/NetworkModel, etc. Compute kernel_demand, memory access, and network route produce Demand, with dependencies and contention handled by the framework. When existing base classes can express it, no need to write your own provider.
2. **Stateful model**: Internal queues, backpressure, pipelines, or state-dependent latency use submit/next_event_time/advance. Bind a new instance for each evaluation according to the documentation; shared controllers use the same instance. Rejected submit must have no side effects; consumer_ready, physical_complete, resource_ready, and queue_available represent different events and cannot substitute for each other.
3. **Independent engine provider**: Complete engines or component groups that must execute jointly implement prepare_request/assess_request/evaluate_request, declaring id/version, component scope, single or joint execution domain, request and result schemas. prepare does not execute simulation; assess explicitly rejects capability gaps; evaluate saves complete request/source/result bindings.

The first two are component behavior expressions; provider is the execution domain boundary. External engines can also be integrated through state models if they can satisfy the state event contract.
Dynamic component networks also require request ID/parent-child relationships, ports, response matching, queue credits, clock domains/same-time phases, and drain contracts;
first confirm that the matching version provides these public capabilities; do not infer support from submit/advance alone, and do not privately advance other components.

## Registration and Verification

When directly reusing existing Accel-Sim native components, first read the [standalone invocation tutorial](../../docs/standalone-components.md)
and the matching version's public factory contract. The [external invocation programs](../../examples/gpu_instantiation/README.md)
consume ICNT and memory static libraries separately; they are not new providers or replacement mechanisms. Install the main wheel first; the public
`native_sdk.load_sdk/compile_consumer/run_consumer` verifies the SDK and links and executes outside the repository;
preserve construction audit, public receipts, and pre/post-installation identity, and record reuse results only after asserting actual execution passes.

Use `register_component` for new hardware types; use `register_component_implementation` for new implementations of existing hardware;
use `register_provider` only for new execution domain providers. These are different from workload's ImplementationDefinition. Independent packages declare entry points:

```toml
[project.entry-points."arch_explorer.components"]
my_extension = "my_extension:register"
```

A parameterless register() only registers descriptions/factories. Use discover_components() to check loaded and failures; failed extensions should not implicitly enable alternative models.

Parameters have units and sources; validate rejects out-of-range and unsupported precision. Components own execution resources; mapping owns splitting, placement, transfers, and lifetimes. Resources already owned by state execution cannot be charged again by outer Demand.

After installing outside the repository, build an instance and execute a minimal supported request, checking resource usage and expected event order; then verify concurrency/rejection or another parameter configuration. Check both actual runs and replays. Explicitly state unmodeled storage, numerics, power, or physical precision; null is not filled with 0.

Accel-Sim's complete joint path and component instantiation path are verified separately. Before adopting the component path, read
`docs/gpu-instantiation.md` and `docs/accelsim-componentization.md`, verify actual ADF port connections,
component independent driving capability, and general routing/clock evidence. It is forbidden to time each component separately and add them up, or to wrap the whole-engine run as each component.
L2/memory should be drivable by non-SM request sources, and the general protocol must not expose mem_fetch pointers or GPU global context; internal native dependencies can remain in the adapter.
Exact integer differentials, backpressure without side effects, no transaction loss/duplication, and system_drained all require empirical proof; they cannot be inferred from equal total time.

When reusing existing native components, verify that the public factory truly creates and owns instances, local state, and destruction, rather than borrowing objects from the whole machine.
Read that version's service dependency and lifecycle contract; memory/interconnect callers only provide their configuration and necessary services, and should not create a GPU host,
SM/kernel/CTA environment, or closures capturing the whole machine. Components and the general Runtime each retain authority; user request sources are only responsible for traffic and response accounting.
Interleaved advancement of two different configurations in the same process, with one destroyed and the other continuing, is required to demonstrate instance isolation; two processes cannot replace this verification.
GPU reassembly must use the same factory. Verify by combining construction call chains with link isolation/construction probes; a single self-reported zero count is insufficient proof.

Foreground completion does not equal physical drain; after active=false or exit=0, there may still be in-flight queue/DRAM work. Preserve the original foreground
foreground_final, then advance the real tail to drain, separately listing drain_ticks/drain_core_cycles/final, without modifying foreground
kernel statistics, fabricating new kernels, or clearing queues. The new component-engine main latency takes the physical end and annotates
physical_system_drain, with foreground and final counts verified separately; the budget covers foreground and tail, and replay does not execute finish.
