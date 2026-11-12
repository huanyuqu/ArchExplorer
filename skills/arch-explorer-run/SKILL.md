---
name: arch-explorer-run
description: Use ArchExplorer to prepare, run, compare, and replay workload simulations on user-specified architectures, explaining result scope and failure causes. Applicable to existing components and workloads; when adding new operators or components, first check the corresponding extension interfaces.
---

# Running an Interpretable Simulation with ArchExplorer

Start from the user's architecture file, workload, and target problem. ADF describes hardware; the workload graph describes computational dependencies; mapping describes splitting, placement, and transfers; components own execution mechanisms; the simulator generates execution plans. `dmc_adf` is an example description factory, not an architecture class users must inherit.

## Locating Installation and Inputs

In the user's environment, run `python -m pip show arch-explorer` and `archexplorer --help`; confirm the actual import path. If not installed, use the user-selected distribution package or checkout. Do not mask uninstalled extensions by modifying PYTHONPATH. Project version numbers cannot replace capability checks.

Run `archexplorer components` and read the implementation versions, sources, inputs, and execution domains under hardware types; Agent uses the same
component_catalog. ADF explicitly binds implementations and their independent parameters, and does not arbitrarily pick the first implementation by component name. SM roofline is
an F/B/P/BW analytical service with explicit IO boundaries, while Accel-Sim SM is a native cluster containing L1/shared; inputs are not interchangeable,
and incompatible domains must not fall back. See `docs/component-implementations.md` and `examples/component_implementations/`.
New implementations of existing components can be installed as external packages and should not modify the CLI/provider list. Component capabilities and limitations are in `docs/component-inventory.md`.

Locate the user's ArchExplorer source distribution or documentation directory (hereafter the project directory), and read as needed:
- `docs/tutorials/03-architecture.md`: ADF files, parameters, units, topology.
- `docs/workload-extensions.md`: new graphs and installed workload extensions.
- `docs/component-extensions.md`: component/provider capabilities.
- `docs/models.md`: implemented scope.
- `docs/gpu-instantiation.md`: read when the task involves GPU component instantiation or differential comparison with original Accel-Sim.
If only a wheel is available without documentation, use the public documentation for the matching release; do not guess interfaces or claim missing capabilities exist.

Preserve user-specified shapes, dtypes, layouts, batches, precision conversions, and mappings. When parameters affecting results are unspecified, state assumptions explicitly or ask; teaching synthetic parameters must be labeled with sources. Hardware parameters include values, units, and sources: ns, GHz, B, B/ns; do not treat clock counts as ns.

## Using Public Entry Points

When you need to preserve common compute/mapping expressions and generate inputs for different targets, read `docs/compilation.md`.
Use `archexplorer task-adapters` to read the public catalog, then `archexplorer compile specs.json --output program.json`,
then `archexplorer prepare program.json --output <new-dir> > public-request.json`,
followed by public `evaluate` for execution/read-only replay. Python equivalents are `workload.api.compile_workload` and
`simulator.api.prepare_compiled`; use explicit legal tiles and AdapterBinding by component path.
Adapter-specific parameters go in bindings[path].options; GPU thread counts do not enter public mapping; execution provider options are passed separately to prepare.
Components can be auto-selected only when they declare a unique compatible item; ambiguity or lack of support must be rejected; do not add new target lists or silently fall back.
Ordinary Kernel/trace, count totals, and public compiled graphs cannot be interchanged, and compiled artifacts cannot be modified and then relabeled with old results.
Compile/prepare does not execute native or mapping search; actual execution, numerical reference, and hardware accuracy are evidenced separately.
The default example for unknown external public compilation is `python -m ae_component_implementations.compilation`,
install the matching main library first, then the independent external package; the exact command is in that package's README.

Existing scenario JSON contains `adf`, `workload`, `mappings`, `provider`, `options`, `scope`. Check capabilities according to the selected provider's documentation; do not arbitrarily replace the engine.

```sh
archexplorer demo prepare --specs scene.json --output run-001
archexplorer demo run run-001
archexplorer evaluate run-001/mapping-000/result.json --output replay-001.json
```

Select a new output directory, preserving the generation request and its referenced files. prepare only prepares; it does not mean simulation passed. After changing inputs, prepare again; on failure or an already-attempted directory, first explain the reason, then decide whether to create one new run according to the user's intent, without unlimited retries.

The corresponding Python entry points are in `arch_explorer.simulator.api`: prepare_request, assess_request, evaluate_request. First check `supported` and `gaps`; only None defaults to an empty options object; all other options values must be objects.

A complete GPU program requires an explicit external engine and its capabilities. CPU event-model results cannot substitute for GPU results; installing a provider does not mean it supports all workloads or cross-engine combinations.

For new graphs/operators, choose public target implementations according to `docs/operator-implementations.md`, and complete external installation first if necessary.
Missing implementations, ambiguity among multiple compatible implementations, and explicitly selected incompatibility are preparation errors; they are not eliminated by changing the model kind, switching providers, or falling back.
Preserve actual semantics/reference/execution ID versions and input sources; event timing, numerical reference, and native comparison are explained separately.

If the task is GPU component instantiation, verify whether the selected version truly connects SM/cluster, interconnect, L2, and DRAM through public ports,
and is advanced by a common transaction/clock framework. Running the same complete engine twice is not component instantiation. Construct the
`accelsim_components` scenario and an independent `accelsim` oracle scenario according to the matching version tutorial; do not rewrite native mechanisms in CLI or scripts.
When capabilities or construction interfaces are missing, clearly report the gap; do not fill it by renaming or fallback.

When the task involves reusing L2/DRAM/ICNT in non-GPU scenarios, first verify the matching version's independent creation factories, minimal services, and ownership contracts.
Request sources only use public transactions/ports, and do not create or borrow whole-GPU instances; responses exactly completing and physical draining are checked separately.
Actually run interleaved advancement and independent destruction of different configuration instances in the same process, preserving inputs, responses, service/component lifecycles, and package provenance.
Independent construction/reuse and GPU original behavior differential are two sets of evidence; passing either cannot substitute for the other; preparation or compilation success does not mean completion.

## Comparison and Delivery

For non-GPU native reuse, you can use [user-side dual-instance invocation](../../docs/standalone-components.md). Select the
matching SDK manifest and ICNT/memory archive target respectively, run `standalone.py` according to the example; preserve external invocation receipts and native statistics,
and additionally use the GPU public comparator to consume behavioral evidence; do not substitute one type of result for the other.

Bound results are checked item by item against architecture, workload, mapping, provider version, and actual task changes. Report latency_ns, resource/compute/transfer ranges, parameter sources, and unmodeled metrics. unknown/null are not filled with 0, and the latencies of two independent subgraphs are not added together and treated as a complete graph simulation.

Distinguish two types of "replay": evaluate of provider results should verify saved results and not re-execute; search/agent results have their own re-evaluation or replay protocols. Check the schema first; do not claim zero execution across the board.

GPU software differential uses the component package's public comparator, preserving both sides' versions/manifests, common trace/configuration/mapping, and scope.
Check per-kernel and overall counts, completion order, declared mechanism metrics, and physical draining; missing values are not filled with zero, and identical total time is not used
as a substitute for mechanism consistency. On mismatch, preserve the originals and specific differences; integer differentials under the same native semantics are not passed by fitting or percentage tolerance.

The new component-engine's main Evaluation.latency_ns represents physical draining, marked as physical_system_drain;
details.native's gpu_tot_sim_cycle/latency_ns(_exact) is the foreground, system_drained_* is the final value.
foreground_final and tail drain_ticks/drain_core_cycles are explained separately, and both differentials must pass. A clock tick is not equal to a
core cycle, and the tail is not an extra kernel; do not force the physical total to equal the per-kernel sum. Exit 0 or active=false is not sufficient to
prove draining; not drained still means failure. Old engine.v1/v2 retain the original latency contract; replay does not re-execute finish.

Deliver reproducible input paths, actual commands, result paths, and failures/limitations. Synthetic event timing, software functional pass, native program execution, and real device accuracy are different conclusions. No need to download weights or enable a physical GPU to complete pure CPU examples.
