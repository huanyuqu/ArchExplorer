# Unified execution: Demand and stateful components

`simulate(graph, scheduler="conservative")` executes the real global earliest-event
loop. The default remains `consistent`. Demand-only nodes retain weighted dominant
fair sharing, FIFO/exclusive ownership, downstream priority, resource-free latency
tails, calendars, storage lifetime and iteration dependencies. Independent branches
progress concurrently; a change in global time is not a synchronization barrier.
The new driver shares allocation algebra and graph bookkeeping with the old driver,
but never calls ConsistentScheduler or its speculative batch/rollback loop.

## Consistent and conservative scheduling

The default `ConsistentScheduler` implements contention-stash scheduling for
Demand tasks: evaluate tasks sharing resources together, split their remaining
work, retain tentative results, and commit only when future competing tasks'
earliest-start bounds make it safe. A newly activated competitor can roll back
uncommitted batches for reevaluation. The component evaluator determines
arbitration; the scheduler determines which tasks overlap and when results can
be committed. Source coverage and independent correctness checks are separate
evidence, so a rollback statistic alone does not establish correctness.

The `conservative` driver advances stateful requests without that speculative
loop. Its lack of arbitrary native-state snapshots does not imply that the
Demand scheduler lacks rollback, and using the conservative path does not
claim that a stateful component executed the speculative algorithm.

## State-model protocol

Import ExecutionRequest, ExecutionEvent, ExecutionModel, DemandAdapter from
`arch_explorer.components.api` (the original protocol types with identical
object identity). A request carries request_id, component,
available_time (ns), operation/payload, operator and iteration. Mapping emits
`Task(name, execution=request)` with ordinary graph dependencies. At runtime the
stable request id gains `:iteration`. IDs must be unique within an evaluation.

Bind `graph.bind_execution(name, factory, resources=(physical_owner_ids, ...))`.
The factory must create a fresh evaluation-local instance. All clients sharing a
controller refer to ONE name. Do not bind the same instance under multiple names.
Explicit owner resources cannot also appear as outer Demand charges. Registration
can advertise `ComponentType.execution_modes=("demand", "stateful")`; existing
components default to demand without any requirement to implement a mutable model.

- `submit(request, now) -> bool`: False is side-effect-free backpressure.
- `next_event_time() -> time | None`: earliest safe observable transition. If no
  lookahead is possible, report the next actual tick; do not invent skipped time.
- `advance(now) -> list[ExecutionEvent]`: only events at that exact time. No calls
  into another component; all cross-component work is mediated by graph requests.
- Event fields: time, phase, serial, request_id, kind, payload. Emitted events are
  consumed in (time, phase, serial, component-name) order. Components must declare
  their local phase ordering and stable serials. Dependent requests are submitted
  after this batch, then newly scheduled same-time events are processed to closure.
  `advance` is component-local; this is not permission to mutate another component
  before its event turn.

Kinds are consumer_ready, physical_complete, resource_ready, queue_available.
consumer_ready completes the graph node; physical/queue release may occur later.
Blocked clients retry on explicit resource/queue events or their available_time,
not on every unrelated tick. TaskRecord release for a request denotes its outer
consumer completion (no outer Demand ownership); physical release is exclusively
in execution_events. Do not interpret those two boundaries interchangeably.
Simulation ends when all graph consumers finish, not when all hardware drains.
next_component_events reports remaining model events; pending_execution counts
accepted requests still awaiting consumer completion, not physical transactions.

Exact rational ns values are accepted by the protocol. A backend exporting rational
trace times must encode them losslessly or disclose its numeric conversion: the core
Trace can contain Fraction values; plain JSON does not serialize them automatically.
Demand-only CLI output remains JSON numeric and backwards compatible. Stateful
CLI serialization and physical-specific provenance/capacity accounting are the
builder/backend integration responsibility. The NeuPIMs profile now reports its
owner, source policy and admission capacities in `details.execution`; its source
clock uses integer ns and its CLI/JSON-lines results support independent replay.
Custom Fraction-based models still need an explicit JSON encoding policy.

Models must not mutate graph demands, capacities or priorities. Unknown owners,
backwards event times, malformed response ownership, non-progress and deadlock
fail explicitly. Side-effect freedom of a rejected submit is a component obligation,
not enforced by copying arbitrary model state. No snapshot/rollback is provided.

## Using the existing CLI

Set `backend_options: {"scheduler": "conservative"}` in an exploration config.
`examples/reusable_detail/demand.json` is a runnable ordinary GSM example:

```bash
archexplorer search --config examples/reusable_detail/demand.json --output winner.json
archexplorer evaluate winner.json --output replay.json
```

The winner retains backend_options and replay uses the same strategy. The opt-in
`pim_offload` instruction mapping now enters the same TaskGraph and automatically
selects `conservative`; adding `memory_execution="transactions"` shares one
controller among ordinary, PIM and host requests. Default aggregate mappings remain
unchanged. See [components](components.md) for the source-profile requirements.

Reusable modules include `components/instruction_ir.py`,
`components/transaction_controller.py`, and the instruction/controller execution
models under `components/stateful/`. `components/execution_profiles.py` keeps source
address layout and instruction/controller policy explicit. Alternate synthetic
topology and non-PIM examples test reuse; they do not establish hardware accuracy
or enable an HSNDI detailed profile. NeuPIMs still requires its documented one-core,
single-sub-batch, FP16 and single-iteration configuration.

## Mapping and assembly contracts

Each `MappingContext` owns an `ExecutionPlan`. A mapping can call
`ctx.plan.require(PlanConstraints(iterations=1, scheduler="conservative",
check_capacity=True, reason="explicit shared-controller program"))`. Omit fields
that are unrestricted. Conflicting requirements fail before scheduling; the
normal `consistent` selection is refined to a required strategy, and an explicit
incompatible strategy refuses. `ctx.bind_execution` declares conservative
execution automatically. Constraints are not inferred from template names,
private model flags or report keys.

`ctx.plan.detail_reporters` contains callables `(simulator, trace) -> dict`, merged
into ordinary Evaluation details after scheduling. An optional
`ctx.plan.result_builder(simulator, trace, include_trace) -> Evaluation` owns a
composition's complete accounting. Full-model tensor residency and request
boundaries belong to mapping; instruction/controller reports belong to their
components. The event driver only consumes this plan. It never treats
`TaskRecord.release_ns` as a storage-free timestamp.

A component type may declare `assembly_hooks=(hook, ...)`, where
`hook(simulator) -> None`. Hooks run once per distinct callable, in first-component
endpoint order, after all models and routes exist and before mapping. They run
only for component types actually instantiated, including networks. New component
authors need no central builder branch or global hook registry. A hook can use
`simulator.add_memory_group(MemoryGroup(path, parts), members)` to register a
striped aggregate. Parts must be actual memory instances, with unique member
paths and no overlapping groups. A component may subclass `MemoryGroup` to own
additional metadata. DRAM/PIM striping and single-bank binding are declared by
their types and implemented in `components/banked_memory.py`.

## Bound event programs and replay

The installed `events` provider, version `1`, participates in the same public
`simulator.api` as external providers. Its single protocol domain accepts
components implementing demand/stateful contracts; `component_types=()` and
`capabilities.open_component_types=true` explicitly describe this open contract.
Provider-only joint domains refuse in ordinary event construction. This does
not model arbitrary cross-engine synchronization.

The provider request schema is `arch-explorer.events-request.v1`; its result is
`arch-explorer.events-result.v1`. Requests echo ADF, workload, mapping and options,
bind exact event-domain/extension source bytes, and record the resolved plan.
Preparation and assessment build and check the graph without invoking the
scheduler or state factories. Options are scheduler, systolic_model (`native`
only for bound programs), check_capacity, iterations and trace. Mapping choices
must be explicit: `auto` mappings perform trial simulations and remain available
through the ordinary evaluation API, while bound program preparation rejects
them before creating output files.

Execution creates an exclusive ATTEMPT receipt, schedules once, and preserves
RAW.json with the complete request, plan, Evaluation and raw Trace. RESULT.json
binds those raw bytes. Replay checks all bindings and original files, the source
and plan, Evaluation consistency and the trace completion boundary, and returns
the original result. It never re-executes the graph. A failed or already attempted
request cannot silently be retried. The JSON provider requires results encodable
as finite JSON numbers; arbitrary Fraction-based component results still need
their own exact encoding/provider contract.

The five old `simulator.events` state implementation modules remain bounded
import-only compatibility layers. All new internal consumers use
`components.stateful`; the implementations and their profile policies exist in
one place. Importing `simulator.events` itself now lazily exposes generic types,
so importing `components.api` does not initialize providers or mappings.

## Optimization boundary

Rates may be cached only while the ordered active Demand contender list is unchanged.
A state or calendar event alone does not change immutable Demand shares. Add/remove
of a contender invalidates the cache. State event order, request acceptance and model
advancement are unchanged. One synthetic timer-heavy benchmark benefited; it is not
a measured NeuPIMs speedup or a promise for other graphs.

Controller optimization skips only provably idle channels until refresh or ingress;
active arbitration and failed issue attempts still advance with their original
ordering. Independent verification retained all 21 reference points, command and
transaction accounting, and micro-traces. A sparse-controller benchmark improved
locally by about 24x with the same trace setting, while end-to-end small-model
measurements were mixed. No whole-model speedup is established. Demand-only
differential and independent concurrency cases found no extra synchronization
delay from the new driver; this is tested coverage, not a proof for arbitrary
user-defined components.
