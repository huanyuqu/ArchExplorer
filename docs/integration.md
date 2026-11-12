# ADF module interfaces

All public search, evaluation and Agent tools share one architecture representation.

| Module | Responsibility |
| --- | --- |
| `adf` | SpaceMatrix, SpacePoint, Param, WorkloadSpec, MappingSpec and registered execution dispatch |
| `components.api` | Hardware types, exact implementation bindings, the shared catalog and installed extension registration |
| `components`, `devices` | Concrete mechanisms, execution providers and optional sourced device seeds |
| `mapping`, `workload` | Public placement/tiles/movement/lifetime, semantic/reference graphs and bound task-adapter coordination |
| `simulator`, `simulator.api` | Generation, generic resources/tasks/dependencies, scheduling and provider request envelopes |
| `components/<hardware>/implementations` | The authoritative mechanism for each hardware type and execution implementation |
| `components/_support/accelsim` | Shared native ABI, engine/build support, provenance and exact historical readers |
| `components/assemblies` | Whole-system composition of the same component factories, including GPU |
| `archs.accelsim.program_demo_specs`, `demo`, CLI | User scene construction, standard specs and complete request/candidate association |
| `structure`, `search`, `archs.adf_seeds` | Generic edits, optional named transforms, nested SA/greedy/random search |
| `agent_env` | Common configuration validation, evaluation, submission, JSON-lines and MCP |
| `cost` | Independent area/cost estimates and explicit unknown metrics |

`search.run_search(config)` and `exploration.explore(config)` are the same callable.
`exploration.replay(result)` dispatches by the saved result contract. Historical
`archexplorer.adf-result.v1` retains its reevaluation behavior. The compiled
search result validates each saved public compiled result, its exact ADF,
workload, effective mapping, objective, options and sources read-only, without
creating an environment, preparing a request or charging another evaluation.
CLI `evaluate` and Agent `evaluate_request` use that same reader.
`exploration.rerun(result, output_dir=new_directory)` is an explicit new
execution with a fresh request identity and budget. Neither path uses a second
simulator or a frontend parameter-conversion formula.

Other request/result schemas go to `simulator.api.evaluate_request`; registered
providers handle their exact historical schemas as well as new generic
envelopes. CLI has no engine-specific schema branches. Agent tool traces retain
their separate `env replay` protocol, and search results retain their own replay.

`simulator.api.prepare_request(adf, workload, mapping, *, output_dir, provider,
options=None)` prepares a bound JSON envelope without running a simulator.
Provider selection is explicit; engine files and limits belong in options.
`assess_request(request)` returns at least `supported: bool` and `gaps: list`.
`evaluate_request(request_or_result)` executes or validates an existing result.
The result contains the complete request and the existing `Evaluation.to_dict()`
under `evaluation`; provider details remain opaque to the frontend. Missing or
incompatible capabilities fail before execution instead of falling back.

`archs.accelsim.program_demo_specs(model)` constructs the tiny ViT/CLIP scene,
including provider `accelsim`, standard objects, mapping candidates and scope.
Provider implementations do not own CLI model names. Other scenes select their
own provider. The demo persists the entire preparation and checks every request
and candidate association, then all capabilities, before evaluating any candidate.
See the [external extension tutorial](component-extensions.md) for a separately
installable component and synthetic provider, explicit registration and entry points.

The public Python helpers `mapping.mapping_space(adf, workload, template, fixed=None)`
and `mapping.best_mapping(adf, workload, templates=None, fixed=None, limit=None)` use
each template's `request_space` and the registry's `mapping_spaces` contract.
Conditional branches remain separate products with only active parameters;
`sampling_defaults` apply unless explicitly overridden by valid selectors.
Unknown, inactive and out-of-domain selectors are rejected, including booleans
where integer choices are required. An inapplicable template still yields no candidates.
`best_mapping` takes `fixed` keyed by template name. Implicit template discovery
skips `multi_device` without a structured request; explicitly selecting it or
providing its request requires all validated placement, sharding and routing fields.

Enumeration preserves template order, branch order, sorted parameter keys and
each domain's choice order. `limit` counts candidates across all branches,
including failed evaluations; `skipped` counts candidates beyond that limit.
Recorded failures do not earn replacement attempts, and equal latencies keep the
first successful candidate. Space discovery performs no backend evaluation.
The winning complete `MappingSpec` can be passed directly to `adf.evaluate` with
the same ADF, workload, backend and options for replay.

Default proposals come from the component catalog and current graph, with no
architecture-family or stage quota. A seed name does not restrict future structures.
Only declared user proposals bound the outer search explicitly. Component domains,
mapping applicability and backend physical requirements still apply to every trial.

`components.api.component_catalog()` is the metadata authority consumed directly
by CLI `components` and Agent `describe_environment.component_catalog`. The
Agent's existing `components` list is a backward-compatible parameter-domain
projection of that same catalog, including custom registries. See
[hardware implementation selection](component-implementations.md).

`SpacePoint.implementation` is an optional exact ImplementationBinding whose
parameters are separate from common hardware parameters. Its ID/version and
configuration participate in serialization, fingerprints and result provenance.
ComponentImplementation describes component execution; workload
ImplementationDefinition still describes operator lowering. Neither requires
adding implementation IDs to the CLI/provider/scheduler.

`workload.api.compile_workload` orchestrates registered operator mappings,
their common graph and component-bound adapters. Mapping-rule callbacks own
expansion, access and inspection. `TaskAdapter.lower/restore` own conversion and saved
input restoration; the component owns timing and resource state. The public
`simulator.api.prepare_compiled` prepares a source-bound request.
CLI `compile` / `prepare` and the Agent tools consume those exact functions,
leaving schemas, numerical/movement semantics and adapter capabilities in their
production owners. See [public compilation](compilation.md). Runtime
`ExecutionPlan` callbacks are not the serialized compilation artifact.

Opt-in `compilation` exploration uses that public compile/prepare/evaluate
pipeline within the same Env and search policies. It saves public compiled v3
results in a distinct exploration format whose replay is source-validated and
read-only. Explicit `rerun` creates new requests and execution; legacy ADF
result replay keeps its earlier reevaluation behavior. See
[attention and bounded exploration](attention.md).

Repeated subgraphs and encoder layers use the workload composition and mapping
registry contracts through the same compiler pipeline. User scenes provide
explicit data, binding specifications and per-instance choices; generic search
and Agent code do not parse layer names or reconstruct rule selections. See
[multi-layer composition and exploration](multilayer-encoder.md).

```sh
python -m arch_explorer search --config examples/adf-dmc.json --output results/sa.json
python -m arch_explorer evaluate results/sa.json --output results/replay.json
python -m arch_explorer env serve --config examples/adf-dmc.json
```

See [structure](structure.md), [Agent protocol](agent.md), [model scope](models.md),
[full-model tutorial](tutorials/05-full-models.md) and [unified execution](unified-execution.md).
