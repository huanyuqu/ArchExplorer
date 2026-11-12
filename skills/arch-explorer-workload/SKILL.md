---
name: arch-explorer-workload
description: Connect new operators or workload graphs to ArchExplorer, write independently installable extensions, and verify mapping and simulation on custom ADFs. Suitable for adding new computational semantics or graph structures; does not require rewriting hardware components for every new model.
metadata:
  id: skill_1b068ddcfd93
---

# New Workload Integration

First find `docs/operator-implementations.md` and
`examples/extensions/operator_implementations/` corresponding to the user's installed version. For compatible single-lower/external-row strategies, see also
`docs/workload-extensions.md`. Read the public API and actual commands before writing extensions; do not modify the main library's model list,
and do not treat automatic PyTorch/ONNX import or arbitrary ISA support as existing functionality.

Distinguish three types of work:
- New graph composed of existing operators: define tensors, operators, and dependencies, directly reusing semantics and execution models.
- Truly new operator: define shape/dtype/attributes, semantics, and numerical reference, then register execution implementations for the required targets.
- New hardware mechanism: connect a component execution model separately; the same operator can have multiple hardware implementations.

## Public Semantic Interface

The extension contract is in `arch_explorer.workload.api`: semantic/graph types, OperatorDefinition, ImplementationDefinition,
ExecutionProgram/ExecutionStep, registration and selection interfaces. First confirm that the matching version actually provides these symbols; if missing, report it,
and do not bypass validation or monkey-patch whitelists.

OperatorDefinition has validate, version, and optional reference(op, inputs), where inputs are read-only ndarrays;
implementation definitions have stable ID/version, operator, target, requirements/dtypes/supports, and lower(op, context).
Semantics do not change with hardware speed; attrs are validated according to the operator's own rules. The old three-parameter lower is a legacy:<op> compatibility adapter,
not a restriction that the same semantics may only have one implementation.

`Workload.to_dict()/from_dict()` saves the graph; `WorkloadSpec('graph', graph.to_dict())` represents an existing graph. Optionally register a builder for parameterized graph generation: `register_workload(kind, builder)`, where builder(params, *, tp=1) returns a Workload. A custom kind is only a convenience entry for graph generation, not a required step for every new graph.

For ordinary single-device events graphs, prefer execution_graph: specify implementations per node, choose first/round_robin,
serialize, and optional memory. Reuse graph dependency/placement/storage foundations; do not copy run_device. When two compatible implementations exist, ambiguity must be
resolved; explicitly selected nonexistence, target/capability/dtype/shape/layout incompatibility should all be rejected before preparing output.
Old GSM/DMC adapters retain their respective limitations; do not silently skip unknown operators. Complex layouts/tiling can extend dedicated strategies;
a fixed tile or an untested selection cannot be called optimal.

Specific schemas, graph inputs/outputs, and topology specifications follow the public tutorials. Preserve branches, shared weights, precision, and dependencies; only parts of the model without dependencies can be parallelized. Reject unknown references, duplicate producers, and dependency cycles; do not split one graph into weighted independent workloads to bypass graph restrictions.

## Implementation and Integration

When different targets share a common compiled expression, use `workload.api.compile_workload/validate_compilation`
and `simulator.api.prepare_compiled`. `compile_graph_v2`'s region computation, indexing, K loops,
tile tensors/assembly, access, and inspection are owned by registered OperatorMapping callbacks; the public compiler only orchestrates and is not rewritten in user scripts.
Public tasks select `TaskAdapter` via `AdapterBinding` along component paths, and its lower/restore handles group input conversion.
`ImplementationDefinition` is still operator lowering, and `ComponentImplementation` still owns the model factory and timing resources.
Public mapping v3 includes compute/memory/rules/operator_rules; per-instance overrides take precedence over kind defaults and unique rules. Rule IDs/versions/options are validated by each rule, dedicated parameters belong to adapter options, and execution options are passed separately to prepare.
Groups supported by the same execution domain retain common dependencies/last readers; when cross-domain bridges are missing, reject rather than independently timing and summing. For examples and FP32/contiguous matmul boundaries, see
`docs/compilation.md` and `docs/attention.md`. New compilation is v3; v1/v2 are explicitly migrated; preserve explicit inputs and both readers of the original DAG, and do not compress the graph into FLOPs and then guess instructions.
Public compilation supports contiguous materialized FP32/FP16/BF16 storage, FP32 accumulation, and per-materialization output rounding; reuse workload.api.round_storage, and do not copy rounding implementations in extensions or user scripts.
Complete pre-LN blocks are composed by the public transformer_encoder_block; multi-layer uses transformer_encoder and compose_workloads. CORE scenarios only generate explicit data/configuration and call public functions; do not precompute inter-layer activations. Default reshape/transpose are materialized; layout_candidates provide complete mappings of materialized/views for the same graph; reshapes that cannot be shared explicitly retain materialization, and do not compute strides yourself or copy fake views.
Same-dtype read-only views preserve bit values and do not add new rounding; different dtypes require cast. resolve_layout provides root storage, element strides/offset, and complete allocation_bytes; BF16 host FP32 arrays are still billed as 2B storage. Read-only positive-stride slice/transpose/shareable reshape requires the adapter to declare readonly_strided; old independent-only external packages explicitly reject aliasing tasks before lowering and do not pretend to have strided capability. Arbitrary as_strided, zero/negative strides, dtype reinterpretation, and alias writes are still rejected.
Current devices still declare FP32 arithmetic services; low-precision promotion/rounding is billed separately as conversion requirements; do not guess acceleration multipliers. Views have no copy but real readers still pay for memory access; roots are allocated only once, and small output views may retain the complete root. Read/write services look at access, allocation looks at allocations, and runtime peaks look at actual allocation/deallocation, not the sum of logical tensors as a substitute. Old non-compiled scopes remain; native precision/ISA goes through public assessment separately, and native is not newly run in this round.
For exact inputs, norm/GELU definitions, and scenarios, see docs/precision-transformer.md and examples/transformer_block/README.md; unknown extensions keep their own precision declarations.
For multi-layer layout comparison with a budget of twelve evaluations, see docs/multilayer-encoder.md and examples/multilayer_encoder/README.md; the single-layout control has a budget of six evaluations. Saved valid rules, root layouts, and resolver/builder/adapter sources must be strictly validated; do not migrate old bound results by replacing fields.
For the complete public ViT entry, see docs/vit-compilation.md and examples/vit_compilation/README.md; only pass images and complete root parameters; graph construction/layout/numerics use the public workload API; do not copy patch or GELU formulas. ViT maintains erf-exact GELU, epsilon=1e-12, and its declared storage boundaries; do not rename a tanh encoder as a substitute; old ViT/CLIP native results are not migrated and claimed as public compilation results. Layout/per-instance mapping/hardware comparison go through the same existing searcher; unknown external packages retain their FP32/contiguous capability scope.
For the complete public CLIP entry, see docs/clip-compilation.md and examples/clip_compilation/README.md; the same graph preserves the image tower, causal text tower, and two rectangular similarity outputs, and only supplies images/token IDs/complete root parameters. QuickGELU, epsilon=1e-5, first-argmax EOT, L2 without epsilon, and FP32 logit_scale order belong to public semantics; do not apply ViT/tanh blocks or move low-precision rounding boundaries. token/EOT remain int32; do not truncate or float them; external illegal NaN/inf cannot be passed through by borrowing internal mask sentinels. Check composition identity and layout_candidates; do not guess tower/layer names or copy index traffic/layout formulas. Output shared-root retention, full embedding capacity, and duplicate index access are checked separately; new BF16 storage is not claimed as equivalent to old CLIP or official mixed precision.
Exploration still calls the existing Env/search; compilation configurations connect to the public compilation flow, each candidate attempt counts toward the budget, and failures are preserved; save compiled-result as read-only evidence.
New compilation search result replay does not execute; when new execution is needed, explicitly rerun. Old ADF results retain their original re-evaluation behavior.

Copy the minimal required parts from independent examples into the user package, using your own namespace. The entry point `arch_explorer.components`'s parameterless register() can register operators, builders, mappings, and components. Registration only declares capabilities; it does not run simulation.

Reuse existing Kernels when sufficient; multi-step implementations return an ExecutionProgram with explicit deps, internal Tensors, and inputs/outputs.
Temporary capacity, last consumers, and deallocation go through the public storage planner; branch temporaries are isolated by the framework. Extend mapping only for special strategies;
operations that cannot be expressed are explicitly rejected, not disguised as matmul or filled with fixed latency. Compute-only examples explicitly state that there is no storage network model.

GPU targets use the component public emitter contract to register target=accelsim implementations; graphs/inputs/references still belong to the workload or
external package. GPU providers do not recognize new model names; do not change kind/metadata to pretend to be ViT/CLIP, and do not create dedicated providers to bypass.
First verify real limitations such as FP32/ISA/single-device/topology; native component paths and oracles use the same inputs, independent outputs, and the public comparator.

## Proving External Usability

Install the main package and extension wheel, leave the source directory, and run using the actual public API and CLI. At minimum cover the new graph and operators the user needs, one trusted small input, mapping variations, and hardware parameter variations. Check that changes enter the real task, not just modify report labels. Check that unsupported dtype/shape/mapping are rejected before execution, and save replayable results.

Numerical reference and timing/resource checks save evidence separately; performance results do not prove tensor values. Bind the versions and source of actual semantics/reference/execution implementations,
mapping, and input bytes; after plugin changes, old results should be rejected on read-only replay. Check registration conflicts and failure rollback;
do not silently switch implementations. TP/multi-device/external engine capabilities are confirmed separately; arbitrary new graphs are not supported by default.
Native foreground/physical tail are compared separately; the main latency takes the physical final, and the tail is not a new kernel; read-only replay does not execute native.
Deliver external packages, ADFs, graphs, mappings, sources, and executable commands; do not modify the general scheduler or CLI identity branches.
