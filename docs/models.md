# Evaluation models and supported scope

The ADF `events` backend builds a task graph from registered components and the
selected mapping. Compute, storage and communication have explicit dependencies
and resource contention. The default consistent scheduler and optional unified
conservative driver share Demand semantics; the latter also drives stateful
components. No artificial barrier is added to stateless workloads.

Parameters carry units and sources. Time is ns; clock is GHz; bandwidth is B/ns
(decimal GB/s); storage is bytes. A source may declare a synthetic assumption or
an unknown reporting clock. Such values do not become device measurements.

[Public compilation](compilation.md) preserves a common graph of contiguous
FP32 matmul regions, K-block loops, masked boundary extents and materialized
assembly through registered rules. Materialized elementwise/reduction,
transpose and broadcast rules support the explicit [attention graph](attention.md).
The [precision and encoder-block contract](precision-transformer.md) adds
FP16/BF16 storage, explicit conversion and FP32 accumulation, plus a complete
synthetic pre-LN block.
The [multi-layer scene](multilayer-encoder.md) connects repeated blocks through
the public composition contract, with a single input and separate per-layer
parameters. Its reference values, synthetic component timings and bounded
search receipts are separate evidence. Its registered materialized/readonly-view
candidates keep one semantic graph and expose root storage plus actual reads.
Positive-stride views require a supporting adapter; arbitrary striding and
alias writes remain unsupported. Existing
noncompiled BF16/FP32 contracts retain their own scope. Component-bound task adapters consume that
expression; dedicated options and conversion rules stay with each adapter.
Events and Roofline can share one supported execution-domain graph without
duplicating the component-IO charge. This does not provide arbitrary ISA,
fusion or cross-domain simulation, and timing receipts alone do not prove
native numerical output equivalence.

The [public ViT classifier](vit-compilation.md) supplies the full image-to-logits
graph, including biased patch projection, leading CLS and position encoding,
independent layers, final normalization and classification. Its exact erf GELU,
epsilon 1e-12 and declared storage boundaries are separate from the earlier
tanh block. The public example compares materialized/view candidates using
existing compilation, search and saved-result validation. Synthetic arithmetic
and SFU service are uncalibrated; they do not establish official mixed precision
or hardware timing accuracy.

The [public CLIP scene](clip-compilation.md) builds one image/text graph with
unequal batches and both similarity outputs. QuickGELU, epsilon 1e-5, causal
attention, first-argmax EOT selection, feature L2 and operation order retain
their declared CLIP semantics. Token/EOT values use int32; logit scale remains
FP32 across the three floating storage dtypes. BF16 is a new declared storage
contract, not legacy or official mixed-precision equivalence. Lookup traffic
uses actual indices while full embedding weights occupy capacity. Layouts and
bounded exploration reuse the same public interfaces; no native execution or
hardware calibration is inferred.

Workloads include supported operators, Transformer stages and complete ViT/CLIP
forward graphs. CLIP's image and text towers and similarity outputs are one graph,
not separately weighted workloads. Supported profiles and omissions are described
in [the full-model tutorial](tutorials/05-full-models.md).

The optional [ONNX frontend](onnx-import.md) imports a bounded standard-domain
IR8/opset17 subset with FP32, FP16 and BF16 storage into the same public
WorkloadSpec and Graph. Explicit Cast and LayerNormalization retain their declared
storage and accumulation contracts. Callers provide
complete runtime roots and explicit shape specialization where needed. Serialized
workloads retain model and external tensor bytes, original node/value identities
and parser provenance, so saved inputs do not depend on the original file path.
The supported operators and early refusals are documented separately; this scope
does not establish arbitrary framework export, all ONNX operators or paper-scale
hardware accuracy.

The [PyTorch exporter](framework-export.md) accepts real CPU modules and exports
their parameters and ordered inputs through the pinned framework exporter.
The complete small vision and dual-tower modules retain actual Conv patch
embedding, class/position tokens and all Transformer blocks. The text tower
uses int64 lookup and first-argmax EOT selection; its two outputs preserve
different image/text batch dimensions. FP16/BF16 are actual storage choices.
The BF16 example explicitly computes patch Conv in FP32 and converts back;
the exported initializer types and Cast nodes determine physical storage.
Explicit resource budgets distinguish logical tensor sizes, materialized roots,
estimated host memory and physical mapping capacity. Full-layer transformer
preflight and MLA admission retain complete requested dimensions; each subsequent
construction, compilation and service phase requires its own execution evidence.
BF16 official evaluator disagreement remains separate from the independent CPU
reference. Hardware comparison requires condition-matched raw measurements;
IPU exp, reciprocal, reduction and conversion costs still require exact codelet
and hardware evidence.

GPU programs use the native Accel-Sim backend. The in-house GPU latency
predictor and its seeds/configuration paths have been removed. The program
frontend requires an installed complete-program provider and an external
engine; preparing a request does not establish successful execution. See the
[native demo](tutorials/05-full-models.md) for tiny/standard profile boundaries,
numerical references and source declarations. Legacy single-kernel requests
remain supported by their original native entry, not as full-model evidence.

The explicit `sm.roofline` component implementation accepts declared operations
and `component_io` bytes, rather than a GPU program. Its ideal-overlap estimate
is separate from native SM cluster execution and does not restore the retired
in-house GPU predictor or introduce a fallback. Implementation-specific inputs,
the traffic boundary and the [selection contract](component-implementations.md)
must be checked before composing work into a graph.

Physical validity is separate from search reachability. Non-GPU DMC,
tensor-parallel, DRAM-PIM, bonded memory, scalar CGRA and tile-private models have
different precision, capacity, topology and timing requirements. Generic
shared-memory mapping may serve non-GPU systolic/vector/PIM hosts; it is not
a GPU timing implementation. A registered
component does not guarantee that every mapping can execute every combination.
See [components](components.md), [CGRA](cgra.md), [manycore](manycore.md) and
[unified execution](unified-execution.md) for the actual boundaries.

Independent [cost estimates](cost.md) provide `area_mm2` and `cost_usd`, source
provenance and missing-mechanism diagnostics. Unknown values remain null. An
active constraint on an unknown metric makes the objective unavailable. With no
such constraint, a latency-only objective is possible; this does not infer cost.
Scores may penalize budget violations, so inspect `score.feasible` separately
from objective rank. A small penalty is not a hard feasibility filter.

[Explicit LLM and MLA examples](../examples/workload_scenarios/README.md)
separate selected-layer performance graphs from numerical decode, and preserve
the different replay contracts of legacy and compiled search results.
