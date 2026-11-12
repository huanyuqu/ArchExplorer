# Complete reference GPU programs

The program API maps a complete small ViT classifier or CLIP vision/text forward
graph to an ordered stream of ordinary SM90 FP32 trace kernels. One external
Accel-Sim process consumes the complete list. Its final cumulative core-cycle
counter supplies the program latency; cumulative kernel counters are never
summed. Host preprocessing, allocation/copy time and CPU reference execution
time are outside that device-cycle result.

```python
from arch_explorer.adf import WorkloadSpec, MappingSpec, from_dict
from arch_explorer.archs.accelsim import program_demo_specs
from arch_explorer.simulator.api import prepare_request, assess_request, evaluate_request

specs = program_demo_specs("clip")
request = prepare_request(
    from_dict(specs["adf"]), WorkloadSpec(**specs["workload"]),
    MappingSpec(**specs["mappings"][0]), output_dir="new-program",
    provider="accelsim", options={"engine": "/absolute/external-engine.json"},
)
assessment = assess_request(request)  # bounded CPU validation only
result = evaluate_request(request)    # the actual external native process
assert evaluate_request(result) == result  # read-only replay; no native launch
```

`arch-explorer.simulator-request.v1` and `arch-explorer.simulator-result.v1`
are the provider-independent public envelopes. Provider `accelsim` version
`1` owns `arch-explorer.accelsim-provider-request.v1` and
`arch-explorer.accelsim-provider-result.v1` payloads. Original options and
complete public specs remain intact; the payload binds the normalized native
request and output directory. A result contains the unchanged request and an ordinary `Evaluation.to_dict()` with
`backend="accelsim"`. Existing single-kernel request/result versions retain
their own API. Legacy program schemas
`arch-explorer.accelsim-program-request.v1` and
`arch-explorer.accelsim-program-result.v1` are registered for exact historical
result replay. The standard ADF evaluation entry also dispatches complete
ViT/CLIP workloads to this program implementation when `backend="accelsim"`.

The workload supplies model dimensions, layer counts, dtype and batches. The
ADF supplies native components, topology, clocks, resources and their sources.
Mapping `accelsim_program` accepts only `threads_per_block` (32, 64 or 128),
changing actual CTA/warp allocation. Runtime options are `timeout_s`,
`max_cycles` and `memory_limit_mb`. Model/mapping/ADF values cannot be hidden in
runtime options. An absent engine is an actionable unsupported/error condition;
the adapter never falls back to a custom GPU throughput model.

Preparation saves the complete graph, deterministic inputs/weights, materialized
shape-only address tables and literals, output data, per-operator numerical
checks, and every kernel trace. The manifest binds their byte hashes to the
workload, mapping, ADF/native configuration and lowering source. Validation
regenerates the bounded lowering, checks all bytes, and checks every CTA/warp's
EXIT closure. Unsupported operators or ISA formats are refused as a complete
program. Large standard profiles retain their complete inspectable graph and
resource gaps without allocating model tensors or launching native simulation.

The emitted algorithm has one output per GPU thread, scalar FFMA matrix
products and reductions repeated by each output thread. All gather-table reads,
input loads and output stores contribute to timing. Tables encode shape/layout
only; embedding lookup and EOT selection compute their data-dependent address
from loaded token/index registers. Views preserve alias addresses and require no
kernel. Other rearrangements are materialized. Activation memory starts
uninitialized: the interpreter rejects use before a producer stores it.

The numerical interpreter executes the emitted register and memory operations,
then checks each operator and final output against the independent complete
reference. FP32 summation order can differ from NumPy/BLAS. SFU operations use
ideal FP32 functions in the interpreter; exact-GELU's erf is implemented by a
multi-instruction polynomial and checked against the exact numerical reference.
There is no invented single erf instruction. Numerical tolerances are reported
separately from timing. Ordinary native traces omit predicate register
scoreboarding; the generated integer guard sources and predicated EXIT masks
remain explicit, and this timing limitation is recorded in the manifest.

These are **reference-generated synthetic** traces with synthetic weights and
consistent virtual addresses. They are not captures of PyTorch/JAX kernels,
not a framework compiler output, and not pretrained-model accuracy evidence.
The trace-driven native engine supplies timing counters, while functional
outputs come from the checked CPU interpretation. Hardware MAPE/worst error
remain null. No GPU query, CUDA initialization, device host or GPU resource
reservation is required to generate or simulate these CPU-native programs.

The external engine binary and libraries are never bundled in the wheel. The
existing pinned engine manifest verifies their content and source identities;
native stdout must also carry the pinned source build identifiers. Successful
results require all ordered launch identities, exact retired thread/warp
instruction counts and issued CTA counts, a matching native clock and normal
termination without cycle limits or deadlock. Replay verifies the staged
configuration, complete kernel list, raw logs and cumulative counters again.

## Replay provenance and semantic identity

The producer's `reference.numpy_version` is retained as environment provenance.
It does not decide program semantics. A consumer still regenerates the complete
bounded program and compares all graph/spec bindings, inputs/weights, tensor and
output hashes, reference steps/coverage/numerical checks, instruction streams,
addresses and resource limits. Only the NumPy version label is excluded from
that transient comparison view. If an actual numerical or semantic field
differs, validation refuses with its JSON pointer and both environment labels;
this is not a guarantee of bitwise equivalence for arbitrary NumPy/BLAS builds.

Source identity remains checked. The consumer accepts its own complete source
binding or the two fixed complete SIM35/SIM35A producer source vectors in
`components.accelsim.source_compat.LEGACY_PROGRAM_SOURCES`.
The SIM35 validator SHA256 is
`eb59dacc3cb18c2b6111b83972bd1eed77ec704519a8d624b2f948edaab4732f`
and SIM35A's validator SHA256 is
`97edd04cbb556497ca5f94f5650f80dc54a2135a907329012d54944a369fdaf9`.
The explicit policy `arch-explorer.accelsim-component-migration.v1` maps
`simulator/accelsim/*.py` to the corresponding component module,
`mapping/accelsim_program.py` to `components/accelsim/lowering.py`, and
`components/accelsim.py` to `components/accelsim/declarations.py`.
Every old source SHA must match its fixed vector; unchanged workload/reference
sources must also match the consumer byte for byte. Full regeneration and raw
counter validation remain mandatory. An unlisted source hash, extra path,
same-name replacement or changed generation source is refused even when the
surrounding artifact reference has been rehashed.

`assess_request` reports `validation_provenance` with the original producer and
actual consumer NumPy/module identities. Replay does not rewrite the producer's
manifest, source hashes, request, result or native originals. A successful
replay returns the unchanged result; no native process is started.

The joint provider owns all native SM/L1/shared/interconnect/L2/DRAM mechanisms
and executes them as one configuration. It exposes no independently timed Python
models. Unsupported execution-domain composition is rejected before native
execution. Its registrar is `arch_explorer.components.accelsim:register`; the
installed `arch_explorer.components` entry point invokes this same function.
Importing or registering it neither queries a GPU nor loads a native engine.

Old `simulator.accelsim` modules and `mapping.accelsim_program` are temporary
forwarding imports. C++ resources and all new internal calls use the component
package. User demo construction belongs to `archs.accelsim`; native configuration
fixtures for component tools live in `components.accelsim.scenarios`.
