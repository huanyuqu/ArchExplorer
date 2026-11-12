# Tile-private memory and BSP scenarios

`ipu_gc200` and `tianjicx` are sourced single-chip **partial analytical
scenarios**. Their dedicated templates express data placement, capacity,
local-memory traffic and phase dependencies. They do not implement Poplar,
machine instructions or a calibrated full-device latency model. Unknown
cost and area remain `null`; use latency-only objectives without cost/area
constraints in the Agent environment.

```python
from arch_explorer import devices
from arch_explorer.adf import WorkloadSpec, MappingSpec, evaluate

adf = devices.ipu_gc200()
work = WorkloadSpec("operator", {
    "op": "matmul", "batch": 1, "m": 16, "k": 32, "n": 64,
    "dtype": "fp16",
})
result = evaluate(adf, work, MappingSpec("ipu_bsp", {
    "tile_m": 16, "tile_n": 128,
}), trace=True)
print(result.latency_ns, result.details["manycore"]["unmodeled"])
```

For TianjicX, use `devices.tianjicx()`, `tianjic_spatial`, and `dtype="int8"`.
GEMM inputs are signed int8 and results are int32. The template updates
output tensor storage accordingly. It does not claim FP16 support.

## Parameters and source scope

Each ADF parameter carries its unit, evidence and source in `Param.source`.
Rates below are vendor/paper ceilings or explicit divisions of such ceilings,
not measured efficiencies. No reference error selects a parameter.

| Resource | GC200 | TianjicX |
| --- | --- | --- |
| Cores / clock | 1472 / 1.325 GHz | 160 / 0.4 GHz |
| Matrix compute | 64 FP16 MAC/cycle/tile, FP32 internal accumulation, FP16 stored result | 128 int8 MAC/cycle/core, int32 accumulation/result |
| Data capacity | 624 KiB/tile | two 64 kB data blocks/core; 128000 B by default |
| SRAM service | rounded 47 TB/s / 1472 = 31.9293 B/ns/tile | 5.12 TB/s / 160 = 32 B/ns/core |
| Exchange | independent 32-bit/cycle transmit and receive per tile = 5.3 B/ns each | 16×10 mesh shape, link timing/contention unknown |
| Global barrier | approximately 150 cycles / 1.325 GHz per modeled boundary | unknown; omitted explicitly |

GC200 data comes from [Graphcore's Hot Chips 2021 presentation](https://hc33.hotchips.org/assets/program/conference/day2/HC2021.Graphcore.SimonKnowles.v04.pdf):
slides 8–9 for tile count/capacity/clock, 12 for data-side SRAM throughput,
13 for accumulation and output formats, 15 for BSP/synchronization, 16 for
exchange endpoints. The [IPU programmer's guide](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html)
specifies up to 64 MAC/cycle using AMP; the [tile parallelism article](https://www.graphcore.ai/posts/exploiting-tile-level-parallelism-on-graphcore-ipus)
describes two FP32 arithmetic operations/cycle, used here for vector
arithmetic. This is not an exponential/reciprocal rate. A single compute
resource serializes tile jobs; distinct AMP/vector instruction overlap and
issue scheduling are not modeled.

GC200 is a die identity. M2000 is a four-IPU machine; Bow is a separate
product/clock scenario. Neither a board label nor a benchmark preset name
establishes which die, clock, compiler or timing boundary a measurement used.
There is no automatic Bow substitution and no Bow preset in this model.

TianjicX count, clock, MAC count and topology come from
[Ma et al., Science Robotics 2022](https://doi.org/10.1126/scirobotics.abk2948),
Table 1 and Fig. 5. Page 7 identifies 5.12 TB/s as chip SRAM read/write
access throughput. It is divided across cores for local SRAM service,
**not assigned to mesh links**. The model counts 2 operations/MAC, giving
2×128×160×0.4 = 16384 GOPS, consistent with rounded 16 TOPS.
[Wang et al., 2025, §4.1/Fig. 7](https://doi.org/10.3389/fnins.2025.1512926)
describes the two data blocks, 128 parallel 8-bit multipliers and 32-bit
accumulators. Total physical 144 kB/core includes other blocks. Source kB/MB
notation does not resolve binary vs decimal units: `capacity_unit="binary"`
explicitly selects 131072 B; the conservative default is decimal, independent
of prediction error. The target benchmark's operand format remains a separate
unknown when its experimental conditions do not state it.

## Components and topology contract

`ipu_tile` / `tianjic_core` require `clock_ghz` (GHz), `precision`,
`accumulation`, `output_precision`, `identity` (strings), and
`macs_per_cycle` (MAC/cycle). Optional `vector_ops_per_cycle` (ops/cycle)
and `sync_cycles` (cycles) describe sourced arithmetic and barrier service.
Compute resources are FIFO and cycle-based. A MAC is two operations.

`tile_exchange` requires integer `tiles` and `topology` (`all_to_all` or
`mesh_unknown`). Optional `endpoint_bandwidth` is B/ns per direction;
`rows` and `cols` must multiply to `tiles`. `mesh_unknown` rejects an assigned
endpoint bandwidth. For known endpoint rates, transfers occupy the source
transmit and destination receive resources, using fair sharing. Disjoint
endpoints can overlap; transfers sharing either direction contend. Same-tile
copies skip the network but still pay SRAM read and write service. Zero-byte
copies create no task.

The GC200 endpoint abstraction is a relaxed throughput schedule. The actual
stateless exchange uses compiler-controlled sends/receiver selection; this
fluid model does not prove that every overlapping transfer schedule is
loadable. Exchange propagation/startup is omitted. For TianjicX the mesh
dimensions are metadata only: there are no invented links, routes, link
capacities or contention penalties. Transfers still occupy source and
destination SRAM. Results explicitly report this missing network timing.

Both templates require homogeneous tile containers as ordered direct
children of one root exchange, each holding one compute point and one private
SRAM. They do not require a DRAM. Memory initialization, host loading,
compilation, code/stack storage and final host readout are outside the boundary.

## Explicit mapping and storage

The finite search space is `tile_m ∈ {1,4,16}`, `tile_n ∈ {32,128,256}`,
`tile_k ∈ {32,64,128}`, `schedule="blocked"`. Explicit mappings that omit
`schedule` retain the original whole-K schedule described below, with defaults
16 and 128; their serialized replay meaning is unchanged. `tile_k` requires
`schedule="blocked"`. Only a single program iteration is supported.
GEMM, MVM and IPU scale/softmax are supported; other operators are rejected.

Each tensor has canonical contiguous element shards across the tile SRAMs.
All canonical tensors, including intermediate/output tensors, remain allocated
for the entire program. This conservative allocation does not reuse dead
intermediates. Every active tile additionally needs one wave's input/output
staging buffer. Both allocations are checked against that tile's capacity;
aggregate spare memory on another tile cannot satisfy the allocation.

Each output block is assigned once, in row-major order across tiles, and
uses the whole K dimension. It gathers its exact A/B element intervals from
their canonical owners, computes locally, then scatters disjoint results to
canonical owners. A is reused across the block's columns and B across its
rows. Other output blocks gather their own inputs; there is no implicit
multicast/cache reuse. Shared-B GEMM and public MVM `(batch,K) @ (M,K).T`
are supported. There is no split-K, so no arithmetic partial-sum reduction is
required; output assembly is a scatter of non-overlapping intervals.

One initial synchronization is followed by waves of:

1. Gather to private staging buffers; all gathers complete before compute.
2. Parallel per-tile compute, with input reads/output writes on local SRAM.
3. Global compute synchronization.
4. Scatter results home, then global exchange synchronization.

For each block with `I` input bytes and `O` output bytes, gather and scatter
each perform a read plus a write, and compute performs `I+O` SRAM bytes.
Total local SRAM service across all tiles is `3*(I+O)` bytes. Compute resource
time is `M*K*N/(MAC_per_cycle*clock_ghz)` ns; simultaneous SRAM and compute
demands are constrained together, not added as two sequential delays. Copies
can additionally be endpoint-limited. No full-K tile fitting means a readable
capacity rejection; streaming K or spilling to invented external memory is
not substituted. Implementation limits are 8192 output blocks/operator and
200000 tasks/program, reported as limits, not hardware incapacity.

## Decode attention and remote KV

### Capacity-aware K blocking

For the blocked strategy use, for example,
`MappingSpec("ipu_bsp", {"schedule": "blocked", "tile_m": 16,
"tile_n": 128, "tile_k": 128})`. TianjicX uses the same keys with
`tianjic_spatial`. These are explicit mapping choices, not fitted hardware
parameters. Search generates this strategy; an old explicit mapping without
the new keys still uses the original strategy.

Canonical tensors remain resident for the entire program. Their placement
uses logical rectangles and element-aligned physical segments. Largest blocks
are assigned to the tile with most free space; a rectangle can span tiles.
This avoids treating logical block fragmentation as physical incapacity.
The number of active compute tiles is bounded by capacity remaining after
canonical storage and the workspace reservation. Per-tile storage is checked
again during event simulation. The policy does not reclaim dead tensors;
a rejection under this policy is not proof that every possible mapping fails.

For an output block of `r × c` elements and K block length `k`, workspace is
`(r+c)*k*input_bytes + 4*r*c`, rounded to a multiple of four bytes. The last
term is an FP32 (IPU) or int32 (TianjicX) accumulator, initialized once and
held on the same tile through all K blocks and final scatter. Each partial
dot gathers only its exact input rectangles and reads and writes `4*r*c`
accumulator bytes. Tail K blocks use their actual lengths. No intermediate
FP16 narrowing or cross-tile partial-sum reduction is performed.

Each K block retains explicit gather, join, compute and global synchronization
events. Thus a subsequent partial dot waits for its predecessor. After the
last K block, IPU final storage reads `4*r*c` bytes and writes `2*r*c` bytes
in the same buffer; forward in-place narrowing is safe. TianjicX already has
its final int32 format. Output scatter and exchange synchronization then
complete the wave, releasing workspace. Conversion instruction time is
unknown; its SRAM traffic is counted. GC200 partial dots use the FP32-memory
rate on Hot Chips 2021 slide 13 (125 TF/s versus 250 TF/s for FP16 memory),
giving 32 rather than 64 MAC/cycle/tile for this internal kernel.

The implementation still creates explicit transfers constrained by source
and destination SRAM and, where known, separate TX/RX resources. Rectangle
intersection uses prefix arithmetic instead of enumerating strided elements.
Network ports are materialized once. Scheduler cut membership uses a set
while preserving member order. No event aggregation, global-network bandwidth
substitution or approximate phase timing is used. The original limits of
8192 output blocks/operator and 200000 tasks/program remain in force.

Details additionally report active tiles, workspace bytes, logical/physical
layout counts, actual per-tile capacity peaks, K starts, partial block and MAC
counts, and accumulator reads/writes. A new layout changes transfer endpoints
and latency; it is not asserted to match the whole-K mapping's prediction.

### IPU distributed partial sums

`ipu_split_k` is a separate IPU-only template. Existing `ipu_bsp` and
`tianjic_spatial` mappings and search spaces keep their previous meanings.
For example:

```python
mapping = MappingSpec("ipu_split_k", {
    "tile_m": 16, "tile_n": 128, "tile_k": 128,
    "partitions": 16, "reduction": "tree",
})
```

The tile-size domains match the blocked template. The partition domain is
`{1,2,3,4,8,16,32}`, filtered by physical tile count and the largest matrix
reduction's number of minimum-size K blocks. Each operator uses
`P = min(partitions, ceil(K/tile_k))`; this explicit saturation creates no
empty partials. Details report requested and effective values and the exact
contiguous K ranges. A requested count outside the finite domain is rejected.
The runtime also rejects insufficient capacity for P simultaneous workspaces.
This is a bounded software mapping space, not a compiler-optimal plan.

K blocks are divided into P balanced contiguous groups. Output group g uses
execution tiles `g*P ... g*P+P-1` in each wave. Each tile initializes a private
FP32 accumulator and gathers its own K segment from the canonical owners;
there is no free operand multicast. Partial dot products retain the sourced
FP32-memory matrix rate described above. Every K round ends with the same
modeled BSP compute boundary.

A fixed binary tree sends right subtree roots to left roots at strides
1, 2, 4, and so on, carrying an unmatched tail forward. For each output
element there are exactly P-1 FP32 additions and P-1 transfers of four bytes.
Each transfer reads source SRAM, writes a separate destination receive buffer
and occupies the corresponding TX/RX endpoints. Each addition reads the
accumulator and receive buffer and writes the accumulator: 12 bytes/element.
A level-wide join precedes additions; a global synchronization follows them.
No reduction or synchronization cost is replaced by zero or by a network-wide
bandwidth. Only root tile zero within the group narrows to FP16 and scatters
the result after the last reduction level.

For requested partitions above one, workspace conservatively adds another
`4*r*c` bytes for a receive buffer to the blocked formula, even when an
operator's effective P saturates at one. Partial and receive buffers remain
held until the wave completes. Canonical placement, all-tensor residency,
per-tile peak checks and implementation limits remain unchanged. With
`partitions=1`, workspace, traffic and latency match the blocked schedule;
there are no reduction tasks. Greater partition counts need not improve
latency because extra transfers, additions and barriers remain explicit.

The internal `fp32_add` kernel uses the existing `vector_ops_per_cycle`
resource (2 FP32 arithmetic operations/cycle for the GC200 preset), not the
AMP MAC rate. Graphcore's [2021 tile parallelism article](https://www.graphcore.ai/posts/exploiting-tile-level-parallelism-on-graphcore-ipus)
describes barrel-scheduled workers and FP32 vector arithmetic. The six worker
contexts do not multiply the per-cycle issue rate by six. The same compute
resource serializes matrix, vector and reduction arithmetic.

The versioned [Poplin MatMul API 3.1.0](https://docs.graphcore.ai/projects/poplar-api/en/3.1.0/poplibs/poplin/MatMul.html)
documents float/half partials and optional multi-stage reduction. This supports
the mechanism, not this particular tree or placement, and does not establish
an experiment's SDK or actual compiled plan. Planner memory proportions are
not physical capacities or efficiency factors in this model. FP32 tree
reassociation changes addition order; reference arithmetic uses tolerance,
not bitwise or compiler-output equivalence. Conversion instruction timing
and the previously listed unknown terms remain omitted explicitly.

`details["manycore"]` records `schedule="split_k"`, `partition_plan`, job
output rectangles/execution tiles, reduction edges/levels/addition counts,
actual capacity peaks and source parameters. Reduction transfers have the
`ReduceTransfer` label and arithmetic has `ReduceAdd`; task IDs allow a full
trace check. The `ipu_split_k` mapping in evaluation provenance and warnings
distinguishes this analytical plan from a recovered library plan. It is
discoverable through the existing mapping registry and sampling interfaces.

### Decode operator

The public operator interface is:

```python
attention = WorkloadSpec("operator", {
    "op": "attention", "batch": 1, "heads": 32, "head_dim": 128,
    "seq_len": 1024, "query_len": 1, "dtype": "fp16",
    "kv_storage": "on-chip",
})
```

Per head it constructs `Q[1,D] @ K[D,S]`, scales by `1/sqrt(D)`, applies
stable row softmax, then multiplies `prob[1,S] @ V[S,D]`. Heads fold into
the batch dimension. K is represented in its logical transposed layout;
physical transpose/conversion costs are excluded. QK/AV accumulate internally
in FP32; all stored tensors are FP16. This is dense one-query decode, not
S×S prefill. Scores, scaled scores and probabilities are separate materialized
tensors. It is a segmented implementation, with no fused/streaming equivalence
claim. Softmax exponential/reciprocal and FP16 conversion timing remain
explicitly unmodeled. Its two matrix phases and byte dependencies are still
scheduled. Full-K AV may exceed private staging capacity for long contexts;
the explicit blocked strategy partitions its sequence reduction while keeping
FP32 accumulators resident. It also partitions QK's head-dimension reduction.
Scale and softmax still operate on full rows with their input/output workspace
accounted for; scores remain materialized and exp/reciprocal timing unknown.

`kv_storage="remote"` keeps K/V in a distinct external store and charges
remote reads, destination SRAM writes and IPU receive resources. The preset
has **no default remote attachment** because board-level bandwidth descriptions
do not establish the benchmark's per-IPU path. It therefore rejects remote KV
unless the caller supplies both `remote_bandwidth=Param(...,"B/ns",source)`
and `remote_capacity=Param(...,"B",source)`. This is an explicit user-sourced
attachment scenario, not evidence that its values match the target machine.
Remote startup/gateway topology remain omitted. Local KV uses tile capacity;
the two modes never silently substitute for each other.

## Results and replay

`Evaluation.details["manycore"]` contains selected mapping, workload semantics,
source parameters, canonical resident bytes, per-tile local read/write,
send/receive/remote bytes, per-job isolated compute time, phase task IDs and
unmodeled terms. `isolated_service_ns` reports resource-work/capacity bounds;
these are **not actual phase durations and must not be summed across tiles**.
Use `trace=True` for task start/end times and `breakdown` for operator/label
spans. Repeated wave spans can enclose other phases; do not sum those spans
as if they were disjoint.

Serialize `adf.to_dict()` as `seed_adf` in an Agent configuration, set
`mapping_templates` to the dedicated template, `backend="events"`,
`cost_budget_usd=null`, `area_budget_mm2=null`, and both penalties to zero.
Evaluate with an explicit mapping, submit the candidate and replay with
`archexplorer evaluate submitted.json --output replayed.json`. The serialized
ADF carries the preset's source parameters and works without a named search
seed. No new architecture/search transforms are implied.

Independent mathematical tests reconstruct matrix output blocks from the
mapping's source intervals, including batched/shared-B GEMM, MVM and both
attention matrix phases. They validate values/dependencies in reference
arithmetic, not device rounding, firmware or executable compiler output.
