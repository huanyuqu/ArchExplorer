# Accel-Sim Componentization: Boundaries and Verification Status

The following preserves the source review and design rationale for the baseline version. This baseline only supports the joint provider of components.accelsim,
and there is no independently connectable SM/L2 implementation yet. The new component instantiation and original differential flow are described in
[GPU Instantiation](gpu-instantiation.md); until actual installation and mechanism verification are completed, the target organization cannot be treated as implemented.
The new workload extension itself also does not expand the native engine's ISA, precision, or graph support scope.

## Review Basis

The project's currently declared engine GPGPU-Sim baseline `c24e8858eee552b3bc2919b147f14e2a88571775` was inspected (the local build additionally includes explicit variant patches), and compared against upstream code. The web dev branch may change, so actual extraction must be bound to a specific version and patches.

- [gpu-sim.cc](https://github.com/accel-sim/gpgpu-sim_distribution/blob/c24e8858eee552b3bc2919b147f14e2a88571775/src/gpgpu-sim/gpu-sim.cc): `next_clock_domain` and `gpgpu_sim::cycle` schedule CORE/L2/DRAM/ICNT; caches check queues before receiving, and responses flow back through the interconnect.
- [l2cache.h](https://github.com/accel-sim/gpgpu-sim_distribution/blob/c24e8858eee552b3bc2919b147f14e2a88571775/src/gpgpu-sim/l2cache.h): memory_sub_partition exposes full/push/top/pop/cache_cycle and L2-to-DRAM queues; it still references gpgpu_sim/config/stats/mem_fetch, and is not yet an independent library interface.
- [shader.h](https://github.com/accel-sim/gpgpu-sim_distribution/blob/c24e8858eee552b3bc2919b147f14e2a88571775/src/gpgpu-sim/shader.h): The SM contains warps, scoreboard, execution pipelines, load/store and synchronization mechanisms, and depends on the GPU context.
- [gpu-sim.h](https://github.com/accel-sim/gpgpu-sim_distribution/blob/c24e8858eee552b3bc2919b147f14e2a88571775/src/gpgpu-sim/gpu-sim.h): sst_gpgpu_sim has external memory access boundaries such as SST_receive_mem_reply and SST_cycle. This shows that splitting and connecting has precedent; this effort did not run or verify SST integration.

## Judgment

Feasible. The reused objects are the original state machines, queues, and cycle advancement mechanisms. Merely wrapping SM/L2 with separate run functions or separately computing a single total delay cannot preserve their interactions. The modules' independent testability and closed-loop coordination within the complete system must both be satisfied.

The transaction boundaries of L2/DRAM/interconnect are generally more suitable for cross-hardware reuse than SM internal mechanisms. When reusing L2, request granularity, address mapping, sector/line, write policy, MSHR/coalescing, atomicity and coherence scope must still be clarified; the original GPU cache does not automatically become a general-purpose cache with CPU coherence. The SM is suitable for other SIMT or GPU-similar architectures, and cannot be reused as an arbitrary NPU matrix array merely by renaming.

## Target Organization (Proposal)

```text
Component package: SM/warp execution → L1 and memory access coalescing → interconnect → L2 partition → DRAM controller
                               ← response and backpressure ←
Common framework: instance connection, transaction routing, clock domains, event ordering, lifecycle
ADF: component instances and parameters, port connections, clock/address mapping declarations
mapping/frontend: operator graph → tiling/program or trace → execution input acceptable to SM
```

Component mechanisms remain in components. Public contracts can be exposed to Python through a C++ adaptation layer, avoiding cross-Python calls for every internal cycle; batch advancement is only allowed for intervals proven to have no visible external events. The first extraction phase should prioritize preserving the original tick order and clock phase, and must not first introduce optimistic time skipping and then compensate with approximate total delay.

## Framework Contracts That Need to Be Added

The current submit/next_event_time/advance mainly consumes pre-declared tasks and reports completion/availability events, and does not yet have complete dynamic sub-request ports and response routing. Cache misses generated during SM execution and their returns affect subsequent issue; pre-enumerating with a static TaskGraph is not sufficient.

A transaction interface independent of GPU names needs to be defined, including at least:

- Request ID, parent request, source/destination ports, read/write/atomic type, address/length/byte mask, ordering constraints, and necessary execution context.
- Accept/reject and credit/queue availability events; rejection has no side effects, all queues are bounded; ownership transfer and response matching are unique.
- Clock domains, same-time phase ordering, cross-domain visibility time, earliest safe event time; when same-time chained events occur, advance to a stable state.
- Request generation and response dispatch are coordinated by the framework; components cannot bypass the scheduler to privately advance another component.
- Consumer readiness, physical completion, resource return, kernel completion, and system drain are modeled separately; the current event graph's consumer all-complete does not mean all physical transactions are drained.
- Each evaluation has independent context/state and statistics, eliminating the dependency on sharing the entire GPU singleton; a cache owns only one copy of real state, without adding an outer Demand layer that double-charges.

The interface needs to be verified for generality on two different component families, rather than using mem_fetch pointers or GPU global context as the common framework protocol. mem_fetch can remain inside the native adaptation layer.

## Recommended Implementation Order and Acceptance

1. Pin the native version, configuration, input, and small program, and save the original complete engine results as the differential baseline.
2. Extract the L2+DRAM subsystem, drive it with a controllable request source, and check hits, misses, coalescing, writeback, queue full, return ordering, and drain; then use a non-SM request source to prove reuse.
3. Separate the interconnect and clock adaptation, verify that contention and backpressure form a closed loop, with no request loss, duplicate completion, or lack of progress.
4. Connect SM/cluster and trace/kernel lifecycle, preserving complete SIMT and instruction dependencies. Reassemble a GPU equivalent to the original configuration, and compare per-boundary requests, cycles, cache statistics, stalls, and completion order; deviations under the same configuration should be explained and fixed, not covered up with fitting coefficients.
5. After behavioral equivalence, replace one component or topology to study new hardware combinations. Functional output and physical accuracy are verified separately.

Trace-driven execution preserves the recorded dynamic instruction and address streams; mapping or architectural changes that alter program paths, addresses, or synchronization behavior may require regenerating inputs, and a single trace cannot be assumed valid for all new hardware. Consistency of native whole-engine results only proves that extraction has not changed covered behavior; real hardware accuracy still requires independent measurement.

The above content is a prior feasibility review, not a completion statement for the new component path. Joint execution is retained as a known path and independent differential
baseline; the new path's installable interface, actual coverage, and limitations are described by the instantiation tutorial and this comparison of original documents.
