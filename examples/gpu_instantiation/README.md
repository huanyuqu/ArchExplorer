# Component GPU versus original-cycle oracle

Run this example with an installed ArchExplorer containing the component GPU
provider, plus a separately built `arch-explorer.accelsim-component-engine.v1`
manifest. It requires CPU simulation, not GPU execution. An older joint engine
manifest cannot substitute for the component build.

The helper uses `archs.gpu_instantiation.case_specs`, generic `demo` preparation
and execution, and the component package's public `compare_results`. It contains
no native model, counters, clock conversion or comparison tolerances. Each case
executes components first, then an independent original-cycle process, with
fresh directories for both mappings (32 and 64 threads per block).

From this directory, with an absolute path to the new engine manifest:

```sh
python workflow.py create --model vit --engine /ABS/NEW/engine.json --output cases/vit.json
python workflow.py topology cases/vit.json --output cases/vit-topology.tsv
python workflow.py run cases/vit.json --output runs/vit
python workflow.py replay runs/vit --output replay-vit.json

python workflow.py create --model clip --engine /ABS/NEW/engine.json --output cases/clip.json
python workflow.py run cases/clip.json --output runs/clip
python workflow.py replay runs/clip --output replay-clip.json
```

`case.json` preserves both complete scenes. `candidate/` and `oracle/` preserve
standard demo manifests, requests and results. Each `comparison-*.json` is the
unmodified public comparison report. `run.json` reaches `compared` only after
every pair passes. Failure preserves available outputs and records `failed`;
prepare a new directory after correcting the cause. Replay verifies the saved
case and invokes the read-only public comparator again. Its output must be
outside the original run tree.

New component-engine results use physical completion for the primary
`Evaluation.latency_ns`. The helper labels this `physical_system_drain` and
copies both foreground and system-drained timings into each comparison's
`timing` presentation. Native foreground cycles and exact ns remain separate
from final physical cycles and exact ns. A real drain tail is a separate
breakdown phase, not another kernel or part of the last kernel. Both foreground
and final/tail comparisons must pass. The authority preserves `foreground_final`,
`drain_ticks` and `drain_core_cycles`; clock ticks are not core cycles. Earlier
engine.v1/v2 results retain their earlier timing contract.

For detailed DRAM with independent SM/ICNT/L2/DRAM clock frequencies:

```sh
python workflow.py create --model vit --engine /ABS/NEW/engine.json --hardware hardware-async.json --output cases/vit-async.json
python workflow.py run cases/vit-async.json --output runs/vit-async
```

The hardware JSON contains public `gpu_adf` keyword arguments. The generated
case exposes all physical connections. User-supplied edits must remain valid
on both sides; missing/wrong/extra connections fail topology validation. Clock
changes alone do not prove pressure coverage: inspect the actual cache, DRAM,
ICNT and backpressure metrics in the comparison, and report unexercised paths.

Keep native executions serial. Defaults bound each process to 600 seconds,
8 GiB and 10 million core cycles; optional `--options limits.json` can reduce
these limits. Native builds use one compilation job. Preserve failed runs.

These are complete **small synthetic** FP32 ViT/CLIP forward programs. CLIP
includes both towers and similarity outputs. Inputs and weights are generated,
and the dynamic trace is reference-generated, not captured from a framework's
CUDA implementation. Equality establishes software consistency for these
programs and configurations; it does not establish hardware timing accuracy.
See [the tutorial](../../docs/gpu-instantiation.md) for scope and API details.

## Standalone transport caller

`standalone_icnt.cpp` is an external C++ caller of the public
`make_icnt(IcntConfig)` factory and the generic dynamic Runtime. It constructs
two native LocalInterconnect instances in one process, uses different buffers
and clocks, and connects ordinary request/reply endpoints. One instance is
destroyed while the other has queued traffic. The echo endpoint is a transport
workload; it does not implement a cache or a memory controller.

With the matching main wheel installed, run from a directory outside the SDK:

```sh
python /ABS/EXAMPLES/standalone.py \
  --sdk /ABS/NATIVE-SDK/SDK.json --target icnt \
  --output /ABS/NEW/standalone-icnt
```

Supply the matching `arch-explorer.accelsim-standalone-sdk.v1` manifest. The
component package validates installed header hashes, actual static archives,
their separate build receipts and dependency closure. An engine manifest or
executable cannot replace this input. The runner delegates linking, constructor
audit and bounded execution to public `native_sdk.compile_consumer` and
`native_sdk.run_consumer`, preserving their receipts in `run.json` alongside
the caller result and unchanged package file hashes. It does not rebuild the
native libraries, copy header aliases or write to the installation. Build and
run failures retain their public process logs and failed receipt.

The caller checks response IDs, addresses, lengths, masks and operations,
retries rejected requests, and requires all Runtime lifetimes and native
packets to drain. Its result preserves the library's own queue, arbitration and
backpressure metrics. The receipt describes this caller execution; it is not a
GPU provider Evaluation, an L2/DRAM reuse result, or the GPU behavior comparison.
Actual results require a matching built library and a successful execution;
syntax checks alone do not establish these properties.

## Non-GPU memory workload

`standalone_memory.cpp` uses public `MemoryOptions` / `MemorySystem`, a
convenience assembly of the same `make_l2` and `make_dram` factories used by the
GPU assembly. The caller includes only `memory_system.hpp`. It needs no shader,
kernel, warp configuration, raw native request pointer, or GPU host. Its request
generator lives in this external program.

```sh
python /ABS/EXAMPLES/standalone.py \
  --sdk /ABS/NATIVE-SDK/SDK.json --target memory \
  --source /ABS/EXAMPLES/standalone_memory.cpp \
  --output /ABS/NEW/standalone-memory
```

Select the **memory** archive target for this source. The caller
warms a cache line, reads it again, writes dirty lines, then streams conflicting
addresses with different sector offsets and lengths. It requires actual native
hits, misses, DRAM reads/writes, dirty writebacks and rejected admissions before
reporting success. It checks every returned ID, address, length, line mask and
read-reply/write-ack type. Public snapshots keep consumer, physical and resource
completion separate; cache contents are not discarded to manufacture drain.

Two systems use different channel counts, caches and asynchronous clocks, with
overlapping addresses. They progress together. While B still has work, the
caller destroys A, rejects an invalid construction, and destroys a third system
with in-flight requests, checking B's state each time. B must then finish with
the same full result as a separately run B control. The ICNT caller likewise
compares its B instance with an isolated control.

These are timing/transaction workloads, not a memory data-value model or GPU
program. The structured result records the actual configurations, requests,
responses and native snapshots. All assertions still require actual installed
execution against matching libraries; a syntax-only check is preparation.
