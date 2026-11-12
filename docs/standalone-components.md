# Calling Native Components in Non-GPU Scenarios

Create components from public factories; the caller owns their lifetimes. Native components handle their own queue, arbitration, cache, or controller mechanisms; the generic `ae_dynamic::Runtime` handles port connections, transaction ownership, clock phases, and completion stages. The caller provides external requests and does not need to inherit GPU classes. Independence should be verified against the factory's actual parameters and dependency closure, then run same-process isolation cases.

The dual-memory and dual-ICNT caller on this page is located in the [user example](../examples/gpu_instantiation/README.md#standalone-transport-caller). It uses actual library interfaces; the matching native static library required for execution is built separately from the public wheel. Code or header availability does not equal actual successful execution.

## Non-GPU Memory Request Sources

The public `MemorySystem` in `memory_system.hpp` is a convenience assembly layer for the fixed source↔L2↔DRAM topology. It calls the same `make_l2` / `make_dram` used in GPU assembly and owns this group's configuration, cycle view, transaction allocation domain, statistics, endpoints, and Runtime. Factories borrow the narrow services each needs; no GPU program or whole-machine object is passed in. When reviewing direct factory service fields and destruction order, use the matching version's component public interfaces and build dependency documentation.

The external caller only provides memory parameters and requests:

```cpp
namespace native = ae_accelsim::standalone;
native::MemoryOptions options;
options.channels = 2;
options.subpartitions_per_channel = 1;
options.cache_sets = 4;
options.cache_ways = 1;
options.controller_queue = 32;
options.simple_dram = false;
options.cache_period = 1.0;
options.dram_period = 2.0;
native::MemorySystem memory(options);
auto request_id = memory.submit(0x1000, 32, false);
memory.tick();
ae_dynamic::Transaction response;
while (memory.receive(response)) {
    // Correlate response.id with the submitted request metadata.
}
```

`native` refers to `ae_accelsim::standalone`. This snippet only demonstrates the interface; one tick does not mean the request is complete. The complete [caller](../examples/gpu_instantiation/standalone_memory.cpp) continues advancing until all requests are received and `drained()` is true, without creating SM, kernel, CTA, or warp services during that time. `receive` only retrieves responses that have already been produced and does not advance caches or controllers. `snapshot()` preserves native L2/DRAM metrics and separate consumer, physical, and resource completion counts; response arrival cannot substitute for draining the entire physical system.

Request sizes are integer multiples of 32B, at most 128B per request, addresses are 32B-aligned, and requests do not cross 128B lines. Responses preserve in-line byte masks; read replies are `read_reply`, write acknowledgments are `write_ack`. `transfer_bytes` is the real transfer length and cannot be unconditionally equated to the logical request's `bytes`. The example validates these relationships and does not claim to simulate memory values.

Multi-sector logical requests must wait for all actual sector replies before returning a single credential, with the same ID as the logical submission and parent_id=0. byte_mask merges valid bytes, and transfer_bytes accumulates real responses: a 32B sector read reply is 40B, a write acknowledgment is 8B; therefore the aggregate traffic for a 64B read is 80B. submitted/received count logical requests, and the three lifecycle counts count actual wire and internal traffic; physical counts cannot be forcibly set to the logical request count. Ready credentials not taken by the user do not prevent physical draining.

The example uses detailed DRAM, a single chiplet, and LRC disabled assembly scope. It issues hot-line reads, writes, and conflicting streams exceeding cache capacity in phases, checking actual hits/misses, reads/writes, dirty-line writebacks, and backpressure. The two systems interleave progress with different channel/cache/clock configurations; destroying or failing to construct the other instance cannot change B; B must ultimately equal a separately run control group with the same configuration. The fixed convenience topology does not represent coverage of all arbitrary direct-factory connections or all native configurations.

The convenience configuration uses sector cache, LRU, writeback, fetch-on-write, and linear index. Users set explicit geometry rather than passing a string of generic GPU configuration items: channels is a power of 2 from 1..64, subpartitions/channel is 1/2/4, cache_sets is a power of 2 from 1..65536, cache_ways is 1..1024, combined cache capacity is at most 1 GiB; controller_queue is 1..65536, and clock cycles are finite and positive. Broader native policies require explicit configuration of the underlying factories.

Borrowing and lifetimes of underlying factories must be checked separately:

| Dependency | Owner and usage |
| --- | --- |
| Configuration and cycle view | Owned by the assembly layer; components read their own parameters and `CycleView`, and do not take time from the GPU |
| Request ID and transaction domain | Each assembly owns its own; responsible for association, allocation, and return; cancellation is not marked as completion |
| Memory statistics | Owned by the assembly layer; the factory receives an explicit statistics reference, and does not require external source configuration of SM/warp |
| L2 | `make_l2` returns a `unique_ptr` that owns the actual instance |
| DRAM | `make_dram` returns a `unique_ptr`, explicitly borrowing this partition's L2 |
| Program/CTA execution services | Only needed by the SM factory; memory and ICNT do not create these services |

Destroy the DRAM that borrows L2 first, then destroy L2, then release statistics/config/context, and finally close the request arena.
`MemorySystem` manages this set of dependencies; callers of the direct factory must follow the same order. The native dependency description for the matching version should list
field purposes that preserve the underlying types item by item, and cannot simply rename the whole-machine host to Context.

## Creating and composing ICNT

The public header `components/accelsim/component_native/icnt_api.hpp` provides:

```cpp
namespace native = ae_accelsim::standalone;
native::IcntConfig config;
config.mode = 2;
config.request_nodes = 1;
config.memory_nodes = 1;
config.local.input_buffer = 2;
config.local.output_buffer = 2;
auto network = native::make_icnt(config);
```

The factory returns a `std::unique_ptr<Icnt>` held by the caller, and does not receive GPU, kernel, or instruction execution services.
mode 2 uses the native LocalInterconnect; mode 1 uses BookSim, which additionally requires its configuration file. The example only selects
mode 2, one pair of request/memory endpoints, and does not extrapolate this use case to mean all BookSim topologies are verified.

`has_buffer(input, transfer_bytes)` is an admission query; `push(input, output, transaction, packet_type)`
submits a request or reply. On rejection the caller retains the transaction and retries, and must not create duplicate logical requests. `advance()` advances the native interconnect by one step,
`peek(output)` views the output head; only after the downstream accepts does `pop(output, result)` occur. Request endpoint numbers come first, memory endpoint numbers
come after; the example uses request 0, memory 1, and replies travel in the opposite direction. `Transaction` stores ID, address, logical length, transfer length,
byte mask, and ordering; read/write request/reply types are specified by the public `IcntPacketType`.

The external network adapter implements the generic Runtime's passive `can_accept/accept/drained`, delegating admission and enqueueing to the native object.
The accept callback only receives the transaction, and the reply is sent in the next explicit phase, avoiding recursive routing. The example is formed through four ordinary connections:

```text
source.request → icnt.request → echo.request
source.reply   ← icnt.response ← echo.reply
```

Each instance registers its own clock and phase, and the Runtime and endpoints remain valid during component use. The phase callback only borrows the corresponding node,
and does not pass the host object that owns all configuration and state to the factory. The echo in the example only produces transfer replies, and does not simulate L2 or DRAM.

## Running, sources, and completion

First install the main library wheel, then invoke the example's `standalone.py` from outside the repository, passing the matching standalone SDK manifest and
`--target memory` or `--target icnt`. The SDK manifest uses
`arch-explorer.accelsim-standalone-sdk.v1`, and stores the actual archives, their respective build receipts, and dependencies;
`native_sdk.load_sdk` verifies the installed header SHA and these references. Preserve the paths and hashes of referenced source files and dependencies.
Building the SDK itself depends on controlled upstream extraction of source code and toolchain; compiling these two external caller programs uses only public headers and static libraries.
Preserve upstream sources, licenses, and NOTICE; the SDK cannot be replaced with a whole-GPU executable, and the wheel does not carry third-party binaries.

The runner uses the public `native_sdk.compile_consumer` / `run_consumer`, and does not duplicate the linking and process management protocol.
The component package stores source/binary/SDK bindings, ELF construction audit, and `PROCESS.json`; the example preserves these public receipts and
file hashes before and after the installation package. By default each CPU subprocess group is limited to at most 600 seconds and 8 GiB, compiles one job, and on failure or timeout reclaims
the entire process group and leaves the original files. `--memory-mib` can lower the run budget, while the compile budget remains 8 GiB. Fail when interfaces or dependencies are missing,
and do not fall back to a Python interconnect model, manual header aliases, or whole-engine wrappers. Check the whole-machine host dependencies against the combined library's source and actual construction evidence,
and cannot conclude independence merely because the filename contains standalone.

The caller program creates two interconnect instances with different buffers and clocks, interleaves their advancement, and uses overlapping transaction IDs and partially identical addresses to verify namespace
isolation. It destroys A while B still has traffic, saves B's native statistics before and after, and verifies that an invalid third construction does not change B, then
completes B. Downstream delayed reception creates backpressure; the program checks response ID, address, length, mask, read/write type, and exactly-once completion.
Finally it requires native packets, flits, credits, and Runtime transactions to all be drained simultaneously.

The pure transfer use case records its own consumption/physical/resource completion after reply delivery. This condition cannot be copied as the completion condition for DRAM:
memory may still have writebacks, controller, or resource return tails, and the memory component's actual completion and drain interfaces should be used.

`run.json` is the run receipt for this external caller program, preserving its assertions and native metrics, and is not the provider's `Evaluation`.
When there is no execution or failure, successful reuse of the standalone library cannot be claimed. The behavior of GPU reassembly remains verified separately using
[the common comparator for components and originals](gpu-instantiation.md#checking-with-the-common-comparator), and cannot be inferred from the ICNT use case.
