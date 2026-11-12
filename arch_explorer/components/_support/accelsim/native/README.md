# Independent native factories

The authority for each installed C++ factory now lives with its component:
SM in `components/sm/implementations/accelsim/native`, L2 in the matching
`l2` directory, DRAM in `dram`, and interconnects in `icnt`. GPU connections
live in `components/assemblies/gpu/native`. This support directory owns only
shared configuration, services, transactions, memory statistics and the
non-GPU memory convenience assembly. `native_resources.native_files()` is the
single installed-resource locator. It supplies the original flat header names
without retaining another copy in the former `accelsim/component_native`.

`l2.cc` and `dram.cc` contain their respective original factory, admission,
queue, metrics and drain methods. `memory_statistics.cc` supplies their shared
configuration checks and statistics factory. Builds compile all three instead
of the former mixed `memory.cc`; public class layouts and headers are unchanged.

These installed source resources expose independently owned SM clusters,
interconnects, L2 subpartitions and DRAM partitions. The fixed local Accel-Sim
and GPGPU-Sim inputs remain the authority for instruction, cache, arbitration
and DRAM mechanisms. No upstream vendor tree or binary is included here.

`memory_api.hpp`, `icnt_api.hpp` and `shader_api.hpp` declare the factories.
Each factory owns the actual native instance. Configuration, clocks,
transaction allocation and necessary statistics are explicit borrowed services
whose lifetimes cover the component. DRAM borrows an explicitly supplied L2
set. There is no GPU host argument or lookup service.

`MemorySystem` is a convenience assembly of these L2/DRAM factories for an
external source. Its options contain memory geometry, queues and timing;
it creates no shader, warp, kernel, CUDA API or GPU environment. Its passive
`receive`, `snapshot` and `drained` operations never advance a mechanism.
Requests and responses retain native identity, address, masks and ownership.
Rejection leaves the source's request queued for its next scheduled phase.

`make_icnt` creates the original per-instance BookSim or Local network.
BookSim uses explicit per-instance route, RNG, pool and traffic state. Its
extracted policy is fly/dest_tag/iq/islip; unsupported policies fail before
network construction. Passive queries do not advance arbitration. Interval
initialization preserves the native meaning and never discards packets.

`GpuAssembly` connects the same factories through
`arch_explorer.simulator.dynamic.runtime.hpp`. Components never receive the
assembly. SMs receive separate CycleView, ProgramDispatch, ShaderCounters,
ShaderTransport and CacheServices interfaces. ProgramExecution owns only
program slots, CTA selection and retirement. The generic Runtime has no GPU,
kernel, model or operator branches. A minimal low-level context supplies
instance identity and an optional instruction-statistics sink; it creates no
CUDA, PTX parser, functional simulator, device-runtime or GPU service.

`native_sdk` exposes installed include paths and source-bound archive receipts.
External C++ callers explicitly compile and execute through bounded CPU process
groups. The GPU archive contains the same objects as the independent image,
excluding only its CLI main. `native_types=True` supplies the corresponding
fixed native configuration/header closure for callers of the raw factories.
MemorySystem and the public ICNT API need only the installed public headers.

`tools/accelsim_native/componentization/native/standalone_build.py` prepares a
new checkout of fixed local inputs, applies the fixed typed patches, performs
controlled service extraction, and compiles with one job. Its explicit source
lists exclude gpu-sim.cc, gpgpusim_entrypoint.cc, accel-sim.cc and the CUDA
runtime host implementation. The resulting image has only components mode.
The frozen original-cycle oracle is a separate unchanged image. The former
borrowed-host builder and hooks are retired; new component requests require
the versioned standalone manifest. Readers reject mixed source/ABI identities.

The first supported GPU assembly is single-chiplet with LRC disabled. SM
clusters include original L1/shared mechanisms. Ordered trace programs use
stream zero. Functional/CDP/SST/perfect-memory, power, visualizer, debug
watchpoints and synthetic L2 flush services are explicitly unsupported.
Pipeline-depth instruction timing comes from the original native options.

Foreground per-kernel counters and the final foreground snapshot retain their
original meaning. Finish then advances actual remaining queues through the
same components until physical drain. It records separate drain ticks and core
cycles, never drops queues or invents fixed tail time, and cannot turn a limit
failure into success. Software consistency does not imply measured hardware
accuracy.

Metrics v2 reports actual accepted transaction lengths in `ports` and every
native allocation in `allocation_domain`. `lifecycle` retains the v1 wire-only
request population. The frozen v1 oracle's ICNT egress observer used its reply
length formula in both directions. `oracle_observations.ports` explicitly
preserves that historical observation; it never changes the actual Runtime
descriptor, queue admission or network byte counters. The reader verifies both
views from the pinned 8B control-packet protocol, and comparison uses the exact
common observation. Raw oracle artifacts remain unchanged.
