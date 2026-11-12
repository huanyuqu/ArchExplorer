# Component library and simulation generation

The simulator generates execution from an architecture description (ADF, see
`arch_explorer/adf/`), a workload and a mapping. Components own the concrete
models and declare compatible execution capabilities. The registered `events`
executor uses the generic task/resource scheduler:

```python
from arch_explorer.archs import dmc_adf
from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate

result = evaluate(dmc_adf(),
                  WorkloadSpec("operator", {"op": "matmul", "m": 16, "k": 64, "n": 64}),
                  MappingSpec("dmc_partition", {"split": "n", "weights": "local", "activations": "local"}))
result.latency_ns, result.breakdown, result.utilization, result.provenance
```

`components.api.discover_components()` registers built-ins and installed
extensions; execution also performs discovery. Component authors import the
public `components.api`, not scheduler internals. See the
[installable extension tutorial](component-extensions.md).
For the current hardware-oriented directory and per-implementation contracts,
run `archexplorer components` and read [component implementations](component-implementations.md).
The tables below explain ordinary model behavior; required input for a selected
implementation comes from the public catalog, not the union of every model's
parameters. See the [component inventory](component-inventory.md) for supported mechanisms
and input requirements.
`evaluate(..., backend="events")` then runs four steps:

1. The hardware builder instantiates the ADF (`build_simulator`): it expands
   `repeat`, binds every SpacePoint to its component type, checks the type's
   conventions and assigns multi-level coordinates. Instance `i` of a repeated
   element is `name[i]`; `coord` lists the endpoint index at each level.
2. The mapping template turns the workload into an event task graph of compute,
   storage, communication and synchronization tasks placed on point instances.
3. The hardware-consistent scheduler executes the graph (below).
4. The backend reports latency, an operator breakdown, resource utilization,
   the provenance of every parameter the component models read, and warnings.
   Raw backend area/cost fields may remain `None`. Agent/search objectives use
   the independent [area/cost model](cost.md), including its assumptions,
   unsupported metrics and separate raw/full basis; see also [agent.md](agent.md).

Options: `scheduler="consistent"|"conservative"`, `systolic_model="native"|"scalesim"|"auto"`,
`check_capacity=True` (raise `CapacityError`; `False` turns it into a warning),
`iterations=N` (override the workload's iteration count), `trace=True`
(per-task records in `result.details["trace"]`). `result.details["stats"]`
always holds the scheduler statistics (commits, rollbacks), task and instance
counts, the scheduler name and the peak occupancy of every memory; both are
part of `result.to_dict()`.

## Units and counting conventions

Time is in ns, sizes in bytes, bandwidths in B/ns (= decimal GB/s). Unit strings
in ADF parameters are `B`, `B/ns`, `ns`, `GHz`, `ops/cycle`, `count` and `1`.

- `tensor_ops_per_cycle`, `simt_ops_per_cycle`: arithmetic operations with an
  FMA counted as two, the convention of vendor peak FLOPS. A SIMT lane retires
  one instruction per cycle, so lane instructions per cycle are
  `simt_ops_per_cycle / 2`.
- `sfu_ops_per_cycle`, `sfu_lanes`: special-function results per cycle.
- A GEMM `C[m,n] = A[m,k] B[k,n]` has `2mkn` operations.
- Cycle-based components (`Conventions.time_base == "cycle"`) return cycles;
  the builder divides by `clock_ghz`. The builder rejects a cycle-based type
  without `clock_ghz` in GHz, unknown time bases and unknown arbitrations.

## CGRA

`cgra_pe` is a scalar uint32 compute component with required `clock_ghz` (GHz)
and `precision` (1), cycle timing and FIFO issue arbitration. `cgra_spatial`
maps GEMM, MVM and two-layer Linear to per-tile and per-link events.
`vectorcgra_4x4` and `vectorcgra_4x4_diag` are sourced, single-array scenarios.
The optional boolean `mesh2d.diagonal` enables KING_MESH neighbor links;
generic routing takes diagonal steps first, then XY (false keeps existing XY).

CGRA evaluations expose `details.cycles.data_load`, `config`, `compute`,
`completion` and `end_to_end`. Unknown physical clock is explicitly marked;
reporting ns are cycle-normalized in that case. See [CGRA evidence, mapping
and counting boundary](cgra.md) for source locations, capacity requirements,
Agent examples and the limits of the analytic model. These presets do not
claim the paper's unpublished compiled schedule or its reported error.

## Components (catalog v2)

For `ipu_tile`, `tianjic_core`, `tile_exchange`, their parameter contracts and
the dedicated `ipu_bsp` / `tianjic_spatial` templates, see
[tile-private memory and BSP scenarios](manycore.md). These presets support
on-chip operation without fabricating a DRAM and explicitly mark incomplete
timing models. The same document describes the public single-query
`operator` workload `op="attention"` and local/remote KV boundaries.

These are the common parameters of ordinary event models; optional ones are marked.
`extensive` parameters are divided by `split` and summed by `merge`.

| component | role | parameters | time base / arbitration | model |
| --- | --- | --- | --- | --- |
| `systolic_array` | compute | `rows`, `cols` (count), `clock_ghz`, `dataflow` (os/ws/is), `precision`; optional `stationary_buffering` (`single` default / `double`, ws/is) | cycle / fifo | GEMM and MVM cycles of SCALE-Sim v3 (below), with `double` the next stationary tile loads during a fold; other kernels are rejected |
| `vector_unit` | compute | `lanes`, `sfu_lanes` (count), `clock_ghz`, `precision` | cycle / fifo | GEMM/MVM: one lane per MAC; vector kernels: `max(SIMT / lanes, SFU / sfu_lanes)` |
| `pim_unit` | compute | `ops_per_cycle` (MAC = 2), `clock_ghz`, `precision`; optional `act_banks` (banks per PIM activate, default 1), `row_buffer` (`shared` default / `dual`), `command_elements` (row elements per MAC command, default one column) | ns / fifo | in-bank GEMV (`mvm`) from the bank's DRAM command timing, all banks of a channel in lockstep (below); other kernels are rejected |
| `sram` | memory | `capacity` (B, ext.), `bandwidth` (B/ns, ext.), `latency` (ns) | ns / fair_share | one port of `bandwidth`; latency paid once per transfer |
| `dram` | memory | `technology` (ddr/gddr/hbm/bonded), `capacity`, `bandwidth` (peak), `latency`; optional `sustained_bandwidth`; optional bank timing `trcd`, `trp`, `tras`, `trrd`, `tfaw`, `tccd`, `tcl` (ns), `row_bytes`, `column_bytes` (B), read only by a `pim_unit` in the same container; optional refresh `trfc`, `trefi` (ns, both or neither) | ns / fair_share | port capacity = `sustained_bandwidth` when given, else `bandwidth`, times `1 - trfc/trefi` with refresh |
| `mesh2d` | network | `rows`, `cols`, `link_bandwidth` (per direction), `hop_latency`; optional `gateway_tile` (default 0) | ns / fair_share | endpoints placed row-major; XY routing; one resource per directed link |
| `bus` | network | `bandwidth`, `latency` | ns / fair_share | one shared resource |
| `crossbar` | network | `port_bandwidth` (per direction), `latency` | ns / fair_share | non-blocking; a transfer uses the source egress and destination ingress port |
| `p2p_switch` | network | `link_bandwidth` (per direction), `links_per_endpoint`, `latency`; optional `pair_links` | ns / fair_share | ports of `links_per_endpoint x link_bandwidth`; with `pair_links`, each directly connected pair is also limited to `pair_links x link_bandwidth` (switchless NVLink boards) |

A network connects the expanded elements of its container (in declaration
order) and the container's gateway to the parent level. A transfer between two
points climbs from the source container through each level's network to the
lowest common container, crosses it and descends. A container without a
network is transparent; the backend reports this as a warning.

### Systolic arrays and SCALE-Sim

`arch_explorer/components/systolic_model.py` reproduces SCALE-Sim v3's dataflow
timing in closed form (`native_gemm`) and can also run SCALE-Sim itself
(`scalesim_gemm`, optional dependency `pip install scalesim`). For dataflow
`os` the spatial dimensions are (m, n) and k is streamed; `ws` maps (k, n) and
streams m; `is` maps (k, m) and streams n. With `folds = ceil(Sr/rows) *
ceil(Sc/cols)`, compute cycles are `folds * (T + rows + cols - 2) - 1` for `os`
and `folds * (2 rows + cols + T - 2) - 1` for `ws`/`is`; SRAM access counts
follow SCALE-Sim's counters. The cross-check over 8x8 to 128x128 arrays, the
three dataflows and eight GEMM shapes (102 cases with SCALE-Sim runtime
permitting) matched exactly; see the WP-SIM-01 delivery report.

SCALE-Sim builds its demand matrices fold by fold, so its runtime grows
quadratically with folds (an 8x8 `os` run of 256x1024x1024 takes minutes).
The backend therefore defaults to `systolic_model="native"`; `"scalesim"` runs
the package and `"auto"` uses it whenever it is installed. SCALE-Sim 3.0.0's
memory simulation fails with NumPy >= 2.5; the adapter only calls its operand
and compute modules, which give the same stall-free cycle count.

The `ws` count includes filling the array with every fold's weights before
its rows stream (part of the `2 rows` term); SCALE-Sim does not overlap that
fill with the previous fold, and the default `stationary_buffering =
"single"` keeps that. For decode GEMMs with a handful of rows (`T` small) the
arrays then spend most cycles loading weights.

`stationary_buffering = "double"` describes an array with a second register
per PE that receives the next fold's stationary tile while the current fold
streams. The TPU v1 matrix unit holds "one 64 KiB tile of weights plus one for
double-buffering (to hide the 256 cycles it takes to shift a tile in)"
(Jouppi et al., ISCA'17, Sec. 2); NeuPIMs' `SystolicWS::issue_ex_inst`
starts a `GEMM_PRELOAD` while the previous instruction is still in the
pipeline and charges the full preload only when the pipeline is empty. A fold
then starts `max(rows, T)` cycles after the previous one: the shadow load
takes `rows` cycles and the array accepts one streamed vector per cycle. Only
the first fold pays the load and only the last the drain:

    compute_cycles = (2 rows + cols + T - 2) - 1 + (folds - 1) max(rows, T)

which equals the single-buffered count for one fold. A batched kernel (`b`
GEMMs) is one chain of `b x folds` folds. The same correction is applied to
SCALE-Sim's own count when `systolic_model` runs the package (the model name
gets `+double-buffered`). SRAM access counts are unchanged; `os` has no
stationary operand and rejects `double`. NeuPIMs' simulator differs in three
details not taken over: a preload starts `rows` cycles after the previous
instruction's start even when that instruction streams longer (two streams
would then share the array), the first preload costs `2 rows - 1` cycles,
and every instruction streams at least 4 cycles. The shadow registers' area
is not in the cost model.

### In-bank PIM (`pim_unit`)

`pim_transform(adf, dram, ["channel", "bank"], [C, B], spec)` splits a DRAM
into `C x B` bank parts `channel_i/<dram>_i/bank_j/<dram>_i_j` and puts the
spec's compute point (a `pim_unit`) and network (`bank_bus`) next to each
bank part. The `pim_unit` reads the bank timing of the DRAM in its own
container; all units of a channel (the container directly above the bank
containers) execute one command stream in lockstep (Newton, MICRO'20; AiM,
Hot Chips 34; NeuPIMs, ASPLOS'24 §5).

Supported kernel: GEMV `mvm (batch, m, k)`, `batch` matrices of `m x k`
stored in the banks' rows times one vector each. Decoder attention uses it
twice per head: `q . K^T` (matrix K, `seq x head_dim`) and `p . V` (matrix
`V^T`, `head_dim x seq`). Its time on a channel of `B` banks, element size
`e`, is NeuPIMs' Algorithm 1 with both latencies taken from DRAM timing:

- `GWRITE` (one row of the vector into the channel's global buffer):
  `max(tRCD + n_col tCCD, tRAS) + tRP`, `n_col = row_bytes / column_bytes`;
  `ceil(batch k e / row_bytes)` of them.
- tile (one row in each of the `B` banks): activate commands open `act_banks`
  banks each, `tRRD` apart, with at most four bank activations in any
  `tFAW` window; after the last activation `tRCD`, `row_bytes / (c e)` MAC
  commands of `max(tCCD, c / MACs per ns)`, the readout `tCL + r ceil(B e /
  column_bytes) tCCD`, then `tRP`, the precharge no earlier than `tRAS` after
  the last activation. `ceil(batch m k e / (row_bytes B))` tiles.

`c = command_elements` is the number of row elements one MAC command covers,
by default one column (`column_bytes / e`). `r` is the number of dot products
in one bank row: `row_bytes / (k e)` when a row holds several whole
`k`-vectors (NeuPIMs issues one `READRES` per head of the row), otherwise 1.

With NeuPIMs' HBM timing (32 banks, 4 per activation, tFAW 30 ns) and one
64 B column per command a tile takes `210 + 14 + 16 x 2 + 16 + 14 = 286` ns.
The `neupims` preset sets `command_elements = 16`, the configuration's
`pim_comp_coverage`: NewtonSim issues one `COMP` per 16 elements, `tCCD`
apart, so a 1 KB row takes 32 commands (64 ns) and a `q . K^T` tile with four
heads per row `210 + 14 + 64 + (14 + 4 x 2) + 14 = 324` ns. `act_banks = 16`
(AiM ACT16) opens a 16-bank channel with one command, so tFAW does not apply.

Refresh (`dram` parameters `trfc`, `trefi`): every `tREFI` the channel is
refreshed for `tRFC` and runs no command. A `GWRITE` or tile accumulates over
an open row and cannot be interrupted, so the controller starts it only if it
ends before the next refresh and otherwise waits for the refresh to finish
(Newton, MICRO'20 §IV: "delay the Newton operation if a refresh request would
mature within the operation's latency"; NeuPIMs §5.2 introduces
`PIM_HEADER` so that the controller knows a GEMV's latency and schedules it
around refresh; NewtonSim `NeuPIMSCommandQueue::CanMeetRefreshDeadline`). A
GEMV's operations (its `GWRITE`s, then its tiles) are packed into the
`tREFI - tRFC` windows. The task duration is computed before scheduling, so
the refresh phase at the GEMV's start is unknown; the duration is the mean
over 64 evenly spaced start phases. For NeuPIMs (`tRFC` 260, `tREFI` 3900)
eleven 324 ns tiles fit a 3640 ns window, about 355 ns per tile over long
GEMVs (+9.4 %: the 260 ns refresh plus the window's unused 76 ns tail).
Host accesses see refresh as a port capacity of `bandwidth (1 - tRFC/tREFI)`
unless a sustained measurement, which includes refresh, is given.

`row_buffer = "dual"` (NeuPIMs) lets normal accesses to a bank proceed during
PIM work; `shared` (single row buffer, Newton and AiM) charges the bank's
port for the whole PIM duration, so host traffic to those banks waits. Not
modeled: command-bus sharing between PIM and host commands (NewtonSim issues
one command per channel clock and serves its PIM queue first; this only
matters while host traffic and PIM work overlap on a channel), and reuse
of a partial sum across tiles in the result registers (every tile reads out).

#### Optional NeuPIMs simulator command semantics

NeuPIMs vector units set `matrix_kernels=false` (the reusable vector component
defaults to true). This matches `NeuPIMSystolicWS::issue_ex_inst` in upstream
f299af3: GEMM/PRELOAD issue to the systolic pipeline. Previously the fastest-unit
selection could put projection/FFN GEMMs on vector lanes, hiding array-size
effects. This capability restriction does not replace SCALE-Sim systolic timing
with a full NeuPIMs instruction scheduler. Upstream SRAM-sized outer tiles,
instruction preload/drain boundaries and transaction queues remain differences.

The ordinary-memory stream interface has aggregate bytes, not bank/row addresses
or per-transaction acceptance timestamps. Consequently GWRITE row preparation
and ordinary-command-induced refresh postponement remain unmodeled. Upstream
`BankState::GetReadyNormalCommand` distinguishes closed rows (ACT), hits (no
ACT/PRE), and conflicts (PRE then ACT). A fixed startup correction cannot
represent all three. WP-SIM-10 stage B therefore freezes an unchanged attention
implementation; no 17 ns or 30 ns correction is introduced.

On the existing 21 NeuPIMs reference cases this dispatch correction produces
no material latency improvement. For the batch-2 diagnostic, FFN-up still has
a shared DRAM service lower bound of 35400 ns plus a 29 ns access tail on both
64 and 128 arrays. Changing the selected compute resource does not remove
that bottleneck. The original simulator's capacity-driven outer tiling and
instruction-ready queues are still absent from this aggregate mapping. This
is a dispatch correction, not completion of the NPU/attention accuracy work.

These are `[simulator]` behaviors from NeuPIMs commit `f299af3`, including
`extern/NewtonSim`, rather than calibrated hardware timings. Only `neupims`
enables them by default. Removing the following optional parameters restores
the previous numerical model; `hsndi` is unchanged.

| Parameter | Component default | `neupims` | Meaning / source |
| --- | --- | --- | --- |
| `ranks` | 1 | 2 | Independent tFAW windows, one channel command per tick; `timing.cc`, `ChannelState::GetReadyCommand`. 32 banks / 4 per activation / 2 ranks gives G_ACT at 0,1,30,31,60,61,90,91 ns. |
| `refresh_mode` | `mean` | `rank_staggered` | `refresh.cc`: first rank refresh at tREFI/ranks, subsequent ranks staggered, per-rank period tREFI. Epoch is simulation time zero and crosses Operation boundaries. |
| `completion_mode` | `atomic` | `commands` | Separate issue, callback, next combined command, PRE eligibility, and consumer dependencies. `IssueCommand`, `UpdateChannelTiming`. |
| `attention_pipeline` | `off` | `heads` | Head softmax can overlap later QK groups; AV partial ADD follows all K chunks, matching the original instruction generator. |
| `write_response` | `physical` | `early` | Attention writes may acknowledge one tick after modeled acceptance while the write drain remains in the graph. `AddTransaction`; other host operators keep their previous semantics. |
| `tccd_s`, `burst_ns`, `trtp` | 1, 2, 5 ns | same | HBM ini / `configuration.cc`. `tRTP` defaults to 5, independently of tRTP_L/S. |

`commands` uses GWRITE `(tCCD_S+tCCD_L)*32` (96 ns here), the formula
implemented in `configuration.cc`; the factor 32 is not asserted to describe
hardware. A combined command with C computations returns at issue+CL+burst,
whereas the next combined command waits until issue+tCCD*(`max(C,6)`+1).
PRE eligibility is a distinct issue+tRTP constraint. For C=32 these are
16, 66, and 5 ns. A separate consumer node adds one modeled response tick.
Rank activation calendar waits are evaluated only after predecessor times
commit, so speculative scheduling never mutates a shared refresh phase.
The conservative header admission estimate follows
`ChannelState::EstimatePIMOperationLatency`; actual refresh may still be
postponed by normal traffic in the original simulator, which is not modeled.
With explicit commands and `mean`, individual command blocks retain a
64-phase expected wait, not a persistent absolute refresh calendar.

`pim_offload` builds QK head groups and AV K chunks from bank row capacity.
Softmax and ADD occupy host vector units. `NeuPIMSystolicWS` uses allocated
byte counts for instruction size: a 1536-element FP16 softmax on 128 lanes
takes `2*(24+1)+3*24=122` cycles. An AV ADD of 128 FP16 outputs takes two
cycles in that implementation, regardless of the number of partial inputs;
this is simulator compatibility, not an ideal mathematical reduction cost.
Write traffic includes the original inclusive end-address transaction.
Trace names expose header, rank-activate, combined-issue, bank-ready,
pre-ready, consumer, softmax, partial-add, and write drain milestones.
`details.pim_commands` reports command and byte counts.

An additional, opt-in source-configuration executor is available through
`evaluate(..., MappingSpec("pim_offload", {"npu_execution": "instructions"}))`.
It uses reusable matrix instruction IR, a double-buffered core executor, and
memory adapters, bound as stateful models in the same TaskGraph as stateless
Demand nodes. The [unified conservative driver](unified-execution.md) handles
requests, backpressure and consumer/physical completion; this path no longer
runs as a separate precomputed timing executor. Mechanisms live in reusable
instruction/controller modules, with NeuPIMs-specific policies kept in explicit
execution profiles. `MatMul` capacity halving and N/K/M loops emit MOVIN,
GEMM_PRELOAD/GEMM, and final-K MOVOUT instructions with valid tail regions.
Weighted matrices use the source constructor's transpose choice: after capacity
selection, exchange M/N and compute B-transpose times A-transpose. Logical output
coordinates are mapped back to the original tensor. This intentionally preserves
all output elements; the upstream MOVOUT code indexes its untransposed output
with the reversed coordinates and can discard rectangular out-of-bounds values.
Workloads without bias tensors do not acquire implicit bias loads.
The source configuration has one compute FIFO; its instruction `size` is
array width/8. This differs from independently scheduling the eight arrays
in the paper-specification preset. Cold and occupied-pipeline timing follow
`NeuPIMSystolicWS::issue_ex_inst`, not a latency multiplier. Scratchpad is
134218 KiB split into two resident slots; aligned operand/output footprint must
fit a half. This is the simulator's configuration, not a hardware SRAM claim:
`Common.cc::initialize_config` assigns both scratchpad capacities from
`sram_size`, ignoring the JSON `spad_size=1024` and `accum_spad_size=1024` fields.
All source-profile constants are reported in provenance.

Adding `"memory_execution": "transactions"` enables shared normal/PIM channel
controllers. READ/WRITE/GWRITE/HEADER/COMPS_READRES use one command bus, ordinary
and PIM row buffers, legal ACT/PRE/G_ACT transitions, separate finite 32-entry read/write/PIM ingress
and 128-entry per-bank command queues, and staggered rank refresh. The clocks
and queues persist across operators. WRITE can acknowledge before physical
issue; combined results can return before the next combined command is legal.
Header completion is internal control progress, not a data callback. Trace
fields distinguish arrival, acceptance, issue, return, bank readiness, and
consumer readiness. Every burst is counted once; this path adds no aggregate
DRAM stream. `details.neupims_transactions` contains counts and queue evidence;
Read coalescing and write forwarding follow `NeuPIMSController::AddTransaction`.
Write drain starts at a full buffer or more than eight writes with empty
command queues; PIM scheduling uses the source two-command threshold. Sparse
writes may remain buffered without an idle flush. Requested burst bytes and
issued DRAM bytes are reported separately, with pending unique writes.
`trace=True` adds command, transaction, and NPU instruction records. The layer
ends at consumer completion; outstanding physical writes are explicitly flagged.

This executor requires fp16 decode, one iteration and sub-batch, `gsm_tiled`,
square 64/128 arrays at 1 GHz, and the homogeneous 32-channel, 2-rank, 16-bank/rank
HBM2 dual-buffer source profile. It rejects other timing profiles, unsupported
GQA, overflowing KV rows, and attention return-slot capacity violations.
The generic vector components still supply non-attention arithmetic.
Normal addresses follow the source's circular cursor and column/channel
permutation, but deduplicated bursts use first-touch order rather than
`robin_hood` hash iteration. KV rows have an explicit disjoint contiguous
layout, not the original whole-model allocator base. Per-bank FIFO service
does not reproduce upstream precharge look-ahead or every queue heuristic.
The controller is an analytical source-based model, not an embedded
NeuPIMs simulator or a hardware calibration. GWRITE's factor 32 comes from
upstream configuration code; it is not independently established hardware timing.

Both options default to `aggregate`. Shared transactions require the instruction
path; compare aggregate, instructions alone, and instructions+transactions for
the staged ablation. Other presets and SCALE-Sim formulas remain unchanged.
The explicit program is evaluated directly; aggregate `run_mapping` rejects
these options rather than silently returning the old graph.

Limitations of the aggregate command path: command mode currently needs dual row buffers, complete rank
activation groups, whole COMP-sized QK heads fitting a row, and vector-unit
host components. Normal/PIM command-bus arbitration, GWRITE normal-row
preparation, exact request acceptance queues, and NPU instruction issue
queues remain analytical approximations. Activation calendar waits are
excluded from resource utilization. Early acknowledgement is implemented
only for the attention program's writes; the aggregate memory interface
does not expose per-transaction host WRITE acceptance elsewhere.

### Hybrid-bonded DRAM

`dram` with `technology = "bonded"` is a DRAM die stack bonded onto a core.
`arch_explorer.devices.bonded_dram(area_mm2, layers)` builds one shard:
capacity `25.6 MiB/mm^2 x area x layers`, bandwidth `20.48 GB/s/mm^2 x area`,
latency not modeled (0 ns). Both densities are derived from Li et al.
(arXiv:2604.08044, Table 3: 8192 banks of 2.5 MiB per 800 mm^2 die;
16 cores x 16 channels x 64 GB/s over the 800 mm^2 footprint), an unreviewed
preprint whose DRAM timing is withheld. The bandwidth is the stack's pin
bandwidth over its footprint and does not grow with layers; `layers` is
limited to the 4 dies of that stack (not a published maximum).
`bond_memory(adf, core, bonded_dram(...))` gives every core instance its own
shard; `dmc_partition` with `weights = "bonded"` keeps each core's weight part
there and streams it through that shard's port.

## Native GPU components

GPU programs use the five `accelsim_*` component families and the native configuration
compiler. Their parameters are not event-model throughput coefficients. See the
[native program tutorial](tutorials/05-full-models.md) and [engine guide](accelsim.md).

## Arbitration and the hardware-consistent scheduler

Every point owns resources: a compute unit (1 ns of occupancy per ns), a memory
port (bytes at its bandwidth), network links or ports. A task demands an
amount of each resource it uses. Alone it takes `max_r demand_r / capacity_r`
plus its latency tail. Concurrent tasks follow the resource's declared
arbitration:

- `fair_share`: processor sharing. A task that streams operands while it
  computes uses several resources at once and progresses at the rate of its
  most loaded one; shares are found by progressive filling of dominant shares
  (dominant resource fairness), which gives n tasks on one resource 1/n each.
  `Task.weight` counts identical work items (the CTAs of one SM) as that many
  shares.
- `fifo` / `exclusive`: one task at a time, without preemption, in arrival
  order. A task acquires all its single-occupancy resources atomically.
- `priority`: only the tasks with the most downstream work (alone time of the
  task plus its longest successor chain) progress; equal priorities share.

The consistent scheduler (HPCA draft, section III-D) issues the activated
tasks of one contention zone (tasks transitively sharing resources) together,
evaluates them until the first completion, truncates the others there
(`v -> v[1], v[2]`), and keeps the evaluated portions in the contention stash
buffer. A batch commits when no task that might contend (queued, or not yet
activated, using the earliest start any of them can still have) can start
before its end; it rolls back when a newly activated task on one of its
resources starts earlier. Successors activate when a completion commits.
Tests compare it, task by task, with an exact global event-by-event reference
on thousands of random graphs (all arbitrations, latencies, iterations and
static tasks) and on mapped transformer graphs.

`scheduler="consistent"` is the default for Demand graphs.
`scheduler="conservative"` uses the same Demand arbitration and also drives
stateful component requests, preserving queue backpressure and separate
consumer-ready and physical-completion events. Instruction/transaction programs
automatically select it; stateless tasks gain no additional barrier. See
[unified execution](unified-execution.md) for the component contract.

Tick semantics: a task runs once per iteration and processes iterations in
order; `static` tasks (weights, KV cache) run once and serve every iteration.
Storage tasks allocate memory when they activate and free it after every
dependent task instance has finished; the backend checks the peak against each
memory's `capacity`. Templates allocate an operator's outputs when the
operator starts (so inputs and outputs are resident together), keep
all-reduce buffers until the last collective step, and stage DMC input copies
in SRAM buffers that live until the consuming compute finishes. Streams carry
a direction: results written by a compute point are routed from the point to
the memory.

## Workloads

`WorkloadSpec("operator", params)`: `op` = `matmul` (`batch`, `m`, `k`, `n`,
`shared_b`), `mvm` (`m`, `k`, `batch`), `softmax`, `layernorm`, `rmsnorm`,
`silu`, `swiglu`, `gelu`, `add` (`rows`, `cols`), `allreduce` (`bytes`), plus
`dtype` (default fp16) and `iterations`.

`WorkloadSpec("transformer", params)`: `model` (registered name or a config
dict), `phase` (`prefill`/`decode`), `batch`, `seq_len`, `layers` (default 1),
`dtype`, `iterations`. Registered models with their sources:
`llama2-7b`, `llama2-70b`, `llama3-70b`, `qwen-72b` (HuggingFace
`config.json`; Qwen's `intermediate_size` 49152 is two 24576-wide
projections), `gpt3-6.7b` (Brown et al., Table 2.1). A layer is RMSNorm or
LayerNorm, QKV projection, attention (QK^T, softmax, AV over the full context;
no causal skipping), output projection, all-reduce (tensor parallel), residual,
norm, FFN up (gate and up fused for SwiGLU) with its activation, FFN down,
all-reduce, residual. With grouped-query attention the attention GEMMs carry
`kv_group`, so the query heads of a group read one K/V head (traffic counts
each K/V head once). Decode attends one new token per sequence against a KV
cache of `seq_len` tokens. Per-element instruction counts of the vector
operators are in `arch_explorer/workload/operators.py`.

## ADF requirements of the builder and the templates

Seed architectures and structure transformations must produce ADFs that
satisfy the rules below; otherwise `build_simulator` or the template raises,
or (where noted) the template reports itself as not applicable.

**Builder (every ADF).**

- The root is a `SpaceMatrix` that does not repeat; at least one compute point
  exists; every `SpacePoint.component` is registered.
- Networks are listed in a matrix's `networks`, cannot repeat, and only the
  first one of a container routes. Its endpoints are the container's
  `elements` expanded by `repeat`, in declaration order (sub-containers,
  compute points, memories and the DRAM all count), plus the gateway to the
  parent level.
- `mesh2d`: `rows x cols` must hold every endpoint. A container with `core`
  repeated 16 times and one `dram` has 17 endpoints, so a 4x4 mesh is rejected
  and 3x6 or 4x5 fits. Endpoint i sits on tile i in row-major order; the
  gateway uses `gateway_tile` (default 0) and may share a tile.
- A container without a network moves data between its endpoints at no cost
  and the backend warns. Give every container whose elements exchange data a
  network (e.g. `bus`).
- A transfer between two points routes through their lowest common container,
  so each compute point needs a network path to the memories it reads.
- **DRAM groups.** A container instance whose subtree holds at least two
  non-bonded `dram` points and nothing else except `pim_unit` points and
  networks is one striped memory (the largest such container; e.g.
  `chip/dram` after `pim_transform`). Templates address it by its path: data
  is spread evenly over its parts, so streaming `n` bytes moves `n / parts`
  through every part's port and route, its capacity is the sum and its access
  latency the largest part latency. Containers inside a group usually have no
  network of their own (`pim_transform` adds none above the bank); the warning
  for them is reported once per indexed level (`channel_*`).
- Every `pim_unit` needs exactly one non-bonded `dram` in its own container
  (its bank) carrying the bank timing parameters; otherwise PIM kernels raise.
- A bonded `dram` (`technology = "bonded"`) belongs to the core container it
  sits in; it never forms a group.

**CGRA template.** `cgra_spatial` requires exactly sixteen `cgra_pe` points
as the sixteen endpoints of one 4×4 `mesh2d` container, plus two equal
power-of-two-capacity SRAM banks outside the array. There is no DRAM device
requirement. Memory requests use the seven boundary-tile ports explicitly;
they do not traverse the generic parent gateway. The preset declares one
word per tick per mesh link and per SRAM read/write port, and all PEs share
one clock. Only uint32, batch one and one invocation are supported. The
resident buffers must fit; use an explicit larger-capacity scenario if needed.

**Devices.** A device is the smallest container whose subtree holds compute
points and a non-bonded `dram` or a DRAM group. The `pim_unit` points of a
group are not device compute (only `pim_offload` uses them) and a bonded
shard does not make its core a device. `gsm_tiled`, `dmc_partition` and
`pim_offload` need exactly one device instance (no `repeat` on it);
`tensor_parallel` uses all of them. The device's main memory is its shallowest
DRAM group or non-bonded `dram`.

**`gsm_tiled` (generic shared-memory compute).**

This template can remain for non-GPU systolic/vector hosts, including PIM systems.
It does not implement a GPU latency predictor. Compute support, parameter domains
and applicability come from the installed template registry. Native GPU workloads
use program lowering and the Accel-Sim backend instead.

**`dmc_partition` (many-core structure).**

- A core is the container directly holding compute points; it must also
  directly hold its own `sram`, otherwise the template is not applicable.
  Cores are usually a `SpaceMatrix` element with `repeat`.
- Matrix operators run on the core's first compute point that supports them;
  vector operators need a `vector_unit` in every core. A core with
  only a `systolic_array` fails on vector operators.
- Resident weights (`weights=local`), the core's output share and its staged
  inputs must fit its SRAM; the backend raises `CapacityError` unless
  `check_capacity=False`.
- The cores and the `dram` share a network, typically a `mesh2d` sized as
  above.
- `weights = "bonded"` is offered when the ADF has a bonded `dram`; every core
  needs one directly in its container (as `bond_memory` creates).

**`pim_offload` (host plus DRAM-PIM, decode only).** One device whose main
memory is a DRAM group with `pim_unit` per bank, and a transformer decode
workload. The host compute points follow the rules of the `inner` template
(`gsm_tiled` or `dmc_partition`); for `dmc_partition` the PIM banks are not
cores. The channels are the containers directly above the bank containers;
each request's K/V cache is assigned to one of them round-robin.

**`tensor_parallel`.** Devices are ranks in endpoint order; `tp` must divide
the device count, the attention heads and the FFN width, and either divide the
KV heads or be a multiple of them. The device containers need a network at the
level above them (`p2p_switch` for NVLink boards). Every rank needs a compute
point that runs vector kernels for the all-reduce reduction, and the inner
template's rules apply to each rank.

The request space is a union of conditional `tp`/`inner` branches. Selecting
`inner=gsm_tiled` exposes only GSM parameters; selecting `dmc_partition`
exposes only DMC parameters, always prefixed with `inner.`. Unknown or inactive
fields are rejected during discovery/validation as well as execution, never
discarded. Each branch uses the intersection of the parameter domains of the
first `tp` devices in actual endpoint order. Single-device applicability is
checked on each device subtree, with that device's root repeat removed; nested
core repeats remain. Optional SM-local tile parameters or bonded weight choices
from unused ranks cannot make a request valid on the selected ranks.

This wrapper supports operator workloads with `tp=1` and transformer workloads
with the degrees described above, on single- or multi-device ADFs. It does not
support ViT, CLIP or π0.5; their complete-model mapping requirements still apply.
Discovery is structural and does not run the simulator to filter candidates;
capacity, kernel support and topology errors can still reject an evaluation.
Abbreviated direct evaluations retain the defaults `tp=max(valid degrees)`,
`collective=ring`, `inner=gsm_tiled`, and the selected inner template's defaults.
Search/replay mappings contain every active key. Sampling now enumerates a
union rather than the former invalid product, so the same random seed can
choose a different candidate; this is a domain correction, not an optimization
quality claim. Historical rejected trials and their evaluation charges remain
unchanged.

## Mapping templates

| template | applies to | parameters | placement |
| --- | --- | --- | --- |
| `gsm_tiled` | non-GPU systolic/vector shared-memory hosts | discover from the registered template | generic event tasks; no GPU timing formula |
| `dmc_partition` | one device whose compute points each have SRAM in their own container | `split` n/m, `weights` local/stream (+ bonded when the ADF has bonded DRAM), `activations` local/dram | output columns or rows partitioned over cores; weights resident in SRAM (capacity-checked), streamed from DRAM, or resident in and streamed from the core's bonded shard; activations kept in producer SRAM and fetched over the network, or written to DRAM |
| `tensor_parallel` | operator (`tp=1`) or transformer, one or more devices | conditional `tp`, `collective` ring/multiring/tree, `inner`, active `inner.*` | Megatron sharding; rank r runs on device r with the inner template; all-reduces become collectives of transfers and reductions |
| `pim_offload` | one device with `pim_unit` banks, transformer decode | `attention` pim/inner, `sub_batches` 1/2, `tp` (the device runs one rank; all-reduces not timed), `inner` gsm_tiled/dmc_partition, `inner.*` | attention GEMVs on the PIM channel holding each request's K/V cache, softmax and all other operators on the host with the inner template; `sub_batches = 2` is NeuPIMs sub-batch interleaving (two halves with shared weights; host order QKV(1), QKV(2), Proj/FFN(1), Proj/FFN(2) while the PIM attention of one half overlaps host work of the other); breakdown labels `LayerNorm+QKV`, `PIM attention` (`Attention` when on the host), `Projection+FFN` |

`arch_explorer.mapping.mapping_space` enumerates a template's candidates and
`best_mapping` evaluates them (optionally limited by a budget that reports how
many candidates were skipped).

## Parameter sources and model assumptions

Non-GPU event components carry explicit parameter units and sources. Unknown clocks
or missing cost parameters remain unknown. DMC stages input pieces in private SRAM;
mesh multicast is not modeled. Attention and collective boundaries depend on each
template. Native GPU configuration provenance and cycle extraction belong to the
[Accel-Sim backend](accelsim.md); the host does not scale historical cycles.

**PIM presets** (`arch_explorer/devices/pim.py`). Evidence tags:
`[paper table]` the device's own paper, `[vendor]` the maker's slides,
`[config]` a number of the paper's published simulator configuration (NeuPIMs
f299af3; numbers only), `[simulator]` a behavior of that simulator's code
(described, not copied), `[third-party]` another team's paper or simulator
configuration for the device, `[preprint, derived]`, `[derived]`,
`[ArchExplorer §V-B2]` (the HSNDI study's own statement and the source it
cites), `[assumed]`, `[not modeled]`. Both presets build the PIM structure
with `pim_structure`, which yields the same tree as `pim_transform` (a test
compares them) without replaying one checked edit per bank.

| preset | structure | options |
| --- | --- | --- |
| `neupims` | `sa` 8 x `systolic_array` (ws, 1 GHz, double-buffered weights), `vu` 8 x `vector_unit` (128 lanes), HBM 32 channels x 32 banks (1 GB, 32 GB/s per channel, NeuPIMs Table 2 timing and refresh), `pim_unit` per bank (16 multipliers, 1 GHz, 4 banks per activation, dual row buffer, 16 elements per MAC command), `npu_fabric` bus | `array` 128 (Table 2) or 64 (Fig. 7(j) variant); `channels`, `banks` for scaled tests |
| `hsndi` | `core` x N (one IPU tile: `amp` vector unit at 1.6 GHz, 256 KB SRAM), GDDR6-AiM channels x 16 banks (512 MiB, 32 GB/s, 2 KB rows, 32 B columns; timing and refresh from CENT/aim_simulator), `pim_unit` per bank (16 BF16 MACs, 1 GHz, ACT16, single row buffer), `exchange` bus | `cores` (default 196), `channels` (default 8; Fig. 12's equal-area point), `core_rate` `paper` (16 FP32 ops/cycle, §V-B2) or `mk1_fp16` (64 ops/cycle), `precision` (default bf16, AiM's format) |

PIM error sources: command-bus sharing between host and PIM commands is not
modeled. AiM's vendor timing (tRCD, tRP, tRAS, tCCD, tFAW) is unpublished,
so `hsndi` uses third-party values, and its refresh counts equal
aim_simulator's generic GDDR6 presets; Newton/NeuPIMs-style readout after
every tile. In the previous atomic model, against Fig. 7(j), PIM attention was about
26 % short at every configuration (WP-SIM-03B report), but that figure is not
a matched reference: the public NeuPIMs simulator run on its own GPT3-7B/TP4
configuration (batch 2, sequence 1536, 128x128 arrays, one decode layer)
agrees with this preset within about 3 % on both NPU segments, while its PIM
attention is about half the figure's value and this preset's is about 47 %
longer than the simulator's (WP-SIM-08 report). The figure's actual
configuration is unknown. The command model above now uses the original
NeuPIMs simulator as the comparison target, not that figure; against the
figure's values its PIM attention is about 53 % short and its layer latency
about 34 % short. The HSNDI
SRAM-PNM values are the IPU Mk1 tile (the cited Hot Chips 33 talk describes
Mk2: 624 KiB, about 32 FP32 ops/cycle at 1.325 GHz), its transcendental rate
is assumed equal to its lane count, and its exchange is one shared bus.

The command model frozen at `7fd4b35` was compared with pristine NeuPIMs
`f299af3` using Operation spans (single decode layer, TP4, 32 channels,
requests assigned channels 0 through batch-1). On 18 GPT3-7B cases
(batch 2/6/12, sequence 1536/3072/6144, array 64/128), PIM attention MAPE
is **5.6024%**, signed mean **-5.6024%**; layer MAPE is **5.8649%**.
On the three GPT3-13B holdouts (batch2, the same sequences, array128),
attention MAPE is **4.7417%** and layer MAPE **1.9148%**. Parameters were
frozen before computing these grid/holdout errors. The remaining negative
attention bias is retained; normal traffic, GWRITE row preparation and
instruction queues have not been replaced by an error correction factor.
The NPU segments agree within about 5 % with 128x128 arrays but are up to
about 18 % short with 64x64 arrays (LayerNorm+QKV and Projection+FFN), where
this preset's Projection+FFN time does not change with the array size; that
gap is outside the PIM command model and remains open.
The private raw logs and configurations are not distributed in the package.
