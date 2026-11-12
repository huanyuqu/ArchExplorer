# AiM shared transaction subsystem

`aim_shared` is an explicit ADF mapping for the GDDR6/AiM memory side of
`hsndi_detail`. It uses one evaluation-local `AimController` per channel through
`ExecutionRequest` and the unified conservative scheduler. Ordinary memory
requests and PIM instructions share the same instance, row state, command
issue slot, external data bus and finite queues. Default `hsndi`/`pim_offload`
and the NeuPIMs profiles are unchanged.

```python
from arch_explorer.adf import WorkloadSpec, MappingSpec, evaluate
from arch_explorer.devices import get
from arch_explorer.mapping.search import mapping_space

# This is an explicit synthetic service time, not an upstream runtime value
# or a manufacturer timing: static 210 cycles at its stated 0.5 ns = 105 ns.
adf = get('hsndi_detail', cores=1, channels=2,
          refresh_service_ns=105.,
          refresh_source='[synthetic] static210 * stated0.5ns; mechanism example')
workload = WorkloadSpec('operator', {
    'op': 'mvm', 'm': 35, 'k': 17, 'batch': 2, 'dtype': 'bf16'})
legal = mapping_space(adf, workload, 'aim_shared')
mapping = next(m for m in legal if m.params['host_traffic'] == 'synthetic_rw')
result = evaluate(adf, workload, mapping, trace=True)
print(result.latency_ns, result.details['aim'])
```

The two active mapping keys are `profile='aim-0f28a07b-effective'` and
`host_traffic='none'|'synthetic_rw'`. Unknown/inactive keys are rejected.
Discovery and execution use the same hardware/workload checks. BF16 GEMV,
one workload iteration, and `1 <= K <= 1024` are supported. The matrix rows
are striped in groups of 16 across local controllers. Each bank holds one
padded 2 KiB matrix row per group. Tail columns and tail banks execute zero
padding; logical and padded operations are reported separately. Batches
reuse resident weights and controller state. Each evaluation creates fresh
controllers. Multiple devices, FP16/BF16 conversion, arbitrary DRAM timing
edits, other AiM opcodes and multi-row accumulation are rejected.

The profile validates ADF bank geometry, timing, units and explicit refresh
provenance, and consumes additional fixed controller rules with separate
provenance. `get('hsndi')` retains the aggregate configuration and is rejected
by this mapping. `get('hsndi_detail')` requires both `refresh_service_ns` and
`refresh_source`; omission, nonpositive/nonfinite service, missing source or
an unsustainable service interval is an error. The physical input is in ns;
the controller rounds it up to its next whole cycle. It is not a fitted
coefficient. A caller-built ADF must satisfy the same contract as the preset.

## Timing and state contract

The fixed source is [arkhadem/aim_simulator
0f28a07bdb83e42b9305ad3d45410ebd3aa2c091](https://github.com/arkhadem/aim_simulator/tree/0f28a07bdb83e42b9305ad3d45410ebd3aa2c091).
These are third-party simulator assumptions, including values marked by the
source as SDK/profiling-derived. They are not manufacturer measurements.
CENT at `4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2` pins AiM
`2eb1ee04f4e8f3c255c9926e7721081726aa4fe4`, whose ACT16 constraints differ.
This profile consistently uses `0f28a07b`; it does not mix those revisions.
Traces from `2eb1ee04` cannot validate this revision. Same-version external
controller comparisons require independent acceptance; source-rule checks
and hand-computed microexamples establish only their declared scope. A whole
upstream run encountering undefined refresh initialization is an observation,
not an authoritative timing reference for that undefined field.

The values below follow the indexed `m_timings` names and the defined
initialization expressions in `GDDR6.cpp:580-716`. They do not simply copy
the preset comments. `rate=2000` recomputes `tCK_ps=1000`; the x16 secondary
tables select `nRRDS=5`, `nRRDL=6`, `nFAW=28`; the refresh interval calculation
sets `nREFI=7800`. The static literal at index 24 is 105 with comment
`nRFCpb`, but its indexed name is `nREFI`; index 26 holds 3333 with comment
`nREFI`, but its name is `nRFCpb`. Per-bank refresh is unsupported.

| Contract | Basis and units |
|---|---|
| 1 ns controller tick; 16 banks; 16K rows/bank; 2 KiB/row | Source organization and defined initialization |
| 32 B burst, 2 cycles, 16 B/ns external port | Source `nBL=2` at effective 1 ns; derived, not a vendor port measurement |
| Cross-group ACT spacing 5 cycles; same-group 6 | x16/rate2000 secondary initialization; no enforced nFAW rule in this source subset |
| RD response 52 cycles; WR 8; WRGB/WRMAC16 3; MAC16 1; RDMAC16 2 | `GDDR6.cpp` command latency assignments |
| ACT16 to MAC16: 56 cycles; PRE: 32; ACT16 activation spacing: 89 | Source command-pair constraints; no efficiency factor |
| Register mode change: TMOD, 32 cycles | Source controller mode switching and `nMODCH` |
| READ/WRITE/active/AiM queues: 33 effective entries each | Source `max_size=32`, enqueue uses `size <= max_size` |
| Ready-first, arrival-order ordinary arbitration; active requests first; PIM FIFO | `AiM_dram_controller.cpp`, `generic_scheduler.cpp` |
| Write-drain high/low marks 0.8/0.2 of declared 32 | Source controller defaults; not fitted |
| Ordinary and PIM queue admission mutually exclusive | Source `send`; already active/in-flight ordinary requests can overlap PIM admission |
| One shared row/bank | ACT/PRE and ACT16/PREA alter the same state; no HBM dual buffer |
| Byte addressing | Explicit channel-local software layout `(row*16+bank)*2048+column*32`; not the upstream physical address mapper |

The supported-command subset has 80 scoped timing pairs, represented as
eligibility deadlines. The external bus additionally reserves actual data
windows `[issue + latency - 2, issue + latency)` to prevent overlapping
32 B transfers with different response latencies. This is a physical
occupancy constraint, not an empirical delay or claim that upstream models
every bus collision identically.

`WRMAC16` resets/biases the bank accumulators, `WRGB` loads the vector,
`MAC16` consumes 32 B from each of 16 banks (512 B, 512 arithmetic operations),
and `RDMAC16` transfers 32 B of BF16 results. Each group fits the 2 KiB global
buffer; readout occurs after all its MACs. No unsupported host reduction or
free conversion is inserted. Internal accumulator numerical accuracy is
not simulated; this is a performance/transaction model.

The detailed ADF records `trcd=36ns`, `trp=32ns`, `tras=54ns`, `tcl=50ns`,
`tccd=2ns`, `trrd=6ns` (same-group; cross-group uses 5ns), and `tfaw=28ns`
(reported but not enforced). Its per-bank bandwidth is the equal-share
description `16B/ns / 16 = 1B/ns`. Its generic PIM issue-rate field is
`clock_ghz=0.5`: 32 operations per bank every two 1ns controller cycles;
this is not an independently measured PU clock. Detailed execution owns
the bank resources and uses the explicit commands, not a second aggregate
bandwidth or arithmetic charge.

Refresh is an explicit **channel-local periodic all-bank extension** with
`tREFI=7800` cycles from the defined source initialization. The source
organization has density 131072Mb, outside the refresh table index domain;
its `density_id=-1` makes the `nRFC` lookup undefined. Neither an observed
zero nor the static 210 is accepted as its effective value. Refresh service
is therefore the explicit caller input described above. The source's generic
`AllBankRefresh` traverses a `rank` level absent from GDDR6; this implementation
does not claim that generic manager runs unchanged. Refresh
becomes pending at absolute multiples of tREFI, waits for active requests,
precharge and recovery, and blocks subsequent bank accesses through tRFC.
No average refresh bandwidth penalty is charged to execution tasks.

Each controller's `parameters` array records the timing name and index,
static literal/comment name, initialized validity/value, selected value,
unit, override order, applicability and excluded fields. Provenance separates
`static_preset`, `defined_effective_initialization`, `caller_configuration`
and `independent_extension`. Command/response rules use the first two;
refresh recovery uses the caller input, while periodic maintenance and
physical bus occupancy are explicitly independent extensions. Undefined
values remain JSON `null`, never zero. This distinction does not upgrade
third-party assumptions into manufacturer evidence.

On each edge, due physical data completions and reader callbacks precede
refresh enqueue and command issue. Reader callbacks retain source FIFO
order and at most one read returns per edge; nonread completions may
coincide. Traces distinguish `data_complete`, consumer `return`, issued
commands, data-bus windows and the earliest legal precharge. A returned
write can leave a later row-recovery deadline: future commands still obey
that deadline. Execution-level physical/resource events report the last
data completion of a submitted stream; consumer readiness waits for its
last callback. State is not reset at an operator boundary.

## Host and accuracy boundary

This mode evaluates GEMV from **controller ingress with weights already
resident and a BF16 vector already available** to BF16 result readout.
Weight loading, IPU compute/casts, host scheduling and exchange timing are
excluded and explicitly warned about. Native IPU BF16 support is not
assumed. `synthetic_rw` injects an independent stream of two ordinary writes
and two ordinary reads per batch/channel to a reserved scratch row; this
is a concurrency stressor, not a measured HSNDI host.

The software feeder admits up to 33 programs, injects at most one command
per client per controller tick, and retries without advancing the program
cursor when a hardware queue or mode is blocked. These are explicit
synthetic frontend choices. A full `submit` returns `False` without changing
IDs, cursors, clocks, counters or traces. Changing this frontend would be a
different declared experiment, not a way to calibrate latency.

The result reports logical/padded operations, resident matrix storage,
buffer/result-transfer sizes, per-controller bytes and traces. The request
ledger distinguishes logical stream commands, hardware-accepted
commands, issued terminal commands, forwarded reads, physical completions,
callbacks returned, and logical commands still pending. Prerequisites such
as ACT/PRE/TMOD and maintenance REF commands appear in the trace but are not
terminal request counts. Fully drained streams obey
`logical = accepted = issued + forwarded = returned = physical_completed`,
with `pending = 0`; a submitted stream is not itself a hardware command.
Mode/queue blocked counts are per-client feeder attempts, not elapsed wall
time or measured host stalls. Component timing, controller policy and host
frontend validation are distinct. The normal
ADF cost model remains independent. `partial=True` and
`paper_configuration_confirmed=False` are retained. No whole-HSNDI,
manufacturer-device, or paper accuracy claim follows from these tests.

## Source attribution

Timing and scheduling facts above are derived from the MIT-licensed AiM
simulator. Attribution for the derived timing subset:

Copyright (c) 2025 University of Michigan.
Authors: Yufeng Gu (yufenggu@umich.edu) and Alireza Khadem (arkhadem@umich.edu).
Reference: “PIM is All You Need: A CXL-Enabled GPU-Free System for LLM
Inference”, ASPLOS 2025.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
