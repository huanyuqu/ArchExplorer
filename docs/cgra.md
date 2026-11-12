# Scalar CGRA timing scenario

`vectorcgra_4x4` and `vectorcgra_4x4_diag` instantiate one 4×4 array and two
SRAM banks. `cgra_spatial` maps uint32 GEMM, MVM and two matrix multiplications
with an SRAM-resident intermediate (`linear2`). These are analytic DFG timing
scenarios. They do **not** reproduce the unpublished mappings used in
ArchExplorer Fig. 7(k,l), compile RTL control words, or establish its 3.10% MAPE.

```python
from arch_explorer.devices import get
from arch_explorer.adf import evaluate, MappingSpec, WorkloadSpec

# Pinned smoke: two banks of 32 words. This small GEMM fits.
adf = get("vectorcgra_4x4_diag")
w = WorkloadSpec("operator", dict(op="matmul", m=2, k=2, n=2, dtype="uint32"))
result = evaluate(adf, w, MappingSpec("cgra_spatial"))
print(result.details["cycles"])

# Explicit larger-capacity scenario for all 24 comparison cases, not paper hardware.
larger = get("vectorcgra_4x4_diag", words_per_bank=128)
```

An omitted `clock_ghz` is tagged **unknown** in provenance. Its numeric value 1
means one native cycle per reporting ns, not a measured 1 GHz clock. Use
`details.cycles`, whose values are invariant when all preset timings are rescaled
with `clock_ghz`. A caller-supplied physical clock remains a scenario assumption.
The engine schedules in ns internally; the cycle report removes that reporting
scale. Area and monetary cost remain unavailable for `cgra_pe`.

## Evidence and applicability

All RTL references below are to
[VectorCGRA commit 20ea53bfd54980acf5bc6eef4b0959a710a4dd1f](https://github.com/tancheng/VectorCGRA/tree/20ea53bfd54980acf5bc6eef4b0959a710a4dd1f),
BSD-3-Clause. This implementation uses independently written models; no RTL code
was copied and no RTL simulation was run. These sources establish this checkout's
structure, not the paper authors' exact hardware/mapping revision.

| Quantity / interpretation | Evidence class and pinned source |
| --- | --- |
| 4×4, 32-bit data, 17 control slots, 8 registers per bank | RTL test configuration: `cgra/test/CgraRTL_gemm_test_from_yaml.py:166–186` |
| Four register banks for the four FU inputs; separate routing and FU crossbars | RTL structure: `tile/TileRTL.py:85–119`, `mem/register_cluster/RegisterClusterRTL.py`, `RegisterBankRTL.py` |
| Scalar ADD and MUL each take one base tick, not a fused MAC | RTL implementation: `fu/single/AdderRTL.py`, `MulRTL.py` call base `Fu` with latency 1 and vector factor power 0 |
| One selected arithmetic/memory operation per tile control step | RTL control structure: `fu/flexible/FlexibleFuRTL.py`, `fu/basic/Fu.py`; the model shares one FIFO issue resource per PE |
| No extra fixed crossbar pipeline delay | RTL combinational datapath: `noc/CrossbarRTL.py`; ready/valid can stall |
| One word per directed neighbor link per tick, first arrival after one tick | RTL `tile/TileRTL.py:119` input `ChannelRTL(latency=1)`; `noc/PyOCN/pymtl3_net/channel/ChannelRTL.py` two-entry queues |
| Four cardinal neighbors, optionally four diagonal neighbors; no wrap | RTL `cgra/CgraRTL.py:61–68,192–268`, MESH / KING_MESH |
| SRAM access only through tiles with x=0 or y=0 (7 ports) | RTL `cgra/CgraRTL.py:72–74,269–272` |
| Two banks of 32 words locally; 256 words global across four CGRAs | RTL test configuration `CgraRTL_gemm_test_from_yaml.py:175–180`; **256 is not single-array local capacity** |
| One read and one write per bank per tick, contiguous address ranges | RTL `mem/data/DataMemWrapperRTL.py:45–56`, `DataMemControllerRTL.py` bank-index shift by per-bank address bits |
| Zero extra SRAM request latency in the selected test | RTL test `CgraRTL_gemm_test_from_yaml.py:423` selects combinational memory; the mapping also accepts an explicit one-tick memory request latency |
| Control ring has 17 nodes, shortest bidirectional routes | RTL `cgra/CgraRTL.py:130–132`, `noc/PyOCN/pymtl3_net/ringnet/RingRouteUnitRTL.py` |

The ring constructor receives `chl_lat=1`, but this pinned `RingNetworkRTL`
connects routers directly and does not consume that argument. The model's one
tick per ring hop is an explicit queue-hop approximation, **not** evidence that
the argument implements a physical pipeline. NI/credit/router handshakes are
not cycle-accurately reproduced.

## DFG, placement and memory choices

The public template space is:

| Choice | Domain / default |
| --- | --- |
| `ii` | 1, 2, 4 / 1; minimum scalar issue spacing at each PE |
| `placement` | `row_major`, `column_major` / row major; output ordinal modulo 16 |
| `routing` | `xy`, `yx` / XY; diagonal steps first in KING_MESH |
| `memory_port` | `striped`, `nearest` / striped |
| `config_words` | 17 only; full control-memory loading scenario |

Here `ii` is **not** the compiler's modulo-scheduled loop II. The upstream smoke
uses a particular compiled II of 17; that value is not reused as a general
compute throughput. Search choices never read reference timings.

For every output and reduction term the graph reads A and B, routes each word
from a memory-connected tile, multiplies, and adds to the local accumulator
(initially zero). The result is routed back and stored. ADD and MUL occupy
separate PE ticks. Register accumulation avoids intermediate SRAM writes.
One output's arithmetic completes before that PE starts its next output;
loads/stores also contend for the issuing boundary PE. Bank read and write
resources are separate FIFO ports. `striped` selects the access tile from
`[0,1,2,3,4,8,12][word_address % 7]`; bank selection remains contiguous, not
modulo-interleaved. `nearest` selects the nearest axis-aligned edge, ties to
y=0. Routing tasks serialize words on each directed link. For a word, link
serialization and registered arrival represent the same tick and are not added
twice. The generic mesh's byte-flow API retains its original bandwidth-plus-tail
semantics; the CGRA word mapper uses the underlying links directly.

This graph is a semantic/traffic-level placement, not a proven realizable
17-word control program. It does not encode GEP/Phi/branch/prologue instructions,
register allocation, every crossbar select, finite FIFO backpressure, or all
ready/valid protocol stalls. The DFG value dependencies and per-resource issue
limits are checked; successful evaluation does not certify RTL compilation.

Inputs, weights and outputs occupy disjoint contiguous SRAM address ranges.
Linear2's second layer reads the first layer's output range directly; no host
round trip is charged. Both sets of weights and the intermediate remain resident.
The API `op="linear2", m=M, k=K, n=[H,N]` means two matrix multiplications,
without bias or activation. GEMM uses `op="matmul", m=M,k=K,n=N`. Public MVM
without `n` means A[M,K] times x[K]; a paper triple with `n` means
A[1,K] times B[K,N]. Batch is restricted to 1.

Capacity checks are mandatory: the default two 32-word banks cannot hold every
paper case. For the comparison, two **128-word** banks are an explicitly changed
scenario: the largest resident footprint is 164 words, so 128 is the smallest
power-of-two per-bank capacity that fits it. It is not a sourced device constant.
No spill/reload model is silently substituted. The original unknown paper SRAM
condition remains unresolved.

## Counting boundary

The pinned `CgraRTL_gemm_test_from_yaml.py:350–419` appends data-preload packets
to the CPU control source, appends per-tile configuration/launch packets, calls
`run_sim`, then reads `sim_cycle_count`. Its `done()` waits for the control and
query sources and the completion sink. The selected smoke has no query/readback
packets and expects a completion packet. There is no subtraction isolating
compute. This motivates reporting a default **preload-through-completion**
boundary, rather than selecting whichever boundary matches the figure.

`details.cycles` reports non-overlapping model stages:

- `data_load`: initial A and every layer's weights, one host packet per word;
  outputs and intermediate activations are not host-preloaded.
- `config`: two count packets, 17 control words and one launch packet per tile,
  one injection per tick, shortest ring delivery, then an all-tiles barrier.
  The packet categories follow `validation/script_generator.py:828–893`.
- `compute`: SRAM requests, scalar operations, routed operands/results and
  layer barriers; includes all output stores.
- `completion`: modeled return at tile 0, one injection plus one ring hop to
  controller 16. The upstream smoke's return tile is 1; tile 0 is this mapping's
  explicit choice, not a claim about the paper schedule.
- `end_to_end`: sum of those four stages.

The model assumes ready control sinks, zero constant/prologue packets, no
reset/harness offset and a global configuration barrier. Upstream can overlap
per-tile launch with later configuration; its actual constant/prologue packet
count depends on the unavailable compiled program. Consequently the numbers
are a declared scenario, not a cycle-exact prediction of `sim_cycle_count`.
Multiple engine iterations are rejected because these boundaries describe one
invocation. A 1×1×1 GEMM with `memory_port="nearest"` is hand-countable:
2 preload + 321 config + 5 compute + 2 completion = **330 cycles**.

## Agent and comparison use

Pass a named preset or its serialized ADF as `seed_adf`; no Agent provider or
paid API is needed. CGRA cost is unknown, so choose a latency-only objective explicitly:

```python
from arch_explorer.agent_env import ExplorationEnv
env = ExplorationEnv({"seed_adf": {"preset": "vectorcgra_4x4_diag"}, "workloads": [w.to_dict()],
                      "mapping_templates": ["cgra_spatial"], "max_evaluations": 4,
                      "mapping_samples": 4, "cost_penalty": 0, "area_penalty": 0})
print(env.evaluate(optimize="mapping"))
```

The same configuration works with the existing JSON-lines / MCP environment.
Runnable synthetic configurations are `examples/adf-cgra.json` and
`examples/adf-cgra-diag.json`. Default structural search uses the same catalog and
generic edits as every other ADF. The environment no longer restricts a run to a
preset-shaped tree or one topology toggle. Explicit parameter choices may also be
submitted; coupled clock/link/SRAM relationships must still be satisfied.

The backend's physical requirements remain authoritative: scalar uint32,
16 PEs, a 4×4 mesh, two supported SRAM banks, valid capacities and matching clock,
link and port timings. Unexecutable edits fail clearly rather than acquiring
unmodeled physical support. Capacity and reporting clock assumptions are recorded
in provenance. The default unknown physical clock is cycle-normalized; do not
interpret a higher assumed clock as measured speedup. Area/cost remain unknown.
See [Agent commands](agent.md#cgra-具名种子与通用搜索) and [generic search](structure.md).

For comparison with externally supplied digitized data:

```bash
python -m validation.cgra_scenario --suites /path/suite-k.json /path/suite-l.json \
  --words-per-bank 128 --output comparison.json
```

This command preserves reference uncertainty, unresolved conditions and source
hashes. It substitutes one fixed `cgra_spatial` scenario for the source suite's
unimplemented `paper-reproduction` mapping, and reports stage counts and signed
and absolute errors. It does not turn the blocked paper-reproduction suite into
a passing validation, choose mappings against references, or run VectorCGRA RTL.
