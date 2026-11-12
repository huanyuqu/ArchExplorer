

# Accel-Sim Native GPU Backend

`backend="accelsim"` compiles ADF components into native configuration, with the external Accel-Sim
advancing cycles uniformly, then returns an `Evaluation`. Python does not simulate a second timing model, nor does it
sum the overlapping cycles of SM, cache, interconnect, and DRAM. Complete small-scale ViT/CLIP is now executed via the
[program-level entry](accelsim-program.md): complete numerical reference, actual mapping,
multi-kernel single stream, and one native process. This path uses explicitly synthesized reference programs and weights;
hardware latency accuracy and framework compilation/kernel selection equivalence are still unverified. The single-kernel
contract and historical calibration interfaces below maintain their respective version boundaries.

Pinned sources are [Accel-Sim v2.0.1](https://github.com/accel-sim/accel-sim-framework/tree/1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea)
and [GPGPU-Sim v2.0.1](https://github.com/accel-sim/gpgpu-sim_distribution/tree/c24e8858eee552b3bc2919b147f14e2a88571775).
The backend does not download tools, does not launch CUDA/GPU kernels, and does not carry native binaries or third-party source code.

## Components and Support Structures

`arch_explorer.archs.accelsim.sm90_adf()` provides the SM90_H100 configuration seed.
All explicit options in the native configuration come from ADF parameters; unlisted native options use the parser defaults of the pinned version above.
`simulator/accelsim/profile.py` records the upstream file and line number for each field; most parameters are fixed to single values.
Synthetic exploration can modify the publicly listed legal subset, but cannot infer real device accuracy for modified configurations from this.

| ADF Object | Native Responsibility | Instances and Constraints |
| --- | --- | --- |
| `gpu/cluster/sm`: `accelsim_sm` | `trace_shader_core_ctx`, scheduler, execution and register resources | One SM per cluster; register block upper limit does not exceed SM capacity |
| `gpu/cluster/l1_shared`: `accelsim_l1_shared` | `ldst_unit`, L1D/shared | One per SM; cache structure, shared capacity and bank configuration are fixed |
| `gpu/icnt`: `accelsim_icnt` | `InterconnectInterface`, BookSim single-level fly network | `network_mode=1`; n=1, subnets=2, single VC; k=SM clusters+L2 subpartitions |
| `gpu/partition/l2`: `accelsim_l2` | `memory_sub_partition`, L2 | Two per DRAM channel; cache configuration and LRC are fixed |
| `gpu/partition/dram`: `accelsim_dram` | `memory_partition_unit`, `dram_t` | 20/40 channels, two chiplet partitions; retains upstream simple DRAM configuration |

GPU root `repeat=1`; the two groups contain only the points in the table, without inner networks or recursive matrices.
SM cluster repeat supports 2/4/8/16/32/64/128/132; DRAM partition repeat supports 20/40.
Except for L2 repeat=2, all other points have repeat=1. When changing group repeat, the interconnect `booksim_k` must be changed synchronously.
Arbitrary meshes, multiple GPUs, heterogeneous SMs, or other connection structures are not supported; illegal structures are rejected before startup.

The legal adjustable subset is per-SM/per-block registers 32768/65536, CTA limit 16/32, L1 latency
41/42/48, shared latency 29/30, L2 ROP latency 237/238, DRAM latency
194/195/200, and BookSim input/VC buffer 128/256/512 flits.
core supports 990/1980/3960 MHz, icnt and L2 must equal twice the core, DRAM
supports 1553/3106 MHz. These are the first-version configuration capability constraints. Native has more options, but the first version does not expose them.

`compile_adf(adf)` returns configuration content and `component_map`. Output is
`gpgpusim.config` and `config_hopper_islip.icnt`; the latter is likewise generated field-by-field from the ADF.
Each mapping entry contains ADF path, parameter, value, unit, source, native configuration key/object, instance count, and constraints.
`config_sha256` computes SHA-256 over a key-sorted JSON object composed of these two files.
Runtime limits are additionally written to `limits.config`, whose hash is preserved in the run record.

Native enumeration defines `network_mode=1` as BookSim, and `2` as LocalInterconnect.
The "local xbar" comment in the H100 configuration cannot replace the code enumeration. This backend maintains mode 1 with its complete
single-level fly configuration and refuses to silently switch modes. The declared components' `build()` is explicitly rejected in the old events backend;
they do not provide a Demand model.

## Installable Real Execution Entry

First independently build the pinned version of `accel-sim.out` and GPGPU-Sim's
`libcudart.so` according to upstream instructions. This is a CPU timing simulator; building requires CUDA headers/tools, GCC, Make,
Bison, Flex, M4, makedepend, zlib, OpenGL, and zstd; running on a GPU is not required.
The upstream Make path also requires adding `util/tracer_nvbit/tracer_tool` to the header search path.
Some systems need `--no-as-needed` because upstream places library arguments before object files.
Local isolated installation and build records are kept by the deployer to avoid setup automatically updating the moving dev/master.

The engine manifest uses `arch-explorer.accelsim-engine.v1`:

```json
{
  "schema": "arch-explorer.accelsim-engine.v1",
  "sources": {
    "accelsim": "1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea",
    "gpgpusim": "c24e8858eee552b3bc2919b147f14e2a88571775"
  },
  "binary": {"path": "/absolute/path/accel-sim.out", "sha256": "actual file hash"},
  "libraries": [{"path": "/absolute/path/libcudart.so", "sha256": "actual file hash"}],
  "build": {"command": "replayable build command", "dependencies": "dependency versions, download sources and hashes"}
}
```

The `libcudart.so` here must be a GPGPU-Sim build artifact, not the CUDA runtime.
The manifest is an identity record for a locally trusted build; hashes alone do not constitute supply chain certification.
Verify the hash of each binary/library before execution; the output must also carry the native build identity of the two pinned commits.
A missing engine raises an actionable `ImportError` and does not fall back to other models.

After installing ArchExplorer, fully public synthetic samples can be generated outside the source tree:

```sh
python -m arch_explorer.components.accelsim.example /tmp/accelsim-example --engine /absolute/path/engine.json
archexplorer evaluate /tmp/accelsim-example/request.json --output /tmp/accelsim-result.json
archexplorer evaluate /tmp/accelsim-result.json --output /tmp/accelsim-replayed.json
```

The CLI request schema is `arch-explorer.accelsim-request.v1`, with top-level exactly
`schema, adf, workload, mapping, options`. `options` accepts `engine`,
`output_dir`, `timeout_s`, `max_cycles`, `memory_limit_mb`; defaults are respectively unset,
temporary directory, 120 seconds, 1000000 core cycles, 4096 MiB.
The result contains the complete request and can be replayed directly by the same CLI.
Paths are interpreted in the current process; absolute paths output by the sample generator are recommended.

The public Python API and CLI use the same backend:

```python
import json
from arch_explorer.adf import WorkloadSpec, MappingSpec, evaluate, from_dict

r = json.load(open('/tmp/accelsim-example/request.json'))
result = evaluate(from_dict(r['adf']), WorkloadSpec(**r['workload']),
                  MappingSpec(**r['mapping']), backend='accelsim',
                  cost_model=False, **r['options'])
print(result.latency_ns, result.details['native'])
```

## Trace and Mapping Contract

`WorkloadSpec.kind` only supports `operator`, and `MappingSpec.template` must be `accelsim_trace`.
The trace applicability module uniformly checks the following fields; the public evaluate, CLI, and result replay all consume it before launching the native engine.
All these fields are required, and unknown fields are rejected; the first version does not additionally expose execution parameters or free metadata.

| Object | Required Fields and Exact Types | Authority and Meaning |
| --- | --- | --- |
| workload.params | `op`: string `trace_kernel`; `kernel`: legal identifier string; `grid`, `block`: three-dimensional positive integer arrays; `accelsim`: file reference object | kernel, grid, block must match the actually parsed trace; only current trace_kernel semantics are accepted |
| workload.params.accelsim | `manifest`: non-empty path string; `sha256`: 64-character lowercase hexadecimal string | SHA-256 of the manifest's raw bytes |
| mapping.params | `trace_id`, `compiler`: non-empty strings; `grid`, `block`: three-dimensional positive integer arrays | Geometry comes from the actual trace; trace_id/compiler are caller source declarations that must match the manifest text |
| manifest.workload | `kind`, `params`, where params are the op/kernel/grid/block above, without the accelsim reference | Fields are likewise required and strictly validated, and separately checked against the request and actual trace |
| manifest.mapping | Complete `template`, `params` | Same mapping contract; actual geometry cannot be bypassed merely by two declarations being equal |
| manifest.kernels[0] | file, two SHA strings, name, grid, block, stream, thread_instructions, ctas | All nested items are type-checked; name, geometry, stream, and completion count must match the actual trace |

grid/block must be JSON arrays containing exactly three positive integers; bool, float, string, tuple, empty/short/long arrays are all rejected.
kernel.stream is a non-negative integer, ctas/thread_instructions are positive integers; Python's True==1 or 1.0==1
cannot substitute for type checking. The v1 actual trace must be one 32-thread CTA, stream 0, and the manifest top-level scope
must be `single-kernel-device-cycles`. workload/mapping cannot additionally declare stream/scope to expand the range.

Legal new kernel names can be used, and must be synchronized across the actual trace, request, and manifest; there is no sample name whitelist.
Arbitrary op, shape, batch, tile, or other uncertified parameters are rejected and will not be treated as execution results for another task.
name/weight remain the display name and suite weight of WorkloadSpec and do not affect execution identity.

Hashes prove file byte consistency and do not certify collection or compilation history. source, compiler, and trace_id are caller declarations:
modifying them and synchronously recomputing the manifest hash only changes the source declaration; software cannot certify the real compiler,
device collection, or history from identical SASS bytes. Execution facts are always checked against the actual trace and cannot be overwritten by source declarations.
This backend's first version does not integrate existing mapping search templates.

The trace schema is `arch-explorer.accelsim-trace.v1`, with a complete example generated by the above command:

- `format="traceg"`, `tracer_version=3`, `isa="sm90"`, pinned engine and profile identity.
- `source` describes kind, GPU, toolchain, and description; the sample is explicitly hand-written synthetic data with no GPU source.
- `commands` specifies `kernelslist.g` and its hash, with content exactly one `kernel-*.traceg` filename.
- `kernels` has exactly one entry, containing original/current file hashes, name, grid/block, stream, CTA, and thread instruction counts.
  The first version's original trace is the same as the consumed trace; unverified derived transformations are not accepted.
- `scope="single-kernel-device-cycles"`.

The first version only accepts one 32-thread CTA, one complete warp, stream 0; supports IADD3, LDG.E,
STG.E, and trailing EXIT. Instructions, registers, complete block, header fields, masks, instruction counts, and hashes
are checked before launch. There is no shared/local addressing, branching, tracez, asynchronous instructions, multi-kernel, concurrency, or memcpy support.
Upstream can support more, but this contract has not yet defined applicability validation for them.
Renaming a single-kernel trace as ViT/CLIP is rejected and cannot produce complete model latency.

## Dynamic Trace v2

`arch-explorer.accelsim-trace.v2` retains the request and mapping fields above, and expands the applicability domain through actual input facts.
Supports SM90 ordinary traceg/tracez v3/v6, complete multi-CTA/warp, legal printable kernel names, dynamic PC repetition,
zero/partial active masks, and per-lane EXIT. All CTAs in the grid and warps in the block must appear,
and each warp stream must consume registration records and EXIT all valid lanes. CTA/warp record order is not used as a name or size whitelist.
The text header must be terminated by a one-line format comment; this is the boundary of the fixed native reader and cannot be replaced with `#BEGIN_TB`.

v2 kernel entries add required `facts`: actual header resources, CTA/warp stream counts, dynamic warp/thread instruction counts,
opcode/mask histograms, repeated PC counts, and per-stream summaries. They are bound item-by-item to the actual file, request, and manifest;
updating hashes cannot disguise different geometry or counts. The Hopper base opcode table in the source comes from the pinned commit;
known opcodes must also satisfy ordinary operand and suffix formats. Unknown opcodes/ISA, Tensor/TMA/asynchronous specialized formats,
cross-CTA clusters, and dynamic resource changes are explicitly rejected and cannot be digested into zero time. Ordinary LDS/STS and BAR.SYNC
have small synthetic native checks; this does not certify all named/asynchronous barriers. Native only establishes a scoreboard for R registers;
UR/P/UP dependencies are not fully modeled; executable software cannot therefore claim real hardware timing correctness.

tracez validation reads according to the fixed `tracez_format.h` footer/index and zstd chunking, verifying offsets, counts,
frames, and complete execution range. Original tracez bytes are passed directly to the native engine; ISA/instructions are not rewritten and predicated-off records are not deleted.
Default boundaries are 64 MiB for the original file, 256 MiB for expanded text, 8 MiB per chunk, 4096 CTAs, and 2 million dynamic records.
The kernel's real register/shared/block resources are checked against the ADF before launch; this check does not add latency corrections.

Externally collected traces can create requests outside the installation package:

```python
from arch_explorer.components.accelsim.trace_v2 import create_request

request = create_request('/data/kernel-77.tracez', '/tmp/native-request',
    source={'kind': 'hardware', 'gpu': 'source device declaration', 'toolchain': 'collection tool version',
            'description': 'source and subrange of the complete kernel'},
    trace_id='external-resource-id', compiler='source compilation declaration', engine='/data/engine.json')
```

Then run the existing `archexplorer evaluate`. The source field is a declaration; the hash is not a collection authenticity certificate.
A single H200 source kernel succeeding under an H100 profile only proves software consumption; it does not represent accuracy on either device.
Real traces and native tools are external resources and are not included in the wheel.

Capability precheck without engine/device:

```sh
python -m arch_explorer.components.accelsim.capabilities /tmp/native-request/request.json
python -m arch_explorer.components.accelsim.capabilities --cubin /data/existing.cubin
python -m arch_explorer.components.accelsim.capabilities /tmp/native-request/request.json --purpose hardware-accuracy --target /data/target-identity.json
```

Returns `passed`/`blocked` with a categorized reason; blocked exits with code 2. Target identity has
three fields: `gpu, isa, evidence_sha256`, referencing an existing device record; the precheck does not query the GPU.
A static cubin can only provide offline ISA evidence and cannot reconstruct a dynamic trace. The pinned native version has no SM120 OpcodeMap;
the Blackwell ISA gap and target physical parameter gaps are reported separately; H100 or zero cost is not assumed by default.

The pinned native `trace_driven.cc` dispatch includes SM35, 60, 61, 70, 75, 80, 86, 90.
This is declared separately from the ordinary SM90 input domain verified by this adapter: SM80/86 etc. return `adapter_isa`,
not that the native side lacks the corresponding ISA; SM120 returns `native_isa`. A static cubin always additionally reports
`dynamic_trace_missing`; having dispatch does not certify all operands, dependencies, or real device timing.
Target, cubin, v1/v2 manifest, and actual trace header reuse the same fact table in `native_isa.py`.

## Read-only protocol review of retained results

`arch_explorer.components.accelsim.protocol` validates closure between existing complete predictions and actual consumed objects,
reusing production `load_trace`, `compile_adf`, `load_engine`, and `parse_stats`.
It does not start the engine, query devices, or create run directories. Software passing only represents post-hoc review closure;
old predictions cannot be declared as frozen before new measurements, nor can hardware windows be opened based on this.

```python
from arch_explorer.components.accelsim.protocol import prepare_review, preflight_protocol

spec = prepare_review('/data/request.json', '/data/result.json',
    '/data/source-origin.json', '/data/source.tar', '/data/history.json',
    '/tmp/new-review')
result = preflight_protocol(spec)
```

`prepare_review` only generates review metadata in a new directory that does not exist; retained results, source archives, and native
run/stdout/stderr/staged inputs must already exist. Native execution source and current reviewer source are recorded separately;
a new reviewer version is not retroactively attributed to an old execution version. CLI equivalent entry point:

```sh
python -m arch_explorer.components.accelsim.protocol /tmp/new-review/protocol.json
```

The protocol is `arch-explorer.accelsim-protocol.v1`; required fields are `schema, purpose,
review_kind, request, frozen, history, case_identity, target, compile_trace_binding`.
Current `review_kind` is fixed as `posthoc-software-review`, and `compile_trace_binding` is fixed as false.
`frozen` is a unique, complete list of `{role,path,sha256}`, binding the actual request, manifest,
trace, commands, ADF, complete config, engine manifest/binary/all libraries, complete prediction,
execution source/provenance/archive, history, native run/stdout/stderr, and every actual staged input.
Missing items, duplicates, extra roles, type errors, or unrelated files after hash recomputation are all rejected.

The prediction must contain the complete `arch-explorer.accelsim-result.v1` envelope; all execution identities, run status,
config/ADF, guards, trace ranges, and values must match the retained records. Only output locations and workload
display names/weights are normalized; timeout, cycle limit, and memory limit participate in identity. Top-level latency, two
JSON native statistics, and hash-bound raw outputs re-parsed by the production parser must close over cycles, frequency, counts, and ranges.
Copying consistent error JSON numbers does not constitute independent native evidence.

The outer envelope remains v1 for recognition by existing CLI result re-ingestion entry points. New result details must declare
`schema=arch-explorer.accelsim-result-details.v2` and
`component_map_schema=arch-explorer.accelsim-component-map.v2`, and include
`device_facts`. When there is no device fact binding, this field must be null; when bound, it must strictly match the entire
report obtained from the current request audit, including original identity, known, unknown, synthetic assumptions,
launch facts, hardware accuracy, and live lease. Missing, extra, mixed versions, or deletion of unknown items
are all rejected; historical device facts do not grant current execution leases, nor do they constitute physical accuracy verification.

The public protocol read-only compatibility covers two limited historical production semantics: SIM28 c2d4bdc and SIM29 94a515c.
Selection is based on the fixed byte digest of production backend/config/component/profile (and
applicable device/constraints) after actual run and archive closure; it cannot be selected solely by caller version labels.
SIM28's old limited parameter domain, topology provenance description, and per-component old clock domain annotations are fully
reconstructed and compared in the original format; SIM29 retains the fact fields, parameter domain, provenance classification, and CORE ready-cycle
annotations of that time. New versions also audit topology fact classification. These projections only handle metadata differences for the above versions;
valid native config bytes/hashes, parameter values/units/provenance, complete provenance, topology instances,
request/engine/trace/guard, native inputs/outputs, and all values still close item by item.
Old annotations are not upgraded to verified physical facts. Unrecognized production bytes or future result versions are conservatively rejected,
requiring explicit new compatibility rules; existing frozen request/prediction/source/run are not rewritten.

Case identity is generated from actual trace bytes and execution ranges, not from manifest description fields;
known identical dynamic flows in tracez/text additionally have a semantic identity, preventing encoding changes from evading known overlap.
The old `arch-explorer.sim27-history.v1` is read-only compatible; the new `arch-explorer.accelsim-history.v1`
references original records with hash-bound parents and accumulates case/semantic/binary registrations without clearing old collections.
Unknown cross-binary or collection associations remain unknown; absence of known overlap does not prove something was not tested.

Execution source archives must bind provenance records, commit labels, and actual executed module bytes; these are verifiable
associations, not signature certificates. Independent measurement/compilation collection associations and new windows are not yet verified; `hardware-accuracy`
still rejects them. Old private draft protocols must be explicitly migrated; historical frozen records or provenance evidence cannot be retroactively altered.

## Time, results, and failures

`latency_ns = final gpu_tot_sim_cycle × 1000 / core_clock_MHz`.
The source is the pinned GPGPU-Sim `src/gpgpu-sim/gpu-sim.cc`: `init_clock_domains()` converts MHz
to Hz, and `gpu_print_stat()` outputs accumulated core cycles. Only the final accumulated count is taken;
a single kernel's `gpu_sim_cycle` must equal it and is not added repeatedly.
The range includes the native kernel launch latency declared by the configuration, device execution, and drain, excluding host CPU
runtime, copies, or other work in the whole model. Other clock domains remain in the configuration and mapping, handled by the native scheduler.

Only reliable completion cycles and retired thread instruction counts are mapped; v2 also verifies native retired warp instruction counts.
`gpu_tot_issued_cta` is an issue count; there is no directly printed completed-CTA counter;
completion is established by the joint verification of complete input EXIT, exact retired counts, and normal exit without truncation; the report explicitly annotates the inference basis.
`breakdown` and `utilization` are empty;
area and cost are `None`. The default cost estimator also marks these new components as unknown.
Result details retain component mapping, capabilities, trace/config/engine/adapter code hashes, native statistics, and evidence directories.

Each execution creates an independent directory, saving configuration, staged trace, command list, stdout/stderr, and `run.json`.
Only a fixed argument array is passed; no shell command field is provided. The upstream legacy reader internally uses popen,
so the staging directory must be a shell-safe absolute path with fixed filenames; original paths are not directly concatenated into the shell.
The native process uses a restricted environment, separate process group, and memory/core dump/CPU/wall timeout limits.
Timeout only terminates this process group; the failure directory is retained.

Non-zero exit, deadlock, max-cycle truncation, missing/duplicate kernels, incomplete statistics, wrong ISA/
hash/provenance fields/mapping cannot produce a successful Evaluation. A native exit code of zero must also pass
complete count checks. Unknown fields and unsupported parameters/topologies are rejected before launch.

## Provenance and licensing

This project's Python integration code is under Apache-2.0, retaining the project NOTICE.
Upstream configuration parameters come from the two pinned commits mentioned above; relevant copyright belongs to Timothy Rogers/Purdue University
and the GPGPU-Sim authors; upstream uses their respective BSD-style licenses.
Source code, binaries, and CUDA Toolkit remain under their respective LICENSE/copyright notices and are not relicensed as Apache-2.0.
This project does not distribute them. When deploying, retain the LICENSE/COPYRIGHT files of downloaded packages and engine source code.
The upstream license text for parameter provenance is retained in source, wheel, and sdist via
`THIRD_PARTY_NOTICES` in `simulator/accelsim/profile.py`.

The [official roadmap](https://accel-sim.github.io/) lists Blackwell as experimental follow-up work.
This entry point only accepts the SM90 subset described above; H100 configurations and synthetic traces cannot prove the ISA, resources, scheduling, clocks, or physical accuracy of RTX PRO 5000
Blackwell. Future work requires separately completing real trace input coverage,
device support audits, and independently frozen hardware verification before advancing to full ViT/CLIP evaluation.


## SM120 typed mechanism variant

`arch-explorer.accelsim-engine.v2` identifies a separate patched engine.
The installable backend verifies the actual binary and simulator-library
SHA-256 against an approved build, both upstream revisions, minimal patch
identities and resulting source trees. The fixed original engine continues
to consume its original SM90 v1/v2 requests. A SM120 trace requires the new
engine and the explicit `sm120_mechanism_adf()` configuration.

This is a synthetic two-SM mechanism topology with forty memory channels
and four independent logical R/UR/P/UP bank spaces. It does not represent
a calibrated Blackwell device. Uniform register accesses count one logical
warp-wide unit; P/R accesses count observed lanes. All timings still come
from the one native resource simulation. There is no Python body time or
shape correction. Area, energy and hardware accuracy remain unknown.

The new trace manifest is `arch-explorer.accelsim-trace.v3`, with actual
binary version 120 and typed native trace version 7. Every instruction
preserves explicit predicate-control identity and the observed pre-predicate
active mask, including all-predicated-off records. Width and lane addresses
are explicit; ordinary LDG/STG widths 16/32/64/128 bits, descriptor UR pairs
and wide R pairs are preserved. Supported bases/modifiers are enumerated
in `typed_trace.py`; specialized/unknown instructions and ordering modes
refuse execution. The native operand collector and scoreboard consume the
typed identities, including multiple outputs and zero/true constants.

For hardware sources, `capture_binding` freezes original SIM28NV1 records,
complete NVBit static metadata, actual cubin, registration, launch and live
preflight. The public reader reconstructs the consumed trace from these
original bytes, checks every static instruction encoding against ELF .text,
full per-warp straight-line PC sequences and lane exits, resource scope and
actual allocation bounds. This capture dialect currently covers complete
straight-line ordinary kernels with predicated EXIT, not arbitrary divergent
control flow. The independent offline capture reader additionally checks
static decoded operands and controlled effective addresses/output values.
Hash binding establishes byte identity; it does not certify a dishonest
provenance declaration or replace independent observation.

`typed_trace.create_request()` has the same source/trace_id/compiler/engine
interface as the v2 creator and accepts `capture_binding` for hardware inputs.
`python -m arch_explorer.components.accelsim.capabilities --cubin FILE
--engine ENGINE.json` can distinguish actual approved SM120 native capability
from the original engine. A static cubin still lacks a dynamic trace. The
hardware-accuracy preflight stays blocked by missing validated physical
parameters even after complete software execution.

Patch/build details are in `tools/accelsim_native/README.md`, and the bounded
NVBit ABI is in `tools/accelsim_native/capture/SPEC.md`. No native binary,
NVBit library, private capture, hardware result or model weight is shipped.


### Device-fact constrained scenarios (SIM29)

The original H100 and SIM28 two-SM fixtures remain attributed software seeds.
The configuration compiler also accepts positive cluster/channel counts and
independent positive integral-MHz core/interconnect/L2/DRAM clocks. This is a
configuration capability, not a calibrated Blackwell profile. No Python latency,
shape, operator, or measured-residual adjustment is applied.

`archs.accelsim.device_scenario(assumptions, evidence)` reads an existing
`arch-explorer.sim28-live-preflight.v1` file without querying a GPU. Its original
SHA256, UUID and CC are bound through an optional `device_evidence` Param. That
Param contains a file reference only; all effective values have one authority in
ADF. The compiler rereads the bound original and rejects changed bytes, units,
source labels, resource values or SM count. A hash proves byte identity, not that
a caller's claimed device history is true. The supported historical importer
currently targets the existing CC12.0 ordinary mechanism; other CCs are refused.

```python
from arch_explorer.adf import Param
from arch_explorer.archs.accelsim import sm120_mechanism_adf, device_scenario
from arch_explorer.components.accelsim.config import compile_adf

# An explicit, complete set of synthetic assumptions is mandatory. The importer
# supplies no missing clock/cache/DRAM/NoC performance values.
a = sm120_mechanism_adf()
a.networks[0].params['booksim_k'] = Param(190, 'count',
    'synthetic assumption: 110 SM endpoints + 40 channels * 2 subpartitions')
a.child('cluster').child('l1_shared').params['gpgpu_adaptive_cache_config'] = Param(
    0, '1', 'synthetic assumption: static shared allocation; cache split unknown')
scene = device_scenario(a, '/data/historical-preflight.json')
compiled = compile_adf(scene)
print(compiled.device_facts)  # known facts, assumptions, unknowns; live_lease=False
```

For the recorded device, factual imports are 110 SMs, register capacity per SM
and per block, shared capacity per SM and per block, and CC12.0. Warp size and
per-block thread bound constrain input applicability. A captured function's
occupancy does not establish the device's maximum threads or CTAs per SM.
Shared initial allocation is derived from the single shared-capacity Param when
omitted; a contradictory explicit alias is rejected. Static allocation is
required for a fact-bound shared capacity because the adaptive cache split has
not been observed. `counterfactual(scene)` explicitly removes the device-fact
claim and labels its retained values synthetic for sensitivity experiments.

Fact claims are a finite semantic mapping from `(native owner, ADF key)` to the
original pointer and unit. Every claimed value, unit, pointer and evidence SHA
must match that mapping; equal numeric values or equal units do not make two
parameters interchangeable. Unobserved fields cannot borrow a fact source.
The same audit supplies component-map classifications, including the SM-count
repeat and the declared static shared-allocation alias. Other repeats remain
explicit scenario topology. File identity is not an execution lease.

The native constraints retained before launch include:

- One core per cluster in this adapter subset; no dummy SM padding. L2
  subpartitions per channel must be a power of two in 1..16, not fixed to two.
- Chiplet count divides both clusters and channels; native IPOLY-MODULO indexing
  requires the power-of-two factor of total subpartitions to be 16, 32 or 64.
- BookSim remains single-stage fly, destination-tag routing and identity endpoint
  mapping; `k = clusters + channels * subpartitions`, bounded to 4096 ports.
  Unsupported topology/route/hash combinations are rejected, not renamed.
- Registers/shared per block cannot exceed their corresponding per-SM capacities;
  the optional shared default must agree. CTA capacity stays within native
  MAX_CTA_PER_SHADER=32. Clock domains need no fixed 2:1 relationship.
- Core MHz is bounded to 4294 because the pinned native frequency getter returns
  uint Hz; other integral-MHz domains are bounded to 1000000. Invalid units,
  booleans, nonpositive frequencies and unsupported compound grammars are refused.

Scalar L1/shared latency fields use core cycles. `gpgpu_l2_rop_latency` and
`dram_latency` are also **core-cycle ready timestamps**, polled by the L2/DRAM
paths; owner names alone do not define their unit. The compiler records that
clock-domain distinction. Native DRAM controller MHz is separate from
`dram_data_command_freq_ratio`; neither is automatically a DDR data rate or
vendor Gb/s. `gpgpu_simple_dram_model=1` is an explicit fixed-service assumption
and bypasses detailed dram_t command counters. Mode0 selects the already
implemented detailed controller. Both retain unknown physical timing. Pinned
sources: `gpu-sim.cc:init_clock_domains/next_clock_domain`,
`l2cache.cc:memory_partition_unit::*dram* / memory_sub_partition::push`,
`shader.cc:shader_core_config::max_cta`, `addrdec.cc/hashing.cc`, and
`intersim2/networks/fly.cpp` at the engine manifest's GPGPU-Sim SHA.

Use `typed_trace.create_request(..., adf=scene)` with the full original typed
trace, then the existing installed API or `archexplorer evaluate` route. A
synthetic trace must retain `kind='synthetic', gpu='none'` and explicit toolchain
and description; a fact-constrained ADF does not turn it into captured execution.
For a pure-CPU scenario-building command:

```sh
python -m arch_explorer.components.accelsim.device assumptions.json historical-preflight.json --output scenario.json
```

No source metadata, successful 110-SM run, or matching API/CLI cycles makes
`hardware-accuracy` pass: same-device cache/network/DRAM/clock and typed-bank
parameters remain unvalidated. Historical preflight evidence is not a live
resource lease. This interface does not authorize device queries or execution.

The source distribution also includes `tools/accelsim_native/calibration/`: pinned public microbenchmark provenance, independently authored new candidates, offline build/static-byte checks, a CPU-only dry-run, conditional service-observable solver and replay. Its raw-record schema and source-reference JSON are shipped with the sources. See that directory's README for explicit budgets and blocked capture/physical-evidence prerequisites. Empty observations remain blocked; synthetic solver exercises are not hardware calibration or a target-specific coefficient table.


## Finite microbenchmark originals and timing

The installed `arch_explorer.components.accelsim.microbundle` module accepts the
versioned SIM30 registry: twelve complete, unrolled dep/quad/chase functions,
one CTA of 32 threads, three fit lengths and one interior timing holdout in each
family. It checks complete ELF instruction bytes, offline resources and an
independent unsigned input/output oracle. This finite registry grants no device
execution permission. Unknown functions, altered input bytes and enlarged budgets
are rejected.

`microcapture` consumes versioned raw SIM30NV1 records plus the lossless original
instruction dictionary/table, actual allocations, outputs and process evidence.
It reconstructs ordinary typed trace v7 through the same renderer as SIM28.
Dynamic global addresses come from the recorded MREF channel and must match an
independent per-lane oracle; no addresses are invented from a shape.
`microrecords` owns raw observation semantics, including actual host/preflight,
loader maps, event handles and float32 milliseconds/bits/nanoseconds closure.
`microexperiment` consumes these original references for fit-only conditional
service intervals. An arbitrary dictionary labelled `hardware` is insufficient.

A CPU-only installed check is:

```sh
python -m arch_explorer.components.accelsim.microcampaign check /absolute/bundle/REGISTRY.json
```

The explicit `run --design ... --handoff ... --output NEW_DIR` operation additionally
requires a newly coordinated handoff, exact installed source/host/tracer identities,
the common lock, no external compute, and all actual functions live before launch.
It is a bounded measurement operation, not a default test. Its append-only journal
freezes all initial native predictions before fit timing, and final intervals and
unchanged native assumptions before heldout timing. Capture and uninstrumented
event timing run in separate processes. Failures retain all originals and the full
attempted/not-started denominator; there are no warmups, retries or replacement
samples. The native engine remains responsible for all simulated cycles.

Event resolution is approximate, and three samples do not establish an error
bound. Service slopes remain conditional composite observations: unknown clocks,
issue mechanisms, caches, queues and instruction-cache effects prevent a physical
L/I or L1/L2/DRAM decomposition. Such intervals are never written into an assumed
native profile as measured parameters. No full-model accuracy claim follows.


The launch environment and the in-process environment are separate observations.
NVBit initialization may remove `LD_PRELOAD` before its helper CPU tools run. A
capture reader accepts only that removal when the exact frozen tracer file is
mapped both before CUDA initialization and before host close; all other variable
changes remain errors. Uninstrumented timing has no such allowance. Both original
environment records remain intact. A failed observation window stays closed even
when a later CPU reader correction can consume its already captured originals.
## Finite measurement outcome replay

`python -m arch_explorer.components.accelsim.microclosure DESIGN WINDOW
--evidence CONTEXT --reader-source SOURCE --output RESULT` is a CPU-only reader.
`CONTEXT` and `SOURCE` are JSON file paths. It never authorizes a device window.
The result separates `replay_status`, `collection_outcome`, every registered
attempt, completed stages, postclosure software recovery and unknown hardware
accuracy. A successfully read failed collection remains failed.

Historical `arch-explorer.sim30-outcomes.v1` needs an explicit closure context
containing hash-bound design/source, journal, observations and existing resource
reaping evidence. New producers write v2 with that same context embedded by
reference. Unknown versions/states, missing/duplicate attempts, orphan scoped
observations, substituted freezes, foreign runs and contradictory cleanup are
rejected. The reader only scans declared campaign roots and the supplied view.
An unrelated sibling directory cannot affect replay. Resource proof consistency
is historical evidence and conveys no current reservation or permission.

Both production `Collector.execute` and installed replay use `microclosure`;
Journal's production methods also replay the complete phase sequence. Prediction
freezes are revalidated against complete native protocols and fit-only inputs.
Fit and heldout observations are closed in both directions to actual process,
start, return and raw-event originals. No new timing can be inferred from a
postclosure native result or from a successful software replay.

`python -m arch_explorer.components.accelsim.microrehearsal --source SOURCE
--bundle REGISTRY --template IMMUTABLE_DELIVERY --output NEW_DIRECTORY
--mode success` drives the same phase orchestration using an explicitly
synthetic process/device boundary from the frozen source archive. Modes
`first_failure`, `timeout` and `quality_refused` stop without replacement samples.
This test command creates no process, device event, GPU lock or real lease and
does not run the native engine. It reads unchanged completed native protocols as
fixtures. The production hardware replay rejects synthetic source/marker roots;
the rehearsal result is always labelled synthetic, with physical parameters
unset and hardware accuracy unvalidated. The approximate event resolution and
conditional interval assumptions remain unchanged.

New finite campaigns use `arch-explorer.measurement-registry.v2`. The registry
declares the work item, compiled function prefix, per-program integer oracle and
bounded budget. Designs, handoffs, leases, receipts, freezes, journal and outcomes
carry that identity with versioned measurement schemas. Stable raw CUDA/NVBit
transport formats remain v1; their version does not identify an experiment.
Historical SIM30 records are still read with their original, bounded contract.
Unknown versions, foreign work items and old handoffs cannot open a new campaign.

New registry rehearsals also require `--native-fixture PLAN`: an explicitly
synthetic plan containing completed public protocols for every new full program,
plus the exact registry, engine, host, tracer and history references. Renaming an
old prediction cannot satisfy it. Complete new static bytes, dynamic addresses,
oracles and synthetic events pass the same parsers used for collection.

Closed new campaigns attach `native_timing_comparison`: every registered fit and
heldout row, original float32 event bits and units, assumed native clock/time,
signed and absolute errors, and summaries with explicit excluded denominators.
Relative errors are fractions (0.10 means 10%). Missing or refused timing stays
visible and never becomes zero error. Conditional composite intervals are
separate from native predictions; physical parameters remain unknown.

`python -m arch_explorer.components.accelsim.microguard selfcheck --root NEW_DIR`
checks CPU descendant stopping/reaping without a GPU lock. Its `run --design
DESIGN --handoff HANDOFF --window NEW_WINDOW --root NEW_SUPERVISION` entry requires
the frozen installation and a fresh actual handoff with an absolute deadline.
It reserves cleanup time, bounds the worker by both wall and monotonic clocks,
and records owned PID/start-tick identities and the final lock recheck. The
handoff deadline never moves when the common lock is later acquired.
