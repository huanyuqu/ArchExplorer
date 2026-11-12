# Non-GPU Synthetic Architectures and Retained Workload Generators

These generators describe systolic/vector resources, workloads, and explicit device interconnects, using `backend="events"`. All hardware values are synthetic assumptions, and results do not represent GPU performance, physical latency accuracy, real weight numerical inference, or robot closed-loop. The native GPU ViT/CLIP demo continues to use the [Accel-Sim program entry](tutorials/05-full-models.md).

Old GPU-specific `arch_explorer.devices.scans` matmul_sweep, softmax_sweep, a100_operators, prefill_sweep, and module CLI have been retired, and paper figure default configurations are no longer public sweep entries. Common evaluate, search, explicit workload/mapping, and retained non-GPU presets continue to use the original interfaces.

## Explicit Hardware Construction Interface

```python
from arch_explorer.archs.adf_seeds import non_gpu_adf

adf = non_gpu_adf(num_cores=2, matrix_precision="bf16",
                  dram_capacity_bytes=64 * 2**30)
print(adf.to_dict())
```

`num_cores` and `dram_capacity_bytes` must be positive integers, rejecting bool; `matrix_precision` is only fp16/bf16/fp32. Each core contains a 32×32, 1 GHz, WS/double-buffered systolic array, with matrix operands using the selected precision; there is also a 16-lane, 4-SFU, 1 GHz FP32 vector unit, explicitly enabling FP32 GEMM/MVM, handling FP32 matrix, vector, conversion, and reduction capabilities. Each core has 4 MiB SRAM and a 128 B/ns local bus; the chip connects each core and DRAM through a mesh constructed by endpoint count. DRAM defaults to 64 GiB, 64 B/ns, 100 ns; parameters are itemized with synthetic sources and units. These values are not balanced against old GPU predictions and are not any vendor device parameters.

Operator precision is specified by the workload contract. A BF16-only array cannot accept FP32 attention; the FP32 vector retains that capability. When the corresponding matrix precision is missing, execution is rejected; FP16, BF16, and FP32 are not treated as interchangeable labels. The original `dmc_adf` is unchanged; `gsm_adf` or old GPU aliases are not provided.

## Generating from Installation Packages and Public Source Code

In a source archive, export, or unpacked sdist, after installing the corresponding wheel, run:

```sh
python tools/generate_pi05.py /tmp/pi05-inputs
python tools/generate_multi_device.py /tmp/multi-device-inputs
python tools/generate_multi_device_execution.py /tmp/multi-device-execution-inputs
```

All three scripts retain the original `OUTPUT_DIRECTORY` parameter and `--help`. Output filenames and Python `configuration()` / `cases()` entries are retained; artifacts use new seeds, not just fixing script imports. Scripts are public source tools, and the wheel provides the production package; scripts do not inject a development working tree into the Python path.

The π0.5 generator first calls the existing `normalize`, with default dtype **bfloat16**, corresponding to the BF16 array; float32 corresponds to the FP32 array, and both always retain the FP32 vector. The input example's three-way vision, prefix KV, action expert, flow, mask, cache, and output shapes come from `openpi.pi05.v1`. FP32 initialized weights, shared BF16 converted copies, operator-level precision, accumulation/reduction, and FP32 output maintain the original graph definition; the entire graph cannot be forced to BF16 or FP32. 64 GiB is an explicit synthetic capacity for holding initialized and converted weights, not an inferred device fact.

```python
from tools.generate_pi05 import configuration

bf16 = configuration(profile="openpi_pi05_dummy", num_steps=1,
                     action_dim=1, action_horizon=1, max_token_len=1)
fp32 = configuration(profile="openpi_pi05_dummy", dtype="float32", num_steps=1,
                     action_dim=1, action_horizon=1, max_token_len=1)
```

Generation and graph/component capability checks do not equal complete events execution. Complete π0.5 execution requires the mapping's applicable/apply to be consistent with the events-declared `openpi.pi05.v1`; explicit errors are raised when unsupported. Execution and ledger boundaries are in [π0.5 documentation](pi05.md). π0.5 native Accel-Sim, multi-device π0.5, or PIM extensions are not provided by this generator.

## Multi-Device, Sharding, and FP32 Communication

`generate_multi_device.configuration(names, workload)` retains any legal unique device names; adjacency directions and asymmetric bandwidth are determined by explicit topology_edges. The default complete ViT/CLIP is FP16; other inputs select array precision by the workload IR's dtype, and the FP32 vector always exists. Empty/duplicate device names and illegal workloads are rejected.

`generate_multi_device_execution.configuration(kind, count, profile, tp)` retains per-operator placement; TP cases shard the real K dimension by rank and record FP32 tree allreduce. Input matrices and final FP16 outputs are not changed to full-graph FP32 due to FP32 partial sums/reductions. count must be a positive integer, tp must be bool; unknown kind, inapplicable profile, or illegal sharding/topology are rejected by the generator and formal request validation.

You can perform small CPU validation from the complete execution generator's `matrix-tp.json`, or independently evaluate through public objects:

```python
from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate, from_dict
from tools.generate_multi_device_execution import configuration

config = configuration("operator", 2, tp=True)
result = evaluate(from_dict(config["seed_adf"]),
                  WorkloadSpec(**config["workloads"][0]),
                  MappingSpec("multi_device", config["mapping_selectors"]["multi_device"]),
                  backend="events", cost_model=False)
print(result.to_dict())
```

Check K shard coverage, FP32 partials, reduction additions, bytes per message, and output dtype. Topology reachability, capacity, precision, and execution support are independent conditions; do not claim the entire workload execution passed just because JSON was generated.
