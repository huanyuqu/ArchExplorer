# Bounded Performance Scenarios and Structural Exploration

First install the release main wheel, and place this directory outside the repository to run. The examples call public data structures, registered mappings, Env, and provider APIs. The default input is one layer of Llama2-7B, batch 1, sequence length 4, using synthetic non-GPU hardware to construct a performance task graph.

Generate the complete configuration and two specific mappings:

```sh
python scene.py emit --input llm.json --output ./llm-input
python scene.py prepare --scenes ./llm-input/SCENES.json --index 0 --output ./llm-request
archexplorer evaluate ./llm-request/REQUEST.json --output ./llm-result.json
archexplorer evaluate ./llm-result.json --output ./llm-readonly.json
archexplorer search --config ./llm-input/CONFIG.json --policy greedy --output ./llm-best.json
archexplorer agent run --config ./llm-input/AGENT-CONFIG.json --policy scripted --max-steps 2 --output ./llm-agent.json
```

The first evaluate executes components, and the second reads the source-bound original receipt. The output directory must be a new directory; preserve the provider subdirectory. Search allows at most four attempted evaluations, and the two mappings share the quota. Explicit SRAM parameter changes also consume the same budget. Check the actual ledger, failures, and termination in the results; do not infer from the returned JSON alone that the budget has been fully used.

Ordinary LLM search BEST's `archexplorer.adf-result.v1` read will re-evaluate. To save the winner's read-only receipt, use the result `best.adf`, `best.workloads[i].workload`, and `best.mappings[i]` to prepare a new events request, then save its result. For compiled graph search replay / rerun usage, see the MLA example below.

Generate grouped SRAM and primitive records:

```sh
python structure.py --output ./sram-structure
```

Four private SRAMs are merged into two intra-group pools, with compute count and total SRAM preserved, and the network is an explicit synthetic configuration. The output SCOPE declares that only structure is generated here; shared access, broadcast reuse, and contention must be supported by formal run components before execution. Ordinary buses must not be treated as implemented hardware multicast.

The three-layer joint exploration example reuses the same Env:

```sh
python joint_search.py --output ./joint-input
archexplorer search --config ./joint-input/CONFIG.json --policy greedy --output ./joint-best.json
```

This independent configuration uses seed 7, three evaluations, and two search steps, first giving a synthetic GSM structure for one layer of Llama2-7B. Candidates include DRAM bandwidth 64→128 B/ns and a GSM→DMC structural transformation composed of primitives; each step samples only one actual mapping from the public gsm_tiled/dmc_partition domain. The results record structures, parameters, and complete mappings that were evaluated but not accepted, without requiring the structural transformation to win. The original private→GSM edit record is in ORIGIN.json. The ordinary search format still restores according to its original re-evaluation contract.

The Python API can use `llm_workload`, `llm_scene`, `llm_search_config`. The generated AGENT-CONFIG uses the same hardware and workload, the first specific mapping in the list, one budget per workload, and explicitly sets two-step evaluate / submit scripts. The offline Agent calls existing tools and does not connect to online model services.

When changing model, phase, batch, seq_len, layers, or dtype, modify all explicit dimensions in llm.json. The registered configurations for Llama2-70B, Llama3-70B, and Qwen-72B are used to construct performance task graphs. Each probe first limits a single process to 120 seconds, 8 GiB, and single-threaded BLAS; if the limit is exceeded, keep the failure and then split.

## Compilation Search for Small MLA

The MLA scenario generates an actual graph from the registered builder, with root values determined by public shapes. The example below defaults to FP32, four heads, short compressed cache, and two compute groups. shared/private and multicast on/off are all explicit parameters.

```sh
python mla.py --sharing grouped --multicast on --output ./mla-input
archexplorer search --config ./mla-input/CONFIG.json --policy greedy --output ./mla-best.json
archexplorer evaluate ./mla-best.json --output ./mla-readonly.json
archexplorer rerun ./mla-best.json --work-dir ./mla-rerun --output ./mla-rerun.json
archexplorer agent run --config ./mla-input/AGENT-CONFIG.json --policy scripted --max-steps 3 --output ./mla-agent.json
```

The evaluate here reads the new compiled-search winner and performs read-only validation according to its schema. rerun uses a new execution directory. Complete source/API consumption examples are:

```python
from arch_explorer.archs.mla import mla_scene
from arch_explorer.workload.api import compile_workload
from arch_explorer.simulator.api import prepare_compiled, evaluate_request

adf, workload, mapping = mla_scene(shared=False, multicast=False, candidate=1, layout='views')
compiled = compile_workload(adf, workload, mapping)
request = prepare_compiled(compiled, output_dir='./mla-api-run', options={'trace': True})
result = evaluate_request(request)
same = evaluate_request(result)
```

`mla_decode` uses absorbed projection and complete explicit cache to perform a single decode. v4 supports external Events; the example uses FP32 data and explicit cache initialization.
