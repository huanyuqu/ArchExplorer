# Agent architecture exploration environment

ArchExplorer provides the same set of stateful tools for Portal Resident, Claude Code, and search strategies.
Models are run by external Agent hosts; this tool only provides Python, JSON-lines, and MCP interfaces, as well as offline
scripted demonstration strategies. No LLM API, model name, or key configuration is required.

## Configuration and startup

`examples/adf-dmc.json` is a general synthetic configuration that can run directly with the bundled
components, mapping templates, and the registered `events` executor. At environment startup,
`arch_explorer.components.api.discover_components()` loads the built-in catalog and installed extensions for
seed validation, tool descriptions, and mapping selection before the first evaluation. `adf.evaluate` also uses registered capabilities.
No additional module registration is needed for the example configuration. Extension authors should see the [standalone component package tutorial](component-extensions.md).

`archexplorer components` and `describe_environment.component_catalog` use the same public
component catalog, listing implementation IDs/versions, provenance, inputs, parameters, and limitations by hardware type. The original `components` field retains
the parameter domain projection needed for structure search; no separate implementation list is created inside the Agent. The `implementation`
field at ADF points carries precise bindings and independent parameters; both the complete `seed_adf` and `create` specs retain this field.
Ordinary `set_parameters` still edits common hardware parameters and does not mix implementation parameters into the common namespace.
The roofline workload for the same SM and the Accel-Sim native program have different contracts; inputs cannot be directly reused after simply changing names.
See [implementation selection and outsourcing examples](component-implementations.md).

Public compilation first uses `task_adapters` to read the registered catalog, then uses the `compile_workload`, `prepare_compiled`, and `evaluate_request` tools:
the first two do not execute and do not consume evaluation budget; the third counts one attempt for a public prepared request, and failures are also charged;
read-only validation of saved results does not consume budget. Pass bindings by component path; adapter-specific parameters go in their options;
public mappings pass compute/memory/rules, with specific options validated by registered rules. The catalog is returned by the public task_adapters; no Agent list is set.
Descriptors are consistent with describe_environment.task_adapters. The operator_mappings tool and the describe_environment field of the same name read the public rules catalog.
Old v1/v2 compilations generate new v3 through explicit migrate_compilation_v1/migrate_compilation_v2 respectively;
old execution results are not migrated or relabeled; missing compatibility items, ambiguity, wrong scope, and adapter-specific parameter errors are rejected by the public contract.
Prepared request results do not replace current exploration candidates or enter best ranking; the original `evaluate` / `submit` retain their original meaning.
For parameters and default matmul/outsourcing calls, see the [public compilation tutorial](compilation.md).

Configure `compilation: {bindings?: object, implementations?: object}` to make existing evaluate/search perform public compilation→preparation→evaluation for each selected mapping;
do not specify the old backend at the same time. backend_options still belong to prepare. Mapping check/compilation/preparation failure in this mode counts as one candidate attempt, with stage and reason retained;
no evaluation cache. mapping_selectors can be a request object or a non-empty list of request objects; the latter passes through the formal request_space item by item.
search.proposals can retain original structure edits, or use `{tool:"set_parameters",args:{parameters:[...]}}`.
New compilation exploration results are read-only validated by evaluate_request and do not replace current candidates or spend new budget; the original evaluate is an explicit new evaluation.
CLI rerun/Python exploration.rerun separately creates new executions. For budget and format details, see [attention and exploration](attention.md).

| Field | Semantics; default when omitted |
| --- | --- |
| seed_adf | Complete ADF JSON, or `{"preset":"dmc|vectorcgra_4x4|vectorcgra_4x4_diag","parameters":{...}}`; required. Python API also accepts an ADF object. |
| workloads | WorkloadSpec JSON list; kind, params, weight, name. weight is a finite positive value; the list is non-empty. |
| backend / backend_options | Registered backend name, default events; options are passed directly to the backend. |
| registration_modules | Additional local plugin registration modules imported after built-in catalog loading; default empty. Only for extension models or explicit test doubles; this is trusted local configuration; submission and replay retain these module names. |
| max_evaluations | Total backend call limit, default 100. |
| cost_budget_usd / area_budget_mm2 | Positive number or null; null disables the constraint. |
| cost_penalty / area_penalty | Non-negative penalty coefficients, default 1.0 each. |
| parameter_samples / mapping_samples | Positive integers, default 4 each. The former includes the baseline parameter version; the latter is the mapping limit per workload. |
| parameter_radius / growth_probability | Default 0.25 / 0.75; rules in structure.md. |
| shrink_probability | Default 0.75; probability of prioritizing scale parameter reduction when known cost or area exceeds budget. |
| parameter_domains | Default []; explicit numeric search domains listing values and source by path/param. Validates existing component constraints and supports authorizing discrete search for originally fixed coordinates. |
| cost_model | Independent area/cost model configuration: basis=raw or full, process_nm / die_count with provenance, overrides by ADF path; see cost.md. |
| random_seed | Integer, default 0. Replayable when the same operation sequence and backend guarantee determinism. |
| mapping_templates | List of restricted template names; when omitted, all registered and applicable templates are used. |
| trace_path | JSONL file path, must not already exist; when omitted, a temporary directory is automatically created. The service outputs the path to stderr. |
| search | max_steps, temperature, cooling, max_relative_degradation, optional proposals. |
| agent | script, max_steps, max_tool_calls, log_path, only for offline scripts. |

The example writes out all algorithm parameters explicitly, with area/cost budgets as null. When constraints are needed, set budgets and penalty coefficients, and first check
the scope and assumptions of the independent cost model; model estimates cannot be treated as quotations.
Configuration is uniformly validated by the ADF environment; unknown fields and missing required fields return ordinary configuration errors.

```sh
archexplorer search --config examples/adf-dmc.json --policy sa --output results/sa.json
archexplorer search --config examples/adf-dmc.json --policy greedy --basic-only --output results/basic.json
archexplorer agent run --config examples/adf-dmc.json --policy scripted --output results/scripted.json
archexplorer evaluate results/scripted.json --output results/replayed.json
```

## Tools and state

### CGRA named seeds and general search

`examples/adf-cgra.json` and `examples/adf-cgra-diag.json` start from ordinary/diagonal mesh respectively,
both using synthetic uint32 GEMM that fits in the default SRAM. Seed parameters are passed directly to `devices.get`;
the default is 32 words per bank; `{"preset":"vectorcgra_4x4","parameters":{"words_per_bank":128}}`
is a caller-explicitly-selected expanded capacity scenario and cannot be called paper hardware parameters. Illegal parameters report errors directly; insufficient capacity does not auto-expand.

```sh
archexplorer search --config examples/adf-cgra.json --policy sa --output results/cgra-sa.json
archexplorer search --config examples/adf-cgra.json --policy greedy --output results/cgra-greedy.json
archexplorer search --config examples/adf-cgra.json --policy random --output results/cgra-random.json
archexplorer agent run --config examples/adf-cgra.json --policy scripted --output results/cgra-agent.json
archexplorer evaluate results/cgra-agent.json --output results/cgra-replayed.json
archexplorer env serve --config examples/adf-cgra.json
archexplorer mcp --config examples/adf-cgra.json
```

The above commands can replace the configuration file with `examples/adf-cgra-diag.json`; all search/Agent results
can be replayed by an independent process `evaluate`. Both examples fix random_seed=17, total budget 12, 4
mapping candidates per time, and at most 6 structure steps. Default structure proposals share the component catalog and basic edits with other ADFs;
within limited steps, unused budget may remain due to physical unsupportedness or duplication. The scripted script explicitly evaluates two topologies,
recovery, and submits best; this is a declared limited demonstration, not the default reachable set.

`describe_environment.search_space` reports `component_graph`. Any legal ADF can be input,
including trees with different names or groupings; component validation and the physical valid domains of mapping/backend still determine whether candidates are executable.

External Agents perform switching based on real tool return values, for example in the same `env serve` process:

```python
import json, subprocess
p = subprocess.Popen(["archexplorer", "env", "serve", "--config", "examples/adf-cgra.json"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def call(tool, args=None):
    p.stdin.write(json.dumps({"id": tool, "tool": tool, "args": args or {}}) + "\n")
    p.stdin.flush()
    response = json.loads(p.stdout.readline())
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]
description = call("describe_environment")
call("evaluate", {"optimize": "mapping"})
call("checkpoint", {"name": "seed"})
# Reuse the explicit, atomic topology edit from this example's scripted actions.
config = json.load(open("examples/adf-cgra.json"))
topology_edit = next(a["args"] for a in config["agent"]["script"] if a["tool"] == "apply_edits")
call("apply_edits", topology_edit)
call("evaluate", {"optimize": "mapping"})
call("restore", {"name": "seed"})
call("submit", {"selection": "best", "path": "results/cgra-tools.json"})
p.stdin.close()
assert p.wait() == 0
```

CGRA uses general component catalog candidates and explicit parameter editing; `parameter_samples` executes according to common rules.
Parameter choices given by users must satisfy component domains and backend physical relationships; unsupported arrays, precisions, capacities, or
clock/bandwidth combinations will report errors at evaluation time. `cgra_spatial`'s ii, placement, routing, memory_port
are still defined by the existing mapping space; ii is the scalar minimum issue interval, not the compiler modulo loop II.

The examples do not set area/dollar budgets; unknown metrics remain null. When these budgets are enabled, the objective is unavailable.
Unknown physical clock is used for cycle normalization; latency_ns cannot be declared as measured nanosecond performance.
Native details.cycles, provenance, and warnings are retained in submission and replay; see [physical boundaries](cgra.md).

### General tools

The sole machine-readable definition source is `arch_explorer.agent_env.schemas.TOOL_SCHEMAS`; all transport uses it.
Python calls such as `env.evaluate(optimize="mapping")` return `{"result": ...}` or
`{"error":{"code":...,"message":...,"details":...}}`. You can also use `env.call(name, args)`.

| Tool | Usage |
| --- | --- |
| describe_environment | Read objectives, budgets, component roles/parameters/units/domain, edit schema, mapping templates, and all tool schemas. |
| get_state | Current ADF JSON, indented tree, specification fingerprint, component area/cost estimates, recent evaluations, and remaining quota. |
| apply_edits | `{"edits":[...]}`; atomically execute 1–3 basic edits or one high-level program. Returns the expanded basic edits. |
| set_parameters | `{"parameters":[{"path":"chip/core/sm","param":"clock_ghz","value":1.2,"source":"synthetic choice"}]}`; constrained to the domain. |
| evaluate | optimize is none, mapping, or parameters_and_mapping. none uses the currently selected mapping; the first time it automatically picks a legal candidate from an applicable template. One mappings per workload can be provided explicitly. |
| checkpoint / restore | `{"name":"baseline"}`; save/restore ADF, mapping, recent evaluations, and edit history. checkpoint may omit name. |
| undo | Undo the most recent operation that successfully changed state, including parameters/mapping selected by evaluate. |
| history | Paginated read of call summaries and backend attempts with offset/limit; default 0/100. |
| best | Returns the complete evaluated candidate with the smallest objective and the quota used. |
| submit | selection is best (default) or current; optional path writes a JSON file that can be replayed independently. |

One evaluation unit is "one backend invocation attempt for one concrete mapping of one workload". Each attempt in the middle/inner layers counts separately, and failures also count. Before starting a parameter version, keep at least enough quota for one complete workload suite; the remaining balance that is less than one suite is retained. Structural edits, parameter settings, viewing, and restore do not consume quota. restore/undo do not refund quota, do not roll back history, and do not reset random number state; best is preserved across restores.

For each workload, output the raw Evaluation, operators sorted by cumulative task time, the resource with the highest utilization, and a provenance source summary. Task times may overlap, so cumulative values cannot be treated as end-to-end latency. Component area/cost breakdown is provided by `arch_explorer.cost.estimate`; the objective uses its `area_mm2/cost_usd`, combined with backend latency. `get_state.costs` and the evaluated candidate's `cost_estimate` include category/path breakdown, sources, assumptions, omitted, and unsupported items. Unknown components cause total area/cost to be null; if this affects enabled constraints, the objective is also null and cannot be claimed to satisfy the budget. DRAM silicon area is listed separately; its absence does not mean the logic die area or purchase price is unknown. The backend's original Evaluation values are preserved as raw output and may still be null; the independent model is not overwritten. See [cost model scope](cost.md) for details.

After structural edits pass ADF syntax and component parameter validation, they may still fail to meet backend construction or mapping requirements. The rules are based on [ADF requirements of the builder and the templates](components.md#adf-requirements-of-the-builder-and-the-templates). When `evaluate` fails, check `error.details.trials`: `cause` preserves the structured reason; if the backend was attempted, `cause.details.attempts` contains the count index, the specific mapping, and the original error. For example, when mesh capacity is insufficient, it includes the network path, endpoint count, and mesh size. When no template is applicable, it returns `no_mapping`, and no backend quota has been consumed at that point; correct the structure based on the current tree in `get_state` and the above constraints, or restore a checkpoint. `history` keeps the complete attempt record; failed backend calls still consume quota; failures do not become the best candidate.

All tool calls and returns are written to JSONL; the first line is the configuration. Replay is possible under the same model version and registered modules:

```python
import json
from arch_explorer.agent_env import ExplorationEnv, replay_trace

env = ExplorationEnv(json.load(open("examples/adf-dmc.json")))
env.checkpoint(name="seed")
env.evaluate(optimize="parameters_and_mapping")
env.submit(path="results/best.json")
restored = replay_trace(json.loads(line) for line in env.trace_path.read_text().splitlines())
assert restored.used == env.used
```

Trace replay verifies results and counts call by call, ignoring submit's external file writes. checkpoints exist only within the current service process; after a service restart, trace replay can be used to recover. Different backend versions or random backends may produce trace_mismatch.

## Claude Code: MCP stdio

Install optional dependencies: run `python -m pip install '.[mcp]'` in the repository root. The MCP transport uses the SDK 1.x low-level server; this version has been verified with 1.30.0. When not installed, the command gives clear installation instructions. The mcpServers / command / args format in the project `.mcp.json` is described in the [Claude Code official MCP documentation](https://code.claude.com/docs/en/mcp). Use absolute paths to avoid being affected by the Claude Code startup directory:

```json
{
  "mcpServers": {
    "archexplorer": {
      "type": "stdio",
      "command": "/absolute/path/to/venv/bin/archexplorer",
      "args": ["mcp", "--config", "/absolute/path/to/adf-dmc.json"]
    }
  }
}
```

After the client connects, read the tool list, then call describe_environment. stdout carries only the MCP protocol; logs go to stderr. The same server process is one exploration session; fair comparison between different Agents requires starting a new process for each with the same configuration.

MCP JSON-RPC frames are validated by the SDK. SDK 1.30.0, verified in this version, discards frames with non-finite ids and continues processing the next valid request; it does not generate a JSON-lines `invalid_request` reply for such frames. Non-finite numeric values in tool parameters return `invalid_arguments`, and the session can continue to be used.

## Portal Resident: JSON-lines stdio

Resident can use its own authorized execution tools to start a local command, holding stdin/stdout pipes until the exploration ends. The same process sends one line of JSON and reads one line of reply in sequence; do not restart the service for each call. This local protocol does not require a Portal account or key, and does not change the Resident's identity.

```python
import json
import subprocess

process = subprocess.Popen(
    ["/absolute/path/to/venv/bin/archexplorer", "env", "serve",
     "--config", "/absolute/path/to/adf-dmc.json"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
)

def call(number, tool, args=None):
    process.stdin.write(json.dumps({"id": number, "tool": tool, "args": args or {}}) + "\n")
    process.stdin.flush()
    reply = json.loads(process.stdout.readline())
    assert reply["id"] == number
    return reply

try:
    print(call(1, "describe_environment"))
    print(call(2, "checkpoint", {"name": "seed"}))
    print(call(3, "evaluate", {"optimize": "parameters_and_mapping"}))
    print(call(4, "apply_edits", {"edits": [{"op": "split", "point": "chip/dram", "num": 2}]}))
    print(call(5, "evaluate", {"optimize": "mapping"}))
    print(call(6, "best"))
    print(call(7, "submit", {"path": "/absolute/path/to/results/agent.json"}))
finally:
    process.stdin.close()
    process.wait(timeout=30)
```

Each line must contain `id` and a string `tool`, with optional `args`. id accepts strings, integers, or null; booleans, decimals, objects, or arrays are not accepted. Parsing rejects NaN, Infinity, -Infinity, and floating-point overflow (such as 1e309), returns `invalid_request`, and continues to the next line; when parsing fails or the id is invalid, the response id is null. For valid frames, the id is echoed back unchanged. When a response cannot be encoded into finite JSON, it returns `invalid_response` and continues; in this case the tool may have already executed, so check history first and avoid directly retrying stateful operations.

Unknown tools, incorrect parameters, and invalid edits return error, and the service continues to accept the next line. The result files of `evaluate` and `submit` can be replayed by an independent process using the same component/backend registration entry points.

## Offline scripts and fair comparison

`agent run --policy scripted` makes no network requests. `agent.script` is a list of `{tool,args}` executed one by one; the example configuration demonstrates viewing, checkpoint, baseline evaluation, structural edits, evaluation, restore, best, submit. Without a script, it repeatedly calls parameters_and_mapping until the evaluation budget or step limit is reached. max_steps defaults to 30, max_tool_calls defaults to max_steps; at the end it exports the best candidate found. The final export does not consume evaluation quota and is listed separately in the trace. Script steps include failed calls.

When comparing SA/greedy/random with Agent, keep the seed ADF, workload/weights, backend and model versions, budget and penalty coefficients, middle/inner layer sample counts, and random seed consistent. Use the actual number of backend calls to plot the best objective trajectory, and report invalid/duplicate proposals and early stopping; do not substitute tool call counts for evaluation counts. To compare structural strategies, explicitly provide a structural edit sequence for scripted; the default strategy without a script only performs parameter/mapping tuning on a fixed structure.

## Exploration task template for Agents

```text
Find a low-latency architecture in the current ArchExplorer environment. First call describe_environment and get_state to confirm workloads, weights, evaluation balance, cost/area budgets, and penalty coefficients. Do not modify these experimental conditions. Check the basis, assumptions, omitted, and unsupported fields of get_state.costs; do not treat the reference model as a complete physical design or an actual purchase quote.

First save a seed checkpoint, then evaluate the baseline with parameters_and_mapping. Based on the component catalog and bottleneck summary, propose 1–3 basic edits or one high-level transformation; record the bottleneck each change is intended to improve. A successful edit does not guarantee the backend can construct or execute it; when evaluation fails, read the cause and backend attempts in error.details.trials, and correct or restore based on get_state and the "ADF requirements of the builder and the templates" section in docs/components.md. Do not treat unevaluated results as performance evidence. After structural changes, evaluate with the same middle/inner layer settings, and restore if necessary; note that restore does not refund quota. Count based on backend evaluation calls.

Stop evaluating when the budget is insufficient. Call best and check feasible, unavailable_constraints, per-workload latency, and provenance. Missing metrics must be explicitly stated in the conclusion. Deliver results with submit(selection="best", path=specified path), and replay independently with archexplorer evaluate. The final report should include baseline and best objectives, actual evaluation count, major structural/parameter/mapping changes, budget availability, and model limitations, with paths to results and the JSONL trace. Do not make additional model API requests; use the exploration tools provided by the current host directly.
```

## Conditional mapping space: aggregation and explicit instructions

`describe_environment().result.mapping_spaces` is the template contract ordered by workload. Ordinary templates with `contract.kind=flat` remain a mapping from parameter names to option lists; conditional templates with `contract.kind=conditional` list mutually exclusive `branches`, `direct_defaults` for the direct API, `sampling_defaults` for default sampling selection, and `unavailable` reasons excluded by hardware/load. `sampling_spaces` gives the active parameter product after filtering under the current configuration, and one value can be selected per key to generate a complete mapping. This discovery step only performs input validation and does not run instructions or transaction timing; rediscovery is required after structural/load changes.

`pim_offload.space(adf, workload)` returns a `ConditionalMappingSpace` and no longer flattens mutually exclusive parameters into a single dictionary. Python callers can use `arch_explorer.adf.registry.mapping_spaces(space, selectors)` to obtain the active product; do not use `space.items()` or take the union of keys across branches. The `.space()` return value of ordinary templates is unchanged.

| Branch | Active parameters |
| --- | --- |
| aggregate / aggregate | attention, sub_batches, tp, inner, two execution fields, and the `inner.*` of the **selected** inner template |
| instructions / aggregate | attention=pim, sub_batches=1, legal tp, inner=gsm_tiled, two execution fields; no `inner.*` |
| instructions / transactions | Same as above; additionally must satisfy source transaction hardware and workload conditions |
| aggregate / transactions | Not supported and does not appear in legal branches |

Default sampling only includes aggregate/aggregate. Explicit instruction execution has higher cost and should be actively selected in small scenarios, for example by adding the following fields to the configuration (ADF is a qualifying NeuPIMs, workload is a supported FP16 decode):

```json
{
  "mapping_templates": ["pim_offload"],
  "mapping_selectors": {
    "pim_offload": {"npu_execution": "instructions", "memory_execution": "transactions"}
  },
  "parameter_samples": 1,
  "mapping_samples": 1,
  "max_evaluations": 1,
  "search": {"max_steps": 1, "proposals": []}
}
```

This is an additional configuration fragment; `seed_adf` and `workloads` are still required. Changing memory_execution to aggregate selects the instruction/bandwidth ablation path. `mapping_selectors` fixes options per template; execution fields not provided still use sampling_defaults, and you cannot write only transactions to let the system implicitly enable instructions. Ordinary flat templates can also use selectors to fix options; no matching branch, unknown keys, or inactive keys all raise errors. Sampling draws a finite index based on the total size of the mutually exclusive product and does not materialize large Cartesian products; count, seed, and failure budget semantics are unchanged.

A complete environment/search mapping must include all active keys of the selected branch, including execution fields. `submit` saves the complete mapping as-is, and CLI `evaluate` validates and replays it under the same conditional contract; inactive keys are not discarded. Direct `adf.evaluate` preserves shorthand compatibility: `MappingSpec('pim_offload', {})` is the original aggregate default, and `{'npu_execution':'instructions'}` defaults to memory=aggregate, attention=pim, sub_batches=1, tp=1, inner=gsm_tiled. When aggregate inner parameters are omitted, the corresponding inner template defaults are used. Shorthand is not a complete Agent mapping; when passing explicit mappings to the environment, all keys should be completed from the discovered active space.

Old results that merged the inner parameters of gsm_tiled and dmc_partition, or carried inner.* into explicit instructions, are now explicitly rejected: these configurations that were previously ignored/mutually conflicting cannot masquerade as equivalent replays by silently deleting keys. Keep the original files and re-evaluate from legal branches to generate new results. When hardware capacity/clock, FP16/iteration, head/GQA, return buffer, and KV address range conditions are not met, unavailable records the reason; it is not automatic expansion or switching physical models. Template discovery cannot predict all runtime capacity/scheduling failures; these are still recorded as actual backend attempts and consume budget.
