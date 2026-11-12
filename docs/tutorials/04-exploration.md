# 4. Control exploration, interpret results, and connect agents

## Fix the problem first, then run the search

The accompanying JSON explicitly limits the teaching experiment:

| Level | Search space in this example |
| --- | --- |
| Structure | Keep 4 vector cores, private SRAM, and DDR DRAM; the top-level `chip/noc` is replaced between the original mesh2d and bus |
| Hardware parameters | `parameter_samples=1`, only the current parameters each round; capacity, compute, and bandwidth do not grow randomly |
| Mapping | Only non-GPU `dmc_partition`, at most 4 legal mapping combinations drawn per round |
| Objective | Latency of a single workload; area/dollar constraints are not enabled, penalty coefficients are zero |
| Stopping condition | At most 16 actual backend attempts and 8 structure proposal steps; seed=17 |

`search.proposals` are two atomic delete/create edit groups, both replacing the same path and preserving the remaining resources. Structures that return to a previously seen fingerprint are skipped, so actual attempts are usually fewer than the upper limit. This only demonstrates "observe results after selected architecture changes", not an exhaustive search of the complete architecture space or a fair policy performance experiment. Bus and mesh share differently; the network input values are synthetic assumptions and do not represent area-equivalent physical implementations.

`mapping_samples=4` is not an exhaustive enumeration of all combinations; each candidate comes from the split/weights/activations space declared by the template. `max_evaluations` counts environment-level backend attempts, not all internal simulation counts or wall-clock cost. Failed backend attempts also consume quota; checkpoint/restore does not refund. Changing only the seed does not guarantee that new hardware is faster. The same source code, configuration, and operation sequence facilitate review, but do not represent statistical significance.

## Entering parameter exploration

First generate several fixed-parameter scenarios according to Section 3, so that the impact of capacity or bandwidth can be clearly attributed. Then add `parameter_domains` to the configuration, for example explicitly authorizing private SRAM capacity options:

```json
[{"path":"chip/core/sram","param":"capacity","values":[524288,1048576,2097152],"source":"synthetic tutorial assumption: 0.5/1/2 MiB private SRAM"}]
```

Place this array in the `parameter_domains` field and increase `parameter_samples` to 2 or more. This increases the parameter candidates and backend calls per round; the baseline is included, and there is no guarantee that a different value is drawn each time. **This array is not a whitelist for all parameters**: parameters for which components already have domains can still be sampled, such as DRAM technology. The current implementation modifies one adjustable coordinate per new parameter version; if other parameters must be strictly fixed, use multiple explicit fixed configurations, `parameter_samples=1`, or set explicit parameters through environment tools before evaluation. Do not describe an unrestricted experiment as "only scanning SRAM". For complete rules, see [three-level search](../structure.md).

## Cost and best feasible result

`best.score` includes objective, feasible, and constraint information. The objective is weighted latency plus area/cost over-limit penalties; if the penalty is too small, the submitted `best` may exceed the budget, and the CLI exit code alone cannot be used to judge whether the design satisfies resource constraints. If a strict budget is required, `best.score.feasible` must be checked; when not satisfied, report "no best candidate satisfying the constraints was delivered this time", and re-audit feasible candidates from the retained attempt records instead of concealing the over-limit.

`best.cost_estimate` is an independent cost model; `raw` and `full` have different scopes, and mm², USD, and the old model mm²eq are not interchangeable. The 7nm in this example is a synthetic reference scenario. Not setting a cost threshold by default does not mean resources have no cost. Before enabling budgets, read [cost.md](../cost.md) and confirm that the components and process used have estimation support; unknown remains null, and enabling constraints with missing metrics produces an unusable objective and must not be filled with zeros.

## How to read feedback

First look at each workload's `evaluation.latency_ns`, `breakdown`, `details`, and `diagnostics`. Task cumulative times may overlap and cannot be directly summed as end-to-end latency; resource utilization depends on the backend definition. For small GEMMs, partitioning, memory capacity/path, local staging, and network contention can all determine the result. To verify an explanation, change only one assumption, keep the remaining conditions, and replay; do not tune parameters to the expected speed.

Save the configuration, code commit, seed, actual evaluations_used, complete results, and the JSONL pointed to by `search.trace_path`. The default trace is in a temporary directory and should be copied to your own experiment directory before cleanup; you can also pre-configure a `trace_path` that does not yet exist. Repeated runs with the same configuration require a new trace file name. `search.trajectory` records structure proposals/acceptances/errors; the complete tool trace can audit evaluation returns and budgets. Saving the best result allows replaying the best candidate, but does not replace saving the full trace.

## Portal Resident / external agent

The same configuration can serve as a stateful tool environment:

```sh
python -m arch_explorer env serve --config examples/tutorials/vision-projection.json
```

Send the following JSON line by line to the process stdin; each line receives a JSON response. Here the architecture is fixed and only mapping is optimized:

```jsonl
{"id":1,"tool":"describe_environment","args":{}}
{"id":2,"tool":"get_state","args":{}}
{"id":3,"tool":"evaluate","args":{"optimize":"mapping"}}
{"id":4,"tool":"submit","args":{"selection":"best","path":"results/tutorials/tools.json"}}
```

After sending EOF to end the service:

```sh
python -m arch_explorer evaluate results/tutorials/tools.json --output results/tutorials/tools-replayed.json
```

Read the feedback each round before deciding the next operation. `search.proposals` only limits the built-in search strategy; it is not a permission or hard constraint for external agent tool calls, and external agents must still comply with the agreed edit sets and resource limits. For MCP installation/Claude Code configuration, Portal Resident subprocess examples, and complete tool definitions, see the [Agent documentation](../agent.md). `agent run --policy scripted` is an offline script, not a call to an LLM.

A task template that can be given to an agent:

> Evaluate the synthetic vision projection with the given configuration; first read the environment and state. Fix the workload, precision, core count, and storage; only allow the two chip/noc replacement groups listed in the configuration, fix hardware parameters, and optimize only mapping each round. Total backend attempts must not exceed 16; do not delete necessary communication or storage. Record a brief reason for the choice before each call, read the returned real latency, constraints, errors, and remaining quota before deciding. Save the trace, check feasible, submit best, and independently replay via CLI. Report uncovered operators and model boundaries, and do not claim complete vision model speed or real device accuracy.

After completing these steps, gradually add workloads, parameter domains, and structure changes according to your own problem, and keep a fixed baseline.
