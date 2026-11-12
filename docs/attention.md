# Attention and bounded compiled exploration

The public example constructs a stable scaled-dot-product attention graph with
explicit FP32 inputs. Its default Q/K/V shapes are `(1,3,4)`, `(1,5,4)` and
`(1,5,3)`. Both matrix products have nondivisible tile boundaries. The key
transpose, row-maximum broadcast and row-sum broadcast are materialized tensors
with real storage and traffic.

The graph is QK-transpose, scale, row-max, broadcast, subtract, exp, row-sum,
broadcast, divide and probability-times-V. The scaled scores are read by max
and subtract; the exponentials are read by sum and divide. Their dependencies
and last-reader lifetime stay in one graph.

```python
from arch_explorer.archs.attention import attention_scene
from arch_explorer.workload.api import compile_workload, execute_reference
from arch_explorer.simulator.api import prepare_compiled, evaluate_request

adf, workload, mapping = attention_scene(input_case="regular")
compiled = compile_workload(adf, workload, mapping)
original_values = execute_reference(workload)
request = prepare_compiled(compiled, output_dir="new-attention", options={"trace": True})
result = evaluate_request(request)
assert evaluate_request(result) == result
```

`input_case="large"` supplies large logits that require stable normalization;
`"constant"` uses zero queries, making each score row constant. These are input
generators, with no embedded expected result or latency. Pass
`hardware=arch_explorer.archs.compilation.roofline_scene` to use the explicit
synthetic Roofline hardware instead of the default events scene.

Numerical reference execution is a CPU check of semantics. Event execution
models component services, memory traffic and dependencies. A numerical
reference comparison does not establish backend numerical execution or device
accuracy. Events express multiply, reduction, exp and reciprocal demands using
the existing sourced synthetic device parameters. Roofline uses an abstract
operation/component-IO boundary; it does not establish hardware SFU throughput.

## Registered mapping rules

`compile_graph_v2` has exactly `compute`, `memory` and `rules`. Rules are selected
by original semantic kind and bind an ID, version and options. For example:

```python
mapping.params["rules"]["matmul"] = {
    "id": "matmul.tiled", "version": "1",
    "options": {"tiles": {"m": 2, "k": 3, "n": 3}},
}
```

Other supported materialized kinds can automatically select their unique rule
when it needs no options. Missing rules, ambiguity, wrong versions and invalid
options reject. `workload.api.operator_mappings()`, CLI `operator-mappings` and
the Agent tool with the same name expose actual callback provenance. The
compiler and search policy contain no operator rule list.

This scene helper supplies contiguous, materialized FP32, rank 1–4 and one
iteration. The [precision/block extension](precision-transformer.md) also
supports FP16/BF16 storage with explicit conversion and FP32 accumulation.
Aliases, strides, implicit broadcasts and unsupported reduction axes reject.
Existing noncompiled paths retain their contracts. Native support is assessed
separately from Events/Roofline; the current native path rejects FP16/BF16 and
requires its own adapter, lowered-input and engine evidence for FP32.

## Existing search, with a compilation pipeline

```python
from arch_explorer.archs.attention import attention_search_config
from arch_explorer.exploration import explore, replay, rerun

config = attention_search_config()
best = explore(config, policy="greedy")
assert replay(best) == best                 # Strictly read-only; no new simulation.
new_run = rerun(best, output_dir="new-run") # Explicit new execution and requests.
```

`compilation: {}` opts an ordinary exploration configuration into public
compile → prepare → evaluate. Optional `bindings` and `implementations` are
forwarded unchanged to the compiler. Omit the legacy `backend` selector in this
mode; components and adapters choose the execution domain. `backend_options`
remain separate provider options for preparation.

The example uses the existing greedy/SA/random search machinery. It supplies
two complete rule-option requests through a `mapping_selectors` request union,
an explicit SRAM parameter domain and parameter proposals through the Agent's
`set_parameters` tool. Each request is checked by the registered template's
formal `request_space`; no rule constraints are copied into search.

The fixed seed is 0, the evaluation budget is 6, the mapping sample bound is 2
and the outer step bound is 3. A bandwidth proposal changes a hardware
parameter. A one-byte SRAM proposal is deliberately infeasible and retains its
diagnostic. The example demonstrates exploration of this bounded candidate
set, with no claim of a global optimum.

Each selected workload/mapping candidate consumes one evaluation unit before
mapping validation, compilation and preparation. Failures keep their phase and
cause and are not refunded. No evaluator cache is enabled. The search policy's
existing duplicate-architecture check avoids new evaluations of those
duplicates. Standalone compile/prepare tools still use zero evaluation units.

Each winner stores the exact effective mapping from its public compiled
result's `request.compilation.source.mapping`, including automatically selected
rules. The attempt ledger retains both `requested_mapping` and
`effective_mapping`; read-only replay compares the saved effective mapping
strictly, and explicit rerun uses that same mapping without guessing rules.

The saved `archexplorer.compilation-search-result.v1` includes each winner's
public `arch-explorer.compiled-result.v3`, actual binding/rule options, source
closure and budget ledger. Python replay, CLI evaluate and Agent
evaluate_request validate this format without preparing, simulating, creating
an environment or charging a new budget. Historical `archexplorer.adf-result.v1`
keeps its earlier reevaluation behavior. `rerun` is a distinct new execution.

The scripted Agent adds `agent` metadata with its policy, tool count, stop
reason and log paths before binding its compiled search result. The reader
validates this metadata and the complete content binding; the log paths are
audit references and are not executed during replay.

```sh
python examples/attention/scene.py --output attention-specs.json --search-config attention-search.json
archexplorer operator-mappings --output mapping-rules.json
archexplorer compile attention-specs.json --output attention-program.json
archexplorer prepare attention-program.json --output new-attention > attention-request.json
archexplorer evaluate attention-request.json --output attention-result.json
archexplorer search --config attention-search.json --policy greedy --output best.json
archexplorer evaluate best.json --output readonly-best.json
archexplorer rerun best.json --work-dir new-winner-run --output rerun-best.json
```

The [external square package](../examples/extensions/operator_mappings/README.md)
adds an unknown semantic operator and mapping after installation of the main
package, then enters these same public APIs and search. The IPU Popnn/GC200
sourced-cost gap remains separately required-incomplete.
