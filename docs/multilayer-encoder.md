# Composed encoder workloads and per-instance exploration

A multi-layer workload is one graph whose edges connect repeated subgraph
instances. The synthetic encoder scene supplies a single `X` and separate
parameters for each pre-LN block. A later block reads the actual tensor produced
by its predecessor. The workload builder owns the full composition; Python
does not run independent blocks and concatenate their answers.

The scene supports two through four blocks, with two- and four-block examples.
Storage is FP32, FP16 or BF16. Accumulation, materialized rounding boundaries,
normalization and GELU follow the [single-block precision contract](precision-transformer.md).
The Events and Roofline hardware instances still declare FP32 arithmetic
service for all three storage types and account for explicit conversions.
Storage dtype does not imply a hardware acceleration multiplier.

The same source graph can use explicit materialization or supported readonly
views. Views preserve their source's stored bits and add no new arithmetic or
rounding boundary. The public layout resolver and mapping registry own this
choice; this example supplies data and asks for complete public candidates.

## Explicit data and standard client objects

```python
from arch_explorer.archs.multilayer import (
    encoder_inputs, encoder_scene, encoder_search_config, encoder_agent_config,
)
from arch_explorer.archs.compilation import roofline_scene

values = encoder_inputs(layers=4, input_case="boundary")
adf, workload, mapping = encoder_scene(layers=4, dtype="bf16", inputs=values,
                                       layout="materialized")
view_adf, same_graph, view_mapping = encoder_scene(
    layers=4, dtype="bf16", inputs=values, layout="views",
)
other_adf, same_workload, other_mapping = encoder_scene(
    layers=4, dtype="bf16", inputs=values, candidate=1,
)
roofline_adf, _, roofline_mapping = encoder_scene(
    layers=4, dtype="bf16", inputs=values, hardware=roofline_scene,
)
```

`values` contains `X` and a `layers` list. Each entry has Wq/Wk/Wv/Wo,
bq/bk/bv/bo, norm1_gamma/norm1_beta/norm2_gamma/norm2_beta, and W1/b1/W2/b2.
Every layer has its own finite, nontrivial values and no activation input.
The public workload API checks the exact shapes and supported layouts.
The scene stores actual numbers, not an implicit random generator or weights
download. Regular, precision-boundary and constant-row input modes are
available; all have different parameters across layers.

The returned ADF, WorkloadSpec and MappingSpec go through the existing public
compiler and simulator interfaces:

```python
from arch_explorer.workload.api import compile_workload
from arch_explorer.simulator.api import prepare_compiled, evaluate_request

compilation = compile_workload(adf, workload, mapping)
request = prepare_compiled(compilation, output_dir="new-model-request", options={"trace": True})
saved = evaluate_request(request)
assert evaluate_request(saved) == saved
```

Compilation and preparation validate the complete composition, placement,
rules and storage constraints without executing a simulator. A saved result
validates read-only. CPU reference values are a separate numerical check;
synthetic component timing is not a device numerical output.

## Compare views with materialization

`layout="materialized"` is the scene and Agent default. `layout="views"` asks
the producer for readonly transpose, slice and shareable reshape rules.
`layout_candidates` returns both policies for the same workload and base
placement; it retains non-layout rules, nondivisible tiles and compute/memory
bindings. A reshape that cannot preserve logical traversal while sharing its
root remains explicitly bound to `reshape.materialized`. The example does not
guess strides, select rules by layer name or silently copy a noncontiguous
array.

```python
from arch_explorer.workload.api import layout_candidates

choices = layout_candidates(workload, mapping)
for policy, selected in choices.items():
    compiled = compile_workload(adf, workload, selected)
    # Public producer inspection: one allocation per actual root storage.
    allocations = compiled["plan"]["allocations"]
    for allocation in allocations:
        print(policy, allocation["storage"], allocation["bytes"],
              allocation["aliases"], allocation["returned"])
    for step in compiled["plan"]["steps"]:
        binding = compiled["plan"]["rules"][step["origin"]]["binding"]
        print(policy, step["origin"], binding, step["access"], step["address_map"])
    request = prepare_compiled(compiled, output_dir="new-" + policy,
                               options={"trace": True})
    result = evaluate_request(request)
    assert evaluate_request(result) == result
```

This comparison keeps the numerical inputs, all layer parameters, hardware and
base mapping fixed. Each policy has a separate new request directory and
source-bound saved result. CPU original/mapped reference checks establish
values separately from the actual component timing and allocation trace.

Inspection distinguishes a logical tensor's `logical_bytes`, its normalized
`layout`, and the whole root's `allocation_bytes`. Strides and offsets are in
storage elements, including for BF16's two-byte storage. Alias roots allocate
once. A small exported slice retains its full root until return; static weight
views keep the existing static root. Neither the sum of logical tensors nor a
view's bounding interval is an observed physical peak.

The built-in Events/Roofline adapters directly read supported positive-stride
layouts; no hidden packing task is assumed. `storage.view` has zero arithmetic,
copy traffic and fresh allocation. Its real consumers still pay for every
logical read, including overlapping views, repeated operands and two distinct
readers. Holes are not useful bytes. Materialization instead reads its source
and writes an independent result. If another implementation needs packing,
it must expose that task, storage, dependency and service explicitly.

Views can remove copies and allocations while retaining a larger root longer.
They need not improve peak usage or latency. Compare the producer's task and
access inspection with the execution's actual allocation/free trace. A
preparation bound is separate from the overlap peak observed at runtime;
capacity failures must retain that phase distinction.

## Bounded search, saved winners and offline Agent

```python
from arch_explorer.exploration import explore, replay, rerun
from arch_explorer.agent_env.driver import run_agent

config = encoder_search_config(layers=2, dtype="fp16", layout="materialized")
config["trace_path"] = "/absolute/run/search.tools.jsonl"
best = explore(config, policy="sa")
assert replay(best) == best
new_execution = rerun(best, output_dir="/absolute/run/new-rerun")

agent_config = encoder_agent_config(layers=2, dtype="fp16", layout="views")
agent_config["trace_path"] = "/absolute/run/agent.tools.jsonl"
agent_result = run_agent(agent_config)
assert replay(agent_result) == agent_result
```

The example offers four complete candidates: both public layout policies for
each of two tile/placement configurations. Search uses the existing public mapping space and
validation callbacks, parameter proposal mechanism, objective and failure-aware
budget. Frontends add no encoder- or layer-name dispatch. Its deterministic
configuration has seed 0 and twelve evaluation units, including a deliberately
impossible one-byte SRAM proposal. Failed preparation does not execute a
simulator or receive a replacement evaluation unit.

`compare_layouts=False` limits the configuration to the selected `layout` and
the original two tile/placement candidates, with six evaluation units. The
`layout` and `candidate` arguments choose the first request; offline Agent
uses that exact request and one evaluation unit. Layout choices are ordinary
registered mappings. Generic search, CLI and Agent code contains no model or
layout-policy dispatch.

The winner stores each workload's public compiled result and complete effective
mapping. Attempt rows retain requested mappings, compile/prepare/evaluation
phases, failures and their budget indices. Replaying a winner checks those
bindings and its saved result; explicit `rerun` alone starts a fresh metered
execution in a new directory. Both keep the exact instance-specific choices.

`compile_graph_v3` has `compute`, `memory`, `rules` and `operator_rules` fields.
An exact instance override takes precedence over a kind default, then the
current unique rule. The scene reads public `operator_names` membership and
actual operator kinds; it does not parse layer names. Use public
`qualified_name(instance, local)` when naming a specific child operator. The
effective mapping saves every resolved instance binding, including its options.

The existing `compile_graph_v2` retains its kind-level meaning and accepts no
`operator_rules`. Public `migrate_mapping_v2` explicitly creates a v3 mapping;
it does not execute or relabel old results. Source changes still invalidate old
results. New execution requires a new compilation and request.

The template's `request_space` is the authority used by existing
`mapping_candidates` and `check_mapping` helpers. Direct registry clients first
import `arch_explorer.mapping`, then use
`MAPPING_TEMPLATES.get("compile_graph_v3").request_space(adf, workload, mapping.params)`.
The example supplies a union of two complete concrete requests.

The [source/configuration command](../examples/multilayer_encoder/README.md)
uses the same functions and the same CLI compile/prepare/evaluate/search/Agent
commands. Its input, search and Agent output files require distinct new paths.

## Materialization and supported scope

Cross-instance bindings reuse the actual tensor; they do not insert a hidden
activation copy. Ordinary materialized operations, real weights, conversions
and temporary tensors still occupy storage. Only declared graph outputs are
kept as outputs. Inspecting an intermediate value in the CPU reference is
separate from retaining every layer output in the execution graph.

The executed graph determines last readers and release times. A fork must have
two real consumers before its last-reader behavior can be tested. Capacity
checks use the mapped allocation contract, while execution records its active
peak; the sum of every tensor's size is not an observed peak. Any conservative
preparation bound must be described as a bound with its cause.

The public composition contract also applies to registered non-Transformer
builders and explicit primitive graphs. Reusing a fragment requires isolated
tensor/operator identities, checked ports and immutable source inputs. Invalid
ports, duplicate instances, conflicting sources, incompatible dtype/shape,
cycles, unsupported layouts and source changes are rejected by the production
owner. A user scene does not repair them or reinterpret their serialized form.

For example, this public primitive fragment can be repeated before and after a
fork. Only the two declared exports are retained as outputs:

```python
from arch_explorer.workload.api import Workload, compose_workloads, workload_from_spec

fragment = Workload("scale_fragment", dtype="fp32")
fragment.tensor("X", (1, 4))
fragment.tensor("Y", (1, 4))
fragment.add("scale", "scale", ["X"], ["Y"], shape=[1, 4], factor=2.0)
instances = [
    {"id": name, "workload": fragment, "inputs": {"x": "X"}, "outputs": {"y": "Y"}}
    for name in ("shared", "left", "right")
]
composed = compose_workloads(
    instances=instances,
    inputs={"X": {"shape": [1, 4], "dtype": "fp32", "kind": "activation",
                  "value": [[1.0, -2.0, 0.5, 0.25]]}},
    bindings=[
        {"to": {"instance": "shared", "port": "x"}, "from": {"input": "X"}},
        {"to": {"instance": "left", "port": "x"},
         "from": {"instance": "shared", "port": "y"}},
        {"to": {"instance": "right", "port": "x"},
         "from": {"instance": "shared", "port": "y"}},
    ],
    outputs={name: {"instance": name, "port": "y"} for name in ("left", "right")},
)
graph = workload_from_spec(composed)
```

Each child declares exact local ports. Root inputs have shape, dtype, kind and
explicit values; a structural child can leave values absent. Every input is
bound once. The compiler derives the dependency order and both real readers of
the shared result. Root exports name the actual tensors without a copy. Nested
builder identities, composition specifications and bindings enter saved provenance.

Readonly tensor aliases use the public root-storage contract. Positive-step,
nonempty slices, valid axis permutations and shareable reshapes preserve dtype
and bits. Negative/zero steps, arbitrary `as_strided`, zero-stride broadcast,
dtype reinterpretation and alias writes are outside the contract. Conflicting
memory placements of a root are rejected rather than silently copied. Saved
layout, nested-builder, rule, adapter and resolver sources remain checked by
read-only validation. Existing graph.v1 has an explicit strict reader; new
graph.v2 serializes element offsets. Reading a new schema is never a new run.

Existing single-block and unknown FP32 extension contracts remain separate.
An adapter declares its supported layouts; an older independent-only adapter
must reject aliases before lowering, or consume an explicitly materialized
configuration. This example adds no strided capability to an external package.
No new native execution, SDK build, KV cache, dropout,
cross-domain bridge, hardware calibration, IPU cost evidence or external
release is provided by this example.
