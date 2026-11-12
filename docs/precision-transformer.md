# Storage precision and an explicit encoder block

The public compiler supports contiguous, materialized FP32, FP16 and BF16
graphs through registered semantics, mapping rules and component-bound task
adapters. The [block example](../examples/transformer_block/README.md) supplies
synthetic data and standard configurations for those same APIs. It represents
one complete pre-LN encoder block with evaluation semantics: noncausal,
without dropout, KV cache or tensor-parallel communication. It supplies no
pretrained-model accuracy or full ViT/CLIP migration result.

## One precision contract

`workload.api.round_storage(value, dtype)` owns numerical storage conversion.
Every external input and materialized arithmetic output uses this same
conversion. External values first become FP32, then round to the declared
storage dtype with round-to-nearest, ties-to-even. Input/output overflow to a
nonfinite value rejects. Subnormal values and signed zero are preserved.

| Tensor storage | Reference array | Charged element bytes |
| --- | --- | --- |
| FP32 | FP32 | 4 |
| FP16 | IEEE half | 2 |
| BF16 | BF16-rounded values carried in an FP32 host array | 2 |

BF16 reuses the existing bit-level RNE implementation. Its canonical digest
contains little-endian BF16 16-bit words. The host array's four-byte elements
do not change the declared two-byte storage or prove device execution.

Arithmetic and reductions accumulate in FP32. Matrix reference evaluation
follows the existing emitted FFMA contract in ascending K order: form the
product plus the FP32 accumulator in FP64 and round once to FP32. Splitting K
into mapping regions does not introduce a low-precision partial-sum boundary.
The selected output storage dtype is applied at the materialized output.
Matmul inputs have the same dtype; an explicit `output_dtype` can select the
output, otherwise it retains the input dtype. Mixed inputs require explicit
cast nodes.

Transpose, broadcast, reshape and same-dtype cast materialize real copies.
They allocate output storage, charge actual read/write bytes and participate
in last-reader release. A reshape is not an alias or stride-only view. A
different-dtype cast promotes through FP32 and rounds at its output. Graphs
with unsupported aliases, noncontiguous strides or implicit broadcasts reject.
The current materialized rules support rank 1–4, one iteration, and the
declared last-axis reduction/keepdims contract.

The default teaching hardware retains `precision=fp32` for both `sm.events`
and `sm.roofline`. FP16/BF16 tensor storage uses that declared FP32 arithmetic
service, plus explicit conversion demand. It receives no assumed low-precision
speedup. An actual low-precision scalar operand load incurs promotion demand;
a low-precision arithmetic output incurs rounding demand. Copy-only operations
need neither; FP16↔BF16 cast incurs both. The component-owned work form carries
the conversion instruction count. Events charges it on the existing SIMT
service; Roofline counts abstract operations within its component-IO boundary.
These are sourced synthetic costs, not measured device instruction costs.

## Complete public graph construction

The public workload API defines the exact input shapes and constructs the
block. It has no implicit weights or fallback data:

```python
transformer_block_input_shapes(*, batch=2, sequence=3, hidden=8,
                               heads=2, mlp_hidden=16)
transformer_encoder_block(*, inputs, batch=2, sequence=3, hidden=8,
                          heads=2, mlp_hidden=16, dtype="fp32",
                          epsilon=1e-5, name="pre_ln_encoder")
```

The first function returns a dictionary of shapes; the second returns
`WorkloadSpec(kind="transformer_block", params=...)` with its complete data.
`workload_from_spec` constructs the ordinary DAG and records its actual
builder, semantics and mapping callbacks in the compilation's sources.
Heads must divide hidden width. The default batch and sequence distinguish
multiple batches and leave masked boundary regions with tiles `{m:2,k:3,n:5}`.

The input dictionary must contain exactly these 17 tensors:

| Keys | Shape |
| --- | --- |
| `X` | `(batch, sequence, hidden)` |
| `Wq`, `Wk`, `Wv`, `Wo` | `(hidden, hidden)` |
| `bq`, `bk`, `bv`, `bo` | `(hidden,)` |
| `norm1_gamma`, `norm1_beta`, `norm2_gamma`, `norm2_beta` | `(hidden,)` |
| `W1`, `b1` | `(hidden, mlp_hidden)`, `(mlp_hidden,)` |
| `W2`, `b2` | `(mlp_hidden, hidden)`, `(hidden,)` |

The graph contains both normalization stages, Q/K/V projections with biases,
materialized head split/transposes, QK, scale and stable max/subtract/exp/sum/div
softmax, AV, head merge, output projection/bias, the first residual, two MLP
projections/biases with GELU and the final residual. Its output is `Y` with
shape `(batch, sequence, hidden)`.

QK first rounds to its selected tensor storage. A visible cast then promotes
the scores to FP32 for scale and the complete stable softmax. Another cast
returns probabilities to the block storage dtype before AV. Residuals and
other materialized arithmetic outputs retain their declared rounding
boundaries; the block is not silently evaluated entirely as FP32.

`affine_norm` normalizes the last dimension with population variance and
positive finite FP32 epsilon, then applies its declared gamma/beta values.
Mean, variance and intermediate arithmetic use FP32. `gelu_tanh` uses

```text
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
```

with FP32 intermediate operations and the declared output-storage boundary.
Its synthetic cost is eight SIMT operations and one tanh SFU per element;
normalization uses seven SIMT operations per element plus three SIMT and one
rsqrt SFU per row. The adapters reuse these existing numerical definitions
and sourced cost declarations. The consumer does not implement them again.

## Scene, API, CLI, Agent and search

`archs.transformer.block_inputs` supplies deterministic values only. `regular`
uses signed small fractions and nontrivial weights, biases and norm parameters;
`boundary` adds ties and nearby representable inputs; `constant` uses constant
activation rows. It obtains shapes from the public API and performs no
rounding or block computation. `block_workload` calls the public builder;
`block_scene` returns standard ADF, WorkloadSpec and MappingSpec objects.

```python
from pathlib import Path
from arch_explorer.archs.transformer import block_scene
from arch_explorer.workload.api import compile_workload, execute_reference
from arch_explorer.simulator.api import prepare_compiled, assess_request, evaluate_request

adf, workload, mapping = block_scene(dtype="bf16")
cpu_reference = execute_reference(workload)  # CPU values, including Y
compilation = compile_workload(adf, workload, mapping)
request = prepare_compiled(compilation, output_dir=Path("new-block-request").resolve(),
                           options={"trace": True})
if not assess_request(request)["supported"]:
    raise ValueError("The selected component contract is unsupported")
result = evaluate_request(request)         # explicit component simulation
assert evaluate_request(result) == result # source-bound, read-only validation
```

The [example commands](../examples/transformer_block/README.md) generate full
source inputs, a search configuration and an Agent configuration. Their
`--inputs` option accepts user data with the same exact public shape contract.
Compilation and preparation alone run neither simulation nor search.

`block_search_config` calls the existing search client with six evaluation
units, seed 0, two mappings and explicit SRAM capacity/bandwidth proposals.
Registered request-space and mapping callbacks validate every request. Compile
and prepare failures retain their actual phase, reason and consumed unit.
There is no block-specific search policy or operator branch. `block_agent_config`
discovers rules, executes one selected mapping and submits through ordinary
Agent tools.

The saved compiled-search winner contains each full public compiled result,
the producer's effective mapping, rule options/sources, provider options and
the complete budget ledger. The original requested mapping stays in the
ledger. Python replay, CLI `evaluate` and Agent `evaluate_request` validate
saved results without new simulation, writes or budget. Scripted Agent
metadata is included in the content binding. Explicit `rerun` uses the saved
effective mapping in a new request/output directory with a new execution
budget. Old `adf-result.v1` retains its historical reevaluation behavior.

## Coverage and evidence boundaries

The unknown [square mapping package](../examples/extensions/operator_mappings/README.md)
retains its explicit FP32-only contract. It expands through a public multiply
rule, with two reads when both input names refer to the same tensor. New
built-in precision support does not expand its declaration or replace its
registered semantics.

CPU original/mapped graph values, Events/Roofline task execution, native
instruction interpretation/execution and hardware measurements have separate
evidence. Native FP16/BF16 is rejected by the current 32-bit load/store and
arithmetic contract. Native FP32 capability depends on its selected adapter,
actual lowered inputs and matching engine assessment. The example above
establishes no hardware performance accuracy or full-model native coverage.
Unsupported KV, aliases/strides and TP communication remain explicit gaps;
the independent IPU Popnn/GC200 cost evidence remains incomplete.

Keep the exact exported source, package/environment identity and original
success/failure receipts when preparing a local distribution. The [release
guide](releasing.md) separates declared dependencies from exercised coverage,
and describes source export → sdist → wheel → outside installation. No remote
publication or package version change is implied by these local checks.

The [multi-layer encoder scene](multilayer-encoder.md) reuses this block and its
precision boundaries through public subgraph composition. It supplies one X
and distinct parameter sets, binds each output to the next input and selects
mapping rules by stable operator instance. The single-block API remains
available with its original complete-input contract.
