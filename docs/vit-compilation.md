# Complete ViT with public compilation and exploration

The runnable [ViT scene](../examples/vit_compilation/README.md) connects a
complete small classifier to the existing public compiler, search and offline
Agent. Images and all parameters are explicit root data. Workload APIs own the
graph, numerical semantics and stable operator identities; registered mapping
rules own expansion, access and layout. The user scene supplies no second
implementation of patch packing, GELU, rounding or peak-memory calculation.

## Python source and execution

```python
from arch_explorer.archs.vit_compilation import vit_scene
from arch_explorer.workload.api import compile_workload, layout_candidates
from arch_explorer.simulator.api import prepare_compiled, evaluate_request

adf, workload, selected = vit_scene(layers=2, dtype='bf16', layout='views')
choices = layout_candidates(workload, selected)
compiled = compile_workload(adf, workload, choices['views'])
request = prepare_compiled(compiled, output_dir='/absolute/vit/new-python-request',
                           options={'trace': True})
result = evaluate_request(request)
readonly = evaluate_request(result)
```

The last call checks the saved execution. It does not prepare a request or
invoke an engine. Existing deterministic source and CPU reference checks are
allowed during readonly validation. Save complete JSON documents with
`allow_nan=False` and retain referenced artifacts and the matching installed
extensions. Changing a builder, mapping, adapter, input or bound result requires
a new compilation; changing checksums does not repair an incompatible source.

`vit_scene`, `vit_workload` and `vit_inputs` provide data/configuration entry
points in `arch_explorer.archs.vit_compilation`. `vit_search_config` and
`vit_agent_config` return ordinary configuration dictionaries for existing
tools. Explicit user data is copied intact and passed to public validation;
the scene never silently fills missing weights or supplies layer answers.

Query the exact root schema before providing your own values:

```python
from arch_explorer.workload.api import vit_classifier_input_shapes
shapes = vit_classifier_input_shapes(batch=2, image_size=4, patch=2, width=4,
                                     layers=2, heads=2, mlp_hidden=7, classes=3)
print(shapes)
```

Roots are a flat dictionary: `images`, `patch_weight`, `patch_bias`, `cls_weight`,
`position_weight`, `norm_gamma`, `norm_beta`, `head_weight`, `head_bias`, plus
sixteen entries per layer prefixed by `block{i}.`. Each layer has `Wq`, `Wk`,
`Wv`, `Wo`, `bq`, `bk`, `bv`, `bo`, `norm1_gamma`, `norm1_beta`, `norm2_gamma`,
`norm2_beta`, `W1`, `b1`, `W2`, `b2`. All are root data; `images` is the sole
activation input. The shape helper validates dimensions. There is no extra
layer input or externally supplied patch tensor. These names and the complete
values are included in the serialized WorkloadSpec.

## Semantics and storage

Patch tokens follow block row then block column. Each patch follows channel,
patch row, patch column in NCHW data. The CLS token precedes image tokens;
position encoding spans all tokens. Each layer has independent parameters,
attention, both residuals and an MLP. Final normalization precedes CLS selection
and classification. GELU is the exact erf form; normalization epsilon is
1e-12. FP32 accumulation, storage rounding and conversion boundaries follow
the public ViT workload contract. Scaled scores are written to their storage
dtype before a fused stable softmax; softmax writes its result once. These boundaries are separate from the
earlier tanh encoder's semantics, and from an official framework's fused or
mixed-precision kernel choices.

Three storage dtypes are available: FP32, FP16 and BF16. BF16's CPU array
container does not change its two-byte modeled storage. Synthetic hardware
continues to declare FP32 arithmetic service, with conversion work accounted
separately. No low-precision speedup is inferred from tensor size.

Norm and softmax use the declared component-local scratch abstraction. Their
internal intermediates are not additional global allocations. The earlier
dedicated native/full-model path used an explicit global-scratch assumption;
its peaks and timing cannot be transferred to this public function-level model.

The two public layout candidates preserve the same original semantic graph
and data. A supported readonly reshape, transpose or selection can share a
root allocation. Packing, concatenation and unshareable transformations stay
explicit. Useful bytes read by consumers, the span of addresses, logical tensor
size and root allocation are different quantities. Allocation/free events
determine the actual peak; summing all tensor sizes does not.

## Inspect actual decisions

```python
for step in compiled['plan']['steps']:
    binding = compiled['plan']['rules'][step['origin']]['binding']
    print(step['origin'], binding, step['address_map'])

print(compiled['plan']['allocations'])
```

The original operator's entry in `plan.rules` owns its resolved binding;
individual steps do not carry a duplicate `rule` field. Complete allocations
show shared roots, real consumers and returned storage. The address map records
the selected access interpretation. Inspect the actual result's service and
allocation trace separately from the CPU values. Removing a copy can leave the
peak unchanged and can still require a packing task elsewhere.

## Bounded search and recovery

The generated configuration uses seed 0, four explicit mappings (two layouts
times two per-instance tile/placement choices), twelve evaluation units and
three bounded hardware proposals. Capacity failures consume budget and keep
their failure phase. `compilation` selects the public compile → prepare →
evaluate loop in the existing Env; there is no ViT-specific evaluator or
provider. The offline scripted Agent discovers mappings, evaluates one
selected candidate and submits it within one unit.

The winner's saved mapping is the producer's full effective choice, including
all resolved instance rules; the ledger keeps the original requested mapping.
The same compiled-result retains source bindings and evaluation. Use
`archexplorer evaluate best.json` in a fresh process for readonly validation,
and `archexplorer rerun best.json --work-dir <new-directory>` for a separate
execution. Budget, request identity and output directory of rerun are new.

The same operations are available through Python:

```python
from arch_explorer.archs.vit_compilation import vit_search_config, vit_agent_config
from arch_explorer.agent_env.environment import ExplorationEnv
from arch_explorer.agent_env.driver import run_agent
from arch_explorer.exploration import explore, replay, rerun

config = vit_search_config(layers=2, dtype='fp32')
config['trace_path'] = '/absolute/vit/python-search.tools.jsonl'
best = explore(env=ExplorationEnv(config), policy='sa')
checked = replay(best)
new_execution = rerun(best, output_dir='/absolute/vit/new-python-rerun')

agent_config = vit_agent_config(layers=2, dtype='fp32', layout='views')
agent_config['trace_path'] = '/absolute/vit/python-agent.tools.jsonl'
agent_result = run_agent(config=agent_config, policy='scripted')
```

## Earlier paths and limits

The [native ViT/CLIP tutorial](tutorials/05-full-models.md) retains its dedicated
program/provider and external-engine contract. Its saved results retain their
original sources and profile identity. The public ViT example uses an explicit
new workload specification, inputs and mapping; it does not convert old
native results into compiled results or automatically switch a backend.
The [multi-layer encoder](multilayer-encoder.md) keeps tanh GELU and its own
normalization and precision contract.

Synthetic CPU reference, Events and Roofline observations establish different
facts. They do not establish standard-scale capacity, native execution,
official mixed-precision equivalence, calibrated erf/SFU costs or physical
device accuracy. Unknown cost and IPU evidence remain separate incomplete
items. This example does not add CLIP, training, dropout, alias writes,
arbitrary striding or cross-domain execution.
