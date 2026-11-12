# Complete CLIP with public compilation and exploration

The [runnable scene](../examples/clip_compilation/README.md) supplies explicit
images, integer token IDs and every model parameter to one complete two-tower
workload. Both similarities are outputs of the same graph: image batch by text
batch, and the transposed text batch by image batch. The example uses unequal
batch sizes so the rectangular relationship is visible.

Workload APIs own construction, validation and numerical semantics. Public
composition metadata supplies stable operator/tensor identities; registered
rules own mapping expansion, indexing, traffic and layout. The user scene
supplies roots and configurations only. Search, CLI and Agent consume the same
public compilation interfaces without identifying the model by name.

## Python entry points and explicit roots

The module `arch_explorer.archs.clip_compilation` exposes `clip_inputs`,
`clip_workload`, `clip_scene`, `clip_search_config` and `clip_agent_config`.
All arguments are keyword-only. Their common dimensions default to
`image_batch=2, text_batch=3, image_size=4, patch=2, width=4, layers=2,
heads=2, mlp=7, context=4, vocab=13, text_width=4, text_heads=2,
text_layers=2, projection=3`. Layers in each tower may independently be 2–4.
`mlp` forwards to the public builder's `mlp_hidden`; text MLP shape comes
directly from its public shape helper.

`clip_inputs` accepts those dimensions and `input_case='regular'` (also
`boundary` or `constant`) and returns the complete flat root dictionary.
`clip_workload` additionally accepts `dtype='fp32'` and `inputs=None`, and
returns the public `WorkloadSpec`. The input pattern controls generated images;
weights are independent finite synthetic values, token IDs include repeats and
maximum ties, and logit scale is explicitly `[0.125]`. Supplied `inputs` are
deep-copied unchanged and receive no generated defaults or integer casts.

```python
from pathlib import Path
from arch_explorer.archs.clip_compilation import clip_inputs, clip_scene
from arch_explorer.workload.api import compile_workload, workload_from_spec
from arch_explorer.simulator.api import prepare_compiled, evaluate_request

roots = clip_inputs()  # Images, integer token IDs and all parameters; no answers.
adf, spec, mapping = clip_scene(inputs=roots, dtype='bf16', layout='views')
graph = workload_from_spec(spec)
print(graph.meta['composition']['instances'])
print(graph.meta['clip']['tensors'])
compiled = compile_workload(adf, spec, mapping)
request = prepare_compiled(compiled, output_dir=Path('/absolute/clip/python-request'),
                           options={'trace': True})
result = evaluate_request(request)
```

`clip_scene` returns `(SpaceMatrix, WorkloadSpec, MappingSpec)`. It adds
`hardware=None, candidate=0, layout='materialized'` to the workload arguments.
For Roofline pass `hardware=arch_explorer.archs.compilation.roofline_scene`.
`clip_search_config` accepts the same arguments plus `compare_layouts=True`
and returns an ordinary existing-search configuration. `clip_agent_config`
accepts the scene arguments and returns an offline scripted Agent configuration.
No wrapper accepts a precomputed tower output, hidden activation or EOT index.

```python
from arch_explorer.archs.clip_compilation import clip_search_config, clip_agent_config
from arch_explorer.agent_env.environment import ExplorationEnv
from arch_explorer.agent_env.driver import run_agent
from arch_explorer.exploration import explore, replay, rerun

config = clip_search_config(dtype='bf16', layout='views')
config['trace_path'] = '/absolute/clip/python-search.tools.jsonl'
best = explore(env=ExplorationEnv(config), policy='sa')
readonly = replay(best)
fresh_execution = rerun(best, output_dir='/absolute/clip/python-rerun')
agent_config = clip_agent_config(dtype='bf16', layout='views')
agent_config['trace_path'] = '/absolute/clip/python-agent.tools.jsonl'
agent_result = run_agent(agent_config, policy='scripted')
```

Use new directories for preparation/rerun and keep complete saved documents
and referenced artifacts. Readonly can run in a separate process using the
same installed source and extensions.

## Model and precision boundaries

CLIP's vision stem has unbiased patch projection, CLS/position inputs and a
pre-norm. Its blocks use QuickGELU with epsilon 1e-5; selecting CLS precedes
post-norm and unbiased feature projection. The text tower uses token lookup,
position input, causal attention, final norm, first-argmax EOT selection and
unbiased projection. Both features receive L2 normalization. The FP32 logit
scale and its exponential follow the public graph's operation and storage
order. Moving the scale across a low-precision matmul changes that contract.

This path retains CLIP's declared FP32/FP16 semantics. BF16 is an explicit new
uniform floating-storage inference contract. It does not establish official
framework mixed precision or legacy BF16 support. FP32 accumulation, RNE
storage boundaries and explicit conversions belong to the workload API;
the scene performs no numerical precomputation. Hardware arithmetic service
remains FP32 for all three storage choices.

`token_ids` and selected EOT indices remain int32 and cost four storage bytes
per element, independently of floating dtype. The public validator rejects
negative/out-of-range, fractional, boolean or string IDs. It does not silently
truncate or reinterpret them. EOT is the first position of the maximum token
ID; repeated maximum IDs do not select the last occurrence.

Causal mask is int32 with 0 meaning keep and 1 meaning blocked. The public
apply-mask operation stores a finite payload; masked softmax consumes that
payload together with the mask and decodes blocked positions locally. External
NaN/inf roots and all-blocked rows remain invalid; finite JSON stays strict.
L2 preserves its declared no-epsilon behavior: invalid zero norms, overflow and
nonfinite values are refused by the owning public operation. The user scene
does not repair them by inserting an epsilon.

Function-level norm/softmax scratch is a component-local abstraction. The old
dedicated model's explicit global scratch assumptions have a different timing
and capacity scope. Old peaks and latency do not transfer merely because both
graphs are called CLIP. QuickGELU/SFU service and conversion costs are synthetic,
not calibrated device instructions.

## Placement, layouts and actual storage

The two placement/tile configurations consume stable composition membership,
not tower/layer-name parsing. They cover every original operator and retain
per-instance nondivisible matmul bindings. `layout_candidates` supplies the
complete materialized/views choices for the same original graph. Unsupported
sharing remains explicit materialization; no hidden packing is inserted by the
scene and no strides or service formula is copied into it.

The explicit matmul tile proposals are `(m,k,n)=(8,8,8)` and `(6,7,10)`,
alternating across public composition membership; candidate 1 swaps the choices
and compute points. Larger proposals keep the teaching graph's expansion
bounded. The generic helper retains its earlier `(2,3,5)/(3,5,3)` defaults for
existing consumers. Registered rules retain all validation and expansion limits;
arbitrary user dimensions or very fine tiling may exceed them.

Inspect the authority for each lowered step:

```python
for step in compiled['plan']['steps']:
    binding = compiled['plan']['rules'][step['origin']]['binding']
    print(step['origin'], binding, step['address_map'])
print(compiled['plan']['allocations'])
```

A rule belongs to the original operator. Steps do not carry a duplicate inline
`rule` field. Address records describe actual logical reads and writes; root
allocations describe storage ownership. Lookup and indexed selection count
real indices and selected elements. Repeated tokens still produce repeated
reads. Capacity retains the full embedding weight, independently of which
rows are read.

In views mode, the two output similarities can share a root, which remains
available until both outputs can return. Materialized mode pays for its real
transpose copy. Graph dependencies connect both towers to similarity work;
resource contention can serialize tasks without a fabricated global barrier.
Actual allocation/free events determine peak memory. Removing copies can leave
the peak unchanged or increase root lifetime; summing all tensor sizes does
not establish a runtime peak.

## Saved exploration and recovery

The generated ordinary search configuration uses a fixed seed, explicit
per-instance/layout candidates and capacity/bandwidth proposals. Each attempted
candidate is charged, including compile/prepare failures. The one-byte capacity
proposal intentionally fails. The original Env performs public compile →
prepare → evaluate and retains the producer's full effective mapping.

The ordinary config sets seed 0, four complete mapping selectors, one parameter
sample and a twelve-unit budget. With `compare_layouts=False` it uses two
selectors and six units. The initial configuration and two existing-tool
proposals set capacity to 1 B and bandwidth to 32 B/ns; declared teaching
domains are capacity `[1,65536]` B and bandwidth `[16,32]` B/ns. Offline Agent
uses one unit and exactly discovery, evaluate and submit tool calls.

The saved winner contains the exact evaluated ADF, workload, rules, layouts,
root values and source-bound compiled result. The ledger also keeps requested
mappings and failure phases. Follow the runnable scene's compile, prepare,
evaluate, search and offline Agent commands. Use fresh paths and preserve
referenced artifacts. A separate process can call `archexplorer evaluate
best.json` read-only; preparation and engine execution are forbidden there,
while deterministic source/reference checks remain allowed. `archexplorer
rerun best.json --work-dir <new-directory>` requests a separate execution with
a new request, directory and budget.

Changing root data, token IDs, selected rules, layout, outputs or sources requires
a fresh compilation/execution. Recomputing a checksum does not migrate a
historical source identity. Matching semantic products can be associated through
explicit source/payload equivalence without relabeling old execution evidence.

## Existing paths and limits

The [native ViT/CLIP tutorial](tutorials/05-full-models.md) retains its original
program-provider, external-engine and result identities. The [public ViT](vit-compilation.md)
retains erf GELU and epsilon 1e-12. The earlier [encoder](multilayer-encoder.md)
retains tanh GELU. Their old timing or native results are separate evidence.

This small synthetic CLIP does not include preprocessing, a tokenizer, weights
download, training/dropout, standard-scale runs, arbitrary striding/alias writes,
cross-domain execution, native/SDK work or GPU capture. Numerical checks,
synthetic component timing and hardware accuracy establish separate facts.
The example makes no calibrated hardware prediction; IPU cost evidence and
official mixed-precision/device equivalence remain separate incomplete items.
