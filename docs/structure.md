# ADF structure editing and three-layer search

`arch_explorer.structure` performs structural editing on the shared `SpaceMatrix` / `SpacePoint` tree.
Calls return a new tree, and the original tree and all its child objects remain unchanged. The ADF contract itself is a mutable dataclass, and callers should not
modify trees that have already been handed to other code as checkpoints or candidates. Each editing step is checked through the component registry, parameter domain, and
network roles; illegal edits raise `StructureError`, whose `to_dict()` contains code, message, and details.
Component evaluators, mapping templates, and event backends are registered by their respective modules.

```python
from arch_explorer.archs.adf_seeds import dmc_adf
from arch_explorer.structure import split, replay

seed = dmc_adf(4)
result = split(seed, "chip/dram", 4)
assert replay(seed, [result.record]) == result.adf
```

The corresponding components must be registered before running; seeds only describe hardware and do not provide simulation models. `dmc_adf(num_cores=4)` contains
vector cores, local SRAM, mesh, and DRAM, with parameters sourced from `synthetic seed`, and is used for non-GPU teaching scenarios.
The GPU uses the ADF from the [native program demo](tutorials/05-full-models.md), not the throughput formula seed.
The CGRA named seeds `vectorcgra_4x4` / `vectorcgra_4x4_diag` are used directly via `devices.get`,
preserving their RTL sources, scenario parameters, and unknown clock markers, without duplicating device constants.

## Six basic primitives

Paths are type-level paths, such as `chip/core/sram`. All instances of the same repeat element share parameters.

| Primitive | Parameters and effects |
| --- | --- |
| `split` | `point, num, algo="even"`. Calls the component `split`; the original point becomes a container with the same name, containing num split points. The parent container inherits the original point's repeat. |
| `merge` | `points, algo="sum", name=None`. Calls the component `merge`, merging points at the same level, of the same type, and with the same repeat. The wrapping layer can be removed again with `remove`. |
| `insert` | `parent, children, name="group", group_size=None`. Places direct children into a new container; networks cannot be children. |
| `remove` | `level`. Children and networks are promoted to the parent level, and each child node's repeat is multiplied by the removed layer's repeat; an error is reported on name conflicts. |
| `create` | `container, spec` or `container, component`, choose one. spec is a complete ADF node JSON; component is an existing path, indicating a move. Networks automatically enter networks. |
| `delete` | `component`. Deletes the point or entire subtree at that path; deleting the root is not allowed. |

`split` is only allowed for compute/memory points; communication points cannot become containers in networks.
Extensive parameters are divided/added according to the component implementation, int resources must divide evenly, and intensive parameters such as clock remain unchanged.
Source descriptions are maintained by the component's split/merge implementation. The root node is preserved; merge/split cannot cross sibling sets.

For grouping repeats use `insert(..., children=["chip/core"], name="chiplet", group_size=4)`.
When core.repeat=16, the result is chiplet.repeat=4 and core.repeat=4. This representation is equivalent to grouping every four cores
into one group, and does not instantiate 16 copies of repeated JSON. group_size must divide repeat evenly. When different groups need to be edited separately,
explicit nodes with different names can be created with create, and then the original repeated node can be deleted. Moving components across different parent physical multipliers
is rejected to avoid silently duplicating or discarding hardware.

## High-level programs

High-level programs return `ProgramResult(adf, records)`, where records consist entirely of basic edits.
`replay(seed, records)` can be expanded independently without executing the high-level program again.

| Program | Structural effect |
| --- | --- |
| `chiplet_transform(chip, cores_per_chiplet, noc_spec, nop_spec)` | Deletes the chip's original network, inserts chiplets by group, creates NoC within groups and NoP between groups. Supports one repeated core or multiple explicit cores. |
| `gsm_to_dmc(chip, shared_memory, parts)` | Splits shared memory, expands the repeat of directly compute-bearing cores, and inserts a tile containing a private memory slice for each core; adds a local bus and global mesh. parts must equal the number of cores. |
| `pim_transform(memory, level, nums, spec)` | Recursively splits memory, inserts at the specified levels, and places spec.compute and spec.network at the lowest level. level/nums can be single values or equal-length lists. |
| `bond_memory(core, memory_spec)` | Binds one dram with technology=`bonded` to each core; if core is a point, it is first wrapped with group_size=1. |

NoC/NoP/bus specs must point to network roles. The default bus and mesh parameters for GSM→DMC are marked as
synthetic seed, and can be explicitly replaced via `bus_spec` and `noc_spec`. `pim_unit` is registered, and legal specs can be used for
`pim_transform`, with supported GEMV executed by `pim_offload`; specific parameter errors are reported when required parameters are missing.
Genuinely unregistered components still return `unsupported_component`. `pim_transform` constructs in a private copy in batches,
validates the entire tree at the end, and preserves individual split/insert/create records for independent replay; failure does not modify the input. The equivalence between hand-written basic edits for the three paper transformation
examples and the normalized high-level programs is shown in `tests/core/test_structure.py`.

## Canonical form

`normalize` processes from leaves toward the root, keeping the root unchanged; when a child container satisfies repeat=1, has no network, has only one
element, and promoting that element would not cause a name conflict, it is expanded. All sibling elements and networks are sorted by name.
Layers with networks, repeats, multiple children, empty containers, and those that would cause name conflicts are preserved; parameter values, source,
and element names are preserved. Normalization is idempotent and does not promise to detect arbitrary graph isomorphism or aliased architectures.

`normalized_fingerprint` uses the shared contract's SHA-256 fingerprint. This fingerprint includes parameters and
provenance, so it is a conservative complete candidate deduplication; trees with different parameters or sources will not be incorrectly merged.
The environment preserves user-edited paths for reference in the next step, and search deduplicates canonical forms before evaluation.

## Three-layer strategy

The SA, greedy, and random modes of `arch_explorer.search.run_search` all call the public utilities of
`ExplorationEnv`. By default the outer layer only generates basic edits from the component catalog and the current graph: split/merge,
insert/remove, create/delete, cross-container moves, and atomic delete/create replacements for categorical parameters.
Duplication can be used for entire subtrees; components that are registered but not present in the seed can also enter candidates. All containers use the same rules,
and names, preset families, stages, and quotas do not control the reachable range. High-level transformations can still be called explicitly, but do not enter the default set.

New components use parameter values with the same name and unit from the current tree or representative values from the declared domain, marked as synthetic source.
Required fixed parameters that cannot be determined from the catalog or existing parameters are not fabricated; in this case the user provides a complete spec.
Representative values are only a finite one-step proposal, not an exhaustive enumeration of all legal graphs, and do not guarantee executability.
`search.proposals` is a user-explicitly-limited set of edits/edit groups; `--basic-only` filters out high-level transformations among them.

Tree/component validation failures do not invoke the backend; actual backend attempts (including failures) count toward the evaluation budget.
CGRA uses the same candidate mechanism as other components, and also allows explicit parameter domains and set_parameters.
The CGRA backend still requires the supported 4×4 scalar uint32 model, 16 PEs, two SRAM banks, capacity, and related
clock/bandwidth relationships; invalid candidates return the real backend reason, and do not represent newly added physical support.
The default unknown clock is a cycle-normalized unit, and clock adjustment results cannot be interpreted as measurement precision.
See [CGRA](cgra.md) for details. checkpoint/restore does not refund consumed quota.

The middle layer contains the current parameter baseline and resamples up to `parameter_samples` parameter versions. Each version modifies one
coordinate of a declared domain: numeric intervals are sampled uniformly within the current value ±`parameter_radius * abs(value)`,
with int allowing at least one step; when the current value is 0, the domain width determines the window. Discrete numeric lists select the candidate closest to the current value
and adjacent items; string lists are sampled uniformly; domain=None remains fixed. All results are validated again.
Each middle-layer round first estimates the current baseline ADF with `arch_explorer.cost.estimate`. As long as any known and enabled
cost/area constraint is exceeded, with probability `shrink_probability` (default 0.75) it preferentially selects capacity, lane count,
array rows/columns, tensor/SIMT/PIM compute unit scale, and samples only at the current value and smaller values.
These parameters affect the current cost formula; clock and simulated bandwidth have not been area-calibrated and are not included in shrink priorities.
Integer boundaries and discrete options may produce unchanged candidates; the bias is not a guarantee of reducing cost every time.
If all enabled metrics are available, none exceed limits, and at least one item still has headroom, with probability `growth_probability` it preferentially selects
capacity, compute, or bandwidth parameters and samples only upward. If some item exceeds its limit, shrink takes priority over headroom in other constraints; unknown metrics
produce no directional judgment. When no bias is triggered, symmetric local exploration still occurs. Each parameter version starts from the same baseline,
and the direction does not change with the previous random sample; the same configuration, random seed, and tool sequence are reproducible.
`parameter_radius`, `growth_probability`, `shrink_probability`, and `cost_model` are explicitly listed in the synthesis configuration.
Existing domain=None defaults remain fixed. Callers can explicitly list positive finite numeric options and source per path in `parameter_domains`
(the seed value must be included), for example `{path:"chip/core/sram", param:"capacity",
values:[524288,1048576,2097152], source:"synthetic private SRAM search space"}`; in this case that coordinate uses
discrete neighbor sampling. All options first pass existing component validation; explicit options can narrow but not expand an existing domain.
Component definitions remain unchanged, and undeclared coordinates still follow the original rules. After structure deletes the corresponding path, that domain no longer participates in sampling.
The example `examples/adf-budget.json` demonstrates non-GPU private SRAM capacity parameters and cost constraints on a real events backend.

The inner layer enumerates the Cartesian product of small `MappingTemplate.space()`; large spaces sample flat indices, avoiding materializing the entire product.
Each workload has at most `mapping_samples` mapping candidates in total, and multiple applicable templates share this quota.
Each workload independently selects the lowest-latency mapping; cost/area is provided by the same ADF cost estimate and does not change with workload.

The objective is `Σ weight*latency_ns + cost_penalty*max(0,cost-cost_budget)
+ area_penalty*max(0,area-area_budget)`. `best` and the middle-layer winner are sorted by this objective; constraint excess,
feasible, and occupancy ratio are output simultaneously. If penalty coefficients are too small, over-budget candidates may be selected, and the caller must check feasible.
When an enabled constraint lacks a metric, the objective is null, and search reports `unavailable_constraint`; when that constraint is not enabled,
search can proceed by latency alone. Unknown metrics always remain null.
`cost_model.basis` selects raw (default) or full; for basis, sources, and unsupported items see [cost.md](cost.md).
Area/cost in the backend Evaluation are preserved as raw backend output, and the search objective uses estimates from the independent cost package.

greedy only accepts moves where the objective does not increase; random accepts all legal candidates with known objectives; SA accepts objective decreases directly,
and for an increase Δ only accepts with probability `exp(-Δ/temperature)` when `Δ <= max_relative_degradation*abs(current_score)`.
After each proposal the temperature is multiplied by cooling. Search settings are algorithm parameters and do not enter the simulation model.
max_steps prevents infinite loops from invalid or duplicate proposals, so the actual evaluation count returned may be less than
the budget. When comparing strategies, use the actual backend call count in the trace as the standard.
