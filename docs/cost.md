# ADF Area and Cost Model

`arch_explorer.cost.estimate(adf)` returns a `CostEstimate`; it is an analytical model driven by public parameters, does not rely on area/cost callbacks of registered components, and does not execute CACTI or any online service. The values are not current quotes, and no constants have been adjusted using area/price data from validated devices. External reference code, papers, and experimental data are not distributed with the package.

```python
from arch_explorer.archs.adf_seeds import dmc_adf
from arch_explorer.cost import estimate

adf = dmc_adf(num_cores=1)
cost = estimate(adf)
print(cost.area_mm2, cost.raw_cost_usd, cost.full_cost_usd)
print(cost.by_category, cost.unsupported)
```

This example uses an explicitly synthesized non-GPU architecture, and the default process remains a reference scenario that has not been validated on devices. Logic area and raw/full cost can be returned; missing DRAM die density/area remains in unsupported and cannot be interpreted as meaning the complete silicon area is known or as an actual quote.

## Outputs and Basis

- `area_mm2`: sum of logic die areas. Includes logic-side memory interfaces, excludes DRAM dies, and is not the package footprint.
- `raw_cost_usd`: wafer allocation without logic die yield plus memory price; corresponds to the raw-die+memory basis.
- `full_cost_usd`: additionally accounts for logic die yield, bumps, organic substrate, and package failure losses in the **RE scenario**. It is not a complete Chiplet Actuary tool and does not include NRE, CoWoS/interposer, supplier margins, or unknown actual assembly steps.
- `cost_usd`: the selected basis, `basis="raw"` by default, or `basis="full"`. Both estimates are always retained separately.
- `by_category`: known logic area subtotals for arithmetic, control_rf, sram, and interfaces_network.
- `by_path`: path, expanded instance count, categorized area, memory cost, and raw die and total raw cost allocated proportionally to logic area. This allocation does not imply that each SpacePoint is manufactured independently.
- `memory`: memory price, sum of DRAM silicon areas, and bonded stack footprint; all are separate from logic die area.
- `provenance`: value/unit/source for input parameters, model parameters, and formula coefficients; `assumptions` is the reference configuration used.
- `unsupported`: unknown components, missing inputs, unused overrides, and their affected metrics. Row area and total cost for unknown components are null and cannot be treated as free components. Sums in categories and memory represent only known items, not complete totals with missing items treated as zero.
- `omitted`: details outside the current formula scope but not independently characterized, such as SFU and physical routing; these limitations are not disguised as measurements.

When DRAM die area is unknown, logic die area and known memory prices are still available. Do not interpret "logic area is available" as "all silicon area in the package is known." The model's cost for the same ADF does not change with workload.
A missing or invalid `dram_usd_gib` only makes the total cost and that row's memory price null; it does not clear logic interface area, and known raw die allocations are retained. `memory.cost_usd` is null when there is an unknown price.

## Configuration and Physical Parameters

Environment, Agent, SA/greedy/random all evaluate through the same cost package. JSON configuration example:

```json
{
  "cost_budget_usd": 100,
  "cost_penalty": 10,
  "growth_probability": 0.75,
  "shrink_probability": 0.75,
  "cost_model": {
    "basis": "raw",
    "process_nm": {"value": 7, "unit": "nm", "source": "LLMCompass 7nm reference scenario"},
    "die_count": {"value": 1, "unit": "count", "source": "one-die design assumption"},
    "overrides": {}
  }
}
```

This fragment is merged into the complete exploration configuration; `process_nm` supports 5 and 7 and does not interpolate other nodes. The default is 7nm with one logic die; `die_count>1` represents an explicit scenario of equal-area, same-process logic dies and does not infer die/chiplet/package boundaries from container hierarchy. repeat multiplies across all ancestors and represents only the number of ADF instances.
When no node is explicitly provided, `assumptions` carries `status="unverified_default"`, indicating that the default node has not been confirmed on devices. Retaining the default calculation is to support synthetic exploration scenarios with undetermined processes; it cannot be used to claim the actual process or area of any device.
Explicitly passing an unsupported 12nm is rejected without performing uncharacterized scaling. The passed source only records the node chosen by the user and does not mean the tool has validated that source. `scope_notes` can carry a list of strings, preserving known configuration limitations in assumptions.

A complete budget example (including an explicit non-GPU private SRAM discrete capacity domain) can be run and replayed directly:

```sh
archexplorer search --config examples/adf-budget.json --policy sa --output results/budget.json
archexplorer evaluate results/budget.json --output results/budget-replayed.json
```

`overrides` takes the form `{"full/ADF/path":{"parameter_name":{"value":...,"unit":...,"source":...}}}`.
The Python API can use `Param`; environment JSON uses the object above. Parameters must have a source; misspelled paths or unused parameters cannot be silently swallowed; when structural transformations invalidate an override, an incomplete estimate is reported. Common cost-only parameters:

| Location | Parameter and unit | Meaning |
| --- | --- | --- |
| Any point | physical_instances / count | Total physical instance count for paths of this type; must not be less than the expanded count of the executed ADF |
| Network | endpoint_count / count | Total physical endpoints (within the current network instance), used when non-execution resources differ from actual resources |
| Network | nvlink_links, pcie4_lanes / count | Number of additional logic-side external interfaces carried by the current network instance; billed as NVLink3 links and PCIe4 lanes respectively. p2p_switch is already billed for NVLink; additional nvlink_links are rejected to prevent duplication |
| DRAM | interface_bytes / B or interface_count / count | Interface capacity ratio or explicit interface count; default is one interface per 16GiB for HBM and one 32-bit interface per GiB for DDR (scenario assumption) |
| DRAM | dram_usd_gib / USD/GiB | Given memory price; GDDR has no built-in price and does not automatically borrow DDR5/HBM prices |
| DRAM | dram_density / MiB/mm2 | Known die capacity density; when not provided, ordinary DRAM silicon area is null |
| bonded DRAM | bonded_layers / count | 1–4 layers, default 4-layer scenario |

Cost overrides do not modify the ADF and do not change simulation results; they are a different configuration from parameter_domains used for actual parameter exploration. Unit mismatches raise errors; prices use GiB to explicitly follow the model convention from the source that treats nominal 80GB HBM as 80 capacity units, and decimal bytes must not be mixed in.

## Logic Area Formulas

Constants are stored in `cost/constants.py`, all using `Param`. The primary sources are the fixed-version
[LLMCompass area model](https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/cost_model/cost_model.py)
and its [GA100 configuration](https://github.com/PrincetonUniversity/LLMCompass/blob/2e015fd2ee750e6cad8d7152df52551e5b41ef20/configs/GA100.json).
The implementation independently rewrites the formulas and does not copy or run external simulators.

Floating-point unit area is `N × 685300 × (mantissa_bits/52)^2 / (density×10^6)`; fraction bits for FP64/32/16 are 52/23/10. The BF16 extension uses 7, explicitly an extension of the same scaling method; INT32 uses 177690 transistors. Densities for 7/5nm are 96.3/138.2 million transistors per mm². systolic_array uses rows×cols FMAs; vector_unit uses lanes FMAs with reference lane control. Unknown precisions or uncharacterized components (including the current cgra_pe) are reported as unsupported.

Native `accelsim_*` GPU components have no independent area/cost estimates, and related metrics remain unknown.
Non-GPU systolic/vector/PIM estimates continue to execute according to each component's contract.

SRAM: `A = blocks × block_bytes × 8 × bitcell_area / efficiency`. A block is at most 512KiB; large capacities are rounded up to complete blocks, and small capacities are at least 4KiB; efficiency uses the source's CACTI table (indexed upward by log2 of capacity).
7/5nm bitcell area is 1.70e-7/1.25e-7 mm²/bit. This is a static CACTI-derived reference table, not running CACTI on every candidate.

mesh/bus/crossbar use the source's 44300000 transistors per expanded endpoint communication proxy and do not claim to cover topology routing area.
HBM2e/DDR5 interfaces use the source's controller transistors plus fixed PHY area; p2p_switch uses the NVLink3 per-link proxy.
Explicit PCIe4 lanes use the same source's 0.48mm² PHY and 3962500 controller transistors; GDDR interfaces currently borrow the DDR5 32-bit proxy and note this in assumptions; GDDR PHY/controller are not yet independently characterized.
Frequency and simulated bandwidth are not directly converted to area; physical models are needed to scale them. Adding network hierarchy explicitly increases interface overhead.

## Manufacturing and Packaging

Raw die uses a 300mm wafer, `gross_dies = π150²/A − π300/√(2A)`,
`raw_die = wafer_price/gross_dies`. 7/5nm wafer prices are 9346/16988 USD from the historical model.
See the [supply chain model used by LLMCompass](https://github.com/PrincetonUniversity/ttm-cas/blob/d98ac0586faaf63f8f4eac94aa7cd50f37421c1f/supply_chain_model.py).
This analytical gross-dies approximation does not further add scribe/edge-loss; an oversized single die is not silently split.

The full RE scenario follows the negative binomial yield and organic packaging cost breakdown of [Chiplet Actuary](https://arxiv.org/html/2203.12268),
with parameters fixed to the [42dd5dee version](https://github.com/Yinxiao-Feng/chiplet-actuary/blob/42dd5deeec2b2f2daf13b1f3ac6042df2d4e3b62/parameter.ini).
`Y_die=(1+A×D/10)^(-10)`, with D for 7/5nm being 0.0009/0.0011 per mm².
full reports raw_die, die_defects, memory, bumps, substrate, package_defects, and wasted_KGD item by item.
Organic substrate area is 4 times the total logic area, at 0.005 USD per unit area; bumps are also 0.005 USD/mm²,
and bonding yield is 0.99^die_count; the multi-die substrate layer factor is calculated according to the source's area thresholds.
Memory is priced as purchased good dies, without re-applying logic-process yield, but scrap cost from packaging failures is included.
This is not A100 CoWoS cost; full cannot be compared with reference prices that exclude packaging to draw same-basis precision conclusions.

## Bonded DRAM and PIM

bonded uses 25.6MiB/mm²/layer, up to 4 layers. Total DRAM silicon area for capacity C is `C_MiB/25.6`,
and the stack footprint is that area divided by the number of layers. 1GiB corresponds to 40mm² total silicon, with a 4-layer footprint of 10mm².
Price uses the HBM reference of 7 USD/GiB; after accounting for 95% bonding yield per layer, it is `C_GiB×7/0.95^layers`,
so 1GiB four-layer is about 8.594 USD. This memory cost already includes stack bonding losses; full does not double-count this loss.
The above normalized parameters come from the paper method given in the task; the unit normalization of the [original 3D-DRAM research](https://arxiv.org/abs/2604.08044)
still needs auditing, and provenance retains that evidence level.

When `pim_unit` is in the same bank group as the sole DRAM, DRAM area/price increases by 20%, and the PU is no longer billed separately as a compute unit on the logic die.
HBM-PIM: capacity is multiplied by 1.2 at HBM2e's 7 USD/GiB. For DRAM-PIM, the AiM equivalent die count takes the maximum of the dies required by capacity, PU count, and compute throughput:
`max(C/512MiB, PU/16, total_ops_per_cycle×GHz/(16×32×1))`.
Source is [SK hynix AiM Hot Chips 34](https://hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf):
4Gb/die, 16PU, 1GHz, 16 MAC/cycle per PU. Price is equivalent capacity × explicit unit price × 1.2,
and external I/O scale is not guessed from internal bank bandwidth. The built-in DDR price of 2.4 USD/GiB comes from LLMCompass; this is not AiM's GDDR quote.
GDDR must provide dram_usd_gib. Without a public DRAM area density, **mm² cannot be derived from 512MiB capacity**; after providing density d,
each equivalent die area is `512/d×1.2 mm²`. This missing value is not filled with zero or disguised as a measurement.

Isomorphic DRAM groups of `pim_transform`/`pim_structure` are aggregated into one row across all expanded parts: capacity, PIM unit count,
throughput, and explicitly provided interface counts all include ancestor repeat, and interfaces and price are computed only once for all capacity in the group. Bank buses within a group
are internal DRAM structure and are not billed as independent logic-die networks; only the 20% PIM overhead assumption remains, with no new internal routing calibration.
The default interface count is still the capacity-ratio scenario (one 1024-bit interface per 16GiB for HBM, one 32-bit proxy per GiB for DDR/GDDR),
**not actual pin counts inferred from bank counts**. For example, NeuPIMs defaults to 32GiB and gets 2 HBM interface units; if modeling its 32 128-bit
channels, a sourced group-level `interface_count=4` should be provided, and the default capacity ratio must not be treated as a physical fact of that device.

Group-level overrides point to the original DRAM container path such as `neupims/hbm` or `hsndi/dram`. `by_path` provides
`member_paths`, `part_instances`, `pim_units`, and `interface_count`. Part-level overrides are not silently discarded;
they are reported as unused overrides. Groups with heterogeneous technology/price/density or PIM placed only in some banks retain per-part processing.
Unsupported items with the same reason and affected metrics are merged; `affected_path_count` is the number of affected type paths (not repeat-expanded instance count),
and `example_paths` lists at most four entries; complete member paths are in the group row.

## Validation and Replay

Non-GPU cost can be checked separately from execution results of the same ADF; use the `dmc_adf(num_cores=1)` example at the beginning for verification.

Native GPU components currently provide no independent area/cost estimates, and unknowns remain null. Static vendor documentation or reference microarchitecture facts
do not constitute an executable GPU latency model; old GPU presets are no longer used to run cost/latency examples.

Submitted files retain the cost_model and complete cost_estimate; replay_result recomputes. When historical results have no cost model configuration,
the current default model is used; old-version failure traces/targets may differ; strict trace replay requires pinning the same model version.
`adf.evaluate` is connected to this model: when the backend does not provide the two metrics, area/cost are filled in; `cost_model=False` disables this.
`details.cost` retains the model, basis, breakdown, and unsupported; for complete assumptions/provenance, see the `estimate` output.
