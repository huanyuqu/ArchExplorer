# Per-Sample Hardware Time Comparison from Complete Sources

`arch_explorer.archs.hardware_comparison.compare_hardware(document)` implements `arch-explorer.hardware-pairs.v1`. The input strictly contains schema and samples; samples strictly contain sample_id, split, measured, predicted. IDs are unique across the file, and split is calibration/development/holdout. Duplicates are all rejected, not just discarding the later row.

Each observation is exactly kind/source/conditions/unit/value/repetitions/raw_values. Source is exactly locator, sha256, license, redistribution, evidence_class, audit. Locator can trace to the original file line, sha256 is lowercase 64-bit, license and redistribution scope are explicit; audit=reviewed/unreviewed. The tool does not verify the authenticity of self-reported audit flags and summaries; source admission requires checking the original.

Conditions on both sides must be completely consistent at the canonical JSON value level, containing exactly the following objects:

| Object | Full fields |
| --- | --- |
| device | vendor/sku/architecture |
| software | sdk/driver/kernel |
| clock | mode/core_hz/memory_hz |
| operation | op/inputs/outputs/attributes |
| precision | inputs/outputs/accumulate/rounding |
| mapping | complete valid mapping, layout, implementation choices |
| timing | start/end/synchronization/warmup/statistic |

inputs/outputs are dictionaries of original names, with values being complete positive shapes, or objects containing shape and role; explicit scalars are `{shape: [], role: 'output'}`. attributes can be a known empty object; other identities reject null, empty containers, or unknown/unspecified/todo. clock is positive and finite; warmup is a non-negative integer; precision inputs/outputs are non-empty dtype arrays. Missing items on the measured side cannot be filled from predictions.

unit is s/ms/us/ns, explicitly converted to ns. repetitions is a positive integer, raw_values length is exactly equal, and value must equal the statistic declared by mean/median/min/max (relative floating-point tolerance 1e-12, absolute tolerance 0). Raw values and computed times/errors must be finite and non-negative; signed error can be negative. Paper means cannot be copied multiple times to fake original repeated data.

Only raw-device-timing/simulator pairs are hardware-pair; synthetic-control/synthetic-control is listed separately. paper-digitization and simulator values cannot enrich measured data. Each row gives P-M, absolute error, relative error; when M=0, both relative items are null, counted as zero_reference, and not included in the MAPE denominator. Aggregate separately by split, evidence class, and both sides' audit combinations, giving sample count, zero count, denominator count, mean, and worst sample ID. The tool does not fit or mix reports across splits.

```sh
archexplorer compare-hardware samples.json --output ERRORS.json
```

Legal rows are written together with specific rejection reasons; when rejected rows exist, the CLI returns 2. Synthetic controls only prove comparator behavior; hardware accuracy requires real records with the same device, precision, shape, mapping, frequency, and timing range that pass source review. The complete frozen sample contract is independently maintained by VAL63; production does not reference QA implementations.
