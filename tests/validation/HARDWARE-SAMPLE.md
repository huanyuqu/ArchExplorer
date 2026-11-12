# Hardware Comparison Sample v1

Frozen protocol `arch-explorer.hardware-pairs.v1`. CORE34's public comparator can implement this protocol independently; the QA reference implementation is `hardware_comparison.py` in the same directory and does not depend on production code. The complete synthetic example is provided by `test_hardware_comparison.pair()`, whose zero SHA and synthetic origin are used only for comparator testing.

Input objects strictly contain `schema` and `samples`. Each sample strictly contains `sample_id` (a string unique across the entire file; all occurrences of duplicate IDs are rejected), `split` (calibration/development/holdout), `measured`, `predicted`. Do not mix aggregation across fit sets, development sets, and holdout sets. Each of the two observations contains all of the following fields:

| Field | Complete value/type |
| --- | --- |
| kind | measured or predicted, consistent with its slot |
| source | Object: locator, sha256, license, redistribution, evidence_class, audit |
| conditions | Complete matching object described below |
| unit | s/ms/us/ns; only time is supported, explicitly converted to ns, not derived from throughput |
| value | Non-negative finite number, equal to the declared statistic of raw_values |
| repetitions | Positive integer |
| raw_values | List of non-negative finite numbers with length equal to repetitions, same unit as unit |

source's locator is the traceable location of the original file and line/sample; sha256 is the lowercase 64-character digest of that source file; license and redistribution must truthfully state the license and public distribution scope; private paper inputs do not enter the public package. evidence_class is raw-device-timing, paper-digitization, simulator, or synthetic-control. audit is reviewed/unreviewed. File digests and self-reported reviewed status do not constitute authentication or authenticity certification; originals must be independently verified. When there are no original repeated records, paper means must not be copied repetitions times. paper-digitization is retained as material pending pairing and must not be passed off as raw-device-timing under this protocol.

conditions fields are exactly as follows, and both sides must be fully identical at the canonical JSON value level; different units may be explicitly converted. The predicted side must not be used to fill in missing measured identities.

| Field | Required subfields and semantics |
| --- | --- |
| device | vendor, sku, architecture (exact device/hardware revision) |
| software | sdk, driver, kernel (exact implementation/version or digest; if there is truly no driver, state the reason for non-applicability) |
| clock | mode, core_hz, memory_hz (positive numbers; real samples/fixed ranges retain originals separately; peak default frequencies must not be filled in) |
| operation | op, inputs, outputs, attributes; inputs/outputs include all shapes and root roles, attributes may be an empty object |
| precision | inputs, outputs, accumulate, rounding; must not write only one dtype |
| mapping | Complete valid mapping, layout, and implementation choice object, not just a name |
| timing | start, end, synchronization, warmup, statistic; statistic=mean/median/min/max, warmup is an explicit count |

All identity fields reject null, empty strings/containers, and unknown/unspecified/todo. Explicit scalars use objects with `shape: []`; unknown shapes remain null and are rejected. Unsupported identities should be rejected first; do not fill in defaults on your own. Empty attributes means a definite no-attribute operation; warmup is a non-negative integer.

A valid hardware pair must have measured=raw-device-timing and predicted=simulator. Synthetic pairs must have synthetic-control on both sides and are reported separately. Per-point outputs: signed_error_ns=P-M, absolute_error_ns=abs(P-M), signed_relative_error=(P-M)/abs(M), absolute_relative_error=abs(P-M)/abs(M). When M=0, both relative items are null and reported separately as zero_reference; do not substitute epsilon for the denominator, and do not include zero references in MAPE. Aggregate by split, evidence class, and audit status, reporting valid count, zero count, relative denominator count, and worst absolute/relative sample IDs. Rejected items retain the specific reason; they must not be silently dropped and then claimed as all passing.

Interface source code has not yet proven hardware accuracy. The verified paper-fig7 original explicitly states not raw measurements; the latency field in MetaDSE architecture configurations is also not equal to raw samples. Only after source review and real predicted identities are complete can they be used as hardware comparison inputs.
