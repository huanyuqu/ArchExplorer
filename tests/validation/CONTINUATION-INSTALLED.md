# VAL64 Additional Consumer for the Unique Candidate

This tool preparation does not import new production source code; `continuation_installed_consumer.py` runs only in the owner's unique wheel/sdist installation environment. The caller first verifies the installation source and frozen input closure against the unique BUNDLE, then runs from a cwd outside the repository, with each stage limited to 120s/8GiB/single thread by `stage_capture.py`. `--candidate-commit` records the identity provided by the caller, and the consumer records the actual site-packages file digest; this string itself does not authenticate the installation package.

```sh
python QA_TESTS/continuation_installed_consumer.py \
  --package FROZEN_CORE35_PACKAGE --case FROZEN_CASE \
  --reference-record EXISTING_QA_REFERENCE/RESULT.json \
  --candidate-commit SOLE_CANDIDATE_COMMIT \
  --candidate 0 --layout views --stage numeric --output NEW_DIRECTORY
```

`numeric` performs real public import_onnx, describe, compile, and CPU reference for the original graph and mapped graph. Check node by node against original value names and declared dtype/logical shape; int32/int64/bool are compared exactly, and indices in floating-point containers are rejected. Each explicit Cast uses an independent conversion check, including zero sign; FP32 records the preset 5e-5 error threshold, low-precision drift and storage grid are recorded separately, and the final pass determination must be combined with the SIM53 formal precision contract. Original inputs, initialization weights, and Constant bytes are computed from real ModelProto types; do not miscalculate all roots as the model's floating-point dtype. FP32 Conv weights in BF16 models are preserved with their original types.

`component` saves real compile/prepare/evaluation results and invocation counts; actual Events/physical shard auditing still requires formal observation adaptation. This stage does not claim CPU numerical pass by borrowing service latency. `search` and `agent` each use one set of explicit placement/layout, max_evaluations at most 3, saving configuration, per-attempt ledger, and winner. Independently verify consumption units, failure units, actual execution markers, original indices, optimal objectives, and the identity of the winner workload/mapping pair against the bound source; do not use new experiments after completion to fabricate final counts for timed-out runs.

`continuation_observations.bind_reference` reuses the source digest of the original reference and frozen package manifest, and does not recompute the entire node graph. Existing FP16 vision wrapper failures continue to record raw_reference_status=failed and reference_requires_review; the saved data can serve as observation reference, but final acceptance must be linked to the previously clarified supplementary classification; do not change the original failed file to pass.

The source-code stage only tests independent comparison and ledger discriminators, and checks the wiring of the six archived reference/input metadata. It does not simulate product return values to execute the consumer, and does not indicate that new operators, components, search, or installation have passed.

The CORE35 formal interface ENTRYPOINTS also specifies the real CLI's `export-torch --input-names`, multiple outputs, `emit --candidate/--layout`, and save/replay/rerun; that part of the protocol already has the author's originals, and QA will only perform real end-to-end consumption on the unique candidate later. Existing `onnx_readonly_probe.py` and `saved_binding_contract.py` are retained as fresh/valid resealing tools; after new production boundaries arrive, verify interception coverage, with pure compilation/rebuild separated from prepare/engine/solver/writes.
