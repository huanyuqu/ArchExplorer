# 7. Baseline Tools and Execution Boundaries

The full GPU demo uses the [native program entry point](05-full-models.md). The old self-built GPU latency model, GSM seeds, and baseline configurations that depend on them have been removed. Existing experimental results, failures, and collection records remain in the original working directory and are not rewritten as new native evidence.

`tools/run_model_baseline.py` still organizes general non-GPU ADF/Agent experiments, recording actual invocations, resource limits, and independent process replay. It does not implement latency formulas. Inputs must be composed of currently supported components and mappings; old GPU prediction seeds or `gpu_sm` / `gpu_kernel_service` components are not accepted. `--max-evaluations` only reduces the experiment budget; it does not shrink the workload, compensate for failures, or guarantee a feasible best candidate.

The two mappings of the new GPU demo generate complete programs and execute them separately under the same workload/ADF, without using old search wins/losses or historical predictions to replace real program results. Full forward pass, numerical reference, execution coverage, and hardware accuracy must be verified separately.
