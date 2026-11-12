# Native GPU Execution and Validation Scope

The [native demo](tutorials/05-full-models.md) shows complete forward semantics, programs, reference outputs, mappings, and native results. Cycles are produced by external Accel-Sim/GPGPU-Sim execution, not fitted from measurements or filled from old predictions.

Complete program execution, reference numerical correctness, official framework equivalence, and physical latency accuracy are different conclusions. Real collection, public traces, and synthetic reference generation report sources separately; missing instructions and execution gaps are explicitly rejected. Historical hardware windows, original logs, and failed results are retained in their respective experiment directories and are not automatically rerun by the public demo. Running the demo does not grant any new real hardware collection or timing permissions.
