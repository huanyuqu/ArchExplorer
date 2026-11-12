# Responsibilities of Explicit Multi-Device Execution

`MappingSpec` saves explicit operator placement, tensor sharding, routing, and local mapping; ADF saves device and interconnect resources. Applicability is checked by the actually installed mapping/backend. Capacity, precision, reduction, dependencies, and communication are not omitted due to distribution across multiple devices. The CLI does not add device family judgment or derive new GPU latency from device count.

After removing self-built GPU predictions, old GPU multi-device configurations and related model timing entries are no longer available. The common `multi_device` data contract and non-GPU capabilities remain, and actual local template available domains should be discovered through the registry. The new ViT/CLIP native demo is a single-stream complete program; it does not claim multi-GPU or π0.5 distributed support.

Device paths, operator identities, sharding, and routing in the input must be fully bound; missing/empty/illegal explicit requests are rejected. Implicit discovery skips unrequested structured templates, and no Evaluation is generated if capability checks fail. Replay uses the original ADF, WorkloadSpec, and complete MappingSpec, without deleting unsupported fields to fake the original execution.
