# Multi-Device Topology and Exploration

The common ADF hierarchy, repeated instances, interconnect, structural editing, and explicit placement/sharding/routing contracts remain. These mechanisms are independent of GPU physical latency implementation. Public multi-GPU configurations that originally contained `gpu_sm` have been removed and will not be silently interpreted as non-GPU hardware. For the new GPU complete program path, see [native demo](tutorials/05-full-models.md).

When creating a non-GPU multi-device ADF, declare compute, memory, and local networks in each device subtree, then declare interconnect in the parent container. Discover templates and conditional branches applicable to the current workload and devices through `describe_environment().result.mapping_spaces`; do not assume any model is executable just because the topology is legal. Structural instance counts include all ancestors' repeat. Area, cost, and physical execution capability are checked separately; unknown metrics cannot be filled with zeros.

Implicit discovery does not request `multi_device` out of thin air; explicit requests must carry complete placement, sharding, and routing. [Execution constraints](multi-device-execution.md), [common interface](integration.md), and [Agent tools](agent.md) describe the entries.
