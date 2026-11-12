# ArchExplorer Workload-Driven Tutorials

The native GPU demo goes from a complete small model to real program execution; the general search tutorial uses a synthetic non-GPU vector many-core. Neither downloads weights or calls paid APIs. Native simulation also requires explicitly installing an external engine.

1. [Installation, first search, and independent replay](01-quickstart.md): closed loop for non-GPU synthetic operators.
2. [Describing your own workload](02-workloads.md): shapes, dependencies, and coverage scope.
3. [Building a non-GPU architecture](03-architecture.md): DMC components, parameters, and evaluation.
4. [Constrained exploration and Agent](04-exploration.md): budgets, constraints, and feasible results.
5. [ViT/CLIP native GPU demo](05-full-models.md): complete small forward pass, ADF, mapping, native results, and new shapes.
6. [Standard model semantic sources](06-full-model-sources.md): not equivalent to native support for standard models.
7. [Scope of baseline tools](07-model-baselines.md): non-GPU search and historical result boundaries.
8. [Integrating your own components or engines](../component-extensions.md): independent packages, registration, mapping, generation, and replay.

The vision/embodied examples in `examples/tutorials/` are only single synthetic projections, not complete models or robot closed loops. The tiny configuration of the default native demo is explicitly declared by the core scenario and does not impersonate standard ViT/CLIP.
