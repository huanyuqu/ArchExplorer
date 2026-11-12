# Validation Boundaries for Model Demos

The public GPU route is complete program native execution; the entry is in the [ViT/CLIP tutorial](tutorials/05-full-models.md). Models require complete outputs and verifiable numerical references; the two mappings should be semantically equivalent, with real program or geometry differences. Native cycles, complete execution counts, clocks, configurations, and sources should be traceable, and results are replayed through the same public entry.

Old `gpu_kernel_service` predictions, service parameter JSON, old GPU calibration tools, and baselines are no longer production entries. Historical measurements and errors remain in their original report locations; native integration cannot rewrite these conclusions. Small model support does not imply standard size support, and synthetic traces do not imply hardware accuracy; gaps remain unknown or unsupported.
