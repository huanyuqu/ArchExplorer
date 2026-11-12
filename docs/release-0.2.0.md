# ArchExplorer 0.2.0

This experimental release provides a Python library and command-line tools for
exploring architecture structure, hardware parameters and workload mappings.
It includes source code, tests, documentation and synthetic examples under
Apache-2.0, with third-party attribution in [NOTICE](../NOTICE).

## Getting started

Use the [installation and CPU-only quick start](../README.md). Python 3.10 or
newer and NumPy are required. ONNX, PyTorch and native simulator integrations
have separate prerequisites documented with each feature. The core quick start
does not require those integrations, model weights, a GPU or a paid API.

## Included capabilities

- Architecture descriptions, component registration and bounded architecture,
  parameter and mapping exploration.
- Workload graphs, component-bound compilation, event simulation and explicit
  performance-model selection.
- Supported tensor layouts, physical storage accounting and typed operations,
  including FP16/BF16 storage with documented accumulation contracts.
- Saved requests/results and independently installable component and operator
  extensions.
- Bounded ONNX import, optional CPU framework export and synthetic Transformer,
  vision and image/text examples.

## Validation and remaining work

Installation and targeted final-fix checks have been performed in the main
environment, Python 3.10 and an installation built from the source distribution.
The repository includes CI for Python 3.10 and 3.12; local checks do not establish
that a remote GitHub Actions run has passed.

Some large workloads time out during simulation or compilation, or exceed memory
capacity.

Earlier implementations have been compared with real GPU measurements and other
simulators. See [supported models](models.md) and
[hardware comparison](hardware-comparison.md) for configurations and comparison
methods.

The public source contains synthetic fixtures rather than private experiment
results, model checkpoints or device-calibration data. Native engines are
external dependencies. Retain the license and attribution when redistributing.
