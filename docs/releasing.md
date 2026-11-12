# Clean source release

The integration checkout preserves the original research repository's Git
history. Its existing remote must remain private during preparation. To create
a public release candidate, export only the reviewed tool sources:

```sh
python tools/export_source.py /absolute/path/to/clean/ArchExplorer
cd /absolute/path/to/clean/ArchExplorer
python -m venv /absolute/path/to/release-venv
/absolute/path/to/release-venv/bin/python -m pip install '.[dev]'
/absolute/path/to/release-venv/bin/python -m pytest -q
/absolute/path/to/release-venv/bin/python -m build
```

`EXPORT_MANIFEST.json` records the relative filename and SHA-256 hash of every
exported file. Inspect this allowlist and the sdist/wheel members. Install the
wheel in a second environment and run the example from an unrelated directory.
Verify that `evaluate` reproduces the selected latency using only the exported
candidate and package. Archive validation reports outside the public source.

Use a new export directory for each candidate build. Reusing a checkout or
export with an old `build/lib` can reintroduce deleted modules into a wheel,
even when the tracked sources are clean. Check the actual wheel and sdist
members against the reviewed source manifest, including removed modules.
Install both artifacts in fresh environments and verify their import paths;
an editable installation pointing to another worktree is not package validation.

The export intentionally contains no Git metadata or commit ancestry. Only
after reviewing the export, license/attribution, model scope and destination,
initialize its public repository and push that clean history. Preparing an
export does not change GitHub visibility or publish the package.

The allowlist contains the Python package, documentation, tests, synthetic
JSON examples, development tools and CI configuration. It excludes paper
experiment runners, figures, results, private calibration JSONs, model weights,
vendored C++ dependencies, API-key files and operational logs.

The local release checks cover these separately installed capabilities:

| Capability | Declared dependency | Verification |
| --- | --- | --- |
| Python library, synthetic Events/Roofline, CPU graph reference, CLI and offline Agent | Python 3.10+; NumPy 1.24+ | Record the actual interpreter and NumPy versions, install outside the source tree, and execute the documented source/configuration files |
| Source build and development checks | setuptools 77+, wheel; optional `dev` dependencies pytest 7+ and build 1+ | Build the sdist from the clean export, then build its wheel and inspect their exact members |
| Bounded ONNX import | Optional `onnx==1.16.2` via the `onnx` extra | Exercise real IR8/opset17 FP32/FP16/BF16 files, external data, explicit root shapes and budgets in the installed package; separately verify the core without ONNX. See [supported scope](onnx-import.md) |
| CPU framework export | Optional `framework` extra: ONNX 1.16.2 and PyTorch 2.5.1 | Select the official CPU PyTorch distribution explicitly before installing the extra. Test actual module and TorchScript export, input order, unchanged caller state and typed downstream execution. Keep framework, official ONNX and independent references separate |
| ScaleSim bridge | Optional `scalesim==3.0.0` | Check separately when that backend is used; installing the main package does not establish this coverage |
| MCP transport | Optional `mcp>=1.30,<2` | Check separately from the offline scripted Agent and JSON-lines protocol |
| Native components | A matching public SDK/engine and its explicit prerequisites | Preserve their source and binary identities; a CPU reference or Events result supplies no native execution evidence |
| Unknown operator/mapping extensions | An independently built extension distribution | Install the main package first, snapshot its files/metadata and console script, install the extension, and exercise its declared API contract without changing those main-package files |

A successful run on one interpreter is evidence for that environment. Keep the
metadata's supported Python range and the environments actually exercised in
the release record as separate facts. Retain Apache-2.0 `LICENSE` and `NOTICE`
in both distributions. Keep the current package version until a coordinated
contract change requires a version update.

Saved compiled results are checked read-only. `rerun` creates a new request and
execution, with its own budget and output directory. Validate both behaviors
explicitly rather than treating a second simulation as result replay. Preserve
failed candidate artifacts and bind any later correction to its own exported
source and installed distribution; unchanged earlier evidence keeps its
original execution identity.
