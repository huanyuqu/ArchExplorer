<p align="center">
  <img src="docs/assets/archexplorer-logo.png" alt="ArchExplorer logo: connected hardware blocks inside an A" width="160">
</p>

<h1 align="center">ArchExplorer</h1>

<p align="center">
  <strong>Automated architecture discovery through unified cross-architecture evaluation</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2503.21297">arXiv paper</a> ·
  <a href="#citation">Citation</a> ·
  <a href="docs/tutorials/01-quickstart.md">Tutorial</a>
</p>

ArchExplorer is an open-source framework for exploring accelerator architectures.
It brings architecture structure, hardware parameters and workload mappings into
one workflow, so designs with different compute, memory and interconnect
organizations can be evaluated through shared interfaces.

- **Compose multi-level hardware.** Describe components and their hierarchy in
  an architecture description file (ADF), from individual compute and memory
  units to larger interconnected systems.
- **Generate and evaluate execution.** Map workload task graphs onto the hardware
  model and use component evaluators and discrete-event scheduling to model
  computation, data movement and resource contention.
- **Explore three design levels together.** Apply structure transformations,
  tune hardware parameters and search workload mappings under explicit budgets.
- **Extend the same workflow.** Register component models, operator mappings or
  simulator providers, including external engines such as Accel-Sim, through
  the public extension interfaces.

**Experimental research software — version 0.2.0.** Start with the CPU-only
synthetic example below; it needs no GPU or external simulator. For the research
background, see our [MLDSE paper](https://arxiv.org/abs/2503.21297).

## Install

Python 3.10+ and NumPy are required. The package contains no model weights,
GPU binaries or paid model API calls.

```sh
git clone https://github.com/huanyuqu/ArchExplorer.git
cd ArchExplorer
python -m venv .venv
. .venv/bin/activate
python -m pip install .
archexplorer --help
```

`python -m arch_explorer` exposes the same CLI.

## Quick start: explore a small synthetic architecture

From the repository root with the environment above activated:

```sh
mkdir -p runs
archexplorer search --config examples/reusable_detail/demand.json --output runs/demo.json
archexplorer evaluate runs/demo.json --output runs/demo-replay.json
```

The first command explores a bounded set of hardware and mapping choices for a
synthetic matrix multiplication workload and saves the selected result. The
second evaluates the saved search result. Reported latencies describe the
configured simulation model; this example does not measure a physical device.
For an explicit workload compilation workflow, see the
[small matrix-multiplication example](examples/compilation/README.md).

## Release status and limits

- The Python library, CLI, synthetic examples and documented extension APIs
  are available for experimentation. APIs and serialized formats may evolve.
- ONNX import supports a documented subset; framework export is optional.
  Native GPU simulation requires a separately installed compatible engine.
- Some large workloads exceed practical runtime or host-memory budgets.
- Earlier implementations have been compared with real GPU measurements and
  other simulators.

See [model scope](docs/models.md) and
[release notes](docs/release-0.2.0.md) for details.

## Optional model frontends

For the optional [ONNX file frontend](docs/onnx-import.md), install
`python -m pip install '.[onnx]'`. It uses ONNX 1.16.2 for a bounded standard-domain
IR8/opset17 subset with FP32, FP16 and BF16 storage. The [MLP and attention examples](examples/onnx_import/README.md)
generate synthetic model files and explicit inputs for the public workflow.

The optional [PyTorch frontend](docs/framework-export.md) exports actual CPU
modules with explicit input order, shapes and weights. Install CPU PyTorch
2.5.1 from its official CPU distribution, then `python -m pip install '.[framework]'`.
The [framework examples](examples/framework_export/README.md) include MLP,
normalized residual decoder blocks, a complete small visual classifier and
an image/text dual tower with integer tokens and rectangular outputs.
See [scale admission](examples/paper_scale/README.md)
for full-size resource requests and [hardware comparison](docs/hardware-comparison.md)
for matching predictions to measured samples.

## Compile a workload with component-bound adapters

The [public compilation tutorial](docs/compilation.md) follows the same matmul
semantics and explicit mapping into a source-bound graph and component-adapted
inputs. CLI `compile` / `prepare`, Python and Agent use the same public API;
neither compilation nor preparation runs simulation or searches for mappings.
See the [matmul and two-reader scene](examples/compilation/README.md) and
[independently installed component example](examples/extensions/component_implementations/README.md#compile-the-same-semantic-graph).
Roofline, events and native models retain distinct work, traffic and accuracy
contracts; selecting an adapter does not make their latencies interchangeable.

The [attention tutorial](docs/attention.md) adds registered operator mappings,
explicit stable-softmax dataflow and bounded exploration through the same
compile/prepare/evaluate APIs. Its [independent operator package](examples/extensions/operator_mappings/README.md)
adds unknown semantics and a mapping rule after main-package installation.
The [complete pre-LN block](docs/precision-transformer.md) extends this workflow
to explicit FP32/FP16/BF16 storage, FP32 accumulation, materialized head movement,
both residuals and the MLP. Its [runnable scene](examples/transformer_block/README.md)
supplies data plus the same CLI, Agent and bounded-search configurations.
The [multi-layer encoder](docs/multilayer-encoder.md) composes repeated blocks
with one model input, distinct layer parameters and per-instance mapping choices;
its [scene command](examples/multilayer_encoder/README.md) uses those same tools.
It compares public materialized/readonly-view candidates for the same input graph,
including explicit fallback for reshapes that cannot share storage. Inspection
separates logical reads, root allocations and actual runtime peaks.
Saved compiled exploration winners support strict read-only validation and a
separate explicit rerun operation.
The [complete synthetic ViT classifier](docs/vit-compilation.md) adds explicit
NCHW images, patch/CLS/position inputs, independent encoder layers and the
classification head through this same workflow. Its [scene](examples/vit_compilation/README.md)
compares layouts and per-instance mappings without changing model semantics;
exact erf GELU and epsilon 1e-12 retain the ViT contract.
The [complete public CLIP example](docs/clip-compilation.md) connects unequal
image/text batches through both towers and returns both rectangular similarity
outputs. Its [scene](examples/clip_compilation/README.md) retains integer tokens,
causal attention, first-maximum EOT selection and QuickGELU while using the same
layout, instance-mapping, search and saved-result workflow.

## Discover and select component implementations

```sh
archexplorer components --output component-catalog.json
```

The public catalog lists each hardware type's implementation IDs, exact versions,
sources, parameters, input contracts and limits. For the same `sm` hardware type,
select an explicit [roofline or native Accel-Sim implementation](docs/component-implementations.md).
Their work inputs and result scopes differ; selection is not an automatic
conversion between operation counts and instruction traces. See the
[hand-calculated SM examples](examples/component_implementations/README.md),
[separately installable third implementation](examples/extensions/component_implementations/README.md)
and [component inventory](docs/component-inventory.md).

## Native GPU program demo

The [ViT/CLIP tutorial](docs/tutorials/05-full-models.md) follows a complete
small forward workload through an ADF, two mappings, program preparation,
capability checks and native Accel-Sim execution. The frontend requires the
complete-program backend and an explicit external engine manifest. Missing
support or engine files cause an error; preparation alone is not simulation.

```sh
archexplorer demo prepare --model vit --engine /absolute/path/engine.json --output runs/vit
archexplorer demo run runs/vit
archexplorer evaluate runs/vit/mapping-000/result.json --output runs/vit-replay.json
archexplorer demo prepare --model clip --engine /absolute/path/engine.json --output runs/clip
archexplorer demo run runs/clip
```

The selected component provider owns the complete forward trace and numerical reference.
Its reported scope distinguishes tiny synthetic reference execution, captured
traces and standard model profiles. Native integration does not establish
framework equivalence, hardware latency accuracy or model accuracy. Standard
ViT-B/16 and CLIP ViT-B/32 execution requires an explicit supported assessment.
No in-house GPU throughput/latency predictor or automatic fallback is provided.
The earlier [single-kernel native interface](docs/accelsim.md) remains separate.

## Non-GPU exploration

These examples use synthetic vector many-core or scalar CGRA hardware:

```sh
archexplorer search --config examples/adf-dmc.json --policy sa --output runs/dmc.json
archexplorer evaluate runs/dmc.json --output runs/dmc-replay.json
archexplorer agent run --config examples/adf-dmc.json --policy scripted --output runs/scripted.json
archexplorer search --config examples/adf-cgra.json --output runs/cgra.json
archexplorer env serve --config examples/adf-dmc.json
```

See the [introductory tutorials](docs/tutorials/README.md), [components](docs/components.md),
[structure/search](docs/structure.md), [Agent tools](docs/agent.md), [model scope](docs/models.md)
and independent [cost model](docs/cost.md). Unknown cost or timing remains unknown.
Synthetic event schedules do not establish device accuracy or search optimality.

## Add a component or provider

The [extension tutorial](docs/component-extensions.md) starts with parameter
declarations, a component model and a standard mapping. The
[separately installable example](examples/extensions/synthetic_components/README.md)
also supplies a small synthetic provider for the generic
`simulator.api.prepare_request / assess_request / evaluate_request` path.
It uses explicit registration or installed entry points, without changing core
source. An ordinary component does not need to implement an external engine.

## Run your own workload graph

The [workload extension guide](docs/workload-extensions.md) starts by reading an
ADF file, then registers an external affine operator, builds a branching DAG
and compares real row mappings on custom synthetic arrays through `events`.
The [independent workload package](examples/extensions/workload_components/README.md)
needs no core operator/model list edits and supplies no replacement provider.
See [user skills](docs/agent-skills.md) for agent workflows that consume these
public interfaces and tutorials.

## Citation

If you use ArchExplorer in your research, please cite our arXiv paper:
[MLDSE: Scaling Design Space Exploration Infrastructure for Multi-Level Hardware](https://arxiv.org/abs/2503.21297).

```bibtex
@misc{qu2025mldse,
  title={MLDSE: Scaling Design Space Exploration Infrastructure for Multi-Level Hardware},
  author={Huanyu Qu and Weihao Zhang and Junfeng Lin and Songchen Ma and Hongyi Li and Luping Shi and Chengzhong Xu},
  year={2025},
  eprint={2503.21297},
  archivePrefix={arXiv},
  primaryClass={cs.AR},
  url={https://arxiv.org/abs/2503.21297},
}
```

The same entry is available in [CITATION.bib](CITATION.bib). GitHub's
**Cite this repository** menu uses [CITATION.cff](CITATION.cff) to point to this paper.

## Development and source export

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python tools/export_source.py /path/outside/this/checkout/ArchExplorer
```

The exporter uses an allowlist and records hashes. Public sources exclude
private calibration, research results, weights, credentials and Git history.
Building or exporting does not publish anything. See [release checks](docs/releasing.md)
and [module interfaces](docs/integration.md). Apache-2.0; retain [NOTICE](NOTICE).
