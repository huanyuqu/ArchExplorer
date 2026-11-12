# Add an implementation to an existing component

This separate distribution registers `example.sm.serial` and
`example.sm.serial_assembly`, version `1`, for the existing hardware type `sm`.
It registers no hardware type, operator, mapping,
provider or CLI branch. Package 0.3.0 also registers the task adapter
`example.serial.events@1`, explicitly declared by the assembly-capable SM.
Install the matching main distribution first:

```sh
python -m pip install /absolute/path/to/arch_explorer-0.2.0-py3-none-any.whl
python -m pip install --no-deps /absolute/path/to/examples/extensions/component_implementations
archexplorer components --output /absolute/path/to/catalog.json
archexplorer task-adapters --output /absolute/path/to/adapters.json
```

Discovery uses the standard `arch_explorer.components` entry point. The returned
catalog contains this implementation alongside built-in SM implementations.
Metadata discovery does not construct native instances or run simulation.
Importing this package alone does not register it. For explicit Python use,
`register()` invokes the exact same registrar as installed discovery.

## Execute and replay through the existing public paths

From outside either source checkout, using a new preparation directory:

```sh
python -m ae_component_implementations --output /absolute/path/to/new-request
archexplorer evaluate /absolute/path/to/new-request/REQUEST.json --output /absolute/path/to/result.json
archexplorer evaluate /absolute/path/to/result.json --output /absolute/path/to/replayed.json
```

The request binds `sm` to the exact implementation ID/version and its own
`ops_per_ns`/`startup_ns` parameters. The common hardware parameter is FP32
precision. Rates and startup are explicit synthetic assumptions with sources.
Change `--ops-per-ns`, `--startup-ns` or the GEMM shape to exercise another real
service demand; the result is not constant.

The implementation accepts FP32 GEMM/MVM `Kernel` inputs and occupies one FIFO
compute resource for `startup_ns + F / ops_per_ns` ns, with one MAC counted as two
arithmetic operations. A direct 2x2x2 GEMM Kernel has F=16 operations and compute
service 3+16/4=7ns. This is the component service, not the whole graph latency:
the existing `gsm_tiled` mapping separately places real memory and transport
tasks on the declared DRAM, SRAM and buses. No memory bytes are charged inside this SM
implementation, so those external tasks are not charged twice.

The scene is a complete ordinary device: a core containing the SM and local
bus, plus shared 4096-byte SRAM, separate 65536-byte DDR memory and device bus. The
mapping moves operands from that DRAM through the declared memories and buses;
these are real mapped tasks. Both buses and both memories use explicit synthetic
rates and latencies. Preparation supplies the legal fixed integer tiles
`tile_m=64` and `tile_n=64`, with `rows_per_cta=4`. The existing mapping charges
boundary memory bytes but pads the compute Kernel's N dimension to `tile_n`.
Thus the default mapped Kernel is 1x2x2x64: F=512 and compute service is
3+512/4=131ns. Including external access latency, this scene takes 143ns with
the matching main distribution. Preparation performs no
automatic tile search or trial execution. Shapes that exceed the declared
memory capacity still reject through the standard mapping checks.

```python
from ae_component_implementations import scene
from arch_explorer.adf import evaluate, fingerprint, from_dict

adf, workload, mapping = scene(ops_per_ns=8.0)
assert from_dict(adf.to_dict()).to_dict() == adf.to_dict()
identity = fingerprint(adf)
result = evaluate(adf, workload, mapping, backend="events", options={"trace": True})
```

`register()` describes a `ComponentImplementation`; this is different from
`workload.api.ImplementationDefinition`, which lowers an operator. The factory
receives a resolved component, materialized point and public BuildContext;
standard `PointModel.param/positive` reads and records implementation parameters.
It returns a fresh ComputeModel per instance. Runtime scheduling remains in the
framework. FP32-only validation, invalid rates/startup and missing parameter
sources reject explicitly; an incompatible native domain never falls back to
this implementation.

Hardware/implementation/parameter mutation, duplicate identity, source binding
and broken-registrar rollback are enforced by the main public API. A copied
result must be replayed with the matching installed implementation source;
changing its ID/version or rate is not a valid replay of the old result.

This synthetic service has no warp, instruction, cache, power or calibrated
hardware behavior. For the explicit roofline versus native-cluster boundary,
see the [main implementation tutorial](../../../docs/component-implementations.md).

## Compile the same semantic graph

With a matching main distribution that provides public compilation, the
explicit `example.sm.serial_assembly@1` implementation selects its declared
`example.serial.events@1` task adapter in the existing events execution domain.
It reuses the original matrix service and adds a narrowly declared synthetic
assembly-issue service. The original GEMM/MVM-only binding stays restricted;
it does not silently start accepting general vector kernels. Both components
and the adapter are registered in one public extension transaction. The main
package does not enumerate these IDs and no new provider is installed.

```sh
python -m ae_component_implementations.compilation --output /absolute/path/new-compiled-run
archexplorer evaluate /absolute/path/new-compiled-run/EXAMPLE-PUBLIC-REQUEST.json --output /absolute/path/compiled-result.json
archexplorer evaluate /absolute/path/compiled-result.json --output /absolute/path/compiled-replay.json
```

This default is an explicit FP32 `3x5 @ 5x7` graph, including numerical inputs,
with `compile_graph_v2`, rule `matmul.tiled@1` tiles `m=2,k=3,n=4`, and storage `device/sram`. Hardware still
has independent DRAM, shared SRAM and two-level communication; the SM charges
compute only. Use `--tile-m 4 --tile-k 2 --tile-n 3` for another mapping or
`--fanout` to retain the product through two subsequent readers. Change rate,
startup or dimensions with the same flags as the legacy example.

The public `compiled.assemble` events input is a FP32
`Kernel('vector', (1, batch*m*n))` with `simt`, `sfu`, `row_simt` and `row_sfu`
all zero. The assembly-capable model accepts only positive 2D shapes and these
zero-arithmetic counters. Its compute resource occupies
`assembly_startup_ns + elements / assembly_elements_per_ns`, using explicit
sourced synthetic parameters (defaults 1ns and 16 element/ns). This models
copy-command issue work, not arithmetic or memory bandwidth. Every assembly
remains a real mapped step with tensor output and dependencies; all memory
bytes, storage and network costs remain with the external mapping.

Use `--assembly-elements-per-ns` and `--assembly-startup-ns` to change that
service. Nonzero vector arithmetic, other precision, malformed dimensions,
unknown operations and missing/invalid sources or rates reject explicitly.
This is not arbitrary vector/SFU support or a native instruction model. The
new source identity also means older saved results cannot be relabeled as
executions of this package.

The directory also saves `EXAMPLE-COMPILATION.json` and
`EXAMPLE-SOURCE-SPECS.json`. Feed the latter to the standard CLI or the same
Python API; generic dispatch never recognizes this package or implementation ID:

```sh
archexplorer compile /absolute/path/new-compiled-run/EXAMPLE-SOURCE-SPECS.json --output /absolute/path/program.json
archexplorer prepare /absolute/path/program.json --output /absolute/path/another-new-run > /absolute/path/public-request.json
archexplorer evaluate /absolute/path/public-request.json --output /absolute/path/another-result.json
```

`compilation_scene()` returns `(adf, workload, mapping)` for Python/Agent use.
The public compiler owns region shapes, masking, K loops, materialized assembly,
storage lifetime and exact bindings. This mapping's masked work is different
from the legacy `gsm_tiled` padded-Kernel accounting above; inspect each
artifact's actual work and traffic rather than reusing its old 131ns service.
There is no automatic roofline/native bridge for this compute-only model.
See the [compilation tutorial](../../../docs/compilation.md) for contracts and
the native numerical/timing evidence boundary.

### Adapter and backend ownership

`adapter.py` supplies conversion and restoration, reusing public events operator
lowering to produce the existing Kernel input. It preserves every declared
group task and uses `traffic=external`; it does not compute service time or
copy the main compiler, mapping, runtime or model. `model.py` remains the sole
authority for the sourced synthetic rates and FIFO compute occupancy.

The component's `adapters` declaration permits automatic selection only for
one compatible adapter. An explicit equivalent binding is:

```json
{"device/core/compute": {"id": "example.serial.events", "version": "1", "options": {}}}
```

The example adapter accepts no options. Unknown options, wrong bindings and
source changes reject before execution. Its group and saved callbacks are
source-bound, so changing the adapter invalidates a saved result even when
the compute model is unchanged. Install the main wheel before this wheel and
verify the main package stays byte-identical; package installation is not a
reason to edit a main-package adapter list.
