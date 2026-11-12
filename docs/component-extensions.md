# From one component to an installable extension

`simulator` is a general simulation generation framework: it generates execution plans based on ADF instances and connections, workloads, and mappings, handling resources, dependencies, scheduling, and lifecycle. The parameters, mechanisms, state machines, and external engine adaptations of specific hardware are owned by `components`. Accel-Sim retains both the original oracle and component instantiation paths; the new path connects SM/cluster, ICNT, L2, and DRAM through the general Runtime. Do not time each component separately and then add them up. The framework selects executors through registered capabilities.

If a new operator only needs existing hardware capabilities, use [external target implementations](operator-implementations.md), registering workload semantics/reference and target lower/support separately; no component changes or new providers are needed. Only enter the component authoring process below when new hardware mechanisms are truly required.

Ordinary components can implement only `ComponentType.build` and `Demand`, without implementing external engine protocols. `arch_explorer.components.api` provides the types component authors need, and re-exports preserve original type identity: `ComponentType`, `ParamSpec`, `Conventions`, `BuildContext`, `PointModel`, `ComputeModel`, `MemoryModel`, `NetworkModel`, `Demand`, `Kernel`, `Provenance`, resource/task types, and `ExecutionModel`, `ExecutionRequest`, `ExecutionEvent`, `DemandAdapter`. Extensions do not need to import internal files from `simulator.events`.

## Declare parameters and capabilities first

`TutorialMac` in the [standalone extension package](../examples/extensions/synthetic_components/README.md) declares `macs_per_ns`, unit `MAC/ns`, and `precision` that only accepts `fp32`. Its build function returns a `ComputeModel`; `kernel_time` only accepts FP32 GEMM/MVM and returns ns by dividing the MAC count by the synthetic rate. Wrong units, boolean rates, out-of-range values, and unsupported kernels should all be rejected. The parameter `source` indicates this is a synthetic assumption. Area and cost are not modeled and remain unknown.

```python
from arch_explorer.components.api import register_component
from ae_synthetic.component import TutorialMac
register_component(TutorialMac)
```

Components declare their capabilities, and mappings declare placement, partitioning, and applicability conditions. Being able to build a component does not mean any mapping can execute it. The example replaces this MAC into a single-core DMC scenario with explicit SRAM/DRAM/network, then runs an FP32 matmul using the existing `dmc_partition` mapping. The example package's `scenes.event_objects()` returns standard ADF, WorkloadSpec, and MappingSpec, which can be passed directly to `adf.evaluate`.

## Actual installation and discovery

Run in the source distribution or checkout root of the main project:

```sh
python -m pip install .
python -m pip install ./examples/extensions/synthetic_components
cd /tmp
python -m ae_synthetic events --output installed-component.json
```

The extension has its own `pyproject.toml` and `src/ae_synthetic`, and is not added to the main project's package discovery list. Its entry point configuration is:

```toml
[project.entry-points."arch_explorer.components"]
tutorial = "ae_synthetic:register"
```

The parameterless `register()` only registers descriptions and factories, and does not run simulations, probe devices, or load external binaries. The same function can also be called explicitly in Python. The framework discovers installed extensions before execution:

```python
from arch_explorer.components.api import discover_components, providers
report = discover_components()
print(report["loaded"], report["failures"])
print(providers())
```

Each failed extension is reported separately, without invalidating unrelated components; duplicate names, provider ids, or incompatible versions must raise explicit errors. Use unique names for your own components and providers. Keep the discovery failure reason for troubleshooting.

## Register a provider when an independent engine is needed

External engines or component groups that must be executed as a whole use `register_provider(provider)`. The object declares a stable `id`, exact `version`, `component_types`, `execution_domain` (`single` or `joint`), `capabilities`, exact `request_schemas` / `result_schemas`, and implements:

```python
prepare_request(adf, workload, mapping, *, output_dir, options=None)  # does not run simulation
assess_request(payload)  # supported: bool, gaps: list
evaluate_request(payload)  # complete request and Evaluation.to_dict()
```

The prepare payload echoes the standard `adf`, `workload`, `mapping`, and `options` dictionaries; other fields are managed by the provider. Results save the complete component request and execution artifact bindings; replay checks sources and results without rewriting the originals. External engine paths and limitations go in options; hardware parameters still come from the ADF, and semantics from WorkloadSpec.

The example's `tutorial.serial_mac` is an independent synthetic execution domain: it only accepts one `tutorial_mac`, FP32 matmul with dimensions 1–64, `tile_m=1|2` for `tutorial_serial`, and `startup_ns`. This mapping is only for this provider and cannot be applied to the event engine. Mixed component domains, BF16, other workloads, and unknown parameters are rejected. It does not simulate memory, numerical output, power, or real hardware.

```sh
python -m ae_synthetic scene --output /tmp/mac-scene.json
archexplorer demo prepare --specs /tmp/mac-scene.json --output /tmp/mac-run
archexplorer demo run /tmp/mac-run
archexplorer evaluate /tmp/mac-run/mapping-000/result.json --output /tmp/mac-replay.json
```

These steps discover the provider through installed entry points, without modifying the main project source or search path. The example has 128 MAC / 8 MAC/ns, plus 5 ns per tile, giving two candidates of 36 ns and 26 ns; this is a public synthetic mechanism example. Changing the mapping changes the actual tile list, and unmodeled metrics remain null.

## General generation and replay entry points

```python
from pathlib import Path
from ae_synthetic.scenes import provider_scene
from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
from arch_explorer.simulator.api import prepare_request, assess_request, evaluate_request

scene = provider_scene()
request = prepare_request(from_dict(scene["adf"]), WorkloadSpec(**scene["workload"]),
                          MappingSpec(**scene["mappings"][0]), output_dir="/tmp/mac-python",
                          provider=scene["provider"], options=scene["options"])
assessment = assess_request(request)
if not assessment["supported"]:
    raise ValueError(assessment["gaps"])
result = evaluate_request(request)
replayed = evaluate_request(result)  # checks saved execution, does not rerun the provider engine
assert replayed["evaluation"] == result["evaluation"]
```

The request outer layer is `arch-explorer.simulator-request.v1`, containing the provider's id/version, three standard specs, options, provider payload, and complete bindings. The result outer layer is `arch-explorer.simulator-result.v1`, containing the complete request, the same provider, the original Evaluation dictionary, provider raw results, and bindings. Details are preserved as-is; `latency_ns` remains ns and is not replaced by Python wall-clock time. Preserve files referenced by the request and their original paths. New results and registered historical exact schemas are both handled by the same resolver; the CLI does not recognize specific engine names.

The demo's candidate set is fully checked before any evaluate: standard objects, candidate labels, paths, all request identities, and preparation records must be consistent; all capabilities are reviewed first, then execution. Duplicate, replaced, missing, or mismatched requests cannot form a successful comparison. After changing any spec or options, prepare to a new directory.

## Stateful components and execution domains

Stateful mechanisms use the public `submit / next_event_time / advance`, keeping backpressure rejection side-effect-free and reporting `consumer_ready`, `physical_complete`, and `resource_ready` separately. Do not mistake consumer readability for physical write-back completion, and do not bypass liveness capacity or resource occupancy. See [joint execution contract](unified-execution.md) for concrete examples.

Execution domains cannot be arbitrarily concatenated. Subcomponents of the Accel-Sim provider require a single native execution; coupling with other independent engines is not supported and should be rejected before execution. Its component implementation is located in `components.accelsim`, and the generic simulator does not select specialized mechanisms based on templates or provider names. User scenario construction, CLI model names, and weight selection also do not belong to the provider. For a complete GPU example, see the [ViT/CLIP tutorial](tutorials/05-full-models.md).
