# Running Your Own Operator Graph from an ADF File

When you need multiple target implementations of the same semantics, explicit selection, or multi-step/temporary tensors, first read the [external execution implementation tutorial](operator-implementations.md) and the [standalone implementation package](../examples/extensions/operator_implementations/README.md). It uses the public execution_graph to handle placement and storage for ordinary graphs; the compatibility example with a single lower and custom row mapping is retained below.

First prepare an ADF file, then read it into an in-memory description, and finally have the framework generate an executable model. The following assumes the file already exists; for first-time use, complete the installation and generate steps in the next section before executing this reading code:

```python
import json
from pathlib import Path
from arch_explorer.adf import from_dict

adf_data = json.loads(Path("branch-inputs/adf.json").read_text())
adf = from_dict(adf_data)
print(adf.to_dict())
```

Here `adf.json` is a hardware description file, and `adf` is an in-memory object containing component types, parameters, and provenance; neither is a running simulator. The build of registered components instantiates the ADF into resources, the mapping turns the operator graph into tasks, and the `events` executor schedules those tasks. After changing the file, you must re-read it and prepare the request again; a prepared model cannot be assumed to update automatically.

## Installing Real External Extensions

The [standalone package](../examples/extensions/workload_components/README.md) for this tutorial contains new operators, graph constructors, a synthetic array, and an external mapping. It does not implement a provider, nor does it modify the main package's operator/model lists, CLI, or scheduler. First install the main package containing `arch_explorer.workload.api`; from a source distribution or checkout root, run:

```sh
python -m pip install .
python -m pip install ./examples/extensions/workload_components
cd /tmp
python -m ae_workload_example generate --output branch-inputs
```

After this, all commands can be run outside the repository. No search path modification is needed, and PyTorch, ONNX, weights, or GPU are not required. Auto-discovery uses the `arch_explorer.components` entry point, target `ae_workload_example:register`. When embedding in Python is needed, the same register can also be called explicitly; it only registers and does not execute simulation.

The generated `adf.json` describes two `tutorial_array`s, one shared SRAM, and a bus; all parameters carry units and synthetic provenance. `affine-graph.json` and `builtin-graph.json` are two independent graphs; `mappings.json` stores two complete mapping selections. These are ordinary input files that can be edited independently, not presets bound to a specific device family.

## First Connect a New Graph That Only Reuses Existing Operators

`builtin-graph.json` has two scale branches and one add merge. The entry point for the new graph is `WorkloadSpec('graph', graph.to_dict())`; there is no need to add a model kind to the main project. Independent scenario weights in the `workloads` list are not equivalent to this graph's dependency relationships.

```python
from arch_explorer.adf import WorkloadSpec
from arch_explorer.workload.api import Workload

graph = Workload.from_dict(json.loads(Path("branch-inputs/builtin-graph.json").read_text()))
spec = WorkloadSpec("graph", graph.to_dict())
```

The graph is a finite JSON DAG that explicitly specifies tensors, operators, inputs/outputs, and parameters; it is not an arbitrary serializable Python object. Loops, dangling tensors, multiple producers, corrupted shape/dtype, and non-finite numbers should be rejected before execution. Inputs initially reside in the selected memory and do not represent simulation of external acquisition of input data. The graph strictly requires topological order and does not auto-sort; forward references and cycles are rejected. This example always writes graphs in topological order.

## Registering Truly New Operators

The public author entry point is `arch_explorer.workload.api`, which retains the same Tensor, Operator, and Workload types. `OperatorDefinition(name, validate, lower)` is the single-lower compatible form, adapted to events as `legacy:<op>`. The new interface can separately register semantic validate/reference/version and the target lower/support of `ImplementationDefinition`. `register_operator(definition)` registers the same definition object; competing names must raise an error; model names are not dispatched within components.

The example's new operation is:

```text
y = scale * x + bias
activation = identity or relu
```

`scale=1.25`, `bias=-0.5`, and the string activation are all valid attributes. The validator checks that input/output are both FP32 `[rows, cols]`, and validates finite coefficients, attribute sets, and shapes. It does not relax the semantics of other built-in operators. The lower returns an existing vector Kernel: one independent multiply and one add per element, plus one more SIMD instruction for ReLU. This is the current synthetic implementation choice, not disguising arbitrary new operators as GEMM; unsupported dtype/tile is explicitly rejected.

An external definition only needs two functions and one stable registration object:

```python
from arch_explorer.workload.api import OperatorDefinition, register_operator
from ae_workload_example.semantics import validate_affine, lower_affine

# The actual package reuses its module-level AFFINE object to avoid creating competing definitions each time.
definition = OperatorDefinition("my_affine", validate_affine, lower_affine)
register_operator(definition)
```

Ordinary multi-step execution can return a public `ExecutionProgram` with internal Tensors, handled by the public graph mapping for dependencies and liveness; special placement strategies or new hardware mechanisms should extend mapping/components. Semantics that cannot be expressed by the target artifact must be rejected, not defaulted to zero latency. This is shape and resource requirement simulation; it does not compute tensor values or prove model numerical correctness. The optional `register_workload(kind, builder)` is only for convenience of construction; the builder's signature is `builder(params, *, tp=1)`. The example `tutorial_branches` explicitly rejects tp>1 and does not silently ignore it.

## Custom Mapping and Hardware

`tutorial_rows` builds tasks through the public MappingTemplate and MappingContext, without calling the private scheduler. It splits each operator by rows and alternates placement between the two arrays; the two candidates have tile_rows of 4 and 16. Each output has independent storage allocation, and consumers depend on the actual producing tasks; the two branches are not impersonated as independent weighted scenarios. Input/output traffic uses SRAM/bus resources, and capacity and liveness are checked by the existing executor; the mapping's public plan enforces capacity checks, and terminal outputs are held until return.

The synthetic array's cycles per tile are `launch_cycles + ceil(SIMD instructions / lanes)`, then converted to ns by clock_ghz. The component only understands kernels, precision, and resources; it does not recognize affine graphs or user model names. Changing lanes or clock should affect its resource requirements; tile granularity changes the real task count and launch overhead.

```sh
python -m ae_workload_example hardware --lanes 8 --clock-ghz 2 --output faster-adf.json
```

This mapping only supports the two specified arrays, shared SRAM/bus, equal-shape 2D FP32 activations, dimensions 1–4096, one iteration, and SIMD lowering without SFU; it does not support aliasing, matrix kernels, other dtypes, multiple devices, or TP. Check mapping applicability before changing hardware. The synthetic formulas in the source are not physically calibrated; unknown costs remain null.

## Prepare, Execute, Compare, and Replay from Files

```sh
python -m ae_workload_example scene --adf branch-inputs/adf.json \
  --graph branch-inputs/affine-graph.json --mappings branch-inputs/mappings.json \
  --output branch-scene.json
archexplorer demo prepare --specs branch-scene.json --output branch-run
archexplorer demo run branch-run
archexplorer evaluate branch-run/mapping-000/result.json --output branch-replay.json
```

Use a new output directory. The scene command reads and validates the passed files, selects `provider='events'`, and does not add a new CLI model branch. Replace the graph path with builtin-graph.json to run a new DAG of existing operators with the same hardware and mapping. Replace the ADF path with faster-adf.json to independently compare hardware changes; do not change the graph at the same time and then attribute the result to hardware.

The same API path also accepts the in-memory description above:

```python
from arch_explorer.adf import MappingSpec
from arch_explorer.simulator.api import prepare_request, assess_request, evaluate_request

mapping = MappingSpec("tutorial_rows", {"tile_rows": 4})
request = prepare_request(adf, spec, mapping, provider="events", output_dir="api-run",
                          options={"trace": True})
assessment = assess_request(request)
if not assessment["supported"]:
    raise ValueError(assessment["gaps"])
result = evaluate_request(request)
replayed = evaluate_request(result)  # Read-only consumption of the original execution receipt; no re-simulation
assert replayed["evaluation"] == result["evaluation"]
```

prepare only generates resources and task plans; only a successful evaluate/run returning an Evaluation is the result of this execution. Compare `details.tutorial_rows.placements`, `details.stats.tasks`, `memory_peak`, breakdown, latency_ns, and parameter provenance. Wall-clock time, passing preflight, or file existence do not substitute for simulation results. Preserve the original paths referenced by the request, the original execution receipt, and the exact installed package; after source code or input changes, a new prepare is required. Source binding failures cannot be bypassed by updating hashes, nor can re-simulation be used to impersonate read-only replay.

When an agent needs to operate this flow, use [user skills](agent-skills.md); for hardware component authors, see the [component extension tutorial](component-extensions.md).
