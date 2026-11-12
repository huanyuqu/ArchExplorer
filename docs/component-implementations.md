# Hardware components and execution implementations

A component identifies a hardware responsibility. An implementation identifies
how that component executes work. Its source and exact version identify the
basis of that implementation. For example, `sm` is the shared hardware type;
`sm.roofline` and `sm.accelsim`, both version `1`, accept different inputs and
have different execution contracts.

```sh
archexplorer components
archexplorer components --output /absolute/path/to/catalog.json
```

The CLI returns `components.api.component_catalog()` directly. Each hardware
entry contains its role, common parameters, directory, coverage statement and
implementation descriptors. Each implementation describes its ID/version,
source, execution domain and modes, input contract, required parameters,
conventions, factory and result scope. Discovering metadata does not construct
instances or execute simulations. A registrar should register descriptors and
factories, not start an engine.

Agent `describe_environment` exposes the same schema under `component_catalog`.
Its existing `components` field is a compatibility projection for structural
search; it adds `domain_kind` to the public common-parameter descriptions.
Neither frontend maintains its own implementation list. The catalog reflects
installed extensions and the environment's actual component registry.

## Code ownership follows hardware responsibility

Hardware directories such as `components/sm`, `systolic_array`, `sram`, `dram`,
`l2` and `icnt` contain their definitions and implementations. Native SM work
lives under `sm/implementations/accelsim`; source-specific ABI, provenance,
historical readers and build support belong under `_support/accelsim`.
Controllers own their request/command services; `assemblies/gpu` composes the
same standalone factories into a complete system. Stateful execution is an
implementation property, not a separate hardware family.

Old public module names may remain thin imports to the authoritative location.
The [component inventory](component-inventory.md) lists supported mechanisms,
limitations and acceptance entry points.

## Bind an exact implementation in the ADF

Common hardware parameters remain in `SpacePoint.params`. Implementation
parameters go in a separate `ImplementationBinding`; each Param retains units
and its source:

```python
from arch_explorer.adf import Param, SpacePoint
from arch_explorer.components.api import ImplementationBinding

point = SpacePoint("compute", "sm", {
    "precision": Param("fp32", "1", "explicit example precision"),
}, implementation=ImplementationBinding("sm.roofline", "1", {
    "peak_ops_per_ns": Param(128, "op/ns", "synthetic declared peak"),
    "bandwidth_bytes_per_ns": Param(64, "B/ns", "synthetic declared component IO rate"),
    "boundary": Param("component_io", "1", "same counted boundary as the work"),
}))
```

The serialized point contains `implementation: {id, version, params}`. It is part
of ADF round trips, fingerprints, candidates and evaluation provenance. Native
SM parameters are not imposed on roofline, and an implementation may declare
the subset of common hardware parameters it consumes. Unknown parameters,
missing required parameters, incompatible domains and ambiguous selections
reject before execution effects. New multi-implementation hardware requires an
explicit binding; legacy defaults preserve the original behavior only.

In an Agent configuration, put the same complete point JSON in `seed_adf`.
Generic `create` edit specs also carry this binding. Common `set_parameters`
edits remain common hardware edits; implementation-specific values are not
silently inserted into the common namespace. Use a complete bound point spec
when constructing an alternative implementation candidate.

## SM roofline: explicit operations and boundary bytes

```python
from arch_explorer.components.sm.implementations.roofline import (
    RooflineWork, evaluate_roofline,
)

result = evaluate_roofline(point, RooflineWork(4096, 256, "component_io"))
assert result.latency_ns == 32
```

The implementation computes `max(F/P, B/BW)` ns, with operations F, boundary
bytes B, peak arithmetic capacity P in op/ns and bandwidth BW in B/ns. One MAC
is two operations. This is an ideal-overlap analytical service estimate with
declared peak-capacity assumptions. It does not infer warp scheduling or native
cycle counts and has no hardware-accuracy claim.

The [public example](../examples/component_implementations/README.md) gives
compute-bound, memory-bound and equal-boundary cases with independent hand
calculations. The work must explicitly identify `component_io`, matching the
implementation binding. A plain operator `Kernel` has no declared boundary
bytes and cannot silently substitute for RooflineWork.

Missing input fails instead of receiving an invented default. For example,
remove `bandwidth_bytes_per_ns` from the binding above and call the same public
evaluator: resolution must reject the incomplete implementation parameters.
Zero/negative or nonfinite capacities, mismatched boundary and F=B=0 also
reject. A new native selection cannot use roofline work as an instruction trace.

This single-component evaluator accounts for its IO bytes itself. A graph mapper
that uses this model must explicitly provide RooflineWork and avoid also charging
the same traffic to external DRAM/ICNT. Existing ordinary mappings do not gain
this contract merely because their compute point is named `sm`. For a
compute-only model with separately mapped memory traffic, use the external
implementation below.

## SM Accel-Sim: native cluster input

The native implementation is `sm.accelsim` version `1`, with the required
native options, program/mapping and port/clock contracts. Its current execution
granularity is an SM cluster containing real L1/shared behavior. A separate
`l1_shared` descriptor contributes parameters to that same owned instance; it
does not establish an independent L1 factory.

Construct the complete native ADF using
`arch_explorer.components.assemblies.gpu.gpu_adf`. The
[SM preparation example](../examples/component_implementations/README.md)
uses that assembly and the existing public scene/request flow. It only prepares
and assesses; actual execution needs a matching external CPU-native engine and
the [GPU workflow](gpu-instantiation.md) bounds. Other standalone memory/ICNT
consumers still use their public factory/SDK contracts.

Changing an implementation binding does not translate operation counts into
native instructions or allow an unsupported execution-domain bridge. Cross-mode
comparisons require their own matching input and result contracts. Native
software consistency does not make roofline latency equal to native latency.

## Install a third implementation without editing the main library

Install the main wheel first, then the
[independent component-implementation distribution](../examples/extensions/component_implementations/README.md).
It adds `example.sm.serial` version `1` to the existing `sm` type. The package
owns its synthetic FP32 GEMM/MVM compute service, parameters, factory and
capabilities. It registers no new hardware type, operator, mapping or provider.
The existing `events` executor and public evaluate/replay CLI consume it.

Component authors use `ComponentImplementation` and
`register_component_implementation` from `components.api`, plus the existing
`arch_explorer.components` entry point. The factory receives a resolved
component, a materialized point and BuildContext. Standard PointModel parameter
access records provenance; a fresh model owns each instance's resources.
The registrar is identical for explicit Python registration and installed
discovery. Duplicate identities or failed registrars cannot partially replace
the registered implementation set.

This object is distinct from workload `ImplementationDefinition`: the latter
lowers an operator into target work. Providers orchestrate execution domains;
mapping owns placement, transport and storage lifetimes. Adding a component
implementation should not add its ID to a CLI, provider or scheduler branch.

Keep source/parameter bindings when replaying results. Historical readers must
either verify the exact saved version/source or reject explicitly; moving a
module does not authorize deleting provenance fields, rewriting old results or
accepting an unrelated SDK. Source export, main installation, extension
installation, actual execution and independent acceptance remain separate
evidence stages.
