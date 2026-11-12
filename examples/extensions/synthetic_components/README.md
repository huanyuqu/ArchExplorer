# Installable synthetic extension

This package lives outside `arch_explorer` and builds its own wheel. It exports
the `arch_explorer.components` entry point `tutorial = ae_synthetic:register`.
Registration only adds declarations and factories; it never runs a simulator,
queries a device or loads a native library. The source uses Apache-2.0 like the
containing project.

From the ArchExplorer source distribution or checkout:

```sh
python -m pip install .
python -m pip install ./examples/extensions/synthetic_components
cd /tmp
python -m ae_synthetic events --output tutorial-events.json
python -m ae_synthetic scene --output tutorial-scene.json
archexplorer demo prepare --specs tutorial-scene.json --output tutorial-run
archexplorer demo run tutorial-run
archexplorer evaluate tutorial-run/mapping-000/result.json --output tutorial-replay.json
```

Use new output directories. These commands use installed packages after leaving
the checkout. No `PYTHONPATH`, `sys.path` changes or edits to ArchExplorer are
needed. Install a framework build with `components.api` and `simulator.api`.

Two extension paths are intentionally shown:

* `TutorialMac` declares a synthetic FP32 GEMM/MVM compute resource in MAC/ns.
  Its `ComputeModel` plugs into the existing event graph, DMC mapping and memory
  components. A standalone ordinary component needs only `register_component`.
* `SerialMacProvider` owns a whole, single-component execution domain. It
  accepts one FP32 matmul, dimensions 1–64, `tutorial_serial` row tiles of 1 or 2,
  and a synthetic `startup_ns` per tile. It rejects mixed hardware, other
  mappings, dtypes, workloads and options. Its payload binds a specification
  artifact and implementation hash. Replay checks the saved execution without
  writing or running it again.

The provider scene performs 128 MACs at 8 MAC/ns. With a 5 ns per-tile startup,
tile sizes 1 and 2 give **36 ns** and **26 ns**. These are synthetic compute-only
schedules, not hardware latency, memory timing or numerical-output evidence.
Area and cost remain unknown. Event-graph timing also includes the explicitly
declared synthetic memory/network demands and need not equal these numbers.

Explicit registration, useful when embedding Python:

```python
from ae_synthetic import register
register()
from ae_synthetic.scenes import event_objects
from arch_explorer.adf import evaluate
print(evaluate(*event_objects(), backend="events").to_dict())
```

For entry-point discovery and failure diagnostics:

```python
from arch_explorer.components.api import discover_components, providers
print(discover_components())  # loaded and per-extension failures
print(providers())
```

See `docs/component-extensions.md` in the main distribution for the authoring
sequence, public types, provider envelope and stateful boundary. Use a unique
organization-prefixed component/provider name for your own package.
