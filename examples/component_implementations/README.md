# Select an SM implementation explicitly

The [component implementation tutorial](../../docs/component-implementations.md)
explains the shared hardware identity and the different input contracts. These
examples use only installed public APIs. Metadata discovery is available through
`archexplorer components`; it performs no native simulation.

## Analytical work

```sh
python /absolute/path/to/sm.py roofline --case compute --output /absolute/path/to/compute.json
python /absolute/path/to/sm.py roofline --case memory --output /absolute/path/to/memory.json
python /absolute/path/to/sm.py roofline --case balanced --output /absolute/path/to/balanced.json
```

Each point is `sm` with an explicit `sm.roofline` version `1` binding. FP32 is
a common hardware parameter; P=128op/ns, BW=64B/ns and boundary=`component_io`
are sourced implementation parameters. Work and hand results are:

| Case | F (operations) | B (component_io bytes) | F/P | B/BW | Estimate (ns) |
| --- | --- | --- | --- | --- | --- |
| compute | 4096 | 256 | 32 | 4 | 32 |
| memory | 256 | 4096 | 2 | 64 | 64 |
| balanced | 1024 | 512 | 8 | 8 | 8 |

The public evaluator owns the formula and scheduling; this caller supplies work
and parameters and serializes its Evaluation. One MAC counts as two operations.
These results are ideal-overlap component estimates. They do not include a
second DRAM/ICNT cost for the same bytes, model warp scheduling, or establish
native-cycle equality.

## Native program preparation

```sh
python /absolute/path/to/sm.py native-prepare --model vit --engine /absolute/path/to/engine.json --output /absolute/path/to/new-native-requests
```

The scene constructor uses the canonical component GPU assembly with explicit
`sm.accelsim` version `1`, native parameters and full program/mapping inputs.
The current factory creates a real SM cluster containing L1/shared. This command
only prepares and assesses requests; it does not run an engine or compare a
roofline number against native cycles. Execution requires a matching external
CPU-native engine and the documented limits in the [GPU workflow](../gpu_instantiation/README.md).

For a genuinely new implementation of the same hardware type, install the
[separate external package](../extensions/component_implementations/README.md)
after the main wheel. Its compute-only model executes through the unchanged
`events` provider and CLI, without adding an implementation list to either.
