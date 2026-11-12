# One semantic graph with component-bound adapters

The examples construct explicit FP32 `3x5 @ 5x7` inputs and tiles `2/3/4`.
Use `--fanout` for an intermediate with two real readers. The registered
`matmul.tiled@1` rule owns regions, reduction blocks and access inspection;
the common plan coordinates storage lifetime. `compile_graph_v2` stores these
tile options in `rules.matmul.options.tiles`.

```sh
python scene.py --output /absolute/path/events-specs.json
python scene.py --example roofline --output /absolute/path/roofline-specs.json
python scene.py --example mixed --output /absolute/path/mixed-specs.json
archexplorer task-adapters
archexplorer compile /absolute/path/events-specs.json --output /absolute/path/program.json
archexplorer prepare /absolute/path/program.json --output /absolute/path/new-run > /absolute/path/request.json
archexplorer evaluate /absolute/path/request.json --output /absolute/path/result.json
archexplorer evaluate /absolute/path/result.json --output /absolute/path/replay.json
```

Events and Roofline use the same ordinary board with explicit component
implementations, shared SRAM, independent DRAM and two communication levels.
Their unique compatible adapters can be selected from component declarations.
Roofline owns its component-IO timing boundary; that boundary is not charged
again externally. Their timing contracts differ.

The mixed scene places the producer and one reader on the events SM and the
other reader on an explicit Roofline SM. Both use the same actual SRAM and
execute in one supported domain graph. It preserves the producer's two readers
across adapter groups; it does not add two independently measured latencies.

Custom hardware uses `scene.py --adf hardware.json --compute board/sm --memory
board/sram --output specs.json`. Supply optional `--bindings bindings.json`
to write exact adapter versions/options into the source specs. The standard CLI
can explicitly replace them with `compile --bindings another-bindings.json`.
No whole-graph target is selected.

Native construction delegates to the component-owned GPU assembly. Create an
explicit binding file whose contents are:

```json
{"component_gpu/cluster/sm": {"id": "accelsim", "version": "1", "options": {"threads_per_block": 32}}}
```

```sh
python scene.py --example native --bindings /absolute/path/native-bindings.json --output /absolute/path/native-specs.json
archexplorer compile /absolute/path/native-specs.json --output /absolute/path/native-program.json
```

The native scene's compute and complete eight-partition striped memory paths
remain `component_gpu/cluster/sm` and `component_gpu/partition/dram`.
Its adapter validates topology and options. A thread count is required there,
never in the common mapping. Engine/SDK/provider settings are separate
`prepare --options`. No engine is bundled and compile/prepare do not run one.

The same source specs enter Agent tools:

```sh
python agent.py /absolute/path/events-specs.json --output /absolute/path/agent-run --execute
```

Optional `--bindings` replaces saved bindings, and `--options` passes provider
options. Without `--execute`, compile/prepare use zero evaluation budget.
Execution uses one evaluation and strict saved-result replay uses none.
The script saves its public envelope/result and separate tool trace.

For an unknown component and adapter, install the
[external package](../extensions/component_implementations/README.md) after the
main package. See the [public API and migration tutorial](../../docs/compilation.md)
for per-task groups, cross-domain rejection, source binding and explicit v1/v2
recompilation. The scripts contain no lowering or timing formulas.
