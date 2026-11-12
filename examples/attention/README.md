# Explicit attention scene

Run `scene.py --output NEW_SPECS.json` to save explicit FP32 Q/K/V inputs and
the stable attention DAG. Optional `--search-config NEW_CONFIG.json` saves the
bounded default search input. Neither operation compiles, prepares or evaluates.
Files must be new, distinct paths.

```sh
python scene.py --output specs.json --search-config search.json
archexplorer compile specs.json --output compilation.json
archexplorer prepare compilation.json --output new-request > request.json
archexplorer evaluate request.json --output result.json
archexplorer search --config search.json --policy greedy --output best.json
archexplorer evaluate best.json --output readonly-best.json
```

Use `--roofline` for the explicit synthetic Roofline hardware, and
`--input-case regular|large|constant` for the numerical input controls. Separate
shape/tile flags are available for source specs; the bounded search tutorial
uses its documented default shapes and two concrete mappings.

The [full tutorial](../../docs/attention.md) specifies materialization,
two-reader dependencies, datatype limits, budgets, source validation and the
distinct explicit rerun operation. Reference values are CPU semantic checks,
not values computed by the event simulator or hardware accuracy evidence.
