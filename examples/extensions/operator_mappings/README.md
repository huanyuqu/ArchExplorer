# Independent operator and mapping rule

This Apache-2.0 package registers `example.square@1`, computing FP32 `X*X`,
and `example.square.materialized@1`. The rule expands to the public multiply
semantic with two explicit reads of X and a materialized output. Its callbacks
own this expression and its traffic. Existing multiply adaptation, component
services and dependency/storage scheduling are reused; the package adds no
provider, scheduler or timing formula.

Install the current main package first, then build and install this independent
package. It requires the registered-operator-mapping and compiled v3 contract.

```sh
python -m pip install /absolute/path/arch_explorer-0.2.0-py3-none-any.whl
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /absolute/path/wheels
python -m pip install --no-deps /absolute/path/wheels/archexplorer_operator_mappings-0.1.0-py3-none-any.whl
archexplorer operator-mappings --operator example.square
python -m ae_operator_mappings --output /absolute/path/new-extension --search-config /absolute/path/search.json --execute
archexplorer search --config /absolute/path/search.json --policy greedy --output /absolute/path/best.json
archexplorer evaluate /absolute/path/best.json --output /absolute/path/readonly-best.json
```

`--execute` explicitly evaluates and validates a saved result. Without that
flag the module only compiles and prepares. `--roofline` uses the public
synthetic Roofline teaching hardware. The output directory and search-config
file must be new paths. The module saves `EXAMPLE-SOURCE-SPECS.json`,
`EXAMPLE-COMPILATION.json`, `EXAMPLE-PUBLIC-REQUEST.json` and, when executed,
`EXAMPLE-PUBLIC-RESULT.json`.

The scene appends the unknown square to the public attention output. It keeps
the original non-square inputs, materialized transpose/broadcast, nondivisible
tiles and two-reader dependencies. The reference hook computes elementwise
FP32 square; it does not provide event-simulator numerical output. Supported
inputs are finite, contiguous FP32 tensors of rank 1–4, with no aliases or
strides; unsupported dtype/shape/options reject.

`ae_operator_mappings.graphs.scene()` returns ordinary ADF/WorkloadSpec/MappingSpec
objects. These enter unchanged API/CLI/Agent tools and the existing exploration
engine described in [the attention tutorial](../../../docs/attention.md).
No main-package list is edited to recognize this operator or rule. Installed
callback sources and generated products participate in the public result's
source binding, so modifying the rule cannot reuse an older result.
