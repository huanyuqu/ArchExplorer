# 3. Build your own non-GPU architecture and simulate

The projection examples in the previous two chapters use a synthetic DMC: 4 vector cores, each with private SRAM, shared DDR DRAM and mesh. The parameter sources are teaching assumptions and have no real GPU meaning. For native GPU components, see the [complete program tutorial](05-full-models.md).

When an ADF file already exists, read the file first and then convert it to an in-memory description; do not treat the file name as the hardware model:

```python
import json
from pathlib import Path
from arch_explorer.adf import from_dict

adf = from_dict(json.loads(Path("my-dmc.json").read_text()))
print(adf.to_dict())
```

The following demonstrates constructing and saving this file for the first time. This ADF object itself has not yet instantiated resources or executed operators; evaluate/prepare generates the execution model based on registered components and mapping. After changing the file, it needs to be re-read and prepared.

```python
import json
from arch_explorer.adf import Param, MappingSpec, WorkloadSpec, evaluate
from arch_explorer.archs import dmc_adf

adf = dmc_adf(num_cores=4)
workload = WorkloadSpec("operator", {"op": "matmul", "m": 196, "k": 192, "n": 576})
mapping = MappingSpec("dmc_partition", {"split": "n", "weights": "local", "activations": "local"})
result = evaluate(adf, workload, mapping)
print(result.to_dict())
adf.child("core").child("sram").params["capacity"] = Param(
    2**21, "B", "synthetic tutorial: 2 MiB per core")
print(evaluate(adf, workload, mapping).to_dict())
with open("my-dmc.json", "w") as file:
    json.dump(adf.to_dict(), file, indent=2)
```

When comparing core counts, call `dmc_adf(num_cores=...)` again so that the number of network endpoints matches the topology. All parameters retain units and sources; when capacity is insufficient, retain the failure and do not disable checks to impersonate the original scenario. When constructing a new ADF, explicitly declare compute, memory, and network; the template checks validity by structure. [Component contracts](../components.md) and [structure editing](../structure.md) explain reuse methods.

To simultaneously connect a new operator graph and a custom array, use the [standalone workload package tutorial](../workload-extensions.md). The hardware there comes from a standalone ADF JSON; lanes and clock_ghz are two explicit parameters, and different tiles of the mapping produce different tasks; components do not branch based on complete model names.

Continue reading: [Exploration and result analysis](04-exploration.md).
