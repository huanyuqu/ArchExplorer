# 1. Installation, search, and replay

The following commands are all executed from the repository containing `pyproject.toml` or a clean source export directory.
Tutorial files are provided with the source code; installing only the wheel will not generate these configurations in the current directory.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m arch_explorer --version
```

Python 3.10+ is required. The only basic dependency is NumPy; this tutorial does not require MCP, SCALE-Sim, GPU, or model downloads.
If Python source code is modified during development, use `python -m pip install -e .`; this tutorial only modifies input files.

## Vision projection: first search

```sh
python -m arch_explorer search --config examples/tutorials/vision-projection.json --policy sa --output results/tutorials/vision.json
python -m arch_explorer evaluate results/tutorials/vision.json --output results/tutorials/vision-replayed.json
```

The first command searches two on-chip network layouts and GEMM mappings on a synthetic non-GPU 4-vector-core DMC.
The second command reads the saved complete ADF, workload, and mapping, and independently simulates the same best candidate, **without re-searching**.
The output directory is created automatically. Results are saved in `results/`; do not submit them as source code or public calibration data.

```python
import json
from pathlib import Path

original = json.loads(Path("results/tutorials/vision.json").read_text())
replayed = json.loads(Path("results/tutorials/vision-replayed.json").read_text())
best = original["best"]
print("Actual backend attempts:", original["evaluations_used"])
print("Objective and constraints:", best["score"])
print("mapping:", best["mappings"])
print("Latency (ns):", best["workloads"][0]["evaluation"]["latency_ns"])
print("Independent cost estimate:", best["cost_estimate"])
print("Diagnostics:", best["workloads"][0]["diagnostics"])
assert best["fingerprint"] == replayed["best"]["fingerprint"]
assert best["mappings"] == replayed["best"]["mappings"]
assert best["score"] == replayed["best"]["score"]
assert best["workloads"][0]["evaluation"]["latency_ns"] == replayed["best"]["workloads"][0]["evaluation"]["latency_ns"]
print("Replay consistent")
```

Save the code as a temporary `.py` file and run it, or paste it into Python in the activated environment.
The same version and same backend should produce the above consistency; when cross-version replay differs, preserve both results and code versions before troubleshooting.
The time unit is ns; divide by 1e6 to get ms. `score.objective` is not necessarily pure latency; see Section 4.

## Embodied policy projection: replace inputs, keep the process

```sh
python -m arch_explorer search --config examples/tutorials/embodied-projection.json --policy greedy --output results/tutorials/embodied.json
python -m arch_explorer evaluate results/tutorials/embodied.json --output results/tutorials/embodied-replayed.json
```

The two inputs only change the workload name and matrix shapes, and results can be checked in the same way.
Choosing different strategies here is only to demonstrate the interface, and cannot be used to compare whether SA or greedy is better: the two workloads are different.

## Common issues

| Symptom | Handling |
| --- | --- |
| `No module named arch_explorer` | Confirm you are using the same virtual environment where the package is installed; install with `python -m pip` |
| Config not found | Run from the source root directory, or pass the absolute path to the config |
| `evaluation_failed` / `invalid_mapping` | First restore the original example, and check trials/attempts, operator and template compatibility in the error |
| Actual call count less than 16 | 16 is the upper limit; duplicate structures are skipped, and max_steps also terminates the search |
| Area/cost is null | Metrics are not covered, so unknown is preserved; cannot change to zero to pass constraints |
| Unexpected results | Check precision, padding, memory/network, mapping, and timing boundaries; do not adjust coefficients first to fit results |

Continue reading: [How to extract workloads from your own models](02-workloads.md).
