"""Generate SYNTHETIC non-GPU systolic/vector multi-device entry fixtures.

With ArchExplorer installed, from a source export or extracted sdist root:
python tools/generate_multi_device.py OUTPUT_DIRECTORY
See docs/non-gpu-generators.md; no real device or measured-link claim.
"""
import json
from pathlib import Path

from arch_explorer.adf import WorkloadSpec
from arch_explorer.archs.adf_seeds import non_gpu_adf
from arch_explorer.devices.topology import compose, link_resource
from arch_explorer.adf.multi_device import validate_request
from arch_explorer.workload.llm import workload_from_spec

SOURCE = "synthetic non-GPU multi-device entry fixture; not measured hardware"


def configuration(names=("north", "south"), workload=None):
    if (not isinstance(names, (tuple, list)) or not names
            or any(type(name) is not str or not name for name in names) or len(set(names)) != len(names)):
        raise ValueError("names must be a nonempty sequence of unique device names")
    payload = workload if workload is not None else {"kind": "vit", "params": {
        "profile": "vit_b16_224", "dtype": "fp16", "batch": 1, "iterations": 1}}
    spec = WorkloadSpec(**payload)
    ir = workload_from_spec(spec)
    # Adjacent edges only; reverse bandwidth is deliberately different.
    # Count, topology, names and capacities are not inferred from a family.
    networks, edges = [], []
    for i in range(len(names) - 1):
        for src, dst, bandwidth in ((i, i + 1, 25.0), (i + 1, i, 7.0)):
            name = f"edge_{src}_{dst}"
            networks.append(link_resource(name, bandwidth, 3.0, source=SOURCE))
            edges.append({"src": f"system/{names[src]}", "dst": f"system/{names[dst]}",
                          "network": f"system/{name}"})
    adf = compose("system", {name: non_gpu_adf(matrix_precision=ir.dtype) for name in names}, networks)
    params = {"device_paths": [f"system/{name}" for name in names],
              "placements": {"*": {"ranks": list(range(len(names))),
                  "mapping": {"template": "gsm_tiled", "params": {
                      "tile_m": 64, "tile_n": 64, "rows_per_cta": 1}}}},
              "shards": [], "collectives": [], "topology_edges": edges,
              "routing": "shortest_hops"}
    validate_request(adf, params)
    return {"seed_adf": adf.to_dict(),
            "workloads": [spec.to_dict()],
            "mapping_templates": ["multi_device"], "mapping_selectors": {"multi_device": params},
            "backend": "events", "max_evaluations": 4, "parameter_samples": 1, "mapping_samples": 1,
            "random_seed": 17, "area_budget_mm2": 1000, "cost_budget_usd": 10000,
            "parameter_domains": [{"path": f"system/{networks[0].name}", "param": "bandwidth",
                                   "values": [7.0, 25.0, 50.0], "source": SOURCE}] if networks else []}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configs = {"two-device.json": configuration(),
               "four-device.json": configuration(("alpha", "beta", "gamma", "delta"),
                   {"kind": "clip", "params": {"profile": "clip_vit_b32", "dtype": "fp16",
                                               "image_batch": 1, "text_batch": 1, "iterations": 1}})}
    for name, config in configs.items():
        (args.output / name).write_text(json.dumps(config, indent=2) + "\n")
