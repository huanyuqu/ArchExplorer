"""pi05 inputs on explicit synthetic non-GPU systolic/FP32-vector hardware.
From export/sdist root with the wheel installed: python tools/generate_pi05.py OUTPUT
See docs/non-gpu-generators.md for mixed-precision and execution boundaries.
"""
import argparse
import json
from pathlib import Path
from arch_explorer.model_specs import normalize
from arch_explorer.archs.adf_seeds import non_gpu_adf


def configuration(**params):
    normalized = normalize(params)
    adf = non_gpu_adf(matrix_precision={"bfloat16": "bf16", "float32": "fp32"}[normalized["dtype"]])
    return {"seed_adf": adf.to_dict(),
            "workloads": [{"kind": "pi05", "params": normalized}],
            "backend": "events", "mapping_templates": ["gsm_tiled"],
            "max_evaluations": 4, "parameter_samples": 1, "mapping_samples": 1, "random_seed": 17}


def cases():
    return {
        "three-cameras.json": configuration(),
        "one-camera.json": configuration(image_masks=[[True], [False], [False]], num_steps=1),
        "two-cameras-tail.json": configuration(batch=2, image_masks=[[True,True],[True,False],[False,True]],
            action_horizon=15, max_token_len=7, token_mask=[[True]*5+[False]*2,[True]*6+[False]],
            num_steps=2),
        "boundary.json": configuration(profile="openpi_pi05_dummy", action_dim=1, action_horizon=1,
            max_token_len=1, token_mask=[[False]], image_masks=[[False],[False],[False]], num_steps=1,
            dtype="float32", iterations=2, noise_seed=0),
        "tail.json": configuration(action_horizon=7, max_token_len=199, num_steps=10),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, cfg in cases().items():
        (args.output/name).write_text(json.dumps(cfg, indent=2)+"\n")
