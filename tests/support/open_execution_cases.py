"""Frozen QA inputs and answers, not a proposed production API.

Target adapters must read actual public plans/results. These cases do not
implement registration, target selection, lowering, or a second scheduler.
"""
SHIFT_SQUARE = {
    "semantic": "qa50.shift_square",
    "formula": "y = (x + bias) ** 2",
    "dtype": "fp32",
    "shape": [2, 4],
    "input": [[-2.0, -1.0, -0.5, 0.0], [0.25, 0.5, 1.0, 2.0]],
    "bias": {"left": 0.5, "right": -0.25},
    "expected_left": [[2.25, 0.25, 0.0, 0.25], [0.5625, 1.0, 2.25, 6.25]],
    "expected_right": [[5.0625, 1.5625, 0.5625, 0.0625], [0.0, 0.0625, 0.5625, 3.0625]],
    "expected_join": [[7.3125, 1.8125, 0.5625, 0.3125], [0.5625, 1.0625, 2.8125, 9.3125]],
    "tensor_bytes": 32,
    "target_cases": {
        "atomic": {"branch_steps": 1, "intermediate_tensors": 0},
        "decomposed": {"branch_steps": 2, "intermediate_tensors": 2,
                       "intermediate_bytes_each": 32},
    },
    # Logical, materialized implementation IO, not predicted DRAM/link traffic
    # or a claim that either instruction stream has run on the target yet.
    "logical_branch_and_join_io_bytes": {"atomic": 224, "decomposed": 352},
    "scope": "exact dyadic numerical answers; no hardware latency prediction",
}
