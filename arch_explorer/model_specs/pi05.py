"""Frozen openpi JAX pi05 input and shape contract; not an executor."""
from copy import deepcopy

CONTRACT_ID = "openpi.pi05.v1"
SOURCE_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
SOURCE_URL = "https://github.com/Physical-Intelligence/openpi/tree/" + SOURCE_COMMIT
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DEFAULTS = {"profile": "openpi_pi05", "dtype": "bfloat16", "batch": 1,
            "action_dim": 32, "action_horizon": 50, "max_token_len": 200,
            "num_steps": 10, "iterations": 1, "noise_seed": 17}
# These are source architecture declarations, not hardware presets.
PROFILES = {
    "openpi_pi05": {"prefix": {"variant": "gemma_2b", "width": 2048, "mlp_dim": 16384},
                   "expert": {"variant": "gemma_300m", "width": 1024, "mlp_dim": 4096},
                   "layers": 18, "heads": 8, "kv_heads": 1, "head_dim": 256},
    "openpi_pi05_dummy": {"prefix": {"variant": "dummy", "width": 64, "mlp_dim": 128},
                         "expert": {"variant": "dummy", "width": 64, "mlp_dim": 128},
                         "layers": 4, "heads": 8, "kv_heads": 1, "head_dim": 16},
}
VISION = {"variant": "So400m/14", "width": 1152, "layers": 27, "mlp_dim": 4304,
          "heads": 16, "head_dim": 72, "patch": 14, "tokens": 256,
          "pool": "none", "position": "learned", "image_resolution": [224, 224]}


def _mask(value, rows, columns, name):
    if (not isinstance(value, list) or len(value) != rows
            or any(not isinstance(row, list) or len(row) != columns
                   or any(type(v) is not bool for v in row) for row in value)):
        raise ValueError(f"{name} must be a bool array of shape [{rows},{columns}]")


def normalize(params):
    if not isinstance(params, dict):
        raise ValueError("pi05 params must be an object")
    unknown = params.keys() - (DEFAULTS.keys() | {"image_masks", "token_mask"})
    if unknown:
        raise ValueError(f"Unknown pi05 fields: {sorted(unknown)}")
    p = {**deepcopy(DEFAULTS), **deepcopy(params)}
    if not isinstance(p["profile"], str) or p["profile"] not in PROFILES:
        raise ValueError("Unknown pi05 profile")
    if p["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("pi05 dtype must be bfloat16 or float32")
    for key in ("batch", "action_dim", "action_horizon", "max_token_len", "num_steps", "iterations"):
        if type(p[key]) is not int or p[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if p["max_token_len"] > 200:
        raise ValueError("This pi05 input contract supports max_token_len <= 200")
    if type(p["noise_seed"]) is not int or not 0 <= p["noise_seed"] < 2**32:
        raise ValueError("noise_seed must be uint32")
    b, t = p["batch"], p["max_token_len"]
    p.setdefault("image_masks", [[True] * b for _ in CAMERAS])
    p.setdefault("token_mask", [[True] * t for _ in range(b)])
    _mask(p["image_masks"], 3, b, "image_masks")
    _mask(p["token_mask"], b, t, "token_mask")
    return p


def expand(params):
    """Return source dimensions, logical storage and shapes, never a task graph."""
    p = normalize(params)
    profile = deepcopy(PROFILES[p["profile"]])
    b, h, a, t = (p[k] for k in ("batch", "action_horizon", "action_dim", "max_token_len"))
    length = 3 * VISION["tokens"] + t
    layers, heads, kv, d = (profile[k] for k in ("layers", "heads", "kv_heads", "head_dim"))
    dp, de = profile["prefix"]["width"], profile["expert"]["width"]
    elem_bytes = 2 if p["dtype"] == "bfloat16" else 4
    prefix_valid = [sum(p["image_masks"][c][i] for c in range(3))*256 + sum(p["token_mask"][i]) for i in range(b)]
    # Logical parameter elements, including biases; physical residency and storage dtype unknown.
    v, m = VISION["width"], VISION["mlp_dim"]
    weights = {
        "vision.shared": 14*14*3*v+v + 256*v + 27*(4*v*v+4*v+4*v+2*v*m+m+v) + 2*v + v*dp+dp,
        "prefix.embedding": 257152*dp,
        "prefix.layers": layers*(2*dp*heads*d+2*dp*kv*d+3*dp*profile["prefix"]["mlp_dim"]+2*dp),
        "prefix.final_norm": dp,
        "expert.layers": layers*(2*de*heads*d+2*de*kv*d+3*de*profile["expert"]["mlp_dim"]+2*(3*de*de+3*de)),
        "expert.final_norm": 3*de*de+3*de,
        "action_in": a*de+de, "time_mlp": 2*(de*de+de), "action_out": de*a+a,
    }
    shapes = {
        "images.each": [b,224,224,3], "vision.patch": [b,16,16,v],
        "vision.tokens.each": [b,256,v], "vision.qkv.each": [3,b,256,16,72],
        "vision.logits.each": [b,16,256,256], "vision.mlp.each": [b,256,m],
        "vision.projected.each": [b,256,dp], "text.ids": [b,t],
        "text.embedded": [b,t,dp], "state.external": [b,a],
        "prefix.tokens": [b,length,dp], "prefix.q": [b,length,heads,d],
        "prefix.kv.each": [b,length,kv,d], "prefix.logits": [b,heads,length,length],
        "prefix.mlp": [b,length,profile["prefix"]["mlp_dim"]],
        "prefix.cache.each": [layers,b,length,kv,d],
        "actions": [b,h,a], "time.sincos": [b,de], "time.condition": [b,de],
        "expert.tokens": [b,h,de], "expert.modulation": [b,3*de],
        "expert.q": [b,h,heads,d], "expert.kv.each": [b,h,kv,d],
        "expert.combined_kv.each": [b,length+h,kv,d],
        "expert.logits": [b,heads,h,length+h],
        "expert.mlp": [b,h,profile["expert"]["mlp_dim"]],
        "output.action_chunk": [b,h,a],
    }
    cache_bytes = 2*layers*b*length*kv*d*elem_bytes
    return {
        "contract": CONTRACT_ID, "source_commit": SOURCE_COMMIT, "source_url": SOURCE_URL,
        "params": p, "vision": deepcopy(VISION), "language": profile, "vocabulary": 257152,
        "shapes": shapes, "prefix_tokens": length, "valid_prefix_tokens": prefix_valid,
        "counts_per_iteration": {"vision_calls": 3, "prefix_passes": 1,
                                "suffix_passes": p["num_steps"], "flow_updates": p["num_steps"]},
        "mask": {"prefix": "valid(query) AND valid(key); fully bidirectional",
                 "prefix_positions": "cumsum(valid)-1",
                 "suffix": "all H suffix queries attend valid prefix keys and all H suffix keys",
                 "suffix_positions": "sum(prefix_valid)+arange(H)",
                 "masked_images_encoded": True, "padding_computed": True},
        "storage": {"prefix_kv_bytes": cache_bytes, "logical_prefix_kv_read_bytes_per_step": cache_bytes,
                    "suffix_kv_bytes_per_layer": 2*b*h*kv*d*elem_bytes,
                    "action_buffer_bytes": b*h*a*4, "weight_elements": weights,
                    "weight_storage_bytes": None,
                    "weight_storage_reason": "checkpoint dtype/residency unknown; elements times per-object storage bytes",
                    "peak_device_bytes": None,
                    "objects": {"prefix.K/V": "one allocation per iteration, immutable across steps; free after output",
                                "suffix.K/V": "per step/layer temporary; concatenated return cache discarded",
                                "weights": "one identity per named object, shared across cameras/steps/iterations",
                                "actions.x": "FP32 loop carry, updated each step"}},
        "precision": {"activation": p["dtype"], "cache": p["dtype"], "images_stem_position": "float32",
                      "gemma_rms_rope": "float32 intermediates then cast back",
                      "gemma_qk_logits": "float32 preferred_element_type; softmax cast to activation",
                      "gemma_other_matmul_accumulation": "backend/compiler dependent, unverified",
                      "action_time_projections": "source NNX Linear defaults float32; checkpoint dependent",
                      "weights": "source initializer float32; checkpoint storage unverified",
                      "siglip_norm_attention_accumulation": "Flax/XLA dependent, unverified"},
        "sampling": {"dt": "-1/num_steps", "initial_time": 1, "condition": "time >= -dt/2",
                     "update": "x_t + dt*v_t", "noise": "external JAX0.5.3 PRNGKey(noise_seed), normal FP32",
                     "noise_shape": [b,h,a], "iterations": "independent full requests; never repeat only suffix"},
        "boundary": {"start": "prepared RGB FP32, token IDs/masks and external noise ready",
                     "end": "complete FP32 action chunk",
                     "excluded": ["resize/normalize", "tokenizer", "state discretization", "noise generation",
                                  "robot I/O", "action application", "training"]},
        "unverified": ["checkpoint weights", "compiled fusion/dead auxiliary outputs",
                       "physical memory peak", "device precision support", "numerical full-model outputs", "latency"],
    }
