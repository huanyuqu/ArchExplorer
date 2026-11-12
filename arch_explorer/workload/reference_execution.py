"""Bounded, deterministic numerical execution of complete ViT/CLIP graphs.

``execute_reference(WorkloadSpec(...))`` returns the graph and every actual
NumPy tensor, including weights and intermediate results. It shares operator
semantics with ``full_primitives.reference``; it does not estimate time, choose
kernels, read pretrained weights, or access a device. The small synthetic
profile is intended for program lowering and numerical checks.

The graph's declared fp16/fp32 storage is preserved at every output boundary.
Matrix/vector arithmetic promotes inputs to fp32 as declared by full_models;
NumPy/libm reduction and transcendental results are a numerical reference,
not a bitwise specification of a GPU implementation.
"""
from __future__ import annotations

from hashlib import sha256, shake_256
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .full_models import SYNTHETIC

if TYPE_CHECKING:
    from arch_explorer.adf.evaluation import WorkloadSpec


REFERENCE_VERSION = "arch-explorer.full-model-reference.v1"
REFERENCE_LIMITS = {
    "max_dimension": 4096,
    "max_batch": 8,
    "max_layers_per_tower": 4,
    "max_width": 64,
    "max_mlp": 256,
    "max_image_size": 32,
    "max_vision_tokens": 65,
    "max_context": 32,
    "max_tensor_elements": 1_000_000,
    "max_total_tensor_bytes": 32 * 1024 * 1024,
    "max_scalar_matmul_operations": 100_000_000,
}
_DTYPES = {"fp16": np.dtype("float16"), "fp32": np.dtype("float32"), "bf16": np.dtype("float32"),
           "int32": np.dtype("int32"), "int64": np.dtype("int64"), "bool": np.dtype("bool")}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _bounded_params(workload):
    """Check dimensions before graph loops, and before any tensor allocation."""
    if workload.kind not in ("vit", "clip"):
        raise ValueError("reference execution supports complete vit/clip only")
    if not isinstance(workload.params, dict):
        raise ValueError("workload params must be a dictionary")
    p = dict(workload.params)
    if p.get("profile") != "synthetic":
        raise ValueError("reference allocation bound: use profile='synthetic'; "
                         "standard profiles are rejected before graph construction")
    overrides = p.get("config", {})
    if not isinstance(overrides, dict) or set(overrides) - set(SYNTHETIC):
        raise ValueError("unknown synthetic config fields")
    c = dict(SYNTHETIC, **overrides)
    if any(type(v) is not int or not 1 <= v <= REFERENCE_LIMITS["max_dimension"]
           for v in c.values()):
        raise ValueError("reference allocation bound: dimensions must be integers in [1,4096]")
    batches = (p.get("batch", 1),) if workload.kind == "vit" else (
        p.get("image_batch", 1), p.get("text_batch", 1))
    if any(type(v) is not int or not 1 <= v <= REFERENCE_LIMITS["max_batch"] for v in batches):
        raise ValueError("reference allocation bound: batch must be an integer in [1,8]")
    if c["image_size"] % c["patch"] or c["width"] % c["heads"]:
        raise ValueError("image/patch and width/heads must divide exactly")
    if workload.kind == "clip" and c["text_width"] % c["text_heads"]:
        raise ValueError("text_width/text_heads must divide exactly")
    checks = {
        "max_layers_per_tower": max(c["layers"], c["text_layers"]),
        "max_width": max(c["width"], c["text_width"]),
        "max_mlp": max(c["mlp"], 4*c["text_width"]),
        "max_image_size": c["image_size"],
        "max_vision_tokens": (c["image_size"]//c["patch"])**2 + 1,
        "max_context": c["context"],
    }
    for key, value in checks.items():
        if value > REFERENCE_LIMITS[key]:
            raise ValueError(f"reference allocation bound: {key}={value} exceeds {REFERENCE_LIMITS[key]}")
    # The full-model builder owns profile/field/dtype/iteration validation.
    return p


def _initial_tensor(tensor, scope, config):
    """SHAKE256 stream bound to model/name/shape/dtype; no global RNG state."""
    binding = _canonical({"version": REFERENCE_VERSION, "scope": scope,
                          "tensor": tensor.name, "shape": list(tensor.shape),
                          "dtype": tensor.dtype})
    size = math.prod(tensor.shape)
    words = np.frombuffer(shake_256(binding).digest(4*size), dtype="<u4")
    if tensor.name == "token_ids":
        vocab = config["vocab"]
        # EOT is the unique highest token at a reproducible, varying position.
        value = (words % max(1, vocab-1)).astype(np.int32).reshape(tensor.shape)
        for row in range(tensor.shape[0]):
            position = (int(words[row*tensor.shape[1]]) + row) % tensor.shape[1]
            value[row, position] = vocab-1
        return value
    value = (words >> 8).astype(np.float32) * np.float32(2.0**-24) - np.float32(.5)
    if tensor.name.endswith(".gamma"):
        value = np.float32(1.) + value * np.float32(.1)
    elif tensor.kind == "weight":
        # Keep dense activations bounded without zero/identity weight stubs.
        fan_in = tensor.shape[-2] if len(tensor.shape) >= 2 else 1
        value = value * np.float32(1. / math.sqrt(fan_in))
    value = value.reshape(tensor.shape)
    if tensor.dtype in ('fp32', 'fp16', 'bf16'):
        from .precision import round_storage
        return round_storage(value, tensor.dtype)
    return value.astype(_DTYPES[tensor.dtype])


def _array_facts(value, tensor):
    # Canonical digest ignores host endianness, strides, and memory addresses.
    raw = np.ascontiguousarray(value, dtype=_DTYPES[tensor.dtype].newbyteorder("<"))
    if tensor.dtype == 'bf16':
        from .precision import storage_bytes
        payload = storage_bytes(value, tensor.dtype)
    else:
        payload = raw.tobytes()
    floating = np.issubdtype(value.dtype, np.floating)
    return {"shape": list(value.shape), "dtype": tensor.dtype,
            "numpy_dtype": value.dtype.name, "elements": int(value.size),
            "bytes": tensor.bytes, "sha256": sha256(payload).hexdigest(),
            "digest_encoding": "C-order little-endian BF16 16bit words" if tensor.dtype == 'bf16' else "C-order little-endian storage bytes",
            **({'host_container_bytes':int(value.nbytes)} if tensor.dtype == 'bf16' else {}),
            "strides_elements": [s//value.itemsize for s in value.strides],
            "alias_of": tensor.alias_of,
            "finite_elements": int(np.isfinite(value).sum()) if floating else int(value.size),
            "negative_infinity_elements": int(np.isneginf(value).sum()) if floating else 0}


def _check_values(op, value, inputs):
    if not np.issubdtype(value.dtype, np.floating):
        return
    if np.isnan(value).any() or np.isposinf(value).any():
        raise ValueError(f"nonfinite numerical result at {op.name}")
    negative = np.isneginf(value)
    if op.op == "causal_mask":
        expected = np.triu(np.ones(value.shape, dtype=bool), 1)
        if not np.array_equal(negative, expected) or np.any(value[~expected] != 0):
            raise ValueError("invalid causal mask sentinel")
    elif op.op == "mask":
        if len(inputs) != 2 or not np.isfinite(inputs[0]).all():
            raise ValueError(f"invalid mask inputs at {op.name}")
        if not np.array_equal(negative, np.broadcast_to(np.isneginf(inputs[1]), value.shape)):
            raise ValueError(f"invalid masked-score sentinel at {op.name}")
    elif negative.any():
        raise ValueError(f"unexpected negative infinity at {op.name}")


def _builtin_preflight(workload):
    _bounded_params(workload)


def _builtin_initialize(workload, graph, external):
    """Model-specific synthetic initialization; selected through ReferenceDefinition."""
    meta = graph.meta['full_model']
    scope = {'kind':workload.kind,'name':workload.name,'config':meta['config'],
             'storage':graph.dtype,
             'external_shapes':{n:list(graph.tensors[n].shape) for n in external}}
    inputs = {n:_initial_tensor(graph.tensors[n],scope,meta['config']) for n in external}
    return {'inputs':inputs,'outputs':list(meta['outputs']),
            'metadata':{'profile':meta['profile'],'config':dict(meta['config']),
                        'initialization':{'algorithm':'SHAKE256 name/shape/dtype/model-bound stream; uint32 high24 -> fp32',
                                          'scope_sha256':sha256(_canonical(scope)).hexdigest(),
                                          'weights':'synthetic bounded dense values; no pretrained data'}}}


def execute_reference(workload, *, inputs=None):
    """Compatibility public entry point; same registered graph reference mechanism."""
    from .reference_api import execute_reference as execute
    return execute(workload,inputs=inputs)
