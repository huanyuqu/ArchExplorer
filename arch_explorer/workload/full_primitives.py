"""Explicit inference primitives for full models, independent of device rates.

Counts describe a straightforward FP32 vector implementation. SFU counts are
abstract function results, NOT calibrated CUDA instruction counts (erf in
particular is not a single hardware instruction). See full_model.timing_gaps.
Data rearrangements are materialized unless explicitly named view.
"""
from dataclasses import dataclass
import math

from arch_explorer.components.base import Kernel
from arch_explorer.workload.operators import Operator
from .precision import bf16 as _bf16

# Per element SIMT/SFU and per row SIMT/SFU, from the formulas below.
# Reductions use conservative N rather than N-1 adds/comparisons.
COSTS = {
    "cast": (1,0,0,0), "reshape": (0,0,0,0), "concat": (0,0,0,0), "mul": (1,0,0,0),
    "rms": (3,0,2,1), "rms_scale": (2,0,0,0),
    "ada_affine": (3,0,0,0), "gated_add": (2,0,0,0),
    "rope": (5,2,0,0), "sincos": (3,2,0,0),
    "silu": (2,2,0,0), "gelu_tanh": (8,1,0,0),
    "positions": (2,0,0,0), "suffix_positions": (1,0,0,0),
    "prefix_mask": (1,0,0,0), "suffix_mask": (1,0,0,0),
    "attention_mask": (1,0,0,0), "flow": (2,0,0,0), "scale_add": (1,0,0,0),
    "copy": (0, 0, 0, 0), "patchify": (0, 0, 0, 0),
    "split_heads": (0, 0, 0, 0), "merge_heads": (0, 0, 0, 0),
    "transpose": (0, 0, 0, 0), "cls_concat": (0, 0, 0, 0),
    "lookup": (1, 0, 0, 0), "select": (1, 0, 0, 0),
    "argmax": (1, 0, 0, 0), "causal_mask": (1, 0, 0, 0),
    "bias": (1, 0, 0, 0), "add": (1, 0, 0, 0),
    "scale": (1, 0, 0, 0), "mask": (1, 0, 0, 0),
    "softmax": (4, 1, 0, 1),
    # mean, center, square, sum, normalize, gamma multiply, beta add;
    # mean and variance reciprocals multiply + epsilon, rsqrt.
    "affine_norm": (7, 0, 3, 1),
    "gelu_exact": (4, 1, 0, 0),  # x*.5*(1+erf(x/sqrt(2)))
    "quick_gelu": (3, 2, 0, 0), # 1.702*x, exp(-z), 1+exp, rcp, x*
    "l2_norm": (3, 0, 0, 1),    # square, sum, multiply rsqrt(sum)
    "exp": (0, 1, 0, 0),
}
ARITHMETIC = {"bias", "add", "scale", "mask", "softmax", "affine_norm",
              "gelu_exact", "quick_gelu", "l2_norm", "exp"}


@dataclass(frozen=True)
class FullOperator(Operator):
    def kernel(self, dtype="fp16", rows=None, tile=None):
        from .extensions import _record_source
        _record_source(FullOperator.kernel)
        if self.op == "view":
            raise NotImplementedError("view aliases storage and has no compute kernel")
        simt, sfu, row_simt, row_sfu = COSTS[self.op]
        # Floating vector operands promoted to FP32, final stored result cast.
        casts = self.attrs.get("casts_per_element", 0)
        return Kernel("vector", (rows or self.attrs["rows"], self.attrs["cols"]),
                      "fp32", simt + casts, sfu, row_simt + self.attrs.get("extra_row_simt", 0), row_sfu)


def _reference(op, values):
    """Small-tensor semantic oracle. No weights, framework, or timing fit needed."""
    import numpy as np
    a = op.attrs
    x = values[0] if values else None
    if op.op == "view":
        return x.transpose(a["axes"]) if a.get("axes") is not None else x.reshape(a["shape"])
    if op.op == "transpose":
        return x.transpose(a["axes"]).copy()
    if op.op == "matmul":
        b = a.get('batch', 1)
        m,k,n = (a[key] for key in ("m","k","n"))
        left=np.asarray(values[0],dtype=np.float32).reshape(b,m,k)
        right=np.asarray(values[1],dtype=np.float32).reshape(-1,k,n)
        indices=np.zeros(b,dtype=int) if a.get("shared_b") else np.arange(b)//a.get("kv_group",1)
        from .precision import matmul_fp32
        return matmul_fp32(left, right[indices])
    if op.op == "cast":
        return x
    if op.op == "concat":
        return np.concatenate(values, axis=a["axis"])
    if op.op in ("copy", "reshape"):
        layout = a.get("layout")
        if layout == "camera mask broadcast":
            return np.broadcast_to(x[:,None],a["shape"]).copy()
        if layout == "BTHD->BHTD":
            return x.transpose(0,2,1,3).reshape(a["shape"]).copy()
        if layout == "BTHD->BHDT":
            return x.transpose(0,2,3,1).reshape(a["shape"]).copy()
        if layout == "NHWC patches stride=patch VALID":
            b,h,w,c = x.shape
            patch = math.isqrt(a["shape"][-1]//c)
            return x.reshape(b,h//patch,patch,w//patch,patch,c).transpose(0,1,3,2,4,5).reshape(a["shape"]).copy()
        return x.copy().reshape(a.get("shape", x.shape))
    if op.op == "patchify":
        b, c, h, w = x.shape
        p = a["patch"]
        return x.reshape(b, c, h//p, p, w//p, p).transpose(0, 2, 4, 1, 3, 5).reshape(b, -1, c*p*p)
    if op.op == "split_heads":
        b, s, h = x.shape
        y = x.reshape(b, s, a["heads"], h//a["heads"]).transpose(0, 2, 1, 3)
        if a.get("key"):
            y = y.transpose(0, 1, 3, 2)
        return y.reshape(a["shape"]).copy()
    if op.op == "merge_heads":
        b, s, h = a["shape"]
        return x.reshape(b, a["heads"], s, h//a["heads"]).transpose(0, 2, 1, 3).reshape(b, s, h).copy()
    if op.op == "cls_concat":
        return np.concatenate((np.broadcast_to(values[1], (x.shape[0], 1, x.shape[-1])), x), axis=1)
    if op.op == "lookup":
        return values[1][x]
    if op.op == "select" and "segment" in a:
        width = a["shape"][-1]; start = a["segment"]*width
        return x[...,start:start+width].reshape(a["shape"])
    if op.op == "positions":
        return np.cumsum(x, axis=-1, dtype=np.int32)-1
    if op.op == "suffix_positions":
        return x.sum(-1, keepdims=True, dtype=np.int32)+np.arange(a["shape"][-1], dtype=np.int32)
    if op.op == "prefix_mask":
        return x[:,:,None].astype(bool) & x[:,None,:].astype(bool)
    if op.op == "suffix_mask":
        h=a["shape"][1]
        return np.broadcast_to(np.concatenate((x.astype(bool),np.ones((len(x),h),bool)),axis=-1)[:,None,:],a["shape"])
    if op.op == "attention_mask":
        mask=np.repeat(values[1],a["heads"],axis=0)
        return np.where(mask,x,np.float32(a["big_neg"]))
    if op.op == "select":
        idx = values[1] if len(values) > 1 else np.zeros(x.shape[0], dtype=int)
        return x[np.arange(x.shape[0]), idx]
    if op.op == "argmax":
        return x.argmax(axis=-1).astype(np.int32)
    if op.op == "causal_mask":
        return np.where(np.triu(np.ones(a["shape"], dtype=bool), 1), -np.inf, 0.0)
    x = x.astype(np.float32)
    if op.op in ("bias", "add", "mask"):
        return x + values[1].astype(np.float32)
    if op.op == "scale":
        if 'divisor' in a:
            divisor = _bf16(a['divisor']) if a.get('divisor_dtype') == 'bf16' else a['divisor']
            return x / divisor
        return x * (values[1] if len(values) > 1 else a["factor"])
    if op.op == "rms":
        return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+a["epsilon"])
    if op.op == "rms_scale":
        return x*(1+values[1].astype(np.float32))
    if op.op == "ada_affine":
        return x*(1+values[1].astype(np.float32))+values[2].astype(np.float32)
    if op.op == "gated_add":
        return x+values[1]*values[2]
    if op.op == "mul":
        return x*values[1]
    if op.op == "flow":
        return x+np.float32(a["dt"])*values[1]
    if op.op == "scale_add":
        return x+np.float32(a["value"])
    if op.op == "silu":
        return x/(1+np.exp(-x))
    if op.op == "gelu_tanh":
        cube = np.float32(np.float32(x*x)*x)
        inner = np.float32(x + np.float32(.044715)*cube)
        tanh = np.tanh(np.float32(math.sqrt(2/math.pi))*inner)
        return np.float32(np.float32(.5)*x*np.float32(np.float32(1)+tanh))
    if op.op == "sincos":
        fraction=np.linspace(0,1,a["shape"][-1]//2,dtype=np.float32)
        period=a["min_period"]*(a["max_period"]/a["min_period"])**fraction
        angle=x[:,None]*(2*np.pi/period)
        return np.concatenate((np.sin(angle),np.cos(angle)),axis=-1)
    if op.op == "rope":
        d=x.shape[-1]; freq=a["max_wavelength"]**(np.arange(d//2,dtype=np.float32)*2/d)
        angle=values[1][:,:,None,None]/freq
        left,right=np.split(x,2,axis=-1)
        return np.concatenate((left*np.cos(angle)-right*np.sin(angle),right*np.cos(angle)+left*np.sin(angle)),axis=-1)
    if op.op == "affine_norm":
        centered = x - x.mean(axis=-1, keepdims=True)
        return centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + a["epsilon"]) * values[1] + values[2]
    if op.op == "softmax":
        if a.get('intermediate_dtype') == 'bf16':
            # JAX BF16 softmax: max/sub/exp in BF16, sum promoted to FP32,
            # row sum cast back to BF16, then BF16 division/result.
            x = _bf16(x)
            z = _bf16(np.exp(_bf16(x - x.max(axis=-1, keepdims=True))))
            return z / _bf16(z.sum(axis=-1, keepdims=True, dtype=np.float32))
        z = np.exp(x - x.max(axis=-1, keepdims=True))
        return z / z.sum(axis=-1, keepdims=True)
    if op.op == "gelu_exact":
        return x * .5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    if op.op == "quick_gelu":
        return x / (1 + np.exp(-1.702*x))
    if op.op == "l2_norm":
        return x / np.sqrt(np.sum(x*x, axis=-1, keepdims=True))
    if op.op == "exp":
        return np.exp(x)
    raise NotImplementedError(op.op)


def reference(op, values):
    """Small numerical lowering oracle; full timing evaluation does not run it.

    BF16 uses round-to-nearest-even on FP32 (returned in an FP32 container).
    Existing full-model operators without result_dtype preserve their oracle.
    """
    import numpy as np
    value = _reference(op, values)
    dtype = op.attrs.get("result_dtype", op.attrs.get("output_dtype"))
    if dtype == "bf16":
        return _bf16(value)
    if dtype:
        return np.asarray(value, dtype={"fp32":np.float32,"fp16":np.float16,"int32":np.int32,"int8":np.int8}[dtype])
    return value
