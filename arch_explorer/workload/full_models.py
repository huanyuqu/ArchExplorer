"""Complete eval-mode ViT classifier and original OpenAI CLIP ViT graphs.

No weights are downloaded. Source files are identified by content hash; these
graphs specify arithmetic and placement, not a claim about a framework's
selected fused kernels or measured hardware accuracy.
"""
from collections import Counter
from dataclasses import replace
import math

from arch_explorer.components.base import dtype_bytes
from arch_explorer.workload.operators import Workload
from arch_explorer.workload.full_primitives import FullOperator, COSTS, ARITHMETIC

SOURCES = [
    {"url": "https://huggingface.co/google/vit-base-patch16-224/resolve/main/config.json",
     "sha256": "abdf73cc4ce79495a83f32a8c9265daa7a1363f5a3efa4704c88a6e7f45ad19d"},
    {"url": "https://raw.githubusercontent.com/huggingface/transformers/v4.13.0/src/transformers/models/vit/modeling_vit.py",
     "sha256": "14818ee658fb1837468a6127ca761df26ab7bf966ab670eccdee4c0bfbd88ced"},
    {"url": "https://raw.githubusercontent.com/openai/CLIP/main/clip/model.py",
     "sha256": "9902cbe5ee90a1da2aa3e6f043e8a23dc1f8831193b963785c9af03d5c7bef2c"},
    {"url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json",
     "sha256": "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770"},
]
PROFILES = {
    "vit_b16_224": dict(image_size=224, patch=16, width=768, layers=12, heads=12, mlp=3072, classes=1000),
    "clip_vit_b32": dict(image_size=224, patch=32, width=768, layers=12, heads=12, mlp=3072,
                         context=77, vocab=49408, text_width=512, text_heads=8, text_layers=12, projection=512),
}
SYNTHETIC = dict(image_size=8, patch=4, width=8, layers=2, heads=2, mlp=16,
                 classes=3, context=5, vocab=17, text_width=8, text_heads=2, text_layers=2, projection=4)


class Graph:
    def __init__(self, name, dtype):
        self.w = Workload(name, dtype=dtype)

    def weight(self, name, shape, dtype=None):
        return self.w.tensor(name, shape, "weight", dtype)

    def primitive(self, name, op, inputs, shape, dtype=None, **attrs):
        if op not in COSTS:
            raise ValueError(f"Unknown full-model primitive {op}")
        out = self.w.tensor(name, shape, dtype=dtype)
        tensors = [self.w.tensors[t] for t in inputs]
        size = math.prod(shape)
        cols = shape[-1]
        rows = size // cols
        read = sum(t.bytes for t in tensors)
        if op in ("bias", "add", "mask", "scale", "affine_norm"):
            # Direct DRAM implementation: broadcast values are read at each
            # use, not fractionally shared across units without a cache.
            for t in tensors:
                aligned = (1,)*(len(shape)-len(t.shape)) + t.shape
                if len(aligned) != len(shape) or any(a not in (1, b) for a, b in zip(aligned, shape)):
                    raise ValueError(f"{name}: invalid broadcast {t.shape} -> {shape}")
            read = sum(size*dtype_bytes(t.dtype) for t in tensors)
        if op == "cls_concat":
            read = tensors[0].bytes + shape[0]*shape[-1]*dtype_bytes(tensors[1].dtype)
        if op == "lookup":
            read = tensors[0].bytes + self.w.tensors[out].bytes
        elif op == "select":
            read = self.w.tensors[out].bytes + sum(t.bytes for t in tensors[1:])
        elif op == "argmax":
            rows, cols = tensors[0].shape
        # FP32 reductions have explicit global scratch: promoted input,
        # result and row statistics. All scratch reads/writes are charged.
        scratch = 0
        if op in ("affine_norm", "softmax", "l2_norm"):
            scratch = 4 * (2 * size + 3 * rows)
        casts = 0
        if op in ARITHMETIC and self.w.dtype == "fp16":
            casts = len(inputs) + 1
        attributes = dict(shape=tuple(shape), rows=rows, cols=cols, read_bytes=read,
                          write_bytes=self.w.tensors[out].bytes, scratch_bytes=scratch,
                          casts_per_element=casts, **attrs)
        self.w.operators.append(FullOperator(name, op, tuple(inputs), (out,), attributes, name))
        return out

    def view(self, name, x, shape, axes=None):
        original = self.w.tensors[x]
        if math.prod(shape) != math.prod(original.shape):
            raise ValueError("view must preserve element count")
        if axes is None and original.strides is not None:
            raise ValueError("reshape view requires contiguous input")
        strides = None
        if axes is not None:
            if sorted(axes) != list(range(len(original.shape))) or tuple(original.shape[i] for i in axes) != tuple(shape):
                raise ValueError("transpose view axes/shape mismatch")
            base = original.strides or tuple(math.prod(original.shape[i+1:]) for i in range(len(original.shape)))
            strides = tuple(base[i] for i in axes)
        out = self.w.tensor(name, shape, dtype=original.dtype)
        self.w.tensors[out] = replace(self.w.tensors[out], alias_of=x, strides=strides)
        self.w.operators.append(FullOperator(name, "view", (x,), (out,),
                                            dict(shape=tuple(shape), axes=axes), name))
        return out

    def mm(self, name, x, y, batch, m, k, n, shared=False):
        expected_a = batch*m*k
        expected_b = (1 if shared else batch)*k*n
        if math.prod(self.w.tensors[x].shape) != expected_a or math.prod(self.w.tensors[y].shape) != expected_b:
            raise ValueError(f"{name}: incompatible matmul shapes")
        out = self.w.tensor(name, (batch, m, n))
        self.w.add(name, "matmul", [x, y], [out], label=name,
                   batch=batch, m=m, k=k, n=n, shared_b=shared)
        return out

    def linear(self, name, x, n, bias=True):
        shape = self.w.tensors[x].shape
        b, m, k = (1, *shape) if len(shape) == 2 else shape
        y = self.mm(name+".mm", x, self.weight(name+".weight", (k, n)), b, m, k, n, True)
        if len(shape) == 2:
            y = self.view(name+".reshape", y, (m, n))
        if bias:
            y = self.primitive(name+".bias", "bias", [y, self.weight(name+".bias_weight", (n,))],
                               self.w.tensors[y].shape)
        return y

    def norm(self, name, x, eps):
        shape = self.w.tensors[x].shape
        return self.primitive(name, "affine_norm", [x, self.weight(name+".gamma", (shape[-1],)),
                              self.weight(name+".beta", (shape[-1],))], shape, epsilon=eps)

    def block(self, name, x, heads, mlp, eps, activation, mask=None):
        b, s, h = self.w.tensors[x].shape
        d = h//heads
        z = self.norm(name+".ln1", x, eps)
        qkv = []
        for key in ("q", "k", "v"):
            proj = self.linear(name+"."+key, z, h)
            shape = (b*heads, d, s) if key == "k" else (b*heads, s, d)
            qkv.append(self.primitive(name+"."+key+".heads", "split_heads", [proj],
                                       shape, heads=heads, key=key == "k"))
        score = self.mm(name+".qk", qkv[0], qkv[1], b*heads, s, d, s)
        score = self.primitive(name+".scale", "scale", [score], (b*heads, s, s), factor=d**-.5)
        if mask:
            score = self.primitive(name+".mask", "mask", [score, mask], (b*heads, s, s))
        prob = self.primitive(name+".softmax", "softmax", [score], (b*heads, s, s))
        av = self.mm(name+".av", prob, qkv[2], b*heads, s, s, d)
        merged = self.primitive(name+".merge", "merge_heads", [av], (b, s, h), heads=heads)
        projected = self.linear(name+".out", merged, h)
        residual = self.primitive(name+".residual1", "add", [x, projected], (b, s, h))
        up = self.linear(name+".up", self.norm(name+".ln2", residual, eps), mlp)
        act = self.primitive(name+"."+activation, activation, [up], (b, s, mlp))
        down = self.linear(name+".down", act, h)
        return self.primitive(name+".residual2", "add", [residual, down], (b, s, h))

    def vision(self, c, batch, clip):
        r, p, h = c["image_size"], c["patch"], c["width"]
        patches = (r//p)**2
        image = self.w.tensor("images", (batch, 3, r, r))
        packed = self.primitive("vision.patchify", "patchify", [image], (batch, patches, 3*p*p), patch=p)
        x = self.linear("vision.patch_projection", packed, h, bias=not clip)
        x = self.primitive("vision.cls", "cls_concat", [x, self.weight("vision.cls_weight", (1, 1, h))],
                           (batch, patches+1, h))
        x = self.primitive("vision.position", "add",
                           [x, self.weight("vision.position_weight", (patches+1, h))], (batch, patches+1, h))
        eps = 1e-5 if clip else 1e-12
        if clip:
            x = self.norm("vision.pre_norm", x, eps)
        for i in range(c["layers"]):
            x = self.block(f"vision.block{i}", x, c["heads"], c["mlp"], eps,
                           "quick_gelu" if clip else "gelu_exact")
        if not clip:
            x = self.norm("vision.final_norm", x, eps)
        x = self.primitive("vision.select_cls", "select", [x], (batch, h), policy="index zero")
        if clip:
            x = self.norm("vision.post_norm", x, eps)
        return self.linear("vision.projection" if clip else "classifier", x,
                           c["projection"] if clip else c["classes"], bias=not clip)


def full_model_workload(kind, params):
    if kind not in ("vit", "clip"):
        raise ValueError("full model kind must be vit or clip")
    p = dict(params)
    expected = "vit_b16_224" if kind == "vit" else "clip_vit_b32"
    profile = p.pop("profile", expected)
    dtype = p.pop("dtype", "fp16")
    iterations = p.pop("iterations", 1)
    if dtype not in ("fp16", "fp32") or type(iterations) is not int or iterations != 1:
        raise ValueError("full models support fp16/fp32 and iterations=1 only")
    if profile == "synthetic":
        c = dict(SYNTHETIC)
        overrides = p.pop("config", {})
        if not isinstance(overrides, dict) or set(overrides) - set(c):
            raise ValueError("unknown synthetic config fields")
        c.update(overrides)
    elif profile == expected:
        c = dict(PROFILES[profile])
    else:
        raise ValueError(f"{kind} profile must be {expected!r} or 'synthetic'")
    batch = p.pop("batch", 1) if kind == "vit" else p.pop("image_batch", 1)
    text_batch = p.pop("text_batch", 1) if kind == "clip" else None
    if p:
        raise ValueError(f"unknown {kind} fields: {sorted(p)}")
    if any(type(v) is not int or v < 1 for v in [*c.values(), batch, *([text_batch] if text_batch else [])]):
        raise ValueError("model dimensions and batches must be positive integers")
    if kind == "clip" and (type(text_batch) is not int or text_batch < 1):
        raise ValueError("text_batch must be a positive integer")
    if c["image_size"] % c["patch"] or c["width"] % c["heads"]:
        raise ValueError("image/patch and width/heads must divide exactly")
    if kind == "clip" and c["text_width"] % c["text_heads"]:
        raise ValueError("text_width/text_heads must divide exactly")
    g = Graph(profile, dtype)
    image = g.vision(c, batch, kind == "clip")
    outputs = [image]
    if kind == "clip":
        b, s, h = text_batch, c["context"], c["text_width"]
        tokens = g.w.tensor("token_ids", (b, s), dtype="int32")
        x = g.primitive("text.lookup", "lookup", [tokens, g.weight("text.token_weight", (c["vocab"], h))], (b, s, h))
        x = g.primitive("text.position", "add", [x, g.weight("text.position_weight", (s, h))], (b, s, h))
        mask = g.primitive("text.causal", "causal_mask", [], (s, s))
        for i in range(c["text_layers"]):
            x = g.block(f"text.block{i}", x, c["text_heads"], 4*h, 1e-5, "quick_gelu", mask)
        x = g.norm("text.final_norm", x, 1e-5)
        eot = g.primitive("text.eot_index", "argmax", [tokens], (b,), dtype="int32",
                          policy="first argmax token id; EOT is highest token id, content-independent scan cost")
        x = g.primitive("text.select_eot", "select", [x, eot], (b, h))
        text = g.linear("text.projection", x, c["projection"], False)
        image = g.primitive("image_features", "l2_norm", [image], (batch, c["projection"]))
        text = g.primitive("text_features", "l2_norm", [text], (b, c["projection"]))
        scale = g.primitive("logit_scale.exp", "exp", [g.weight("logit_scale", (1,), "fp32")], (1,), dtype="fp32")
        image = g.primitive("image_scaled", "scale", [image, scale], (batch, c["projection"]))
        text = g.primitive("text_transposed", "transpose", [text], (c["projection"], b), axes=(1, 0))
        logits = g.mm("similarity", image, text, 1, batch, c["projection"], b)
        logits = g.view("logits_per_image", logits, (batch, b))
        outputs = [logits, g.view("logits_per_text", logits, (b, batch), axes=(1, 0))]
    weights = [t for t in g.w.tensors.values() if t.kind == "weight"]
    g.w.meta["full_model"] = dict(
        profile=profile, model_kind=kind, synthetic=profile == "synthetic", config=c,
        sources=SOURCES[:2] if kind == "vit" else SOURCES[2:],
        inputs={k: list(t.shape) for k, t in g.w.tensors.items() if k in ("images", "token_ids")},
        input_boundary="processed NCHW RGB images and int32 token ids resident in device DRAM; CPU preprocessing/tokenizer excluded",
        outputs={k: list(g.w.tensors[k].shape) for k in outputs},
        arithmetic=dict(storage=dtype, accumulation="fp32", vector="fp32",
                        cast="matrix epilogue casts once; vector load promotion/final cast charged; logit_scale fp32",
                        policy="declared uniform-storage inference variant, not official mixed-precision kernel equivalence"),
        parameter_elements=sum(math.prod(t.shape) for t in weights),
        parameter_bytes=sum(t.bytes for t in weights),
        operator_coverage=dict(Counter(o.op for o in g.w.operators)),
        timing_gaps=["SFU function-result abstraction, especially erf, is not a calibrated instruction implementation",
                     "No framework fusion/compiler/kernel-choice equivalence or hardware latency accuracy claim",
                     "FP32 reduction scratch is explicitly materialized in global memory; no free register-cache assumption"],
        eval_mode=True, missing_operators=[], supported_mapping="single-device gsm_tiled",
    )
    return g.w
