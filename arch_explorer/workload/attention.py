"""Dense single-query decode attention, with explicit scaled QK/softmax/AV."""
from arch_explorer.workload.operators import Workload


def decode_attention(params):
    p = dict(params)
    p.pop("op", None)
    dtype = p.pop("dtype", "fp16")
    if dtype != "fp16":
        raise ValueError("decode attention currently requires fp16 inputs and stored intermediates; fp32 accumulation")
    b, h, d, s, q = (p.pop("batch", 1), p.pop("heads", 32), p.pop("head_dim", 128),
                     p.pop("seq_len"), p.pop("query_len", 1))
    if any(type(v) is not int or v < 1 for v in (b, h, d, s, q)) or q != 1:
        raise ValueError("decode attention requires positive integer dimensions and query_len=1")
    storage = p.pop("kv_storage", "on-chip")
    if storage not in ("on-chip", "remote"):
        raise ValueError("kv_storage must be on-chip or remote")
    iterations = p.pop("iterations", 1)
    if type(iterations) is not int or iterations < 1 or p:
        raise ValueError(f"invalid attention iterations or unknown parameters: {sorted(p)}")
    w = Workload("decode_attention", dtype=dtype, iterations=iterations,
                 meta={"attention": True, "kv_storage": storage, "query_len": 1,
                       "batch": b, "heads": h, "head_dim": d, "seq_len": s,
                       "accumulation": "fp32", "stored_precision": "fp16",
                       "scale": d ** -0.5, "implementation": "segmented, materialized scores; not fused"})
    for name, shape, kind in (("Q", (b*h, 1, d), "activation"), ("K", (b*h, d, s), "kv"),
                              ("V", (b*h, s, d), "kv"), ("scores", (b*h, 1, s), "activation"),
                              ("scaled", (b*h, 1, s), "activation"), ("probs", (b*h, 1, s), "activation"),
                              ("out", (b*h, 1, d), "activation")):
        w.tensor(name, shape, kind)
    w.add("QK", "matmul", ["Q", "K"], ["scores"], "QK", batch=b*h, m=1, k=d, n=s)
    w.add("scale", "scale", ["scores"], ["scaled"], "Scale", rows=b*h, cols=s)
    w.add("softmax", "softmax", ["scaled"], ["probs"], "Softmax", rows=b*h, cols=s)
    w.add("AV", "matmul", ["probs", "V"], ["out"], "AV", batch=b*h, m=1, k=s, n=d)
    return w
