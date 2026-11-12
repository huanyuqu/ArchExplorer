"""Transformer decoder-layer workloads with tensor-parallel sharding.

Model configurations come from the public HuggingFace ``config.json`` files
or the GPT-3 paper; each carries its source. ``transformer_workload`` emits
the operators one tensor-parallel rank executes for ``layers`` decoder layers
(Megatron-style: attention heads and FFN columns split across ranks, an
all-reduce after the attention output projection and after the FFN down
projection). Labels group operators as in Fig. 7(i): QKV, Attention, O proj,
AllReduce, FFN up, FFN down, plus Norm and Residual.

Attention is the unfused sequence QK^T, softmax, AV over the full ``ctx``
(no causal-mask skipping). Prefill attends over ``seq_len`` tokens; decode
generates one token per sequence against a KV cache of ``seq_len`` tokens.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace

from arch_explorer.workload.operators import Workload

HF = "https://huggingface.co"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int          # width of one FFN projection
    layers: int
    vocab: int
    gated: bool                # SwiGLU (gate and up projections) vs. plain MLP
    norm: str                  # rmsnorm | layernorm
    activation: str            # silu | gelu
    source: str

    def to_dict(self):
        return asdict(self)


MODELS = {
    "llama2-7b": ModelConfig("llama2-7b", 4096, 32, 32, 128, 11008, 32, 32000, True, "rmsnorm", "silu",
                             f"{HF}/NousResearch/Llama-2-7b-hf/blob/8efe6c9b93655b934e27bd9981e3ec13e55ae7c2/config.json "
                             "(mirror of meta-llama/Llama-2-7b-hf)"),
    "llama2-70b": ModelConfig("llama2-70b", 8192, 64, 8, 128, 28672, 80, 32000, True, "rmsnorm", "silu",
                              f"{HF}/NousResearch/Llama-2-70b-hf config.json @ d955dd853c3a "
                              "(mirror of meta-llama/Llama-2-70b-hf)"),
    "llama3-70b": ModelConfig("llama3-70b", 8192, 64, 8, 128, 28672, 80, 128256, True, "rmsnorm", "silu",
                              f"{HF}/NousResearch/Meta-Llama-3-70B config.json @ fdbf133d993d "
                              "(mirror of meta-llama/Meta-Llama-3-70B)"),
    "qwen-72b": ModelConfig("qwen-72b", 8192, 64, 64, 128, 24576, 80, 152064, True, "rmsnorm", "silu",
                            f"{HF}/Qwen/Qwen-72B config.json @ b8e18ac61df6; intermediate_size 49152 is split "
                            "into two 24576-wide projections (modeling_qwen.py)"),
    "gpt3-6.7b": ModelConfig("gpt3-6.7b", 4096, 32, 32, 128, 16384, 32, 50257, False, "layernorm", "gelu",
                             "Brown et al., Language Models are Few-Shot Learners, arXiv:2005.14165 Table 2.1 "
                             "(d_ff = 4 d_model; GPT-2 vocabulary)"),
}

PHASES = ("prefill", "decode")


def model_config(model) -> ModelConfig:
    if isinstance(model, ModelConfig):
        return model
    if isinstance(model, str):
        if model not in MODELS:
            raise KeyError(f"Unknown model {model!r}; registered: {sorted(MODELS)}")
        return MODELS[model]
    if isinstance(model, dict):
        data = dict(model)
        data.setdefault("name", "custom")
        data.setdefault("source", "user-supplied config")
        return ModelConfig(**data)
    raise TypeError("model must be a registered name, a dict or a ModelConfig")


def transformer_workload(model, phase="prefill", batch=1, seq_len=2048, layers=1, tp=1, dtype="fp16",
                         iterations=1, resource_budget=None) -> Workload:
    """Operators of one tensor-parallel rank for ``layers`` decoder layers."""
    config = model_config(model)
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    for name, value in (("batch", batch), ("seq_len", seq_len), ("layers", layers), ("tp", tp)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.heads % tp or config.intermediate % tp:
        raise ValueError(f"tp={tp} does not divide {config.name}'s {config.heads} heads / FFN width")
    if config.kv_heads % tp and tp % config.kv_heads:
        raise ValueError(f"tp={tp} incompatible with {config.kv_heads} KV heads")
    budget=None
    if resource_budget is not None:
        facts=transformer_preflight(model,phase,batch,seq_len,layers,tp,dtype,iterations,resource_budget)
        budget=facts['budget']
        if not facts['admitted']:
            raise ValueError('transformer resource_budget refused: '+', '.join(v['field'] for v in facts['violations']))
    w = Workload(f"{config.name}-{phase}-b{batch}-s{seq_len}-tp{tp}", dtype=dtype, iterations=iterations,
                 meta={"model": config.to_dict(), "phase": phase, "batch": batch, "seq_len": seq_len,
                       "layers": layers, "tp": tp})
    if budget is not None:w.meta['resource_budget']=budget
    return _populate(w,config,phase,batch,seq_len,layers,tp)


def _populate(w,config,phase,batch,seq_len,layers,tp):
    """One descriptor path shared by the graph builder and scalar preflight."""
    hq = config.heads // tp
    hkv = max(1, config.kv_heads // tp)     # KV heads are replicated when tp > kv_heads
    hd, hidden, inter = config.head_dim, config.hidden, config.intermediate // tp
    q_len = seq_len if phase == "prefill" else 1
    tokens = batch * q_len
    x = w.tensor("x0", (tokens, hidden))
    for layer in range(layers):
        p = f"l{layer}."
        norm1 = w.tensor(p + "norm1", (tokens, hidden))
        w.add(p + "norm1", config.norm, [x], [norm1], "Norm", rows=tokens, cols=hidden)
        qkv_width = (hq + 2 * hkv) * hd
        w_qkv = w.tensor(p + "w_qkv", (hidden, qkv_width), kind="weight")
        qkv = w.tensor(p + "qkv", (tokens, qkv_width))
        w.add(p + "qkv", "matmul", [norm1, w_qkv], [qkv], "QKV", batch=1, m=tokens, k=hidden, n=qkv_width,
              shared_b=True)
        kv = w.tensor(p + "kv_cache", (2, batch, seq_len, hkv, hd), kind="kv")
        scores = w.tensor(p + "scores", (batch * hq, q_len, seq_len))
        # GQA: kv_group consecutive query heads read the same K/V head.
        group = hq // hkv
        w.add(p + "attn_qk", "matmul", [qkv, kv], [scores], "Attention", batch=batch * hq, m=q_len, k=hd,
              n=seq_len, kv_group=group,
              input_elements=[batch*hq*q_len*hd, batch*hkv*hd*seq_len])
        probs = w.tensor(p + "probs", (batch * hq, q_len, seq_len))
        w.add(p + "softmax", "softmax", [scores], [probs], "Attention", rows=batch * hq * q_len, cols=seq_len)
        ctx = w.tensor(p + "context", (tokens, hq * hd))
        w.add(p + "attn_av", "matmul", [probs, kv], [ctx], "Attention", batch=batch * hq, m=q_len, k=seq_len,
              n=hd, kv_group=group,
              input_elements=[batch*hq*q_len*seq_len, batch*hkv*seq_len*hd])
        w_o = w.tensor(p + "w_o", (hq * hd, hidden), kind="weight")
        attn_out = w.tensor(p + "attn_out", (tokens, hidden))
        w.add(p + "o_proj", "matmul", [ctx, w_o], [attn_out], "O proj", batch=1, m=tokens, k=hq * hd, n=hidden,
              shared_b=True)
        if tp > 1:
            reduced = w.tensor(p + "attn_reduced", (tokens, hidden))
            w.add(p + "allreduce1", "allreduce", [attn_out], [reduced], "AllReduce", bytes=w.tensors[attn_out].bytes)
            attn_out = reduced
        h = w.tensor(p + "h", (tokens, hidden))
        w.add(p + "residual1", "add", [x, attn_out], [h], "Residual", rows=tokens, cols=hidden)
        norm2 = w.tensor(p + "norm2", (tokens, hidden))
        w.add(p + "norm2", config.norm, [h], [norm2], "Norm", rows=tokens, cols=hidden)
        up_width = 2 * inter if config.gated else inter
        w_up = w.tensor(p + "w_up", (hidden, up_width), kind="weight")
        up = w.tensor(p + "up", (tokens, up_width))
        w.add(p + "ffn_up", "matmul", [norm2, w_up], [up], "FFN up", batch=1, m=tokens, k=hidden, n=up_width,
              shared_b=True)
        act = w.tensor(p + "act", (tokens, inter))
        w.add(p + "act", "swiglu" if config.gated else config.activation, [up], [act], "FFN up", rows=tokens,
              cols=inter)
        w_down = w.tensor(p + "w_down", (inter, hidden), kind="weight")
        down = w.tensor(p + "down", (tokens, hidden))
        w.add(p + "ffn_down", "matmul", [act, w_down], [down], "FFN down", batch=1, m=tokens, k=inter, n=hidden,
              shared_b=True)
        if tp > 1:
            reduced = w.tensor(p + "ffn_reduced", (tokens, hidden))
            w.add(p + "allreduce2", "allreduce", [down], [reduced], "AllReduce", bytes=w.tensors[down].bytes)
            down = reduced
        out = w.tensor(f"x{layer + 1}", (tokens, hidden))
        w.add(p + "residual2", "add", [h, down], [out], "Residual", rows=tokens, cols=hidden)
        x = out
    return w


class _Statistics:
    """Metadata-only sink: never constructs a Tensor, Operator or numeric root."""
    def __init__(self,dtype):
        self.dtype=dtype;self.tensors={};self.produced=set();self.operators=0;self.edges=0
        self.matrix_fmas=0;self.vector_elements=0;self.allreduce_bytes=0

    def tensor(self,name,shape,kind='activation',dtype=None):
        from arch_explorer.components.base import dtype_bytes
        dtype=self.dtype if dtype is None else dtype
        self.tensors[name]=SimpleNamespace(shape=shape,kind=kind,bytes=math.prod(shape)*dtype_bytes(dtype))
        return name

    def add(self,name,op,inputs,outputs,label='',**attrs):
        self.operators+=1;self.edges+=len(inputs)+len(outputs);self.produced.update(outputs)
        if op=='matmul':self.matrix_fmas+=attrs['batch']*attrs['m']*attrs['k']*attrs['n']
        elif op=='allreduce':self.allreduce_bytes+=attrs['bytes']
        else:self.vector_elements+=attrs['rows']*attrs['cols']


def transformer_preflight(model, phase='prefill', batch=1, seq_len=2048, layers=1, tp=1,
                          dtype='fp16', iterations=1, resource_budget=None):
    """Inspect a complete one-rank performance graph before graph allocation.

    Logical weights/KV are declarations, with no invented numeric values. A
    metadata-only one-layer sink uses the same descriptors as the real builder;
    full-layer counts scale that sink without constructing a large graph.
    """
    from .resource_budget import normalize_budget
    config=model_config(model)
    if phase not in PHASES:raise ValueError(f'phase must be one of {PHASES}')
    for key,value in (('batch',batch),('seq_len',seq_len),('layers',layers),('tp',tp),('iterations',iterations)):
        if type(value) is not int or value<1:raise ValueError(key+' must be a positive integer')
    for key in ('hidden','heads','kv_heads','head_dim','intermediate','layers','vocab'):
        if type(getattr(config,key)) is not int or getattr(config,key)<1:
            raise ValueError('model '+key+' must be a positive integer')
    if config.heads%tp or config.intermediate%tp:
        raise ValueError(f"tp={tp} does not divide {config.name}'s {config.heads} heads / FFN width")
    if config.kv_heads%tp and tp%config.kv_heads:
        raise ValueError(f'tp={tp} incompatible with {config.kv_heads} KV heads')
    if config.heads//tp%max(1,config.kv_heads//tp):raise ValueError('query heads must group exactly into KV heads')
    if dtype not in ('fp32','fp16','bf16'):raise ValueError('transformer preflight dtype must be fp32/fp16/bf16')
    if type(config.gated) is not bool or config.norm not in ('rmsnorm','layernorm') or config.activation not in ('silu','gelu'):
        raise ValueError('unsupported transformer norm/activation/gating configuration')
    budget=normalize_budget({} if resource_budget is None else resource_budget)
    one=_populate(_Statistics(dtype),config,phase,batch,seq_len,1,tp)
    x=one.tensors['x0'];roots=[t for n,t in one.tensors.items() if n not in one.produced and n!='x0']
    tensor_bytes=x.bytes+layers*sum(t.bytes for n,t in one.tensors.items() if n!='x0')
    root_bytes=x.bytes+layers*sum(t.bytes for t in roots)
    tensors=1+layers*(len(one.tensors)-1);operators=layers*one.operators
    metadata=256*(tensors+operators)+64*one.edges*layers
    metrics=dict(operators=operators,tensors=tensors,root_bytes=root_bytes,materialized_root_bytes=0,
        weight_bytes=layers*sum(t.bytes for t in roots if t.kind=='weight'),
        kv_bytes=layers*sum(t.bytes for t in roots if t.kind=='kv'),temporary_bytes=tensor_bytes-root_bytes,
        total_tensor_bytes=tensor_bytes,root_python_bytes=0,estimated_host_bytes=12*metadata,
        max_rank=max(len(t.shape) for t in one.tensors.values()),
        max_dimension=max(max(t.shape) for t in one.tensors.values()),
        max_tensor_elements=max(math.prod(t.shape) for t in one.tensors.values()),
        matrix_fmas_per_iteration=one.matrix_fmas*layers,total_matrix_fmas=one.matrix_fmas*layers*iterations,
        vector_elements_per_iteration=one.vector_elements*layers,allreduce_bytes_per_iteration=one.allreduce_bytes*layers)
    fields={'operators':'max_steps','max_rank':'max_rank','max_dimension':'max_dimension',
            'max_tensor_elements':'max_tensor_elements','total_tensor_bytes':'max_total_tensor_bytes',
            'materialized_root_bytes':'max_root_bytes','estimated_host_bytes':'max_host_bytes'}
    violations=[dict(field=k,required=metrics[k],limit=budget[b],budget_field=b) for k,b in fields.items() if metrics[k]>budget[b]]
    return dict(schema='arch-explorer.transformer-preflight.v1',
        request=dict(model=config.to_dict(),phase=phase,batch=batch,seq_len=seq_len,layers=layers,tp=tp,dtype=dtype,iterations=iterations),
        budget=budget,metrics=metrics,admitted=not violations,violations=violations,
        semantics='one tensor-parallel rank; full attention; declared weights and KV; no numeric roots',
        plan=dict(status='mapping-required',minimum_task_instances=operators*iterations,replicas=None,physical_peak_bytes=None),
        stages=dict(graph='workload_from_spec(WorkloadSpec(kind="transformer",params),tp=tp)',
            compile='compile_graph does not support compact KV performance graphs; no fabricated numeric roots',
            service='existing MappingTemplate/prepare_request provider path; actual task/capacity/traffic only after preparation'))


def workload_from_spec(spec, tp=1) -> Workload:
    """Compatibility entry point for the unique public workload dispatcher."""
    from .api import workload_from_spec as build
    return build(spec, tp=tp)
