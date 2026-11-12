from non_gpu_fixture import shared_adf
"""Independent small-array reference, parameter totals and execution contracts."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from arch_explorer.adf import WorkloadSpec, MappingSpec, evaluate
from arch_explorer.adf.evaluation import WORKLOAD_KINDS
from arch_explorer.devices import get
from arch_explorer.mapping import mapping_space
from arch_explorer.simulator.events.backend import CapacityError, run_mapping
from arch_explorer.workload.full_models import Graph, full_model_workload
from arch_explorer.workload.full_primitives import reference


@pytest.fixture(autouse=True)
def entry_contract(monkeypatch):
    # SIM-12's baseline predates CORE-09's two-kind registration. This fixture
    # only enables that fixed entry contract for isolated simulation tests.
    # On the combined tree CORE-09 supplies it without a patch.
    if not {"vit", "clip"} <= WORKLOAD_KINDS:
        monkeypatch.setattr("arch_explorer.adf.evaluation.WORKLOAD_KINDS", WORKLOAD_KINDS | {"vit", "clip"})


def data_and_execute(w):
    rng = np.random.default_rng(203)
    produced = w.producers()
    data = {}
    for name, t in w.tensors.items():
        if name in produced:
            continue
        if name == "token_ids":
            data[name] = np.array([[1, 16, 3, 0, 0], [2, 3, 4, 16, 0], [16, 1, 2, 0, 0]], dtype=np.int32)[:t.shape[0]]
        else:
            data[name] = (rng.standard_normal(t.shape)*.1).astype(np.float32)
            if name.endswith(".gamma"):
                data[name] += 1
    initial = dict(data)
    for op in w.operators:
        if op.op == "matmul":
            a = op.attrs
            x = data[op.inputs[0]].reshape(a["batch"], a["m"], a["k"])
            y = data[op.inputs[1]].reshape(1 if a["shared_b"] else a["batch"], a["k"], a["n"])
            value = x @ y
        else:
            value = reference(op, [data[n] for n in op.inputs])
        t = w.tensors[op.outputs[0]]
        assert value.shape == t.shape, op.name
        data[op.outputs[0]] = value.astype(np.int32 if t.dtype == "int32" else np.float32)
    return initial, data


def independent_model(c, inputs, clip):
    """Monolithic formula, no graph operators, primitive refs or shape attrs."""
    v = inputs
    def linear(x, name, bias=True):
        z = x @ v[name+".weight"]
        return z + v[name+".bias_weight"] if bias else z
    def norm(x, name, eps):
        return (x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1, keepdims=True)+eps)*v[name+".gamma"]+v[name+".beta"]
    def block(x, name, heads, causal, eps):
        b, s, h = x.shape
        z = norm(x, name+".ln1", eps)
        q, k, value = [linear(z, name+"."+p).reshape(b, s, heads, h//heads).transpose(0, 2, 1, 3)
                        for p in ("q", "k", "v")]
        scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(h//heads)
        if causal:
            for i in range(s):
                scores[..., i, i+1:] = -np.inf
        probs = np.exp(scores-scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        attn = (probs @ value).transpose(0, 2, 1, 3).reshape(b, s, h)
        x = x + linear(attn, name+".out")
        z = linear(norm(x, name+".ln2", eps), name+".up")
        z = z/(1+np.exp(-1.702*z)) if clip else z*.5*(1+np.vectorize(math.erf)(z/math.sqrt(2)))
        return x + linear(z, name+".down")
    images = v["images"]
    p = c["patch"]
    # Explicit loops make a channel/patch permutation mistake observable.
    patches = np.stack([images[:, :, i:i+p, j:j+p].reshape(images.shape[0], -1)
                        for i in range(0, c["image_size"], p) for j in range(0, c["image_size"], p)], axis=1)
    image = linear(patches, "vision.patch_projection", not clip)
    image = np.concatenate([np.broadcast_to(v["vision.cls_weight"], (len(images), 1, c["width"])), image], axis=1)
    image += v["vision.position_weight"]
    if clip:
        image = norm(image, "vision.pre_norm", 1e-5)
    for i in range(c["layers"]):
        image = block(image, f"vision.block{i}", c["heads"], False, 1e-5 if clip else 1e-12)
    if not clip:
        image = norm(image, "vision.final_norm", 1e-12)[:, 0]
        return [linear(image, "classifier")]
    image = linear(norm(image[:, 0], "vision.post_norm", 1e-5), "vision.projection", False)
    ids = v["token_ids"]
    text = v["text.token_weight"][ids] + v["text.position_weight"]
    for i in range(c["text_layers"]):
        text = block(text, f"text.block{i}", c["text_heads"], True, 1e-5)
    text = norm(text, "text.final_norm", 1e-5)
    text = text[np.arange(len(ids)), ids.argmax(-1)]
    text = linear(text, "text.projection", False)
    image /= np.linalg.norm(image, axis=-1, keepdims=True)
    text /= np.linalg.norm(text, axis=-1, keepdims=True)
    logits = (np.exp(v["logit_scale"])*image) @ text.T
    return [logits, logits.T]


@pytest.mark.parametrize("kind", ["vit", "clip"])
def test_complete_small_graph_against_independent_model(kind):
    p = dict(profile="synthetic", dtype="fp32")
    p.update(dict(image_batch=2, text_batch=3) if kind == "clip" else dict(batch=2))
    w = full_model_workload(kind, p)
    inputs, values = data_and_execute(w)
    expected = independent_model(w.meta["full_model"]["config"], inputs, kind == "clip")
    for name, result in zip(w.meta["full_model"]["outputs"], expected):
        np.testing.assert_allclose(values[name], result, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kind,total,layers", [("vit", 86567656, 12), ("clip", 151277313, 24)])
def test_named_complete_parameter_and_layer_counts(kind, total, layers):
    w = full_model_workload(kind, {})
    meta = w.meta["full_model"]
    assert meta["parameter_elements"] == total
    assert sum(o.name.endswith(".qk") for o in w.operators) == layers
    assert sum(o.name.endswith(".av") for o in w.operators) == layers
    assert meta["operator_coverage"].get("quick_gelu" if kind == "clip" else "gelu_exact") == layers
    assert meta["parameter_bytes"] == total*2 + (2 if kind == "clip" else 0)
    json.dumps(meta)


def test_primitive_layouts_views_and_selection():
    g = Graph("probe", "fp32")
    x = g.w.tensor("x", (2, 3))
    out = g.view("transpose", x, (3, 2), axes=(1, 0))
    a = np.arange(6).reshape(2, 3)
    b = reference(g.w.operators[-1], [a])
    assert np.shares_memory(a, b)
    np.testing.assert_array_equal(b, a.T)
    assert g.w.tensors[out].strides == (1, 3)
    with pytest.raises(ValueError, match="contiguous"):
        g.view("bad", out, (6,))
    g.primitive("copy", "copy", [x], (2, 3))
    assert not np.shares_memory(a, reference(g.w.operators[-1], [a]))


def test_execution_lookup_traffic_alias_lifetime_capacity_and_repeatability():
    spec = WorkloadSpec("clip", dict(profile="synthetic", image_batch=2, text_batch=3))
    mapping = MappingSpec("gsm_tiled", dict(tile_m=64, tile_n=64))
    adf = shared_adf()
    result = evaluate(adf, spec, mapping)
    assert result.latency_ns == evaluate(adf, spec, mapping).latency_ns
    info = result.details["full_model"]
    assert info["outputs"] == {"logits_per_image": [2, 3], "logits_per_text": [3, 2]}
    lookup = next(o for o in info["operators"] if o["op"] == "lookup")
    assert lookup["read_bytes"] == 3*5*4 + 3*5*8*2
    assert info["tensors"]["text.token_weight"]["allocated_bytes"] == 17*8*2
    assert info["tensors"]["logits_per_text"]["allocated_bytes"] == 0
    assert info["memory_peak"]["chip/dram"] > info["parameter_bytes"]
    assert any(value > 0 for name, value in info["resource_demands"].items() if ":link" in name)
    assert sum(o.get("scratch_traffic_bytes", 0) for o in info["operators"]) > 0
    sim, _ = run_mapping(adf, spec, mapping)
    graph = sim.mapping.graph
    out = next(t for t in graph.tasks if t.name == "chip:similarity")
    ret = next(t for t in graph.tasks if t.name == "full_model.return")
    assert any(s == out.id for s, _ in graph.preds[ret.id])
    # Shrink actual physical DRAM, leaving all model dimensions unchanged.
    dram = next(e for e in adf.elements if e.name == "dram")
    params = dict(dram.params, capacity=replace(dram.params["capacity"], value=128))
    low = replace(adf, elements=[replace(e, params=params) if e.name == "dram" else e for e in adf.elements])
    with pytest.raises(CapacityError):
        evaluate(low, spec, mapping)


@pytest.mark.parametrize("kind,p", [
    ("vit", {"batch":0}), ("clip", {"text_batch":0}), ("vit", {"dtype":"bf16"}),
    ("vit", {"profile":"clip_vit_b32"}), ("clip", {"layers":1}), ("vit", {"iterations":2}),
    ("vit", {"profile":"synthetic", "config":{"width":7}}),
])
def test_invalid_models_reject(kind, p):
    with pytest.raises(ValueError):
        full_model_workload(kind, p)


def test_mapping_guards_and_iterations():
    adf = shared_adf()
    spec = WorkloadSpec("vit", {"profile":"synthetic"})
    for name in ("dmc_partition", "tensor_parallel", "pim_offload", "cgra_spatial", "ipu_bsp"):
        assert mapping_space(adf, spec, name) == []
        with pytest.raises(ValueError, match="not applicable"):
            evaluate(adf, spec, MappingSpec(name, {}))
    with pytest.raises(ValueError, match="iterations=1"):
        evaluate(adf, spec, MappingSpec("gsm_tiled", {"tile_m":64, "tile_n":64}), iterations=2)


def test_view_and_copy_have_distinct_physical_allocations_and_traffic(monkeypatch):
    import arch_explorer.mapping.full_models as mapper
    def graph(copy):
        g = Graph("storage-probe", "fp16")
        x = g.w.tensor("x", (32, 64))
        middle = (g.primitive("middle", "copy", [x], (32, 64)) if copy
                  else g.view("middle", x, (32, 64)))
        y = g.primitive("y", "scale", [middle], (32, 64), factor=2)
        g.w.meta["full_model"] = dict(outputs={y:[32, 64]}, parameter_bytes=0)
        return g.w
    results = []
    for copying in (False, True):
        monkeypatch.setattr(mapper, "workload_from_spec", lambda spec, c=copying: graph(c))
        results.append(evaluate(shared_adf(), WorkloadSpec("vit", {"profile":"synthetic"}),
                                MappingSpec("gsm_tiled", {})))
    view, copy = [r.details["full_model"] for r in results]
    size = 32*64*2
    assert view["tensors"]["middle"]["allocated_bytes"] == 0
    assert copy["tensors"]["middle"]["allocated_bytes"] == size
    assert copy["resource_demands"]["chip/dram:port"] - view["resource_demands"]["chip/dram:port"] == 2*size
    assert results[1].latency_ns > results[0].latency_ns
    assert view["memory_peak"]["chip/dram"] == 2*size
    assert copy["memory_peak"]["chip/dram"] == 2*size


def test_tiny_cache_cannot_supply_free_cross_cta_reuse(monkeypatch):
    import arch_explorer.mapping.full_models as mapper
    g = Graph("cache-probe", "fp16")
    x = g.w.tensor("x", (1, 128, 64))
    y = g.mm("y", x, g.weight("w", (64, 128)), 1, 128, 64, 128, True)
    g.w.meta["full_model"] = dict(outputs={y:[1, 128, 128]}, parameter_bytes=64*128*2)
    monkeypatch.setattr(mapper, "workload_from_spec", lambda spec: g.w)
    adf = shared_adf()
    cache = next(e for e in adf.elements if e.name == "l2")
    tiny = replace(adf, elements=[replace(e, params=dict(cache.params,
                   capacity=replace(cache.params["capacity"], value=1))) if e.name == "l2" else e
                   for e in adf.elements])
    spec = WorkloadSpec("vit", {})
    mapping = MappingSpec("gsm_tiled", {"tile_m":64, "tile_n":64})
    normal = evaluate(adf, spec, mapping).details["full_model"]
    small = evaluate(tiny, spec, mapping).details["full_model"]
    assert normal["resource_demands"]["chip/dram:port"] == (128*64+64*128+128*128)*2
    assert small["resource_demands"]["chip/dram:port"] == (4*(64*64+64*64)+128*128)*2
    assert small["memory_peak"]["chip/l2"] == 1


def test_fp16_output_conversion_runs_on_the_producing_unit(monkeypatch):
    import arch_explorer.mapping.full_models as mapper
    from arch_explorer.components.base import Kernel
    g = Graph("cast-probe", "fp16")
    x = g.w.tensor("x", (1, 2, 3))
    y = g.mm("y", x, g.weight("w", (3, 4)), 1, 2, 3, 4, True)
    g.w.meta["full_model"] = dict(outputs={y:[1, 2, 4]}, parameter_bytes=24)
    monkeypatch.setattr(mapper, "workload_from_spec", lambda spec: g.w)
    sim, _ = run_mapping(shared_adf(), WorkloadSpec("vit", {}),
                         MappingSpec("gsm_tiled", {"tile_m":64, "tile_n":64}))
    unit = "chip/core[0]/vector"
    fused = next(t for t in sim.mapping.graph.tasks if t.name == "y@"+unit)
    expected = sim.points[unit].kernel_demand(Kernel("vector", (1, 8), "fp32", simt=1))
    matrix = sim.points[unit].kernel_demand(Kernel("gemm", (1, 2, 3, 64), "fp16"))
    assert fused.demand[unit] == matrix.resources[unit] + expected.resources[unit]
    assert not any(".cast@" in t.name for t in sim.mapping.graph.tasks)


def test_fp32_variant_on_explicit_fp32_vector_hardware():
    from arch_explorer.adf import Param, SpaceMatrix, SpacePoint
    def p(value, unit):
        return Param(value, unit, "synthetic full-model regression, not measured hardware")
    adf = SpaceMatrix("test", [
        SpacePoint("vpu", "vector_unit", dict(lanes=p(32, "count"), sfu_lanes=p(8, "count"),
                    clock_ghz=p(1.0, "GHz"), precision=p("fp32", "1"))),
        SpacePoint("dram", "dram", dict(technology=p("ddr", "1"), capacity=p(2**30, "B"),
                    bandwidth=p(64.0, "B/ns"), latency=p(5.0, "ns"))),
    ], [SpacePoint("bus", "bus", dict(bandwidth=p(32.0, "B/ns"), latency=p(1.0, "ns")))])
    result = evaluate(adf, WorkloadSpec("vit", {"profile":"synthetic", "dtype":"fp32"}),
                      MappingSpec("gsm_tiled", {"tile_m":64, "tile_n":64}))
    assert result.latency_ns > 0
    assert result.details["full_model"]["arithmetic"]["storage"] == "fp32"
