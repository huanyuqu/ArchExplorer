"""Small explicit attention inputs and public graph examples.

Only scene data and graph composition live here. Operator semantics, mapping
rules, adaptation and component service times belong to their public owners.
"""
from copy import deepcopy
import math


def attention_inputs(*, batch=1, queries=3, keys=5, head_dim=4, value_dim=3,
                     input_case="regular"):
    """Return contiguous FP32 Q/K/V data, without computing attention answers.

    ``large`` creates logits that need stable normalization; ``constant`` has
    zero queries so every score in a row is equal. The default shape is not
    square and leaves boundary tiles in both matrix products.
    """
    dimensions = (batch, queries, keys, head_dim, value_dim)
    if any(type(n) is not int or n < 1 for n in dimensions):
        raise ValueError("attention dimensions must be positive integers")
    if input_case not in ("regular", "large", "constant"):
        raise ValueError("input_case must be regular, large or constant")

    def values(rows, cols, offset):
        return [[[(b * 37 + r * cols + c + offset) % 13 / 4.0 - 1.5
                  for c in range(cols)] for r in range(rows)] for b in range(batch)]

    q, k, v = values(queries, head_dim, 2), values(keys, head_dim, 7), values(keys, value_dim, 11)
    if input_case == "constant":
        q = [[[0.0 for _ in range(head_dim)] for _ in range(queries)] for _ in range(batch)]
    elif input_case == "large":
        q = [[[1024.0 * x for x in row] for row in entry] for entry in q]
        k = [[[1024.0 * x for x in row] for row in entry] for entry in k]
    return deepcopy({"Q": q, "K": k, "V": v})


def attention_workload(*, batch=1, queries=3, keys=5, head_dim=4, value_dim=3,
                       input_case="regular"):
    """Materialized, stable scaled-dot-product attention with two real readers."""
    from arch_explorer.adf import WorkloadSpec
    from arch_explorer.workload.api import Workload
    inputs = attention_inputs(batch=batch, queries=queries, keys=keys, head_dim=head_dim,
                              value_dim=value_dim, input_case=input_case)
    graph = Workload("explicit_stable_attention", dtype="fp32")
    q = graph.tensor("Q", (batch, queries, head_dim))
    k = graph.tensor("K", (batch, keys, head_dim), kind="weight")
    v = graph.tensor("V", (batch, keys, value_dim), kind="weight")
    kt = graph.tensor("KT", (batch, head_dim, keys))
    graph.add("transpose_keys", "transpose", [k], [kt], shape=[batch, head_dim, keys], axes=[0, 2, 1])
    shape = [batch, queries, keys]
    scores = graph.tensor("scores", shape)
    graph.add("scores", "matmul", [q, kt], [scores], batch=batch, m=queries, k=head_dim, n=keys)
    scaled = graph.tensor("scaled", shape)
    graph.add("scale_scores", "scale", [scores], [scaled], shape=shape, factor=1.0 / math.sqrt(head_dim))
    row_shape = [batch, queries, 1]
    maximum = graph.tensor("row_max", row_shape)
    graph.add("row_max", "reduce_max", [scaled], [maximum], shape=row_shape, axis=-1, keepdims=True)
    max_grid = graph.tensor("max_grid", shape)
    graph.add("broadcast_max", "broadcast", [maximum], [max_grid], shape=shape)
    centered = graph.tensor("centered", shape)
    graph.add("center", "sub", [scaled, max_grid], [centered], shape=shape)
    exponentials = graph.tensor("exponentials", shape)
    graph.add("exponentiate", "exp", [centered], [exponentials], shape=shape)
    row_sum = graph.tensor("row_sum", row_shape)
    graph.add("row_sum", "reduce_sum", [exponentials], [row_sum], shape=row_shape, axis=-1, keepdims=True)
    sum_grid = graph.tensor("sum_grid", shape)
    graph.add("broadcast_sum", "broadcast", [row_sum], [sum_grid], shape=shape)
    probabilities = graph.tensor("probabilities", shape)
    graph.add("normalize", "div", [exponentials, sum_grid], [probabilities], shape=shape)
    output = graph.tensor("attention", (batch, queries, value_dim))
    graph.add("weighted_values", "matmul", [probabilities, v], [output],
              batch=batch, m=queries, k=keys, n=value_dim)
    graph.meta["inputs"] = inputs
    return WorkloadSpec("graph", graph.to_dict())


def attention_scene(*, hardware=None, batch=1, queries=3, keys=5, head_dim=4,
                    value_dim=3, input_case="regular", tile_m=2, tile_k=3, tile_n=3):
    """Use public semantic rules on an explicit synthetic teaching board.

    ``hardware`` is an ordinary scene constructor, for example ``events_scene``
    or ``roofline_scene``. Components and their declared adapters own execution.
    """
    from .compilation import events_scene
    hardware = events_scene if hardware is None else hardware
    adf, _, mapping = hardware(m=queries, k=head_dim, n=keys,
                              tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)
    workload = attention_workload(batch=batch, queries=queries, keys=keys, head_dim=head_dim,
                                  value_dim=value_dim, input_case=input_case)
    return adf, workload, mapping


def attention_search_config(*, hardware=None, workload=None, input_case="regular"):
    """Six evaluation units, fixed seed, two mappings and parameter proposals.

    This is input to the existing search engine, not an alternative optimizer.
    Registered mapping rules validate both concrete requests. The one-byte SRAM
    proposal deliberately preserves a rejected candidate in the audit trail.
    """
    from .compilation import compile_mapping
    adf, original, first = attention_scene(hardware=hardware, input_case=input_case)
    second = compile_mapping(compute="board/core/sm", memory="board/sram", tile_m=3, tile_k=2, tile_n=2)
    source = "bounded attention search example; explicit synthetic hardware proposal"
    proposals = [{"tool": "set_parameters", "args": {"parameters": [{
        "path": "board/sram", "param": key, "value": value, "source": source}]}}
        for key, value in (("capacity", 1), ("bandwidth", 32.0))]
    return {"seed_adf": adf.to_dict(), "workloads": [(original if workload is None else workload).to_dict()],
            "compilation": {}, "backend_options": {"trace": True},
            "mapping_templates": [first.template],
            "mapping_selectors": {first.template: [first.params, second.params]},
            "max_evaluations": 6, "parameter_samples": 1, "mapping_samples": 2,
            "parameter_domains": [
                {"path": "board/sram", "param": "capacity", "values": [1, 65536], "source": source},
                {"path": "board/sram", "param": "bandwidth", "values": [16.0, 32.0], "source": source}],
            "random_seed": 0, "search": {"max_steps": 3, "max_proposals": 3, "proposals": proposals}}
