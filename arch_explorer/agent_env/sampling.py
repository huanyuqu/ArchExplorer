"""Bounded mapping sampling and local, budget-aware parameter proposals."""
from copy import deepcopy
from dataclasses import replace
import math
import json

from arch_explorer.adf import MappingSpec, SpacePoint, walk
from arch_explorer.adf.multi_device import TEMPLATE as MULTI_DEVICE
from arch_explorer.adf.registry import ConditionalMappingSpace, mapping_spaces, validate_mapping_params
from arch_explorer.structure import check
from .errors import EnvError


def _requests(selector):
    if selector is None or isinstance(selector, dict):
        return [selector]
    if isinstance(selector, list) and selector and all(isinstance(item, dict) for item in selector):
        return selector
    raise EnvError("invalid_mapping_space", "A mapping selector must be an object or nonempty union of objects")


def mapping_contract(template, adf, workload, selector):
    descriptions = []
    for request in _requests(selector):
        space = template.request_space(adf, workload, request)
        descriptions.append({"contract": space.to_dict() if isinstance(space, ConditionalMappingSpace)
                             else {"kind": "flat", "parameters": space},
                             "sampling_spaces": mapping_spaces(space, request)})
    return {"kind": "request_union", "requests": descriptions} if isinstance(selector, list) else descriptions[0]


def mapping_candidates(adf, workload, registry, count, rng, names=None, selectors=None):
    candidates = []
    selectors = {} if selectors is None else selectors
    if not isinstance(selectors, dict) or any(name not in (names or registry.names()) for name in selectors):
        raise EnvError("invalid_mapping_space", "mapping_selectors must name selected mapping templates")
    for name in names or registry.names():
        if name == MULTI_DEVICE and name not in selectors:
            # Implicit registration discovery is not a structured request.
            # Explicit selection must fail, even if other templates can run.
            if names and name in names:
                raise EnvError("invalid_mapping_space", "multi_device requires an explicit structured request")
            continue
        template = registry.get(name)
        if not template.applicable(adf, workload):
            if name in selectors:
                raise EnvError("invalid_mapping_space", f"Selected mapping {name} does not apply")
            continue
        for request in _requests(selectors.get(name)):
            try:
                spaces = mapping_spaces(template.request_space(adf, workload, request), request)
            except ValueError as error:
                raise EnvError("invalid_mapping_space", f"{name}: {error}") from error
            products = [(sorted(space), space, math.prod(len(v) for v in space.values())) for space in spaces]
            size = sum(product[2] for product in products)
            # Sample a public request product, without materializing a large product.
            indices = range(size) if size <= count else sorted(rng.sample(range(size), count))
            for index in indices:
                for keys, space, length in products:
                    if index < length:
                        break
                    index -= length
                values = {}
                for key in reversed(keys):
                    index, digit = divmod(index, len(space[key]))
                    values[key] = space[key][digit]
                candidates.append(MappingSpec(name, values))
    candidates = list({json.dumps(candidate.to_dict(), sort_keys=True, allow_nan=False): candidate
                       for candidate in candidates}.values())
    if not candidates:
        raise EnvError("no_mapping", "No applicable registered mapping template")
    if len(candidates) > count:
        candidates = rng.sample(candidates, count)
    return candidates


def check_mapping(adf, workload, mapping, registry):
    template = registry.get(mapping.template)
    if not template.applicable(adf, workload):
        raise EnvError("invalid_mapping", f"{mapping.template} does not apply")
    try:
        validate_mapping_params(template.request_space(adf, workload, mapping.params), mapping.params, complete=True)
    except ValueError as error:
        raise EnvError("invalid_mapping", f"{mapping.template}: {error}") from error


def parameter_variant(adf, registry, rng, radius, grow=False, shrink=False, domains=None):
    if grow and shrink:
        raise ValueError("A parameter proposal cannot grow and shrink simultaneously")
    result = deepcopy(adf)
    domains = domains or {}
    tunable = [(path, node, key, replace(spec, domain=domains.get((path, key), spec.domain)))
               for path, node in walk(result) if isinstance(node, SpacePoint)
               for key, spec in registry.get(node.component).params.items()
               if key in node.params and (spec.domain is not None or (path, key) in domains)]
    if not tunable:
        return result
    # Mutate one coordinate at a time to keep variants near the candidate.
    preferred = [item for item in tunable if item[2] in {
        "capacity", "bandwidth", "link_bandwidth", "port_bandwidth", "lanes", "rows", "cols",
        "tensor_ops_per_cycle", "simt_ops_per_cycle", "sfu_ops_per_cycle", "clock_ghz"}]
    smaller = [item for item in tunable if item[2] in {
        "capacity", "lanes", "rows", "cols", "tensor_ops_per_cycle", "simt_ops_per_cycle", "ops_per_cycle"}
        and item[3].kind in {"int", "float"}]
    pool = smaller if shrink and smaller else preferred if grow and preferred else tunable
    path, node, key, spec = rng.choice(pool)
    param, domain = node.params[key], spec.domain
    if isinstance(domain, tuple):
        lo, hi = domain
        span = abs(param.value) * radius if param.value else (hi - lo) * radius
        if spec.kind == "int":
            span = max(1, span)
        lower, upper = max(lo, param.value - span), min(hi, param.value + span)
        if grow:
            lower = max(lower, param.value)
        if shrink:
            upper = min(upper, param.value)
        if spec.kind == "int":
            lower, upper = math.ceil(lower), math.floor(upper)
            value = rng.randint(lower, upper) if lower <= upper else param.value
        else:
            value = rng.uniform(lower, upper)
    elif isinstance(domain, list) and domain:
        if spec.kind in {"int", "float"}:
            choices = sorted(set(domain))
            center = min(range(len(choices)), key=lambda i: abs(choices[i] - param.value))
            nearby = choices[max(0, center - 1):center + 2]
            if grow:
                nearby = [x for x in nearby if x >= param.value] or nearby
            if shrink:
                nearby = [x for x in nearby if x <= param.value] or nearby
            value = rng.choice(nearby)
        else:
            value = rng.choice(domain)
    else:
        return result
    node.params[key] = replace(param, value=value, source=f"parameter search from {param.source}")
    check(result, registry)
    return result
