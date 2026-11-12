"""Exhaustive or budget-limited mapping selection inside a template's space.

``mapping_space`` lists the concrete ``MappingSpec`` candidates of a template
(a union of active request products, optionally restricted by ``fixed``);
``best_mapping`` evaluates them with a backend and returns the
fastest. Candidates are enumerated in a deterministic order; ``limit`` keeps
the first ``limit`` of them and reports how many were skipped, so a budget
never silently truncates the search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools

from arch_explorer.adf.evaluation import MappingSpec, evaluate
from arch_explorer.adf.multi_device import TEMPLATE as MULTI_DEVICE
from arch_explorer.adf.registry import MAPPING_TEMPLATES, mapping_spaces


@dataclass
class MappingSearchResult:
    best: MappingSpec
    evaluation: object
    evaluated: list = field(default_factory=list)   # (MappingSpec, latency_ns or error string)
    skipped: int = 0


def mapping_space(adf, workload, template, fixed=None):
    """List complete active mappings using the template's formal request contract."""
    instance = MAPPING_TEMPLATES.get(template)
    if not instance.applicable(adf, workload):
        return []
    space = instance.request_space(adf, workload, fixed)
    candidates = []
    for branch in mapping_spaces(space, fixed):
        keys = sorted(branch)
        candidates.extend(MappingSpec(template, dict(zip(keys, values)))
                          for values in itertools.product(*(branch[k] for k in keys)))
    return candidates


def best_mapping(adf, workload, templates=None, fixed=None, limit=None, backend="events", **options):
    """Evaluate every candidate mapping (up to ``limit``) and return the fastest."""
    names = templates or MAPPING_TEMPLATES.names()
    fixed = {} if fixed is None else fixed
    if not isinstance(fixed, dict) or any(name not in names for name in fixed):
        raise ValueError("fixed must name selected mapping templates")
    candidates = []
    for name in names:
        # Registration discovery is not a structured request. Explicit selection
        # or a supplied request (even empty/invalid) must reach formal validation.
        if name == MULTI_DEVICE and not templates and name not in fixed:
            continue
        candidates += mapping_space(adf, workload, name, fixed.get(name))
    skipped = 0
    if limit is not None and len(candidates) > limit:
        skipped = len(candidates) - limit
        candidates = candidates[:limit]
    best, evaluated = None, []
    for spec in candidates:
        try:
            result = evaluate(adf, workload, spec, backend=backend, **options)
        except (ValueError, NotImplementedError) as error:
            evaluated.append((spec, f"error: {error}"))
            continue
        evaluated.append((spec, result.latency_ns))
        if best is None or result.latency_ns < best[1].latency_ns:
            best = (spec, result)
    if best is None:
        raise ValueError(f"No valid mapping among {len(candidates)} candidates")
    return MappingSearchResult(best[0], best[1], evaluated, skipped)
