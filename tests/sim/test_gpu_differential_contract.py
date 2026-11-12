"""VAL49 comparator preparation, not a component implementation acceptance.

All observations here are deliberately synthetic test fixtures. Real native
parsers and dynamic-protocol adapters wait for the owner's public contract.
"""
from copy import deepcopy
import hashlib
import json
import os
import subprocess

import pytest

from gpu_differential import BINDINGS, COUNTERS, PENDING, compare, compare_files


@pytest.fixture
def runs():
    expected = {"bindings": {name: hashlib.sha256(name.encode()).hexdigest() for name in BINDINGS},
        "clocks_hz": {"core": 1980000000, "l2": 1320000000, "dram": 850000000,
                      "icnt": 1100000000},
        "kernels": [{"name": name, "uid": i + 1, "thread_instructions": 128,
                     "warp_instructions": 4, "ctas": 1} for i, name in enumerate(("branch", "join"))]}
    rows = []
    for work, cycles in zip(expected["kernels"], (101, 109)):
        counters = dict.fromkeys(COUNTERS, 0)
        counters.update(cycles=cycles, thread_instructions=128, warp_instructions=4, ctas=1,
                        l2_hits=3, l2_misses=2, dram_reads=1, dram_writes=1,
                        icnt_requests=4, icnt_responses=4, backpressure_cycles=3)
        rows.append({"name": work["name"], "uid": work["uid"], "counters": counters})
    oracle = {"schema": "qa.gpu-differential-observation.v1", "role": "original_accelsim",
        "run_id": "independent-oracle-run", "bindings": deepcopy(expected["bindings"]),
        "exit_code": 0, "normal_exit": True, "truncated": False, "system_drained": True,
        "pending": dict.fromkeys(PENDING, 0), "clocks_hz": deepcopy(expected["clocks_hz"]),
        "kernels": rows, "totals": {key: sum(row["counters"][key] for row in rows) for key in COUNTERS}}
    candidate = deepcopy(oracle)
    candidate.update(role="composed_components", run_id="independent-component-run")
    return expected, oracle, candidate


def test_exact_comparison_uses_rational_clock_conversion_and_does_not_certify_extraction(runs):
    result = compare(*runs)
    assert result["latency_ns_fraction"] == [3500, 33]
    assert result["architecture_and_raw_normalization_audit"] == "separately_required"


@pytest.mark.parametrize("identity", sorted(BINDINGS))
def test_different_source_variant_trace_configuration_or_mapping_is_rejected(runs, identity):
    expected, oracle, candidate = runs
    candidate["bindings"][identity] = "0" * 64
    with pytest.raises(ValueError, match="different common inputs"):
        compare(expected, oracle, candidate)


@pytest.mark.parametrize("metric", sorted(COUNTERS))
def test_one_integer_difference_is_not_hidden_by_total_time_or_tolerance(runs, metric):
    expected, oracle, candidate = runs
    candidate["kernels"][0]["counters"][metric] += 1
    candidate["totals"][metric] += 1
    with pytest.raises(ValueError):
        compare(expected, oracle, candidate)


@pytest.mark.parametrize("replacement", [None, 0.0, True, -1, float("nan"), float("inf")])
def test_unknown_float_or_boolean_counter_is_not_coerced_to_zero(runs, replacement):
    expected, oracle, candidate = runs
    candidate["kernels"][0]["counters"]["l2_hits"] = replacement
    with pytest.raises(ValueError, match="exact integer"):
        compare(expected, oracle, candidate)


def test_missing_counter_is_not_assumed_zero_even_when_both_sides_omit_it(runs):
    expected, oracle, candidate = runs
    for document in (oracle, candidate):
        del document["kernels"][0]["counters"]["backpressure_cycles"]
    with pytest.raises(ValueError, match="missing/unscoped"):
        compare(expected, oracle, candidate)


@pytest.mark.parametrize("failure", ["missing", "reverse", "duplicate", "zero_exit_truncation",
    "drain_flag_only", "pending_write", "aggregate", "same_run", "wrong_clock"])
def test_incomplete_or_forged_success_is_rejected(runs, failure):
    expected, oracle, candidate = runs
    if failure == "missing": candidate["kernels"].pop()
    elif failure == "reverse": candidate["kernels"].reverse()
    elif failure == "duplicate": candidate["kernels"][1] = deepcopy(candidate["kernels"][0])
    elif failure == "zero_exit_truncation": candidate["truncated"] = True
    elif failure == "drain_flag_only": del candidate["pending"]
    elif failure == "pending_write": candidate["pending"]["physical_inflight"] = 1
    elif failure == "aggregate": candidate["totals"]["ctas"] += 1
    elif failure == "same_run": candidate["run_id"] = oracle["run_id"]
    elif failure == "wrong_clock": candidate["clocks_hz"]["dram"] += 1
    with pytest.raises(ValueError):
        compare(expected, oracle, candidate)


def test_recorded_comparison_is_read_only_and_refuses_replaced_or_shared_files(runs, tmp_path, monkeypatch):
    expected, oracle, candidate = runs
    refs = []
    for name, document in (("oracle", oracle), ("candidate", candidate)):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "normalized.json"
        path.write_text(json.dumps(document))
        refs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    def forbidden(*args, **kwargs):
        raise AssertionError("comparison executed a subprocess")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    before = {ref["path"]: open(ref["path"], "rb").read() for ref in refs}
    assert compare_files(expected, *refs)["status"] == "equal_normalized_observations"
    assert before == {path: open(path, "rb").read() for path in before}
    with pytest.raises(ValueError, match="same evidence"):
        compare_files(expected, refs[0], refs[0])
    path = refs[1]["path"]
    with open(path, "ab") as output:
        output.write(b" ")
    with pytest.raises(ValueError, match="evidence bytes differ"):
        compare_files(expected, *refs)
