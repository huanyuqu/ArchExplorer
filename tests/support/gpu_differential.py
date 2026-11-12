"""QA-only strict comparison of explicitly normalized native observations.

This is not a production schema/parser or proof of component extraction. A
future adapter must cite each raw statistic's units, reset/cumulative scope,
and kernel attribution; it must never fill missing counters with zero. Native
run receipts and source/routing audits remain separate required evidence.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


BINDINGS = {"engine_baseline", "variant", "configuration", "program", "traces",
            "inputs", "mapping", "isa_scope"}
COUNTERS = {"cycles", "thread_instructions", "warp_instructions", "ctas",
            "l2_hits", "l2_misses", "dram_reads", "dram_writes",
            "icnt_requests", "icnt_responses", "backpressure_cycles"}
PENDING = {"kernel_queue", "requests", "responses", "physical_inflight", "held_resources"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unsigned(value, label, *, positive=False):
    require(type(value) is int and value >= (1 if positive else 0),
            label + " must be an exact integer")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _bindings(value):
    require(type(value) is dict and set(value) == BINDINGS, "incomplete input/source binding")
    require(all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in value.values()),
            "binding must contain explicit SHA256 digests")


def _validate(document, expected, role):
    require(type(document) is dict, "observation must be an object")
    require(document.get("schema") == "qa.gpu-differential-observation.v1", "unknown QA schema")
    require(document.get("role") == role, "wrong execution role")
    _bindings(document.get("bindings"))
    require(document["bindings"] == expected["bindings"], "different common inputs/source/variant")
    require(type(document.get("run_id")) is str and document["run_id"], "run identity required")
    require(document.get("exit_code") == 0 and type(document["exit_code"]) is int,
            "native process failed")
    require(document.get("normal_exit") is True, "normal native termination not established")
    require(document.get("truncated") is False, "truncated/unknown completion")
    require(document.get("system_drained") is True, "system not physically drained")
    pending = document.get("pending")
    require(type(pending) is dict and set(pending) == PENDING, "drain evidence missing")
    for key, value in pending.items():
        require(unsigned(value, key) == 0, "system_drained contradicts pending " + key)
    clocks = document.get("clocks_hz")
    require(type(clocks) is dict and set(clocks) == {"core", "l2", "dram", "icnt"},
            "all native clock domains required")
    for key, value in clocks.items():
        unsigned(value, key + " Hz", positive=True)
    require(clocks == expected["clocks_hz"], "native clock configuration differs")
    kernels = document.get("kernels")
    require(type(kernels) is list and len(kernels) == len(expected["kernels"]) > 0,
            "missing/extra kernels")
    require(len({row.get("uid") for row in kernels}) == len(kernels), "duplicate kernel completion")
    totals = dict.fromkeys(COUNTERS, 0)
    for row, work in zip(kernels, expected["kernels"]):
        require(row.get("name") == work["name"] and row.get("uid") == work["uid"],
                "kernel identity/order differs")
        unsigned(row["uid"], "kernel uid", positive=True)
        counters = row.get("counters")
        require(type(counters) is dict and set(counters) == COUNTERS,
                "missing/unscoped mechanism counters")
        for key in COUNTERS:
            unsigned(counters[key], key, positive=key == "cycles")
            totals[key] += counters[key]
        for key in ("thread_instructions", "warp_instructions", "ctas"):
            require(counters[key] == work[key], "incomplete retirement/CTA accounting")
    require(type(document.get("totals")) is dict and set(document["totals"]) == COUNTERS,
            "aggregate counter set incomplete")
    for key, value in document["totals"].items():
        unsigned(value, "total " + key)
        require(value == totals[key], "per-kernel/aggregate accounting differs")
    return Fraction(totals["cycles"] * 10**9, clocks["core"])


def compare(expected, oracle, candidate):
    """Compare exact observations; extraction/routing evidence is not inferred."""
    _bindings(expected["bindings"])
    oracle_ns = _validate(oracle, expected, "original_accelsim")
    candidate_ns = _validate(candidate, expected, "composed_components")
    require(oracle["run_id"] != candidate["run_id"], "same execution reused as both sides")
    for key in ("kernels", "totals", "clocks_hz"):
        require(oracle[key] == candidate[key], "oracle/candidate " + key + " differ")
    require(oracle_ns == candidate_ns, "different exact ns conversion")
    return {"status": "equal_normalized_observations", "expected_binding": digest(expected),
            "latency_ns_fraction": [oracle_ns.numerator, oracle_ns.denominator],
            "architecture_and_raw_normalization_audit": "separately_required"}


def compare_files(expected, oracle_ref, candidate_ref):
    """Read two distinct recorded observations; never start a process."""
    paths = []
    documents = []
    for ref in (oracle_ref, candidate_ref):
        require(type(ref) is dict and set(ref) == {"path", "sha256"}, "raw evidence reference required")
        path = Path(ref["path"])
        require(path.is_file() and not path.is_symlink(), "original evidence file required")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "evidence bytes differ")
        paths.append(path)
        documents.append(json.loads(raw))
    require(not paths[0].samefile(paths[1]), "same evidence reused as both executions")
    return compare(expected, *documents)
