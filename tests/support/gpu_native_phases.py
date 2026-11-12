"""Independent QA checks for SIM38's published native metrics, without execution.

Kernel snapshots precede native update_stats. Sum only the documented current
cycles/instructions into foreground totals; mechanism arrays are snapshots,
not additive per-kernel counters. Final includes real physical tail work.
"""
from fractions import Fraction
import math

from gpu_differential import require, unsigned


SCALARS = {"cycles", "total_cycles", "instructions", "total_instructions", "total_cta",
           "stall_core_to_icnt", "stall_icnt_to_l2", "stall_l2_to_icnt", "stall_icnt_to_core",
           "icnt_injected", "icnt_ejected", "icnt_packet_bytes", "native_busy", "pending"}
L2 = {"index", "accesses", "misses", "hits", "pending_hits", "reservation_fails",
      "port_available_cycles", "data_port_busy_cycles", "fill_port_busy_cycles"}
DRAM = {"index", "commands", "activity", "nop", "activate", "precharge", "read", "write",
        "writeback", "requests"}


def snapshot(value):
    require(type(value) is dict and set(value) == SCALARS | {"l2", "dram", "system_drained"},
            "incomplete native mechanism snapshot")
    for name in SCALARS:
        unsigned(value[name], name)
    require(type(value["system_drained"]) is bool, "explicit physical drain flag required")
    for name, fields in (("l2", L2), ("dram", DRAM)):
        rows = value[name]
        require(type(rows) is list and bool(rows), "complete native partition array required")
        for index, row in enumerate(rows):
            require(type(row) is dict and set(row) == fields, "native partition fields differ")
            for key, count in row.items():
                unsigned(count, key)
            require(row["index"] == index, "partition index set/order incomplete")


def validate_phases(value, expected_uids, core_hz, *, max_cycles=None, max_ticks=None):
    require(value.get("schema") == "arch-explorer.accelsim-component-metrics.v1", "native schema differs")
    require(value.get("mode") in ("components", "oracle"), "native mode required")
    unsigned(core_hz, "core Hz", positive=True)
    periods = value.get("clock_periods")
    require(type(periods) is dict and set(periods) == {"core", "icnt", "l2", "dram"},
            "four native clock periods required")
    require(all(type(period) in (int, float) and math.isfinite(period) and period > 0
                for period in periods.values()), "invalid native clock period")
    for key in ("drain_ticks", "drain_core_cycles", "original_cycle_calls", "component_ticks"):
        unsigned(value.get(key), key)
    require(value["original_cycle_calls"] == 0 if value["mode"] == "components"
            else value["component_ticks"] == 0, "wrong execution path")
    calls = value["component_ticks"] if value["mode"] == "components" else value["original_cycle_calls"]
    require(calls > 0 and calls >= value["drain_ticks"], "tail exceeds complete tick accounting")
    kernels = value.get("kernels")
    require(type(kernels) is list and [k.get("uid") for k in kernels] == expected_uids,
            "kernel identity/order differs (tail is not a kernel)")
    require(bool(expected_uids) and len(set(expected_uids)) == len(expected_uids), "complete kernel set required")
    for kernel in kernels:
        unsigned(kernel["uid"], "uid", positive=True)
        snapshot(kernel["metrics"])
    foreground, final = value.get("foreground_final"), value.get("final")
    snapshot(foreground)
    snapshot(final)
    for phase in (foreground, final):
        require(len(phase["l2"]) == len(kernels[0]["metrics"]["l2"])
                and len(phase["dram"]) == len(kernels[0]["metrics"]["dram"]), "phase topology differs")
    for current, total in (("cycles", "total_cycles"), ("instructions", "total_instructions")):
        require(sum(k["metrics"][current] for k in kernels) == foreground[total],
                "kernel sums belong to foreground, not physical final")
    require(kernels[-1]["metrics"]["total_cta"] == foreground["total_cta"], "foreground CTA differs")
    require(final["total_cycles"] == foreground["total_cycles"] + value["drain_core_cycles"],
            "physical core-cycle difference differs")
    for key in ("total_instructions", "total_cta"):
        require(final[key] == foreground[key], "tail fabricated retirement/CTA")
    require(final["system_drained"] and final["native_busy"] == 0 and final["pending"] == 0,
            "physical work still pending")
    if value["drain_ticks"] == 0:
        require(final == foreground and value["drain_core_cycles"] == 0, "zero tail changed the snapshot")
    if max_cycles is not None:
        require(final["total_cycles"] <= unsigned(max_cycles, "cycle limit", positive=True),
                "combined foreground/finish cycle budget exceeded")
    if max_ticks is not None:
        require(calls <= unsigned(max_ticks, "tick limit", positive=True), "combined tick budget exceeded")
    return {"foreground_ns": Fraction(foreground["total_cycles"] * 10**9, core_hz),
            "physical_ns": Fraction(final["total_cycles"] * 10**9, core_hz)}


def compare_phases(oracle, component, expected_uids, core_hz, **limits):
    require(oracle.get("mode") == "oracle" and component.get("mode") == "components", "independent modes required")
    left = validate_phases(oracle, expected_uids, core_hz, **limits)
    right = validate_phases(component, expected_uids, core_hz, **limits)
    for key in ("clock_periods", "kernels", "foreground_final", "drain_ticks", "drain_core_cycles", "final"):
        require(oracle[key] == component[key], "phase difference: " + key)
    require(left == right, "exact foreground/physical ns differ")
    return {"status": "equal_phase_observations", **left,
            "input_receipts_and_extraction_audit": "separately_required"}
