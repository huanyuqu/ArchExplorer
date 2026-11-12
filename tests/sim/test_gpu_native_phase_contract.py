"""Synthetic phase fixtures are not native-run acceptance evidence."""
from copy import deepcopy
from fractions import Fraction

import pytest

from gpu_native_phases import DRAM, L2, SCALARS, compare_phases, validate_phases


def state(cycles, total, instructions, total_insn, ctas, *, pending=0):
    value = dict.fromkeys(SCALARS, 0)
    value.update(cycles=cycles, total_cycles=total, instructions=instructions,
                 total_instructions=total_insn, total_cta=ctas, pending=pending,
                 native_busy=int(bool(pending)), system_drained=not pending,
                 l2=[dict.fromkeys(L2, 0)], dram=[dict.fromkeys(DRAM, 0)])
    return value


def observations(nonzero=True):
    foreground = state(0, 210, 0, 256, 2, pending=4 if nonzero else 0)
    final = state(17, 227, 0, 256, 2) if nonzero else deepcopy(foreground)
    # Native mechanism snapshots may progress after all kernels have retired.
    if nonzero:
        final["dram"][0]["writeback"] = 12
    base = {"schema": "arch-explorer.accelsim-component-metrics.v1", "mode": "oracle",
        "original_cycle_calls": 1100, "component_ticks": 0,
        "clock_periods": {"core": 1 / 1980000000, "icnt": 1e-9, "l2": 1e-9, "dram": 2e-9},
        "kernels": [{"uid": 1, "metrics": state(101, 101, 128, 128, 1)},
                    {"uid": 2, "metrics": state(109, 210, 128, 256, 2)}],
        "foreground_final": foreground, "final": final,
        "drain_ticks": 91 if nonzero else 0, "drain_core_cycles": 17 if nonzero else 0}
    component = deepcopy(base)
    component.update(mode="components", original_cycle_calls=0, component_ticks=1100)
    return base, component


@pytest.mark.parametrize("nonzero", [True, False])
def test_foreground_kernel_sums_and_physical_final_are_separate(nonzero):
    oracle, component = observations(nonzero)
    result = compare_phases(oracle, component, [1, 2], 1980000000)
    assert result["foreground_ns"] == Fraction(3500, 33)
    assert result["physical_ns"] == Fraction((227 if nonzero else 210) * 50, 99)
    if nonzero:
        assert component["drain_ticks"] != component["drain_core_cycles"]
        assert component["final"]["total_cycles"] != sum(k["metrics"]["cycles"] for k in component["kernels"])


@pytest.mark.parametrize("change", ["final_writeback", "drain_ticks", "final_stall", "clock"])
def test_equal_foreground_does_not_hide_physical_difference(change):
    oracle, component = observations()
    if change == "final_writeback": component["final"]["dram"][0]["writeback"] += 1
    elif change == "drain_ticks": component["drain_ticks"] += 1
    elif change == "clock": component["clock_periods"]["dram"] *= 2
    else: component["final"]["stall_icnt_to_l2"] += 1
    with pytest.raises(ValueError, match="phase difference"):
        compare_phases(oracle, component, [1, 2], 1980000000)


@pytest.mark.parametrize("change", ["flag_only", "missing_phase", "core_difference", "tail_as_kernel",
    "retire_in_tail", "cta_in_tail", "tick_type", "old_engine", "wrong_mode", "zero_tail_mismatch"])
def test_forged_or_incomplete_phase_success_is_rejected(change):
    _, component = observations()
    if change == "flag_only": component["final"]["pending"] = 4
    elif change == "missing_phase": del component["foreground_final"]
    elif change == "core_difference": component["drain_core_cycles"] += 1
    elif change == "tail_as_kernel": component["kernels"].append({"uid": 3, "metrics": component["final"]})
    elif change == "retire_in_tail": component["final"]["total_instructions"] += 1
    elif change == "cta_in_tail": component["final"]["total_cta"] += 1
    elif change == "tick_type": component["drain_ticks"] = True
    elif change == "old_engine": component["schema"] = "arch-explorer.accelsim-engine.v2"
    elif change == "wrong_mode": component["original_cycle_calls"] = 1
    else: component["drain_ticks"] = component["drain_core_cycles"] = 0
    with pytest.raises(ValueError):
        validate_phases(component, [1, 2], 1980000000)


@pytest.mark.parametrize("limits", [{"max_cycles": 220}, {"max_ticks": 1050}])
def test_budget_covers_foreground_and_finish_without_reset(limits):
    _, component = observations()
    with pytest.raises(ValueError, match="budget exceeded"):
        validate_phases(component, [1, 2], 1980000000, **limits)
