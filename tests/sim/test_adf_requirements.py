"""The ADF requirements documented in docs/components.md ("ADF requirements of the builder and the templates")."""
import pytest

from arch_explorer.adf import MappingSpec, Param, SpaceMatrix, SpacePoint, WorkloadSpec, evaluate
from arch_explorer.adf.registry import MAPPING_TEMPLATES
from arch_explorer.simulator.events import build_simulator

S = "test fixture"


def p(value, unit):
    return Param(value, unit, S)


def systolic():
    return SpacePoint("pe", "systolic_array", {"rows": p(16, "count"), "cols": p(16, "count"),
                                                "clock_ghz": p(1.0, "GHz"), "dataflow": p("ws", "1"),
                                                "precision": p("fp16", "1")})


def vpu():
    return SpacePoint("vpu", "vector_unit", {"lanes": p(32, "count"), "sfu_lanes": p(8, "count"),
                                             "clock_ghz": p(1.0, "GHz"), "precision": p("fp16", "1")})


def sram(name="sram"):
    return SpacePoint(name, "sram", {"capacity": p(2**24, "B"), "bandwidth": p(128.0, "B/ns"), "latency": p(1.0, "ns")})


def dram():
    return SpacePoint("dram", "dram", {"technology": p("ddr", "1"), "capacity": p(2**32, "B"),
                                       "bandwidth": p(64.0, "B/ns"), "latency": p(80.0, "ns")})


def mesh(rows, cols):
    return SpacePoint("noc", "mesh2d", {"rows": p(rows, "count"), "cols": p(cols, "count"),
                                        "link_bandwidth": p(32.0, "B/ns"), "hop_latency": p(1.0, "ns")})


def bus():
    return SpacePoint("bus", "bus", {"bandwidth": p(64.0, "B/ns"), "latency": p(1.0, "ns")})


def chip(core_elements, rows, cols, cores=16):
    return SpaceMatrix("chip", [SpaceMatrix("core", core_elements, repeat=cores), dram()], [mesh(rows, cols)])


SOFTMAX = WorkloadSpec("operator", {"op": "softmax", "rows": 64, "cols": 64})


def test_mesh_must_hold_every_expanded_endpoint_including_dram():
    with pytest.raises(ValueError, match="17 endpoints do not fit a 4x4 mesh"):
        build_simulator(chip([systolic(), vpu(), sram()], 4, 4))
    build_simulator(chip([systolic(), vpu(), sram()], 3, 6))


def test_container_without_network_warns():
    adf = chip([vpu(), sram()], 4, 5)
    result = evaluate(adf, SOFTMAX, MappingSpec("dmc_partition", {}))
    assert any("chip/core has no network" in warning for warning in result.warnings)


def test_root_cannot_repeat():
    with pytest.raises(ValueError, match="root cannot repeat"):
        build_simulator(SpaceMatrix("chip", [vpu(), sram(), dram()], [bus()], repeat=2))


def test_dmc_cores_need_their_own_sram():
    template = MAPPING_TEMPLATES.get("dmc_partition")
    assert template.applicable(chip([systolic(), vpu(), sram()], 4, 5), SOFTMAX)
    shared = SpaceMatrix("chip", [SpaceMatrix("core", [systolic(), vpu()], repeat=16), sram("pool"), dram()],
                         [mesh(3, 6)])
    assert not template.applicable(shared, SOFTMAX)


def test_vector_operators_need_a_vector_capable_unit():
    with pytest.raises(NotImplementedError):
        evaluate(chip([systolic(), sram()], 4, 5), SOFTMAX, MappingSpec("dmc_partition", {}))
    gpu = SpaceMatrix("gpu", [systolic(), sram("l2"), dram()], [bus()])
    with pytest.raises(NotImplementedError):
        evaluate(gpu, SOFTMAX, MappingSpec("gsm_tiled", {}))
    gpu = SpaceMatrix("gpu", [vpu(), sram("l2"), dram()], [bus()])
    assert evaluate(gpu, SOFTMAX, MappingSpec("gsm_tiled", {})).latency_ns > 0


def test_device_templates_need_exactly_one_device():
    gpu = SpaceMatrix("gpu", [vpu(), sram("l2"), dram()], [bus()], repeat=2)
    server = SpaceMatrix("server", [gpu], [bus()])
    assert not MAPPING_TEMPLATES.get("gsm_tiled").applicable(server, SOFTMAX)


def bonded():
    return SpacePoint("bonded", "dram", {"technology": p("bonded", "1"), "capacity": p(2**26, "B"),
                                         "bandwidth": p(16.0, "B/ns"), "latency": p(0.0, "ns")})


def test_bonded_shards_and_dram_groups_do_not_make_devices():
    from arch_explorer.mapping.base import adf_devices
    # A core with a bonded shard is not a device; the chip with the main DRAM is.
    assert adf_devices(chip([vpu(), sram(), bonded()], 4, 5)) == [("chip", 1)]
    # Main memory split into banks: one group, the chip is the device and the group its memory.
    banks = SpaceMatrix("dram", [SpaceMatrix("bank", [dram()], [bus()], repeat=4)])
    adf = SpaceMatrix("chip", [SpaceMatrix("core", [vpu(), sram()], [bus()], repeat=2), banks], [mesh(1, 3)])
    sim = build_simulator(adf)
    assert list(sim.memory_groups) == ["chip/dram"] and adf_devices(adf) == [("chip", 1)]
    assert evaluate(adf, SOFTMAX, MappingSpec("dmc_partition", {"activations": "dram"})).latency_ns > 0


def test_pim_unit_needs_one_bank_dram_in_its_container():
    pim = SpacePoint("pim", "pim_unit", {"ops_per_cycle": p(32, "ops/cycle"), "clock_ghz": p(1.0, "GHz"),
                                         "precision": p("fp16", "1")})
    two = dram()
    two.repeat = 2
    adf = SpaceMatrix("chip", [vpu(), sram(), SpaceMatrix("bank", [two, pim], [bus()])], [bus()])
    from arch_explorer.components.base import Kernel
    with pytest.raises(ValueError, match="exactly one sibling dram"):
        build_simulator(adf).points["chip/bank/pim"].gemv_time(Kernel("mvm", (1, 8, 8), "fp16"), 1)
