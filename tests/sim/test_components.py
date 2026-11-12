"""Component library v1, conventions binding and the hardware builder."""
import pytest

import arch_explorer.simulator.events  # noqa: F401  (registers components)
from arch_explorer.adf import COMPONENTS, ComponentType, Conventions, Param, ParamSpec, SpaceMatrix, SpacePoint
from arch_explorer.adf.registry import Registry
from arch_explorer.components import Kernel, systolic_model
from arch_explorer.components.base import GATEWAY, ComputeModel
from arch_explorer.simulator.events.builder import build_simulator

S = "unit test"

CATALOG = {  # PLAN.md component catalog v1: role, required parameters with units, extensive ones
    "systolic_array": ("compute", {"rows": "count", "cols": "count", "clock_ghz": "GHz", "dataflow": "1",
                                   "precision": "1"}, set()),
    "vector_unit": ("compute", {"lanes": "count", "sfu_lanes": "count", "clock_ghz": "GHz", "precision": "1"}, set()),
    "sram": ("memory", {"capacity": "B", "bandwidth": "B/ns", "latency": "ns"}, {"capacity", "bandwidth"}),
    "dram": ("memory", {"technology": "1", "capacity": "B", "bandwidth": "B/ns", "latency": "ns"},
             {"capacity", "bandwidth"}),
    "mesh2d": ("network", {"rows": "count", "cols": "count", "link_bandwidth": "B/ns", "hop_latency": "ns"}, set()),
    "bus": ("network", {"bandwidth": "B/ns", "latency": "ns"}, set()),
    "crossbar": ("network", {"port_bandwidth": "B/ns", "latency": "ns"}, set()),
    "p2p_switch": ("network", {"link_bandwidth": "B/ns", "links_per_endpoint": "count", "latency": "ns"}, set()),
}


def p(value, unit):
    return Param(value, unit, S)


def sram(name="sram", capacity=2**20, bandwidth=64.0):
    return SpacePoint(name, "sram", {"capacity": p(capacity, "B"), "bandwidth": p(bandwidth, "B/ns"),
                                     "latency": p(2.0, "ns")})


def dram(**extra):
    params = {"technology": p("hbm", "1"), "capacity": p(2**30, "B"), "bandwidth": p(100.0, "B/ns"),
              "latency": p(50.0, "ns")}
    params.update({k: p(v, "B/ns") for k, v in extra.items()})
    return SpacePoint("dram", "dram", params)


def systolic(rows=16, cols=16, dataflow="ws", clock=2.0):
    return SpacePoint("pe", "systolic_array", {"rows": p(rows, "count"), "cols": p(cols, "count"),
                                               "clock_ghz": p(clock, "GHz"), "dataflow": p(dataflow, "1"),
                                               "precision": p("fp16", "1")})


def mesh(rows, cols):
    return SpacePoint("noc", "mesh2d", {"rows": p(rows, "count"), "cols": p(cols, "count"),
                                        "link_bandwidth": p(32.0, "B/ns"), "hop_latency": p(1.5, "ns")})


def chip(cores=4, rows=2, cols=3):
    core = SpaceMatrix("core", [systolic(), sram()], repeat=cores)
    return SpaceMatrix("chip", [core, dram()], [mesh(rows, cols)])


def test_catalog_names_units_and_extensive_flags_follow_plan():
    for name, (role, required, extensive) in CATALOG.items():
        component = COMPONENTS.get(name)
        assert component.role == role
        declared = {key: spec.unit for key, spec in component.params.items() if spec.required}
        assert declared == required, name
        assert {key for key, spec in component.params.items() if spec.extensive and spec.required} == extensive
        if "clock_ghz" in component.params:
            assert component.conventions.time_base == "cycle"


def test_builder_expands_repeat_assigns_coordinates_and_records_provenance():
    sim = build_simulator(chip())
    assert sorted(sim.points) == sorted([f"chip/core[{i}]/{n}" for i in range(4) for n in ("pe", "sram")]
                                        + ["chip/dram"])
    assert sim.points["chip/core[2]/sram"].coord == (2, 1)
    assert sim.points["chip/dram"].coord == (4,)
    assert sim.container("chip/core[3]").endpoint == 3
    sim.points["chip/core[0]/pe"].kernel_demand(Kernel("gemm", (1, 8, 8, 8)))
    rows = {(row["path"], row["param"]) for row in sim.provenance.as_list()}
    assert ("chip/core/pe", "rows") in rows
    assert all(row["source"] == S for row in sim.provenance.as_list() if row["param"] != "gemm_model")
    assert any(row["param"] == "gemm_model" and "SCALE-Sim" in row["source"] for row in sim.provenance.as_list())


def test_systolic_cycles_are_scalesim_cycles_over_clock():
    sim = build_simulator(chip())
    pe = sim.points["chip/core[0]/pe"]
    demand = pe.kernel_demand(Kernel("gemm", (3, 40, 50, 60)))
    cycles = systolic_model.native_gemm(16, 16, "ws", 40, 50, 60).compute_cycles
    assert demand.resources[pe.unit.name] == pytest.approx(3 * cycles / 2.0)
    with pytest.raises(NotImplementedError, match="vector_unit"):
        pe.kernel_demand(Kernel("vector", (4, 4), simt=1))
    with pytest.raises(NotImplementedError, match="fp16"):
        pe.kernel_demand(Kernel("gemm", (1, 4, 4, 4), "fp32"))


def test_vector_unit_rates():
    vec = SpacePoint("vpu", "vector_unit", {"lanes": p(32, "count"), "sfu_lanes": p(4, "count"),
                                            "clock_ghz": p(0.5, "GHz"), "precision": p("fp16", "1")})
    sim = build_simulator(SpaceMatrix("dev", [vec, dram()]))
    vpu = sim.points["dev/vpu"]
    softmax = Kernel("vector", (10, 100), simt=4, sfu=1, row_sfu=1)
    assert vpu.kernel_demand(softmax).resources[vpu.unit.name] == pytest.approx(max(4000 / 32, 1010 / 4) / 0.5)


def test_mesh_xy_routing_and_hierarchy_route():
    sim = build_simulator(chip(cores=4, rows=2, cols=3))
    noc = sim.root.networks[0]
    demand = noc.route(0, 4)          # core[0] at (0,0) -> dram at (1,1): X hop first, then Y hop
    assert set(demand.resources) == {"chip/noc:link(0, 0)->(0, 1)", "chip/noc:link(0, 1)->(1, 1)"}
    assert demand.latency_ns == pytest.approx(3.0)
    assert set(noc.route(4, 0).resources) == {"chip/noc:link(1, 1)->(1, 0)", "chip/noc:link(1, 0)->(0, 0)"}
    route = sim.route("chip/core[0]/sram", "chip/core[3]/sram")   # core 3 sits at tile (1, 0)
    assert set(route.resources) == {"chip/noc:link(0, 0)->(1, 0)"}
    assert any("core has no network" in w for w in sim.warnings)
    transfer = sim.transfer("chip/dram", "chip/core[1]/sram", 1000)
    assert transfer.resources["chip/dram:port"] == 1000 and transfer.resources["chip/core[1]/sram:port"] == 1000
    assert transfer.latency_ns == pytest.approx(50.0 + 1.5 * len(noc.route(4, 1).resources))
    with pytest.raises(ValueError, match="do not fit"):
        build_simulator(chip(cores=6, rows=2, cols=3))


def test_bus_crossbar_and_p2p_resources():
    def server(network):
        dev = SpaceMatrix("dev", [systolic(), dram()], repeat=3)
        return SpaceMatrix("srv", [dev], [network])
    bus = build_simulator(server(SpacePoint("net", "bus", {"bandwidth": p(10.0, "B/ns"), "latency": p(1.0, "ns")})))
    assert set(bus.root.networks[0].route(0, 2).resources) == {"srv/net:bus"}
    xbar = build_simulator(server(SpacePoint("net", "crossbar", {"port_bandwidth": p(10.0, "B/ns"),
                                                                 "latency": p(1.0, "ns")})))
    assert set(xbar.root.networks[0].route(0, 2).resources) == {"srv/net:out0", "srv/net:in2"}
    p2p = build_simulator(server(SpacePoint("net", "p2p_switch", {
        "link_bandwidth": p(25.0, "B/ns"), "links_per_endpoint": p(12, "count"), "latency": p(0.0, "ns"),
        "pair_links": p(4, "count")})))
    route = p2p.root.networks[0].route(1, 2)
    resources = p2p.resources
    assert resources["srv/net:out1"].capacity == 300 and resources["srv/net:pair1->2"].capacity == 100
    assert set(route.resources) == {"srv/net:out1", "srv/net:in2", "srv/net:pair1->2"}
    assert p2p.route("srv/dev[0]/dram", "srv/dev[2]/dram").resources.keys() >= {"srv/net:pair0->2"}


def test_dram_sustained_bandwidth_replaces_peak_and_is_checked():
    sim = build_simulator(SpaceMatrix("dev", [systolic(), dram(sustained_bandwidth=80.0)]))
    assert sim.points["dev/dram"].port.capacity == 80.0
    with pytest.raises(ValueError, match="exceeds"):
        build_simulator(SpaceMatrix("dev", [systolic(), dram(sustained_bandwidth=120.0)]))


def test_builder_rejects_incompatible_conventions_and_unknown_components():
    class NoClock(ComponentType):
        name = "no_clock"
        role = "compute"
        params = {"ops": ParamSpec("ops/cycle")}
        conventions = Conventions(time_base="cycle")

        def build(self, point, context):
            return ComputeModel(self, point, context)

    class BadBase(NoClock):
        name = "bad_base"
        params = {"ops": ParamSpec("ops/cycle"), "clock_ghz": ParamSpec("GHz")}
        conventions = Conventions(time_base="tick")

    registry = Registry("component", ComponentType)
    for component in (NoClock, BadBase):
        registry.register(component)
    for name in ("sram", "dram"):
        registry.register(type(COMPONENTS.get(name)))
    with pytest.raises(ValueError, match="clock_ghz"):
        build_simulator(SpaceMatrix("d", [SpacePoint("u", "no_clock", {"ops": p(1, "ops/cycle")}), dram()]), registry)
    with pytest.raises(ValueError, match="time_base"):
        build_simulator(SpaceMatrix("d", [SpacePoint("u", "bad_base", {"ops": p(1, "ops/cycle"),
                                                                       "clock_ghz": p(1.0, "GHz")}), dram()]), registry)
    with pytest.raises(KeyError, match="Unknown component"):
        build_simulator(SpaceMatrix("d", [SpacePoint("u", "warp_drive", {})]))
    with pytest.raises(ValueError, match="no compute"):
        build_simulator(SpaceMatrix("d", [dram()]))
    assert GATEWAY == -1
