import pytest

from arch_explorer.adf import (Conventions, ComponentType, Evaluation, MappingSpec, Param, ParamSpec,
                               SpaceMatrix, SpacePoint, WorkloadSpec, evaluate, find, fingerprint,
                               from_dict, instance_count, register_backend, validate)
from arch_explorer.adf.registry import Registry


class FakeDram(ComponentType):
    name = "fake_dram"
    role = "memory"
    params = {"capacity_bytes": ParamSpec("B", "int", extensive=True),
              "bandwidth": ParamSpec("B/ns", extensive=True),
              "latency": ParamSpec("ns")}

    def build(self, point, context):
        return None


class FakeBus(ComponentType):
    name = "fake_bus"
    role = "network"
    params = {"bandwidth": ParamSpec("B/ns")}
    conventions = Conventions(arbitration="fair_share")

    def build(self, point, context):
        return None


def registry():
    result = Registry("component", ComponentType)
    result.register(FakeDram)
    result.register(FakeBus)
    return result


def dram(name="dram"):
    return SpacePoint(name, "fake_dram", {
        "capacity_bytes": Param(8 * 2**30, "B", "test"),
        "bandwidth": Param(1555.0, "B/ns", "test"),
        "latency": Param(100.0, "ns", "test")})


def tree():
    core = SpaceMatrix("core", [dram("sram")], repeat=4)
    return SpaceMatrix("chip", [core, dram()], [SpacePoint("noc", "fake_bus", {"bandwidth": Param(64.0, "B/ns")})])


def test_round_trip_paths_and_fingerprint():
    chip = tree()
    validate(chip, registry())
    again = from_dict(chip.to_dict())
    assert again == chip
    assert fingerprint(again) == fingerprint(chip)
    assert find(chip, "chip/core/sram").component == "fake_dram"
    assert instance_count(chip, "chip/core/sram") == 4


def test_validation_rejects_wrong_units_and_duplicates():
    bad = dram()
    bad.params["latency"] = Param(100.0, "cycle", "test")
    with pytest.raises(ValueError, match="unit"):
        validate(SpaceMatrix("chip", [bad]), registry())
    with pytest.raises(ValueError, match="Duplicate"):
        validate(SpaceMatrix("chip", [dram(), dram()]))
    with pytest.raises(ValueError, match="network"):
        validate(SpaceMatrix("chip", [], [dram("noc")]), registry())


def test_split_divides_only_extensive_parameters_and_merge_inverts():
    component = registry().get("fake_dram")
    parts = component.split(dram(), 4)
    assert [p.value("capacity_bytes") for p in parts] == [2 * 2**30] * 4
    assert parts[0].value("bandwidth") == pytest.approx(1555.0 / 4)
    assert parts[0].value("latency") == 100.0
    merged = component.merge(parts)
    assert merged.value("capacity_bytes") == 8 * 2**30
    assert merged.value("bandwidth") == pytest.approx(1555.0)


def test_cycle_components_must_declare_clock():
    class Bad(FakeDram):
        name = "bad"
        conventions = Conventions(time_base="cycle")
    with pytest.raises(ValueError, match="clock_ghz"):
        Bad().validate(dram())


def test_backend_dispatch():
    with pytest.raises(NotImplementedError):
        evaluate(tree(), WorkloadSpec("operator", {"op": "matmul"}), MappingSpec("none"), backend="missing")
    register_backend("contract-test", lambda adf, workload, mapping: Evaluation(latency_ns=1.0))
    result = evaluate(tree(), WorkloadSpec("operator", {}), MappingSpec("none"), backend="contract-test")
    assert result.latency_ns == 1.0 and result.backend == "contract-test" and result.cost_usd is None
    assert result.area_mm2 is None and result.details["cost"]["unsupported"]  # fake components have no cost model


def test_cost_estimate_fills_unset_area_and_cost_only():
    from arch_explorer.cost import estimate
    from arch_explorer.archs import dmc_adf
    adf = dmc_adf()
    expected = estimate(adf)
    register_backend("contract-cost", lambda adf, workload, mapping: Evaluation(latency_ns=1.0))
    result = evaluate(adf, WorkloadSpec("operator", {}), MappingSpec("none"), backend="contract-cost")
    assert (result.area_mm2, result.cost_usd) == (expected.area_mm2, expected.cost_usd)
    assert result.details["cost"]["basis"] == "raw"
    full = evaluate(adf, WorkloadSpec("operator", {}), MappingSpec("none"), backend="contract-cost",
                    cost_model={"basis": "full"})
    assert full.cost_usd == estimate(adf, options={"basis": "full"}).cost_usd
    skipped = evaluate(adf, WorkloadSpec("operator", {}), MappingSpec("none"), backend="contract-cost",
                       cost_model=False)
    assert skipped.area_mm2 is None and skipped.cost_usd is None and "cost" not in skipped.details
    register_backend("contract-own-cost",
                     lambda adf, workload, mapping: Evaluation(latency_ns=1.0, area_mm2=1.0, cost_usd=2.0))
    own = evaluate(adf, WorkloadSpec("operator", {}), MappingSpec("none"), backend="contract-own-cost")
    assert (own.area_mm2, own.cost_usd) == (1.0, 2.0) and "cost" not in own.details


def test_events_backend_loads_on_first_use():
    import json
    import subprocess
    import sys
    from arch_explorer.archs import dmc_adf
    adf = json.dumps(dmc_adf().to_dict())
    code = ("import json, sys\n"
            "from arch_explorer.adf import MappingSpec, WorkloadSpec, evaluate, from_dict\n"
            "from arch_explorer.adf.evaluation import backends\n"
            "assert 'events' not in backends()\n"
            "adf = from_dict(json.loads(sys.stdin.read()))\n"
            "r = evaluate(adf, WorkloadSpec('operator', {'op': 'softmax', 'rows': 1024, 'cols': 1024}),"
            " MappingSpec('gsm_tiled', {}))\n"
            "assert r.backend == 'events' and r.latency_ns > 0 and isinstance(r.details, dict)\n")
    subprocess.run([sys.executable, "-c", code], input=adf, text=True, check=True)
