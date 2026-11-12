from copy import deepcopy
import math
import random

import pytest

from arch_explorer.adf import COMPONENTS, Param, SpaceMatrix, SpacePoint, find
from arch_explorer.agent_env import ExplorationEnv, replay_result
from arch_explorer.agent_env.sampling import parameter_variant
from non_gpu_fixture import shared_adf
from arch_explorer.cost import estimate
from arch_explorer.cost.constants import CONSTANTS, DEFAULTS
from arch_explorer.cost.formulas import manufacturing, raw_die_cost, register_area, sram_area
from arch_explorer.search.nested import run_search


def param(value, unit="1"):
    return Param(value, unit, "explicit synthetic test assumption").to_dict()


def dram(tech="ddr", capacity=512 * 2**20):
    return SpacePoint("memory", "dram", {"technology":Param(tech, source="test"),
        "capacity":Param(capacity, "B", "test"), "bandwidth":Param(32, "B/ns", "test")})


def test_constants_and_assumptions_have_sources():
    assert all(p.source and p.unit for p in [*CONSTANTS.values(), *DEFAULTS.values()])
    estimate(shared_adf())  # Standard component names need no component callbacks.


def test_cache_equations_and_units():
    assert sram_area(512 * 1024) == pytest.approx(512 * 1024 * 8 * 1.7e-7 / 0.602)
    assert sram_area(1024 * 1024) == 2 * sram_area(512 * 1024)
    assert sram_area(100) == sram_area(4096)
    assert sram_area(0) == 0
    assert sram_area(4096, 5) / sram_area(4096, 7) == pytest.approx(1.25 / 1.70)
    assert register_area(4, 16384, 32, 4) > 0
    with pytest.raises(ValueError):
        sram_area(-1)


def test_raw_and_full_cost_formulas():
    area = 100
    gross_dies = math.pi * 150**2 / area - math.pi * 300 / math.sqrt(2 * area)
    raw = 9346 / gross_dies
    assert raw_die_cost(area) == pytest.approx(raw)
    full = manufacturing(area, 7)
    yield_die = (1 + 0.0009 * area / 10)**-10
    expected = (raw / yield_die + 7 + 100 * 0.005 + 100 * 4 * 0.005) / 0.99
    assert full["total_usd"] == pytest.approx(expected)
    assert full["die_yield"] == pytest.approx(yield_die)
    assert manufacturing(200, 0, dies=2)["raw_die_usd"] == pytest.approx(2 * raw)


def test_nested_repeats_and_unknown_components_are_not_free():
    leaf = SpacePoint("cache", "sram", {"capacity":Param(1024 * 1024, "B", "test")}, repeat=3)
    tree = SpaceMatrix("root", [SpaceMatrix("tile", [leaf], repeat=4)])
    result = estimate(tree)
    assert result.area_mm2 == pytest.approx(12 * sram_area(1024 * 1024))
    assert result.by_path[0]["instances"] == 12
    tree.elements.append(SpacePoint("unknown", "unmodeled"))
    result = estimate(tree)
    assert result.area_mm2 is None and result.cost_usd is None
    assert result.by_path[-1]["area_mm2"] is None
    assert result.unsupported[-1]["path"] == "root/unknown"
    assert result.by_category["sram"] > 0  # Only a known partial subtotal.


def test_bonded_capacity_layers_and_yield():
    tree = SpaceMatrix("root", [dram("bonded", 2**30)])
    original = deepcopy(tree.to_dict())
    result = estimate(tree)
    assert result.memory["die_area_mm2"] == pytest.approx(40)
    assert result.memory["bonded_footprint_mm2"] == pytest.approx(10)
    assert result.cost_usd == pytest.approx(7 / 0.95**4)
    result = estimate(tree, options={"overrides":{"root/memory":{"bonded_layers":param(2, "count")}}})
    assert result.memory["bonded_footprint_mm2"] == pytest.approx(20)
    assert result.cost_usd == pytest.approx(7 / 0.95**2)
    assert tree.to_dict() == original
    invalid = estimate(tree, options={"overrides":{"root/memory":{"bonded_layers":param(5, "count")}}})
    assert invalid.cost_usd is None and invalid.unsupported


@pytest.mark.parametrize("tech,expected", [("ddr", 1.44), ("hbm", 4.2)])
def test_pim_overhead_and_explicit_memory_area(tech, expected):
    unit = SpacePoint("pu", "pim_unit", {"ops_per_cycle":Param(32, "ops/cycle", "test"),
        "clock_ghz":Param(1, "GHz", "test"), "precision":Param("bf16", source="test")}, repeat=16)
    tree = SpaceMatrix("root", [dram(tech), unit])
    result = estimate(tree)
    assert result.memory["cost_usd"] == pytest.approx(expected)
    assert result.memory["die_area_mm2"] is None
    assert any(row["metrics"] == ["memory_die_area_mm2"] for row in result.unsupported)
    result = estimate(tree, options={"overrides":{"root/memory":{"dram_density":param(16, "MiB/mm2")}}})
    assert result.memory["die_area_mm2"] == pytest.approx(512 / 16 * 1.2)
    assert not result.unsupported
    assert result.by_path[1]["area_mm2"] == 0  # PU overhead already on memory die.
    if tech == "ddr":
        unit.params["ops_per_cycle"] = Param(64, "ops/cycle", "test")
        larger = estimate(tree)
        assert larger.memory["cost_usd"] == pytest.approx(2 * expected)


def test_gddr_has_no_invented_market_price():
    tree = SpaceMatrix("root", [dram("gddr")])
    result = estimate(tree)
    assert result.cost_usd is None
    assert result.area_mm2 > 0
    assert result.by_path[0]["raw_die_usd"] > 0
    assert result.memory["cost_usd"] is None
    assert result.by_path[0]["memory_usd"] is None
    assert all("area_mm2" not in issue["metrics"] for issue in result.unsupported)
    priced = estimate(tree, options={"overrides":{"root/memory":{"dram_usd_gib":param(3, "USD/GiB")}}})
    assert priced.memory["cost_usd"] == 1.5
    assert priced.area_mm2 == result.area_mm2


def test_unpriced_memory_and_process_scope_are_independent_of_area():
    from arch_explorer.devices import get
    tree = shared_adf()
    find(tree, "chip/dram").params["technology"] = Param("gddr", "1", "synthetic unpriced memory")
    result = estimate(tree)
    assert result.area_mm2 == pytest.approx(sum(result.by_category.values()))
    assert result.area_mm2 > 0 and result.cost_usd is None
    assert any("dram_usd_gib" in issue["reason"] and issue["metrics"] == ["cost_usd"] for issue in result.unsupported)
    assert any(a.get("status") == "unverified_default" for a in result.assumptions)
    explicit = estimate(tree, options={"process_nm":param(5, "nm")})
    assert not any(a.get("status") == "unverified_default" for a in explicit.assumptions)
    with pytest.raises(ValueError, match="5 or 7"):
        estimate(tree, options={"process_nm":param(12, "nm")})
    invalid_price = estimate(tree, options={"overrides":{"chip/dram":{"dram_usd_gib":param(3, "USD")}}})
    assert invalid_price.area_mm2 == result.area_mm2 and invalid_price.cost_usd is None





def test_grouped_pim_presets_cost_once_and_report_missing_evidence_once(monkeypatch):
    from arch_explorer.components.memory import Dram
    from arch_explorer.devices import neupims, hsndi
    monkeypatch.setitem(COMPONENTS._items, "dram", Dram)
    for tree, group_path, part_count, interfaces, memory_usd in (
        (neupims(), "neupims/hbm", 1024, 2, 32 * 7 * 1.2),
        (hsndi(), "hsndi/dram", 128, 4, None),
    ):
        before = tree.to_dict()
        result = estimate(tree)
        row = next(r for r in result.by_path if r["path"] == group_path)
        assert row["part_instances"] == part_count
        assert len(row["member_paths"]) == part_count
        assert row["pim_units"] == part_count
        assert row["interface_count"] == interfaces
        if memory_usd is None:
            assert row["memory_usd"] is None
        else:
            assert row["memory_usd"] == pytest.approx(memory_usd)
        assert row["pim_overhead"] == pytest.approx(0.2)
        assert result.area_mm2 > 0
        assert len(result.unsupported) == (1 if memory_usd else 2)
        assert all(issue["affected_path_count"] == part_count for issue in result.unsupported)
        assert all(len(issue["example_paths"]) <= 4 for issue in result.unsupported)
        assert tree.to_dict() == before
        priced = estimate(tree, options={"overrides":{group_path:{"dram_usd_gib":param(3, "USD/GiB"),
                                                        "dram_density":param(16, "MiB/mm2")}}})
        assert priced.area_mm2 == result.area_mm2
        capacity_gib = 32 if memory_usd else 4
        assert priced.memory["cost_usd"] == pytest.approx(capacity_gib * 3 * 1.2)
        assert priced.memory["die_area_mm2"] == pytest.approx(capacity_gib * 1024 / 16 * 1.2)


def test_memory_group_aggregation_preserves_ancestor_repeats_and_cost_sums(monkeypatch):
    from arch_explorer.components.memory import Dram
    from arch_explorer.devices import neupims
    monkeypatch.setitem(COMPONENTS._items, "dram", Dram)
    tree = neupims(channels=2, banks=2)
    baseline = estimate(tree)
    tree.repeat = 3
    result = estimate(tree)
    assert result.area_mm2 == pytest.approx(3 * baseline.area_mm2)
    assert result.memory["cost_usd"] == pytest.approx(3 * baseline.memory["cost_usd"])
    assert sum(row["area_mm2"] for row in result.by_path) == pytest.approx(result.area_mm2)
    assert sum(row["raw_cost_usd"] for row in result.by_path) == pytest.approx(result.raw_cost_usd)


def test_model_options_and_override_validation():
    for options in [{"process_nm":param(3, "nm")}, {"basis":"bad"}, {"die_count":param(0, "count")},
                    {"process_nm":{"value":7,"unit":"nm","source":""}}]:
        with pytest.raises(ValueError):
            estimate(shared_adf(), options=options)
    result = estimate(shared_adf(), options={"overrides":{"bad/path":{"capacity":param(8, "B")}}})
    assert result.cost_usd is None and "Unused" in result.unsupported[-1]["reason"]


def test_physical_counts_cannot_undercount_active_hardware_and_paths_sum():
    tree = shared_adf()
    invalid = estimate(tree, options={"overrides":{"chip/core/vector":{"physical_instances":param(1, "count")}}})
    assert invalid.area_mm2 is None and invalid.cost_usd is None
    result = estimate(tree)
    assert sum(row["raw_cost_usd"] for row in result.by_path) == pytest.approx(result.raw_cost_usd)
    assert sum(row["area_mm2"] for row in result.by_path) == pytest.approx(result.area_mm2)




def test_shrink_sampler_reduces_size_and_preserves_domains():
    tree = shared_adf()
    baseline = estimate(tree).cost_usd
    candidates = [parameter_variant(tree, COMPONENTS, random.Random(i), 0.1, shrink=True) for i in range(80)]
    assert all(estimate(candidate).cost_usd <= baseline for candidate in candidates)
    assert any(estimate(candidate).cost_usd < baseline for candidate in candidates)
    for candidate in candidates:
        assert find(candidate, "chip/core/vector").params["clock_ghz"] == find(tree, "chip/core/vector").params["clock_ghz"]
    with pytest.raises(ValueError):
        parameter_variant(tree, COMPONENTS, random.Random(0), 0.1, grow=True, shrink=True)


def test_budget_pressure_and_cost_options_survive_search_replay(config):
    config.update(cost_budget_usd=0.1, area_budget_mm2=None, shrink_probability=1.0,
                  parameter_samples=4, mapping_samples=1, max_evaluations=16,
                  cost_model={"basis":"full"})
    env = ExplorationEnv(config)
    base = estimate(env.adf, options=config["cost_model"]).cost_usd
    result = env.evaluate(optimize="parameters_and_mapping")["result"]
    assert all(t["score"]["metrics"]["cost_usd"] <= base for t in result["trials"])
    assert any(t["score"]["metrics"]["cost_usd"] < base for t in result["trials"])
    search = run_search(config, policy="sa")
    assert search["cost_model"] == {"basis":"full"}
    assert search["best"] == replay_result(search)["best"]


def test_explicit_domains_validate_and_replay_after_path_deleted(config):
    config["parameter_domains"] = [{"path":"chip/l2", "param":"capacity", "values":[8*2**20,16*2**20],
                                    "source":"search-space design"}]
    env = ExplorationEnv(config)
    outside = env.set_parameters(parameters=[{"path":"chip/l2", "param":"capacity", "value":12*2**20}])
    assert outside["error"]["code"] == "invalid_parameters"
    assert "result" in env.apply_edits(edits=[{"op":"delete", "component":"chip/l2"}])
    assert "result" in env.evaluate()
    payload = env.submit()["result"]
    assert replay_result(payload)["best"] == payload["best"]
    for values in ([], [-1, 16*2**20], [8*2**20], [float("nan"), 16*2**20]):
        config["parameter_domains"][0]["values"] = values
        with pytest.raises(ValueError):
            ExplorationEnv(config)
