from non_gpu_fixture import shared_adf
import pytest

from arch_explorer.adf import COMPONENTS, MAPPING_TEMPLATES, register_backend
from arch_explorer.agent_env.environment import load_registrations
from core_fakes import FakeMapping, component_types, fake_backend


@pytest.fixture(autouse=True)
def fake_registry(monkeypatch):
    # Load production registration initializers before temporarily replacing
    # their registries. This also works after integration with WP-SIM-01.
    load_registrations({})
    monkeypatch.setattr(COMPONENTS, "_items", component_types())
    monkeypatch.setattr(MAPPING_TEMPLATES, "_items", {FakeMapping.name: FakeMapping})
    register_backend("core-test", fake_backend)


@pytest.fixture
def config(tmp_path):
    return {"seed_adf": shared_adf().to_dict(),
            "workloads": [{"kind": "operator", "params": {"operations": 4096}, "name": "small", "weight": 1.0}],
            "backend": "core-test", "max_evaluations": 24, "random_seed": 17,
            "parameter_samples": 2, "mapping_samples": 2, "parameter_radius": 0.1,
            "growth_probability": 1.0, "cost_budget_usd": 1000.0, "area_budget_mm2": 500.0,
            "cost_penalty": 3.0, "area_penalty": 2.0,
            "search": {"max_steps": 100, "temperature": 1.0, "cooling": 0.95, "max_relative_degradation": 0.25}}
