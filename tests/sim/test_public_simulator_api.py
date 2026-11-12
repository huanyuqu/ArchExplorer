"""Exercise public preparation, receipts and typed inputs without native execution."""
from copy import deepcopy
import hashlib
import json

import pytest

from arch_explorer.adf import MappingSpec, WorkloadSpec
from arch_explorer.archs import dmc_adf
from arch_explorer.components.api import get_provider, register_provider
from arch_explorer.simulator.api import assess_request, evaluate_request, prepare_request


def prepare(output, options=None):
    return prepare_request(dmc_adf(1),
        WorkloadSpec("operator", {"op": "matmul", "batch": 1, "m": 4, "k": 8, "n": 4}),
        MappingSpec("dmc_partition", {"weights": "local", "activations": "local"}),
        output_dir=output, provider="events", options=options)


def seal(value):
    value.pop("binding", None)
    value["binding"] = {"sha256": hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()}
    return value


@pytest.mark.parametrize("options", [[], False, 0, "", [1], True, 1, "invalid"])
def test_non_object_options_rejected_before_preparation(tmp_path, options):
    output = tmp_path / "unprepared"
    with pytest.raises((TypeError, ValueError), match="options.*JSON object"):
        prepare(output, options)
    assert not output.exists(), "invalid input must not create provider artifacts"


@pytest.mark.parametrize("options", [None, {}])
def test_default_options_are_objects_and_prepare_does_not_execute(tmp_path, options):
    output = tmp_path / "prepared"
    request = prepare(output, options)
    assert request["options"] == {} and request["payload"]["options"] == {}
    assert {p.name for p in output.iterdir()} == {"REQUEST.json"}
    assert assess_request(request) == {"supported": True, "gaps": []}
    assert {p.name for p in output.iterdir()} == {"REQUEST.json"}


def test_real_events_receipt_replays_without_writes_or_second_attempt(tmp_path):
    output = tmp_path / "events"
    request = prepare(output, {"trace": True})
    result = evaluate_request(request)
    assert result["request"] == request
    assert result["evaluation"]["latency_ns"] > 0
    assert (output / "RAW.json").is_file()
    before = {p.name: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in output.iterdir()}
    assert evaluate_request(result) == result
    after = {p.name: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in output.iterdir()}
    assert before == after
    with pytest.raises(FileExistsError):
        evaluate_request(request)


def test_rehashed_outer_inputs_cannot_change_the_prepared_payload(tmp_path):
    request = prepare(tmp_path / "events")
    changed = deepcopy(request)
    changed["workload"]["params"]["m"] += 1
    seal(changed)
    with pytest.raises(ValueError, match="payload differs from generic workload"):
        assess_request(changed)


def test_idempotent_registration_requires_the_same_provider_object():
    provider = get_provider("events")
    register_provider(provider)
    assert get_provider("events") is provider
    with pytest.raises(ValueError, match="version/identity conflict"):
        register_provider(type(provider)())
    assert get_provider("events") is provider


def test_public_types_preserve_scheduler_and_component_identity():
    from arch_explorer.components import api
    from arch_explorer.components.base import Demand
    from arch_explorer.simulator.events.engine import Task, TaskGraph, Resource, simulate
    from arch_explorer.simulator.events.protocol import ExecutionEvent, ExecutionRequest
    for actual in (Demand, Task, TaskGraph, Resource, simulate, ExecutionEvent, ExecutionRequest):
        assert getattr(api, actual.__name__) is actual


def test_joint_native_domain_cannot_fall_back_to_ordinary_events(tmp_path):
    from arch_explorer.components.accelsim.scenarios import sm90_adf
    output = tmp_path / "unsupported-domain"
    with pytest.raises((ValueError, TypeError, NotImplementedError), match="(?i)domain|provider|accelsim"):
        prepare_request(sm90_adf(), WorkloadSpec("operator",
            {"op": "matmul", "batch": 1, "m": 4, "k": 8, "n": 4}),
            MappingSpec("gsm_tiled", {}), output_dir=output, provider="events")
    assert not output.exists()


def test_missing_provider_fails_locally_without_preparing_or_replacing_events(tmp_path):
    provider = get_provider("events")
    output = tmp_path / "unavailable"
    with pytest.raises(NotImplementedError, match="unknown/unavailable provider"):
        prepare_request(dmc_adf(1), WorkloadSpec("operator", {"op": "matmul", "m": 4, "k": 8, "n": 4}),
            MappingSpec("dmc_partition", {}), output_dir=output, provider="qa.absent.provider")
    assert not output.exists()
    assert get_provider("events") is provider
