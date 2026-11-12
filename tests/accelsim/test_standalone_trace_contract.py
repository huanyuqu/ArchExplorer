"""Synthetic unit controls for the QA observer; not native/factory acceptance."""
from copy import deepcopy

import pytest

from standalone_trace import audit_trace, assert_isolated, config_digest
from standalone_cases import memory_requests


def fixture():
    plan = {"instances": {"a": {"period": 2, "queue": 1},
                          "b": {"period": 5, "queue": 2}},
            "requests": [{"instance": name, "id": 1, "operation": "read",
                          "address": address, "bytes": 32}
                         for name, address in (("a", 4096), ("b", 8192))]}
    rows = []
    for name, address, end in (("a", 4096, 7), ("b", 8192, 11)):
        rows += [dict(instance=name, tick=0, kind="create",
                      configuration_sha256=config_digest(plan["instances"][name])),
                 dict(instance=name, tick=1, kind="offer", request_id=1, accepted=False,
                      ownership_before={"pending": [], "payload": [1, 2]},
                      ownership_after={"pending": [], "payload": [1, 2]}),
                 dict(instance=name, tick=2, kind="offer", request_id=1, accepted=True),
                 dict(instance=name, tick=end, kind="complete", request_id=1, response_to=1,
                      response_id=101, operation="read", address=address, bytes=32),
                 dict(instance=name, tick=end, kind="physical", pending=[],
                      native_busy=1, drained=False),
                 dict(instance=name, tick=end+2, kind="physical", pending=[],
                      native_busy=0, drained=True),
                 dict(instance=name, tick=end+2, kind="destroy")]
    # This is submission/observer order, deliberately not a global timestamp sort.
    a, b = rows[:7], rows[7:]
    for row in rows:
        if row["kind"] == "offer":
            row.update(operation="read", address=4096 if row["instance"] == "a" else 8192,
                       bytes=32)
    return plan, [*a[:3], *b[:3], *a[3:], *b[3:]]


def test_local_ids_and_clocks_are_instance_scoped_and_tail_is_real():
    plan, rows = fixture()
    result = audit_trace(plan, rows)
    assert result["a"]["completed"] == result["b"]["completed"] == [1]
    assert result["a"]["rejected"] == result["b"]["rejected"] == 1
    assert_isolated([r for r in rows if r["instance"] == "b"], rows, "b")


@pytest.mark.parametrize("defect,reason", [
    ("rejected-consumed", "response without pending"),
    ("duplicate", "response without pending"),
    ("wrong-parent", "response parent"),
    ("cross-instance", "response payload"),
    ("lost", "pending ownership"),
    ("reject-mutation", "rejection changed"),
    ("false-drain", "claimed drain"),
    ("early-destroy", "destroy before"),
    ("clock", "clock moved backwards"),
    ("after-destroy", "event after destruction"),
    ("config", "configuration identity"),
    ("offer-payload", "offered payload"),
])
def test_bad_observations_cannot_pass_as_reuse(defect, reason):
    plan, rows = fixture()
    create, reject, offer = rows[:3]
    complete = next(r for r in rows if r["instance"] == "a" and r["kind"] == "complete")
    tail = next(r for r in rows if r["instance"] == "a" and r["kind"] == "physical")
    if defect == "rejected-consumed":
        rows.remove(offer)
    elif defect == "duplicate":
        rows.insert(rows.index(complete)+1, deepcopy(complete))
    elif defect == "wrong-parent":
        complete["response_to"] = 2
    elif defect == "cross-instance":
        complete["address"] = 8192
    elif defect == "lost":
        rows.remove(complete)
    elif defect == "reject-mutation":
        reject["ownership_after"]["payload"][0] = 9
    elif defect == "false-drain":
        tail["drained"] = True
    elif defect == "early-destroy":
        tail["kind"] = "destroy"
    elif defect == "clock":
        complete["tick"] = 0
    elif defect == "after-destroy":
        rows.append(dict(instance="a", tick=50, kind="physical", pending=[],
                         native_busy=0, drained=True))
    elif defect == "config":
        create["configuration_sha256"] = "0" * 64
    else:
        offer["address"] = 8192
    with pytest.raises(ValueError, match=reason):
        audit_trace(plan, rows)


def test_failure_returns_ownership_and_does_not_count_as_completion():
    plan, rows = fixture()
    plan["allow_return"] = ["a"]
    index = next(i for i, r in enumerate(rows) if r["instance"] == "a" and r["kind"] == "complete")
    rows[index:index+1] = [dict(instance="a", tick=7, kind="failure", reason="injected service error"),
                           dict(instance="a", tick=7, kind="return", request_id=1,
                                ownership_returned=True)]
    result = audit_trace(plan, rows)
    assert result["a"]["completed"] == [] and result["a"]["returned"] == [1]
    assert result["b"]["completed"] == [1]
    assert_isolated([r for r in fixture()[1] if r["instance"] == "b"], rows, "b")


def test_dependency_is_a_completed_response_not_submission_order():
    plan, rows = fixture()
    plan["requests"].append(dict(instance="a", id=2, operation="write", address=4096,
                                 bytes=32, after=[1]))
    rows.insert(3, dict(instance="a", tick=3, kind="offer", request_id=2, accepted=True,
                        operation="write", address=4096, bytes=32))
    with pytest.raises(ValueError, match="before dependency response"):
        audit_trace(plan, rows)


def test_another_instance_cannot_silently_advance_a_local_clock_or_queue():
    _, rows = fixture()
    control = deepcopy([r for r in rows if r["instance"] == "b"])
    rows[-1]["tick"] += 1
    with pytest.raises(ValueError, match="instance isolation"):
        assert_isolated(control, rows, "b")


def test_pressure_inputs_exceed_capacity_and_keep_read_dirty_hot_dependencies():
    for instance, lines, base in (("a", 2, 4096), ("b", 4, 36864)):
        requests = memory_requests(instance, base=base, line_bytes=128, resident_lines=lines)
        assert [r["operation"] for r in requests[:3]] == ["read", "write", "read"]
        assert requests[1]["after"] == [requests[0]["id"]]
        assert requests[2]["after"] == [requests[1]["id"]]
        assert len({r["address"] for r in requests}) > lines
        assert all(dep < r["id"] for r in requests for dep in r["after"])
        assert set(requests[-1]["after"]) == {r["id"] for r in requests[3:-1]}
        assert all(r["bytes"] == len(r["byte_mask"]) == 32 for r in requests)


def test_byte_masks_are_part_of_the_observed_request_identity():
    plan, rows = fixture()
    plan["requests"][0]["byte_mask"] = [1, 0] * 16
    for row in rows:
        if row["instance"] == "a" and row["kind"] == "offer":
            row["byte_mask"] = [1] * 32
    with pytest.raises(ValueError, match="offered byte mask"):
        audit_trace(plan, rows)


@pytest.mark.parametrize("available", [True, False])
def test_passive_queries_preserve_time_payload_and_ownership(available):
    plan, rows = fixture()
    before = {"clock": 1, "owned": [], "payload": [1, 2]}
    row = dict(instance="a", kind="query", tick=1, available=available,
               state_before=before, state_after=deepcopy(before))
    rows.insert(2, row)
    audit_trace(plan, rows)
    row["state_after"]["clock"] = 2
    with pytest.raises(ValueError, match="passive query mutated"):
        audit_trace(plan, rows)
