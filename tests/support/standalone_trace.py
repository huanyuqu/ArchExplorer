"""QA observer format, NOT a factory API or proof of native independence.

A future installed-public-API driver records calls/results in this format.
Local request IDs are scoped by instance; a driver must retain its wire-ID map.
Ticks are instance-local observations, not core cycles or wall time. Reject
snapshots contain payload/queue/ownership state, excluding diagnostic counters.
Native counters, constructor/link witnesses and actual cache behavior require
separate evidence; a synthetic transcript passing here proves none of them.
"""
import hashlib
import json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _uint(value):
    _require(type(value) is int and value >= 0, "unsigned integer required")
    return value


def config_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def audit_trace(plan, events):
    """Check observed ownership, response identity and physical destruction.

    plan['instances'] maps labels to actual configurations (opaque here).
    plan['requests'] supplies independently frozen client requests. Explicit
    'after' dependencies concern response receipt, not arbitrary FIFO service.
    'allow_return' names failure-control instances: their accepted requests may
    return ownership instead of completing, but are never counted as success.
    """
    configurations = plan["instances"]
    _require(type(configurations) is dict and bool(configurations), "instances required")
    expected = {}
    for request in plan["requests"]:
        instance, rid = request["instance"], _uint(request["id"])
        _require(instance in configurations, "unknown planned instance")
        key = instance, rid
        _require(key not in expected, "duplicate planned request")
        _require(request["operation"] in ("read", "write"), "read/write required")
        _uint(request["address"])
        _require(_uint(request["bytes"]) > 0, "positive byte count required")
        if "byte_mask" in request:
            mask = request["byte_mask"]
            _require(type(mask) is list and len(mask) == request["bytes"]
                     and all(type(bit) is int and bit in (0, 1) for bit in mask),
                     "one binary mask entry per logical byte required")
        expected[key] = request
    for (instance, rid), request in expected.items():
        _require(all((instance, dep) in expected and dep != rid
                     for dep in request.get("after", [])), "invalid dependency")
    allowed_returns = set(plan.get("allow_return", []))
    _require(allowed_returns <= configurations.keys(), "unknown failure control")
    states = {}
    for event in events:
        name, kind, tick = event["instance"], event["kind"], _uint(event["tick"])
        _require(name in configurations, "unplanned observed instance")
        if kind == "create":
            _require(name not in states, "duplicate instance creation")
            _require(event["configuration_sha256"] == config_digest(configurations[name]),
                     "configuration identity differs")
            states[name] = dict(tick=tick, pending=set(), accepted=set(), completed=set(),
                                returned=set(), replies=set(), rejected=0,
                                drained=False, destroyed=False, failed=False)
            continue
        _require(name in states, "event before creation")
        state = states[name]
        _require(not state["destroyed"], "event after destruction")
        _require(tick >= state["tick"], "instance clock moved backwards")
        state["tick"] = tick
        if kind == "query":
            _require(type(event["available"]) is bool, "explicit availability required")
            _require(event["state_before"] == event["state_after"], "passive query mutated state")
        elif kind == "offer":
            rid = _uint(event["request_id"])
            _require((name, rid) in expected, "unplanned request")
            _require(rid not in state["accepted"], "accepted request offered again")
            _require(not state["failed"], "offer after instance failure")
            request = expected[name, rid]
            _require(all(event[key] == request[key] for key in ("operation", "address", "bytes")),
                     "offered payload identity differs")
            if "byte_mask" in request:
                _require(event["byte_mask"] == request["byte_mask"], "offered byte mask differs")
            _require(type(event["accepted"]) is bool, "explicit accept/reject required")
            if not event["accepted"]:
                _require(event["ownership_before"] == event["ownership_after"],
                         "rejection changed payload/queue/ownership")
                state["rejected"] += 1
            else:
                _require(all(dep in state["completed"] for dep in request.get("after", [])),
                         "request accepted before dependency response")
                state["accepted"].add(rid)
                state["pending"].add(rid)
                state["drained"] = False
        elif kind == "complete":
            rid = _uint(event["request_id"])
            _require(rid in state["pending"], "response without pending accepted request")
            request = expected[name, rid]
            _require(_uint(event["response_to"]) == rid, "response parent differs")
            reply = _uint(event["response_id"])
            _require(reply not in state["replies"], "response identity reused")
            _require(all(event[key] == request[key] for key in ("operation", "address", "bytes")),
                     "response payload identity differs")
            state["pending"].remove(rid)
            state["completed"].add(rid)
            state["replies"].add(reply)
            state["drained"] = False
        elif kind == "failure":
            _require(name in allowed_returns and bool(event["reason"]), "unplanned failure")
            state["failed"] = True
            state["drained"] = False
        elif kind == "return":
            rid = _uint(event["request_id"])
            _require(state["failed"] and rid in state["pending"], "unowned failure return")
            _require(event["ownership_returned"] is True, "ownership not returned")
            state["pending"].remove(rid)
            state["returned"].add(rid)
            state["drained"] = False
        elif kind == "physical":
            pending = event["pending"]
            _require(type(pending) is list and len(pending) == len(set(pending))
                     and all(type(rid) is int for rid in pending)
                     and set(pending) == state["pending"], "pending ownership differs")
            busy = _uint(event["native_busy"])
            _require(type(event["drained"]) is bool, "explicit drain state required")
            _require(not event["drained"] or (not pending and busy == 0),
                     "claimed drain while work remains")
            state["drained"] = event["drained"]
        elif kind == "destroy":
            _require(not state["pending"] and state["drained"], "destroy before physical drain")
            state["destroyed"] = True
        else:
            raise ValueError("unknown observation kind")
    _require(set(states) == set(configurations), "missing instance")
    for name, state in states.items():
        required = {rid for instance, rid in expected if instance == name}
        _require(state["destroyed"], "instance not destroyed")
        _require(required == state["completed"] | state["returned"],
                 "planned request lost or never accepted")
    return {name: {key: sorted(value) if isinstance(value, set) else value
                   for key, value in state.items() if key != "tick"}
            for name, state in states.items()}


def assert_isolated(control, interleaved, instance):
    """Compare independently recorded local observations, not summed counters."""
    def project(events):
        rows = [dict(event) for event in events if event["instance"] == instance]
        _require(bool(rows) and rows[0]["kind"] == "create", "isolation control missing")
        origin = rows[0]["tick"]
        for event in rows:
            event["tick"] -= origin
        return rows
    _require(project(control) == project(interleaved), "instance isolation differs")
