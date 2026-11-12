"""Synthetic checks of the independent observer authority, not native runs."""
from copy import deepcopy

import pytest

from standalone_wire import validate_wire_views


def case():
    # Two read requests and one write, each 32B. 8B control is preserved.
    a = dict(source="icnt", source_port="request_mem0", destination="l2_0",
             destination_port="request", accepted=3, logical_bytes=96, transfer_bytes=56)
    h = dict(a, transfer_bytes=88)
    reply = dict(source="l2_0", source_port="reply", destination="icnt",
                 destination_port="reply_mem0", accepted=3, logical_bytes=96, transfer_bytes=88)
    oracle = dict(schema="arch-explorer.accelsim-component-metrics.v1", ports=[h, reply])
    component = dict(schema="arch-explorer.accelsim-component-metrics.v2",
                     ports=[a, deepcopy(reply)], oracle_observations=deepcopy(oracle),
                     allocation_domain=dict(allocated=7, destroyed=7, live=0))
    return component, oracle


def test_actual_and_historical_lengths_are_retained_separately():
    component, oracle = case()
    differences = validate_wire_views(component, oracle)
    assert len(differences) == 1
    assert differences[0]["components"] == 56 and differences[0]["oracle"] == 88


@pytest.mark.parametrize("defect,reason", [
    ("transfer", "conservation"), ("control", "8B control"),
    ("observer", "original oracle"), ("count", "count or logical"),
    ("logical", "count or logical"), ("other-edge", "unrelated wire"),
    ("duplicate", "duplicate wire"), ("missing", "identity set"),
    ("leak", "allocation domain"), ("free", "allocation domain"),
    ("boolean", "unsigned wire"),
])
def test_wire_differences_cannot_hide_other_mechanism_corruption(defect, reason):
    component, oracle = case()
    row = component["ports"][0]
    if defect == "transfer": row["transfer_bytes"] += 1
    elif defect == "control": row["transfer_bytes"] = 0
    elif defect == "observer": component["oracle_observations"]["ports"][0]["transfer_bytes"] += 1
    elif defect == "count": row["accepted"] += 1
    elif defect == "logical": row["logical_bytes"] += 32
    elif defect == "other-edge": component["ports"][1]["transfer_bytes"] += 1
    elif defect == "duplicate": component["ports"].append(deepcopy(row))
    elif defect == "missing": component["ports"].pop()
    elif defect == "leak": component["allocation_domain"]["live"] = 1
    elif defect == "free": component["allocation_domain"]["destroyed"] = 6
    else: row["accepted"] = True
    with pytest.raises(ValueError, match=reason):
        validate_wire_views(component, oracle)
