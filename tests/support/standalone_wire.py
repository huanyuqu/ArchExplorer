"""Independent QA accounting for the pinned v1 control-packet observer.

For request ICNT egress: actual = 8*N + write_data, while the old reply-sized
observer = 8*N + read_data. Their sum must be 16*N + logical_data. Other edges
are unchanged. This checks both raw views; it does not erase wire differences.
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(rows):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in ("source", "source_port", "destination", "destination_port"))
        require(key not in result, "duplicate wire")
        for field in ("accepted", "logical_bytes", "transfer_bytes"):
            require(type(row[field]) is int and row[field] >= 0, "unsigned wire count required")
        result[key] = row
    require(bool(result), "wire observations missing")
    return result


def validate_wire_views(component, oracle):
    require(component["schema"] == "arch-explorer.accelsim-component-metrics.v2",
            "explicit standalone v2 required")
    require(oracle["schema"] == "arch-explorer.accelsim-component-metrics.v1",
            "pinned original v1 required")
    observer = component["oracle_observations"]
    require(observer["schema"] == oracle["schema"], "historical observer version differs")
    actual, historical, original = map(_index, (component["ports"], observer["ports"], oracle["ports"]))
    require(actual.keys() == historical.keys() == original.keys(), "wire identity set differs")
    differences = []
    for key, row in actual.items():
        old, native = historical[key], original[key]
        require(old == native, "pinned observation differs from original oracle")
        require(row["accepted"] == old["accepted"] and row["logical_bytes"] == old["logical_bytes"],
                "wire request count or logical width differs")
        n, logical, length = (row[k] for k in ("accepted", "logical_bytes", "transfer_bytes"))
        if key[0] == "icnt" and key[1].startswith("request_mem"):
            require(8*n <= length <= 8*n + logical, "actual request length violates 8B control")
            require(length + old["transfer_bytes"] == 16*n + logical,
                    "read/write observer conservation differs")
        else:
            require(length == old["transfer_bytes"], "unrelated wire length changed")
        if length != old["transfer_bytes"]:
            differences.append(dict(wire=list(key), components=length, oracle=old["transfer_bytes"]))
    allocation = component["allocation_domain"]
    require(all(type(allocation[k]) is int and allocation[k] >= 0
                for k in ("allocated", "destroyed", "live")), "allocation counts invalid")
    require(allocation["allocated"] > 0 and allocation["allocated"] == allocation["destroyed"]
            and allocation["live"] == 0, "actual allocation domain not empty")
    return differences
