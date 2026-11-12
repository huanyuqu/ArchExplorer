from copy import deepcopy

import pytest

from arch_explorer.adf import COMPONENTS, Param, SpaceMatrix, SpacePoint, find, fingerprint, instance_count
from arch_explorer.archs.adf_seeds import bus, dmc_adf, memory, mesh
from non_gpu_fixture import shared_adf
from arch_explorer.structure import (StructureError, apply_edit, bond_memory, check, chiplet_transform, create,
                                     delete, gsm_to_dmc, insert, merge, normalize, pim_transform, remove, replay, split)


def same(left, right):
    assert fingerprint(normalize(left)) == fingerprint(normalize(right))


def test_split_merge_conservation_and_immutable_inputs():
    original = dmc_adf()
    before = deepcopy(original)
    divided = split(original, "chip/dram", 4)
    assert original == before
    for child in find(divided.adf, "chip/dram").elements:
        assert child.value("capacity") == 2**28
        assert child.value("bandwidth") == 16.0
        assert child.value("latency") == 100.0
    merged = merge(divided.adf, [f"chip/dram/dram_{i}" for i in range(4)], name="dram")
    flattened = remove(merged.adf, "chip/dram")
    assert find(flattened.adf, "chip/dram").value("capacity") == 2**30
    same(replay(original, [divided.record, merged.record, flattened.record]), flattened.adf)
    # Output mutation must not leak into an earlier tree or record.
    flattened.adf.elements.clear()
    assert original == before and divided.adf.elements


def test_repeat_group_and_remove_preserve_physical_count():
    original = dmc_adf(16)
    grouped = insert(original, "chip", ["chip/core"], name="chiplet", group_size=4)
    assert find(grouped.adf, "chip/chiplet").repeat == 4
    assert find(grouped.adf, "chip/chiplet/core").repeat == 4
    assert instance_count(grouped.adf, "chip/chiplet/core/vector") == 16
    same(remove(grouped.adf, "chip/chiplet").adf, original)
    with pytest.raises(StructureError, match="divisible"):
        insert(original, "chip", ["chip/core"], group_size=3)


@pytest.mark.parametrize("edit,code", [
    ({"op": "delete", "component": "chip/missing"}, "path_not_found"),
    ({"op": "split", "point": "chip/core", "num": 2}, "wrong_role"),
    ({"op": "split", "point": "chip/noc", "num": 2}, "wrong_role"),
    ({"op": "remove", "level": "chip"}, "root_edit"),
    ({"op": "create", "container": "chip/core", "component": "chip"}, "cycle"),
    ({"op": "create", "container": "chip/core", "component": "chip/dram"}, "repeat_mismatch"),
    ({"op": "merge", "points": ["chip/dram", "chip/dram"]}, "invalid_selection"),
])
def test_structured_errors_are_atomic(edit, code):
    original = dmc_adf()
    before = original.to_dict()
    with pytest.raises(StructureError) as caught:
        apply_edit(original, edit)
    assert caught.value.code == code
    assert original.to_dict() == before


def test_create_move_and_delete():
    original = dmc_adf(1)
    moved = create(original, "chip/core", component="chip/dram")
    assert find(moved.adf, "chip/core/dram").component == "dram"
    with pytest.raises(KeyError):
        find(moved.adf, "chip/dram")
    added = create(original, "chip/core", spec=memory("other"))
    same(delete(added.adf, "chip/core/other").adf, original)


def test_normalization_idempotent_ordering_and_safe_collisions():
    original = dmc_adf(4)
    nested = insert(original, "chip", ["chip/dram"], name="wrapper").adf
    same(nested, original)
    nested.elements.reverse()
    same(nested, original)
    canonical = normalize(nested)
    assert canonical == normalize(canonical)
    # A container carrying a fabric is a meaningful spatial level.
    wired = create(nested, "chip/wrapper", spec=bus()).adf
    assert fingerprint(normalize(wired)) != fingerprint(canonical)


def test_chiplet_high_level_matches_handwritten_basic_program():
    root = dmc_adf(16)
    noc, nop = mesh(cols=4), mesh("nop", cols=5)
    expected = replay(root, [
        {"op": "delete", "component": "chip/noc"},
        {"op": "insert", "parent": "chip", "children": ["chip/core"], "name": "chiplet", "group_size": 4},
        {"op": "create", "container": "chip/chiplet", "spec": noc.to_dict()},
        {"op": "create", "container": "chip", "spec": nop.to_dict()},
    ])
    actual = chiplet_transform(root, "chip", 4, noc, nop)
    same(actual.adf, expected)
    same(replay(root, actual.records), actual.adf)


def test_pim_structure_matches_paper_program_and_validates_registered_spec():
    root = SpaceMatrix("chip", [memory("dram", technology="ddr")])
    # A registered vector test double plays the injected PU role; no PIM evaluator claim.
    pu = find(dmc_adf(), "chip/core/vector").to_dict()
    expected_edits = [{"op": "split", "point": "chip/dram", "num": 4}]
    for i in range(4):
        target = f"chip/dram/channel_{i}"
        expected_edits.extend([
            {"op": "insert", "parent": "chip/dram", "children": [f"chip/dram/dram_{i}"], "name": f"channel_{i}"},
            {"op": "create", "container": target, "spec": pu},
            {"op": "create", "container": target, "spec": bus("pim_bus").to_dict()},
        ])
    actual = pim_transform(root, "chip/dram", "channel", 4, pu)
    assert actual.records == expected_edits
    same(actual.adf, replay(root, expected_edits))
    from arch_explorer.components.pim import PimUnit
    COMPONENTS.register(PimUnit)  # core tests otherwise use an isolated registry
    pu = SpacePoint("pim", "pim_unit", {"ops_per_cycle":Param(32, "ops/cycle", "test"),
        "clock_ghz":Param(1, "GHz", "test"), "precision":Param("fp16", source="test")}).to_dict()
    actual = pim_transform(root, "chip/dram", "bank", 4, pu)
    assert actual.adf.to_dict() == replay(root, actual.records).to_dict()
    del pu["params"]["ops_per_cycle"]
    with pytest.raises(StructureError, match="requires parameter ops_per_cycle") as caught:
        pim_transform(root, "chip/dram", "bank", 4, pu)
    assert caught.value.code == "invalid_adf"


def test_large_pim_transform_batches_validation_and_preserves_direct_tree(monkeypatch):
    from arch_explorer.components.pim import PimUnit
    from arch_explorer.devices.pim import pim_structure
    from arch_explorer.structure import transforms
    COMPONENTS.register(PimUnit)
    root = SpaceMatrix("chip", [memory("dram", technology="ddr")])
    before = deepcopy(root.to_dict())
    pu = SpacePoint("pim", "pim_unit", {"ops_per_cycle":Param(32, "ops/cycle", "test"),
        "clock_ghz":Param(1, "GHz", "test"), "precision":Param("fp16", source="test")})
    calls = []
    original_check = transforms.check
    def counted_check(tree, registry):
        calls.append(tree.name)
        return original_check(tree, registry)
    monkeypatch.setattr(transforms, "check", counted_check)
    result = pim_transform(root, "chip/dram", ["channel", "bank"], [32, 32], pu.to_dict())
    assert calls == ["chip", "spec", "chip"]
    assert len(result.records) == 3137
    direct = pim_structure(root.elements[0], ["channel", "bank"], [32, 32], pu, bus("pim_bus"))
    assert find(result.adf, "chip/dram").to_dict() == direct.to_dict()
    assert root.to_dict() == before


def test_gsm_to_dmc_matches_basic_privatization():
    root = shared_adf(2)
    core = find(root, "chip/core").to_dict()
    edits = [{"op": "split", "point": "chip/l2", "num": 2}, {"op": "remove", "level": "chip/l2"},
             {"op": "delete", "component": "chip/core"}]
    for i in range(2):
        edits.append({"op": "create", "container": "chip", "spec": {**core, "name": f"core_{i}", "repeat": 1}})
    for i in range(2):
        edits.extend([
            {"op": "insert", "parent": "chip", "children": [f"chip/core_{i}", f"chip/l2_{i}"], "name": f"tile_{i}"},
            {"op": "create", "container": f"chip/tile_{i}", "spec": bus("local_bus").to_dict()}])
    edits.extend([{"op": "delete", "component": "chip/noc"},
                  {"op": "create", "container": "chip", "spec": mesh(cols=3).to_dict()}])
    actual = gsm_to_dmc(root, "chip", "chip/l2", 2)
    same(actual.adf, replay(root, edits))
    assert sum(find(actual.adf, f"chip/tile_{i}/l2_{i}").value("capacity") for i in range(2)) == 2**24


def test_bond_memory_and_multilevel_pim():
    root = dmc_adf(4)
    result = bond_memory(root, "chip/core", memory("bonded", technology="bonded"))
    assert instance_count(result.adf, "chip/core/bonded") == 4
    same(result.adf, replay(root, result.records))
    pu = find(root, "chip/core/vector").to_dict()
    result = pim_transform(root, "chip/dram", ["channel", "bank"], [2, 2], pu)
    same(result.adf, replay(root, result.records))
    check(result.adf)


def test_network_roles_and_integer_partition_errors():
    root = dmc_adf()
    with pytest.raises(StructureError) as caught:
        chiplet_transform(root, "chip", 2, memory(), mesh())
    assert caught.value.code == "wrong_role"
    find(root, "chip/dram").params["capacity"] = Param(7, "B", "test")
    with pytest.raises(StructureError, match="divisible"):
        split(root, "chip/dram", 2)


def test_split_repeated_point_and_bond_point_preserve_instance_count():
    root = SpaceMatrix("chip", [memory("dram", technology="ddr")])
    root.elements[0].repeat = 3
    result = split(root, "chip/dram", 2).adf
    assert instance_count(result, "chip/dram/dram_0") == 3
    assert instance_count(result, "chip/dram/dram_1") == 3
    core = find(dmc_adf(), "chip/core/vector")
    core.repeat = 4
    root = SpaceMatrix("chip", [core])
    result = bond_memory(root, "chip/vector", memory("bonded", technology="bonded")).adf
    assert instance_count(result, "chip/vector_bonded/vector") == 4
    assert instance_count(result, "chip/vector_bonded/bonded") == 4


def test_public_edit_records_use_transport_schema_without_null_defaults():
    from arch_explorer.agent_env.schemas import BASIC_SCHEMAS, validate_arguments
    root = dmc_adf()
    divided = split(root, "chip/dram", 2)
    records = [divided.record,
               merge(divided.adf, ["chip/dram/dram_0", "chip/dram/dram_1"]).record,
               insert(root, "chip", ["chip/dram"]).record,
               create(root, "chip", spec=memory("extra")).record,
               delete(root, "chip/dram").record,
               remove(root, "chip/core").record]
    for record in records:
        validate_arguments(record, {"oneOf": BASIC_SCHEMAS})
        assert all(value is not None for value in record.values())
