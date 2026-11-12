"""Non-power-of-two, gateway, shared-link and no-free-broadcast hand cases."""
import pytest
from mla_multicast_expectations import endpoint_path,tree_transfer,hierarchical_transfer


def transfer(count,source,destinations,*,multicast=True):
    return tree_transfer(count,source,destinations,12,path='board/tree',
                         link_bandwidth=4,port_bandwidth=6,hop_latency=.5,multicast=multicast)


def test_five_endpoints_use_actual_leaves_and_integer_common_ancestor():
    assert endpoint_path(5,0,4) == [8,4,2,1,3,6,12]
    assert endpoint_path(5,-1,2) == [1,2,5,10]
    assert endpoint_path(5,4,-1) == [12,6,3,1]
    assert endpoint_path(3,0,2) == [4,2,1,3,6]
    with pytest.raises(ValueError):endpoint_path(5,0,5)


def test_one_endpoint_has_two_directional_ports_and_zero_tree_hops():
    assert endpoint_path(1,-1,0) == [1]
    result = transfer(1,-1,[0])
    assert result['demand_bytes'] == {'board/tree:out-1':12,'board/tree:in0':12}
    assert result['isolated_cutthrough_busy_ns'] == 2
    assert result['startup_tail_ns'] == 0


def test_three_endpoints_multicast_charges_shared_trunk_and_each_branch_once():
    actual = transfer(3,-1,[0,1])
    assert actual['demand_bytes'] == {
        'board/tree:edge1_to_2':12,'board/tree:edge2_to_4':12,'board/tree:edge2_to_5':12,
        'board/tree:out-1':12,'board/tree:in0':12,'board/tree:in1':12}
    assert actual['isolated_cutthrough_busy_ns'] == 3
    assert actual['startup_tail_ns'] == 1
    unicast = transfer(3,-1,[0,1],multicast=False)
    assert unicast['demand_bytes']['board/tree:edge1_to_2'] == 24
    assert unicast['demand_bytes']['board/tree:out-1'] == 24
    assert unicast['isolated_cutthrough_busy_ns'] == 6
    assert unicast['startup_tail_ns'] == 1


@pytest.mark.parametrize('targets',[[0,0],[-1],[],[3]])
def test_invalid_targets_never_become_free_broadcast(targets):
    with pytest.raises(ValueError):transfer(3,-1,targets)


def test_frozen_hierarchy_root_gateway_and_memory_ports_from_plain_input():
    import hashlib,json
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'fixtures/workload_scenarios/INPUTS.json'
    raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='e32af5b84faf48159c92eda84ff36ef3566a58a0dc2818030b0aeda5226ac052'
    adf=json.loads(raw)['scenarios']['grouped_multicast']['adf']
    destinations=['board/group[0]/sram','board/group[1]/sram']
    result=hierarchical_transfer(adf,'board/dram',destinations,80,multicast=True)
    demand=result['demand_bytes']
    assert demand['board/dram:port']==80
    assert demand['board/tree:out0']==80
    assert demand['board/tree:edge8_to_4']==80
    assert demand['board/tree:in3']==demand['board/tree:in4']==80
    for group in range(2):
        assert demand[f'board/group[{group}]/sram:port']==80
        assert demand[f'board/group[{group}]/tree:out-1']==80
        assert demand[f'board/group[{group}]/tree:in0']==80
    assert len(demand)==22
    assert result['isolated_cutthrough_busy_ns']==5
    # Root endpoint0 leaf8 to group1 endpoint4 leaf12: 6 edges; group
    # gateway1 to SRAM endpoint0 leaf4: 2 edges; DRAM startup 4 ns.
    assert result['startup_tail_ns']==12
    unicast=hierarchical_transfer(adf,'board/dram',destinations,80,multicast=False)
    assert unicast['demand_bytes']['board/dram:port']==160
    assert unicast['demand_bytes']['board/tree:edge8_to_4']==160
    assert unicast['isolated_cutthrough_busy_ns']==10
