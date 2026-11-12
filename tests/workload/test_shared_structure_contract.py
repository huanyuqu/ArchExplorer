"""Installed production pooling invariants, outside core's fake-registry suite."""
from copy import deepcopy
import pytest


def test_public_pooling_replaces_root_network_and_preserves_resource_count():
    from arch_explorer.archs.adf_seeds import dmc_adf,bus
    from arch_explorer.adf import walk,SpacePoint,SpaceMatrix
    from arch_explorer.structure import share_private_memory,replay,normalized_fingerprint
    seed=dmc_adf(4);before=deepcopy(seed.to_dict())
    memory=next(node for _,node in walk(seed) if isinstance(node,SpacePoint) and node.component=='sram')
    capacity=memory.value('capacity');bandwidth=memory.value('bandwidth')
    result=share_private_memory(seed,'chip',2,'sram',bus('group_bus').to_dict(),bus('root_bus').to_dict())
    assert seed.to_dict()==before
    assert normalized_fingerprint(replay(seed,result.records))==normalized_fingerprint(result.adf)
    groups=[node for node in result.adf.elements if isinstance(node,SpaceMatrix) and node.name.startswith('memory_group_')]
    assert len(groups)==2
    for group in groups:
        pools=[node for node in group.elements if isinstance(node,SpacePoint) and node.component=='sram']
        cores=[node for node in group.elements if isinstance(node,SpaceMatrix)]
        assert len(pools)==1 and len(cores)==2
        assert pools[0].value('capacity')==2*capacity
        assert pools[0].value('bandwidth')==2*bandwidth
        assert all(not any(isinstance(n,SpacePoint) and n.component=='sram' for n in core.elements) for core in cores)
    assert [n.name for n in result.adf.networks]==['root_bus']


@pytest.mark.parametrize('group_size',[0,True,3])
def test_pooling_bad_group_has_no_partial_edit(group_size):
    from arch_explorer.archs.adf_seeds import dmc_adf,bus
    from arch_explorer.structure import share_private_memory,StructureError
    seed=dmc_adf(4);before=deepcopy(seed.to_dict())
    with pytest.raises(StructureError):
        share_private_memory(seed,'chip',group_size,'sram',bus('group_bus').to_dict(),bus('root_bus').to_dict())
    assert seed.to_dict()==before
