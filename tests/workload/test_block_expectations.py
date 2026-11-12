"""Hand block cases discriminate clipping, guard traffic and seeded FMA."""
from copy import deepcopy
import hashlib,struct
import numpy as np
import pytest
from block_expectations import partition_grid,partition_coverage,payload_digest,load,zero,matmul_accumulate


def test_partition_edges_cover_rectangles_without_flat_interval_shortcut():
    grid=partition_grid([3,5],[2,3]);r=partition_coverage([3,5],grid)
    assert [b['bytes'] for b in grid]==[24,16,12,8] and r['bytes']==60
    assert r['flat_indices_by_fragment'][0]==[0,1,2,5,6,7]
    with pytest.raises(ValueError,match='hole'):partition_coverage([3,5],grid[:-1])
    with pytest.raises(ValueError,match='overlap'):partition_coverage([3,5],grid+[grid[0]])
    bad=deepcopy(grid);bad[0]['bytes']=0
    with pytest.raises(ValueError,match='byte width'):partition_coverage([3,5],bad)
    assert payload_digest([[0.,-0.,1.,-2.]])==hashlib.sha256(struct.pack('<ffff',0.,-0.,1.,-2.)).hexdigest()


def test_load_counts_guard_reads_and_materialized_output():
    result,io=load([[0,1,2],[3,4,5]],starts=[0,1],shape=[2,2],guards=([[7,8]],),guard_names=['fence'])
    np.testing.assert_array_equal(result,[[1,2],[4,5]])
    assert io['reads']==[('source',(0,1)),('source',(0,2)),('source',(1,1)),('source',(1,2)),('fence',(0,0)),('fence',(0,1))]
    assert io['read_by_tensor']=={'source':16,'fence':8} and io['write_bytes']==16
    result,io=zero([2,3]);assert not result.any() and io['read_bytes']==0 and io['write_bytes']==24


def test_seed_is_present_before_first_fma_and_storage_is_new():
    seed=np.array([[[2.**24]]]);original=seed.copy()
    result,io=matmul_accumulate([[[1,1]]],[[[1],[-2.**24]]],seed)
    np.testing.assert_array_equal(result,[[[0.]]])
    np.testing.assert_array_equal(seed,original)
    assert not np.shares_memory(seed,result)
    assert io['read_by_tensor']=={'C':4,'A':8,'B':8} and io['write_bytes']==4
    with pytest.raises(ValueError,match='shapes'):matmul_accumulate([[[1]]],[[[1]]],[[1]])
