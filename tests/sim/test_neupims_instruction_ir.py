import pytest
from arch_explorer.components.neupims_instructions import (
    Instruction,issue_compute,matmul_tiles,outer_shape,footprint)


@pytest.mark.parametrize('width,expected',[(64,(127,261)),(128,(255,525))])
def test_cold_and_hot_preload(width,expected):
    inst=Instruction('GEMM_PRELOAD',(),size=width//8)
    assert issue_compute(0,inst,width,width)==expected
    start,end=expected
    assert issue_compute(1,inst,width,width,(start,end,inst.size))==(start+width,end+width)
    # Empty after retirement is cold. A nonempty FIFO remains hot even if
    # its modeled finish has passed (retirement is one instruction per tick).
    assert issue_compute(end,inst,width,width,None)==(end+start,2*end)
    assert issue_compute(end,inst,width,width,(start,end,inst.size))==(start+width,end+width)


def test_capacity_and_tail_bytes():
    assert outer_shape(64,64,64,64,49152)==(64,64,64)
    with pytest.raises(ValueError,match='half SPAD'):
        outer_shape(64,64,64,64,49151)
    tiles=list(matmul_tiles(2,65,67,64,1024*1024))
    assert len(tiles)==1
    code=tiles[0].instructions
    assert sum(i.region.bytes for i in code if i.opcode=='MOVIN' and i.region.tensor=='a')==2*65*2
    assert sum(i.region.bytes for i in code if i.opcode=='MOVIN' and i.region.tensor=='b')==65*67*2
    assert sum(i.region.bytes for i in code if i.opcode=='MOVOUT')==2*67*2
    assert len([i for i in code if i.opcode=='GEMM_PRELOAD'])==4
    assert [i.shape for i in code if i.opcode=='GEMM_PRELOAD'][-1]==(2,1,3)


def test_outer_split_and_reuse():
    dims=outer_shape(128,256,128,64,196608)
    assert footprint(dims,64,2)<=98304
    tiles=list(matmul_tiles(128,256,128,64,196608))
    assert len(tiles)>1
    assert any(t.accum for t in tiles)
    # Each output element stored once, after all K tiles.
    assert sum(i.region.bytes for t in tiles for i in t.instructions if i.opcode=='MOVOUT')==128*128*2


def test_weighted_transpose_reuse_and_logical_output():
    # Source constructor enables B.T @ A.T: 192x64 @ 64x2 has one
    # PRELOAD followed by two GEMM, instead of three PRELOADs for 2x64x192.
    tiles=list(matmul_tiles(2,64,192,64,1024*1024,transposed=True))
    code=tiles[0].instructions
    assert [i.opcode for i in code if i.opcode.startswith('GEMM')]==['GEMM_PRELOAD','GEMM','GEMM']
    assert sum(i.region.bytes for i in code if i.opcode=='MOVIN')==(2*64+64*192)*2
    outputs=[i.region for i in code if i.opcode=='MOVOUT']
    assert {(r,c) for t in outputs for r in range(t.row,t.row+t.rows)
            for c in range(t.col,t.col+t.cols)}=={(r,c) for r in range(2) for c in range(192)}
    assert sum(t.bytes for t in outputs)==2*192*2
    # K splitting with an asymmetric tail must retain every mathematical MAC
    # and write each logical output exactly once.
    tiles=list(matmul_tiles(3,257,129,64,98304,transposed=True))
    code=[i for t in tiles for i in t.instructions]
    assert sum(i.shape[0]*i.shape[1]*i.shape[2] for i in code if i.opcode.startswith('GEMM'))==3*257*129
    assert sum(i.region.bytes for i in code if i.opcode=='MOVOUT')==3*129*2


def test_source_parsed_capacity_matches_observed_instruction_counts():
    from arch_explorer.components.neupims_instructions import SOURCE_SPAD_BYTES
    # Existing original b2/128 QKV observation: 768 compute schedules,
    # 32 PRELOAD and 736 GEMM. Common.cc reads sram_size, not spad_size.
    assert SOURCE_SPAD_BYTES==137439232
    tiles=list(matmul_tiles(2,4096,3072,128,SOURCE_SPAD_BYTES,transposed=True))
    assert len(tiles)==1
    code=tiles[0].instructions
    assert sum(i.opcode=='GEMM_PRELOAD' for i in code)==32
    assert sum(i.opcode=='GEMM' for i in code)==736
    assert tiles[0].footprint<=SOURCE_SPAD_BYTES//2
