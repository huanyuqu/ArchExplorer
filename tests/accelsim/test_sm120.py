"""Typed stream and capture-shape oracles. All raw fixtures are SYNTHETIC."""
import json,struct
import pytest
from arch_explorer.simulator.accelsim.typed_trace import instruction,register,opcode
from arch_explorer.simulator.accelsim.capture_input import read_records,operands
from arch_explorer.archs.accelsim import sm90_adf,sm120_mechanism_adf
from arch_explorer.simulator.accelsim.config import compile_adf
from arch_explorer.simulator.accelsim.trace_stream import facts

H={'lineinfo':0,'nregs':24}
def test_types_constants_and_number_spaces():
    assert [register(x) for x in ('R0','UR0','P0','UP0')]==[('R',0),('UR',0),('P',0),('UP',0)]
    assert [register(x) for x in ('RZ','URZ','PT','UPT')]==[('R',255),('UR',255),('P',7),('UP',7)]
    for invalid in ('R256','UR256','P8','UP8','BAD0','R-1'):
        with pytest.raises(ValueError):register(invalid)

@pytest.mark.parametrize('size,suffix',[(2,'.U16'),(4,''),(8,'.64'),(16,'.128')])
def test_width_vector_and_exact_lane_addresses(size,suffix):
    dst=['R'+str(i) for i in range(max(1,size//4))]
    mask=0x15  # noncontiguous lanes 0,2,4; not compressed into adjacent lanes
    line=f"0010 {mask:x} {len(dst)} "+' '.join(dst)+f" LDG.E{suffix} 4 UR4 UR5 R10 R11 {size} 0 100000 100100 100200 0 Guard PT Active {mask:x}"
    assert instruction(line,H)==(16,mask,'LDG','LDG.E'+suffix,size)
    with pytest.raises(ValueError):instruction(line.replace('100200','ffffffffffffffff'),H)
    if size>4:
        with pytest.raises(ValueError):instruction(line.replace(str(len(dst))+' '+' '.join(dst),'1 R0',1),H)

@pytest.mark.parametrize('line',[
    '0010 ffffffff 1 R24 MOV 0 0 0',
    '0010 ffffffff 1 R0 LDG.E 0 0 0',
    '0010 ffffffff 1 R0 LDG.E.STRONG.SYS 0 4 0 0',
    '0010 ffffffff 1 R0 LDG.E.U16 0 4 0 0',
    '0001 ffffffff 1 R0 MOV 0 0 0',
    '0010 100000000 1 R0 MOV 0 0 0',
    '0010 ffffffff 1 R0 TMA.SPECIAL 0 0 0',
    '0010 ffffffff 1 R0 MOV 0 0 0 extra'])
def test_native_unsupported_inputs_never_default_to_cost(line):
    with pytest.raises(ValueError):instruction(line,H)

def test_mechanism_profile_explicit_coupling_and_original_preset():
    old=compile_adf(sm90_adf());new=compile_adf(sm120_mechanism_adf())
    assert '-sim28_typed_regs 1\n' not in old.text
    assert '-sim28_typed_regs 1\n' in new.text
    assert '-gpgpu_coalesce_arch 120\n' in new.text
    assert '-gpgpu_n_clusters 2\n' in new.text and '-gpgpu_n_mem 40\n' in new.text
    assert 'NOT a Blackwell physical profile' in new.text
    a=sm120_mechanism_adf();a.child('partition').repeat=20
    with pytest.raises(ValueError):compile_adf(a)

def synthetic_capture():
    rows=[]
    for cta in range(4):
        for warp in range(4):
            rows.append(struct.pack('<8I32Q',0,cta,0,0,warp,0xffffffff,0,0,*([0]*32)))
    h=struct.pack('<8s14I',b'SIM28NV1',1,64,288,len(rows),len(rows),4096,0,1,4,1,1,128,1,1)
    return h+b''.join(rows)

def test_all_predicated_off_observation_is_not_dropped():
    grid,block,streams=read_records(synthetic_capture())
    assert grid==[4,1,1] and block==[128,1,1] and len(streams)==16
    assert all(len(v)==1 and v[0]['mask']==0 for v in streams.values())

@pytest.mark.parametrize('offset,value',[(32,1),(36,2),(12,63),(24,17),(28,4095),(8,2),(52,64)])
def test_raw_capture_bounds_overflow_and_single_launch(offset,value):
    b=bytearray(synthetic_capture());struct.pack_into('<I',b,offset,value)
    with pytest.raises(ValueError):read_records(bytes(b))

def test_raw_active_address_and_sequence_rejections():
    b=bytearray(synthetic_capture())
    struct.pack_into('<Q',b,64+32,0x100000)  # address in a lane with exec mask 0
    with pytest.raises(ValueError):read_records(bytes(b))
    b=synthetic_capture()
    with pytest.raises(ValueError):read_records(b[:-288])
    with pytest.raises(ValueError):read_records(b+b'\0')

def reg(kind,num,nbytes=4):
    return {'type_name':kind,'nbytes':nbytes, 'reg':{'num':num}} if kind in ('REG','UREG') else {'type_name':kind,'nbytes':1,'pred':{'num':num}}
def ins(op,operands,size=0,**updates):
    return dict(opcode=op,operands=operands,size=size,is_tma_mem=False,is_tma_tensor=False,is_extended=False,
        has_pred=False,pred_uniform=False,pred_num=-1,mref_count=0,memory_space=0,is_load=False,is_store=False,**updates)

def test_multi_output_uniform_predicate_and_control_dependencies():
    i=ins('UIADD3',[reg('UREG',0),reg('UPRED',0),reg('UPRED',7),reg('UREG',1),reg('UREG',255),reg('UREG',2)])
    dst,src,width=operands(i)
    assert dst==['UR0','UP0','UP7'] and src==['UR1','UR255','UR2'] and width==0
    i=ins('MOV',[reg('REG',1),reg('REG',255)])
    i.update(has_pred=True,pred_uniform=True,pred_num=0)
    assert operands(i)[1]==['R255']

def test_actual_blackwell_iadd64_preserves_both_halves():
    i=ins('IADD.64',[reg('REG',8),reg('REG',6),{'type_name':'IMM_UINT64','nbytes':8}])
    assert operands(i)==(['R8','R9'],['R6','R7'],0)

def test_wide_multiply_add_reads_the_complete_64_bit_addend():
    # nbytes=4 cannot discard the semantic high half of the WIDE addend.
    i=ins('IMAD.WIDE.U32',[reg('REG',4),reg('REG',7),{'type_name':'IMM_UINT64','nbytes':4},reg('REG',4)])
    assert operands(i)==(['R4','R5'],['R7','R4','R5'],0)

def test_descriptor_pair_wide_address_and_memory_vector_are_distinct():
    mref={'type_name':'MREF','nbytes':8,'mref':{'has_ra':True,'ra_num':10,'ra_mod':6,
        'has_ur':False,'ur_num':0,'ur_mod':0,'has_desc':True,'desc_ureg_num':4}}
    i=ins('LDG.E.128',[reg('REG',0),mref],16)
    i.update(mref_count=1,memory_space=3,is_load=True,is_extended=True)
    dst,src,size=operands(i)
    assert dst==['R0','R1','R2','R3'] and src==['R10','R11','UR4','UR5'] and size==16

