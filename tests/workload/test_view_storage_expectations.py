"""VAL58 expectation self-checks; these are not production acceptance."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

import view_storage_expectations as oracle
from view_storage_expectations import (
    AddressMap,Allocation,Reader,Root,STORAGE_BYTES,canonical_scalar_bytes,
    check_placements,event_peaks,hand_fork,interval_peaks,lifetimes,
    operand_read_bytes,rectangular_tiles,root_map,write_bytes,
)


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_holey_slice_keeps_distinct_logical_span_and_full_root_bytes(dtype):
    root=Root('storage',(4,6),dtype)
    view=root_map(root).slice(((1,4,2),(1,6,2)))
    assert view.shape==(2,3) and view.steps==(12,2) and view.first==7
    assert view.addresses()==[7,9,11,19,21,23]
    unit=STORAGE_BYTES[dtype]
    assert view.read_bytes==6*unit
    assert view.enclosing_span_bytes==17*unit
    assert root.bytes==24*unit
    assert view.byte_addresses()==[7*unit,9*unit,11*unit,19*unit,21*unit,23*unit]
    assert len({view.read_bytes,view.enclosing_span_bytes,root.bytes})==3


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_nested_permutation_and_slice_keep_one_root_and_relative_offset(dtype):
    root=Root('storage',(4,6),dtype)
    first=root_map(root).slice(((1,4,2),(1,6,2)))
    transposed=first.transpose((1,0))
    nested=transposed.slice(((1,3,1),(0,2,1)))
    assert transposed.addresses()==[7,19,9,21,11,23]
    assert nested.addresses()==[9,21,11,23]
    assert nested.first==9 and nested.steps==(2,12)
    assert nested.root is first.root is root
    table=list(range(24))
    assert nested.values(table)==[9,21,11,23]
    assert root_map(root).transpose((1,0)).addresses()==[
        0,6,12,18,1,7,13,19,2,8,14,20,3,9,15,21,4,10,16,22,5,11,17,23]


def test_noncontiguous_reshape_uses_complete_logical_order_and_never_implicit_copy():
    root=Root('root',(2,3))
    transposed=root_map(root).transpose((1,0))
    with pytest.raises(ValueError):transposed.reshape((6,))
    assert transposed.reshape((1,3,2)).addresses()==[0,3,1,4,2,5]
    assert root_map(root).reshape((3,2)).addresses()==[0,1,2,3,4,5]
    # Positive-stride noncontiguity alone is not proof that sharing is impossible.
    spaced=root_map(Root('larger',(4,6))).slice(((1,2,1),(0,6,2)))
    assert spaced.addresses()==[6,8,10]
    assert spaced.reshape((3,)).steps==(2,)
    with pytest.raises(ValueError):spaced.reshape((2,2))


def test_rank_one_and_four_maps_keep_element_units_and_axis_order():
    line=root_map(Root('line',(9,),'bf16')).slice(((2,9,3),))
    assert line.addresses()==[2,5,8] and line.byte_addresses()==[4,10,16]
    cube=root_map(Root('rank4',(2,2,2,3))).transpose((3,0,2,1))
    assert cube.shape==(3,2,2,2)
    assert cube.addresses()==[0,6,3,9,12,18,15,21,1,7,4,10,13,19,16,22,2,8,5,11,14,20,17,23]
    assert cube.slice(((1,3,1),(0,2,1),(0,1,1),(1,2,1))).addresses()==[7,19,8,20]


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_boundary_tiles_and_overlapping_operands_keep_every_real_read(dtype):
    root=Root('root',(4,6),dtype)
    view=root_map(root).slice(((1,4,2),(1,6,2)))
    rows=rectangular_tiles(view,(2,2));unit=STORAGE_BYTES[dtype]
    assert [row['shape'] for row in rows]==[(2,2),(2,1)]
    assert [row['element_addresses'] for row in rows]==[[7,9,19,21],[11,23]]
    assert [row['read_bytes'] for row in rows]==[4*unit,2*unit]
    assert operand_read_bytes(view,view)==12*unit
    overlap=view.slice(((0,2,1),(1,3,1)))
    assert overlap.addresses()==[9,11,21,23]
    assert operand_read_bytes(view,overlap)==10*unit
    assert len(set(view.addresses())|set(overlap.addresses()))==6


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_view_selects_stored_bit_patterns_without_copy_rounding_or_conversion(dtype,monkeypatch):
    root=Root('root',(2,4),dtype)
    source=[oracle.store(x,dtype) for x in (0.,-0.,1.,1+2**-8,1+2**-11,-1.5,2**-130,65504.)]
    view=root_map(root).transpose((1,0)).slice(((1,4,2),(0,2,1)))
    assert view.addresses()==[1,5,3,7]
    expected=[canonical_scalar_bytes(source[i],dtype) for i in (1,5,3,7)]
    def forbidden(*args,**kwargs):raise AssertionError('view gather introduced a storage-rounding operation')
    with monkeypatch.context() as patch:
        patch.setattr(oracle,'store',forbidden)
        selected=view.values(source)
    assert [canonical_scalar_bytes(x,dtype) for x in selected]==expected
    assert all(len(x)==STORAGE_BYTES[dtype] for x in expected)


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('swap',[False,True])
def test_real_alias_readers_both_hold_full_root_until_last_completion(dtype,swap):
    case=hand_fork(dtype,swap_readers=swap)
    rows={row['name']:row for row in case['intervals']};unit=STORAGE_BYTES[dtype]
    assert rows['root']['bytes']==24*unit
    assert rows['root']['release']==7 and rows['root']['last_reader']==7
    assert sorted(reader.end for reader in case['readers'] if reader.name!='merge')==[3,7]
    assert case['peaks']=={'memory':36*unit}
    assert case['read_bytes']==24*unit and case['write_bytes']==18*unit
    assert len(case['allocations'])==4  # No allocation for either alias descriptor.
    assert sum(row['bytes'] for row in case['intervals'])==42*unit
    assert case['peaks']['memory']<sum(row['bytes'] for row in case['intervals'])


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_exporting_small_slice_holds_full_root_and_capacity_uses_actual_live_peak(dtype):
    normal=hand_fork(dtype);exported=hand_fork(dtype,retain_slice=True)
    unit=STORAGE_BYTES[dtype]
    assert normal['peaks']['memory']==36*unit
    assert exported['peaks']['memory']==42*unit
    root=next(row for row in exported['intervals'] if row['name']=='root')
    assert root['release']==10 and root['held_to_return'] and root['bytes']==24*unit
    peak=normal['peaks']['memory'];individual=max(item.root.bytes for item in normal['allocations'])
    assert individual==24*unit and individual<=peak-1<peak
    assert individual-1<individual
    # The first inequality distinguishes individual allocation lower bounds
    # from runtime overlap; it does not guess a future compiler rejection phase.


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_explicit_materialization_adds_traffic_but_can_reduce_root_lifetime_peak(dtype):
    view=hand_fork(dtype);copy=hand_fork(dtype,materialize=True);unit=STORAGE_BYTES[dtype]
    assert view['read_bytes']==24*unit and copy['read_bytes']==30*unit
    assert view['write_bytes']==18*unit and copy['write_bytes']==24*unit
    assert next(row for row in copy['intervals'] if row['name']=='root')['release']==2
    assert copy['peaks']=={'memory':30*unit}
    assert view['peaks']=={'memory':36*unit}
    assert len(copy['allocations'])==len(view['allocations'])+1


def test_weight_alias_and_control_dependency_keep_one_physical_allocation():
    weight=Root('weights',(4,6),'bf16',resident=True)
    view=root_map(weight).slice(((1,4,2),(1,6,2)))
    rows=lifetimes([Allocation(weight,0,0)],[Reader('read',(view,),1,2)],finish=10)
    assert len(rows)==1 and rows[0]['bytes']==48 and rows[0]['release']==10
    root=replace(weight,resident=False)
    view=root_map(root).slice(((1,4,2),(1,6,2)))
    rows=lifetimes([Allocation(root,0,0)],[Reader('read',(view,),1,2)],finish=10,controls=[(view,8)])
    assert rows[0]['last_reader']==8 and rows[0]['release']==8


def test_ordered_memory_events_detect_duplicate_alias_allocation_and_tied_transients():
    events=[('allocate','root','m',96),('allocate','left','m',24),('allocate','right','m',24),
            ('free','root','m',96),('allocate','output','m',24),
            ('free','left','m',24),('free','right','m',24),('free','output','m',24)]
    assert event_peaks(events)=={'m':144}
    # Reversing an allocation/free pair can create a transient at the same
    # clock. Observations must retain event order, not sort ties to force a peak.
    tied=events[:3]+[events[4],events[3]]+events[5:]
    assert event_peaks(tied)=={'m':168}
    with pytest.raises(ValueError):event_peaks(events[:1]*2)
    with pytest.raises(ValueError):event_peaks([('free','missing','m',96)])


def test_same_root_placement_conflict_alias_write_and_bad_producer_order_refuse():
    root=Root('root',(4,6));view=root_map(root).slice(((1,4,2),(1,6,2)))
    assert check_placements([(view,'a'),(view.transpose((1,0)),'a')])=={'root':'a'}
    with pytest.raises(ValueError):check_placements([(root_map(root),'a'),(view,'b')])
    with pytest.raises(ValueError):write_bytes(view)
    assert write_bytes(root_map(Root('copy',view.shape)))==24
    with pytest.raises(ValueError):lifetimes([Allocation(root,0,5)],[Reader('early',(view,),2,4)],finish=10)
    with pytest.raises(ValueError):lifetimes([Allocation(root,0,0)]*2,[],finish=10)


@pytest.mark.parametrize('failure',['negative-offset','past-root','zero-step','negative-step','empty',
    'axis-repeat','axis-missing','slice-past-end','rank5','boolean-step','count-change'])
def test_independent_arithmetic_refuses_invalid_descriptors(failure):
    root=Root('root',(4,6));base=root_map(root)
    bad={
      'negative-offset':lambda:AddressMap(root,(1,),(1,),-1),
      'past-root':lambda:AddressMap(root,(2,),(1,),23),
      'zero-step':lambda:base.slice(((0,4,0),(0,6,1))),
      'negative-step':lambda:base.slice(((0,4,-1),(0,6,1))),
      'empty':lambda:base.slice(((1,1,1),(0,6,1))),
      'axis-repeat':lambda:base.transpose((0,0)),
      'axis-missing':lambda:base.transpose((0,)),
      'slice-past-end':lambda:base.slice(((0,5,1),(0,6,1))),
      'rank5':lambda:base.reshape((1,1,1,4,6)),
      'boolean-step':lambda:base.slice(((0,4,True),(0,6,1))),
      'count-change':lambda:base.reshape((2,11)),
    }
    with pytest.raises(ValueError):bad[failure]()


def test_oracle_process_imports_no_production_or_numpy_layout_implementation():
    directory=Path(__file__).resolve().parent
    code='''
import json,sys
sys.path.insert(0,sys.argv[1])
from view_storage_expectations import hand_fork
assert hand_fork()['peaks']=={'memory':144}
assert not any(n=='numpy' or n.startswith('numpy.') or n=='arch_explorer' or n.startswith('arch_explorer.') for n in sys.modules)
print(json.dumps({'production_imports':False,'numpy_imports':False}))
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(directory)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'production_imports':False,'numpy_imports':False}
