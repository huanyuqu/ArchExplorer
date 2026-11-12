"""Independent VAL58 addressing and lifetime arithmetic, not a public schema.

These QA descriptors describe hand-derived facts. They do not parse production
Tensor/layout fields, construct workloads, or import a production resolver.
Addresses and strides are in elements; byte conversion uses storage dtype.
"""
from dataclasses import dataclass
from itertools import product
from math import prod
import struct

from precision_transformer_expectations import store

STORAGE_BYTES={'fp32':4,'fp16':2,'bf16':2,'int32':4}


def dimensions(shape):
    result=tuple(shape)
    if not 1<=len(result)<=4 or any(type(v) is not int or v<1 for v in result):
        raise ValueError('positive rank 1..4 shape required')
    return result


def row_major_steps(shape):
    shape=dimensions(shape)
    return tuple(prod(shape[i+1:]) for i in range(len(shape)))


def indices(shape):
    return product(*(range(n) for n in dimensions(shape)))


@dataclass(frozen=True)
class Root:
    name:str
    shape:tuple
    dtype:str='fp32'
    memory:str='memory'
    resident:bool=False

    def __post_init__(self):
        object.__setattr__(self,'shape',dimensions(self.shape))
        if not isinstance(self.name,str) or not self.name or not isinstance(self.memory,str) or not self.memory:
            raise ValueError('explicit storage and memory names required')
        if self.dtype not in STORAGE_BYTES:raise ValueError('unknown storage dtype')
        if type(self.resident) is not bool:raise ValueError('resident flag required')

    @property
    def elements(self):return prod(self.shape)

    @property
    def bytes(self):return self.elements*STORAGE_BYTES[self.dtype]


@dataclass(frozen=True)
class AddressMap:
    root:Root
    shape:tuple
    steps:tuple
    first:int=0
    shared:bool=True

    def __post_init__(self):
        object.__setattr__(self,'shape',dimensions(self.shape))
        object.__setattr__(self,'steps',tuple(self.steps))
        if len(self.steps)!=len(self.shape) or any(type(s) is not int or s<1 for s in self.steps):
            raise ValueError('one positive element stride per axis required')
        if type(self.first) is not int or self.first<0:raise ValueError('nonnegative element offset required')
        last=self.first+sum((n-1)*s for n,s in zip(self.shape,self.steps))
        if last>=self.root.elements:raise ValueError('last addressed element exceeds the root allocation')

    @property
    def elements(self):return prod(self.shape)

    @property
    def read_bytes(self):return self.elements*STORAGE_BYTES[self.root.dtype]

    @property
    def enclosing_span_bytes(self):
        return (1+sum((n-1)*s for n,s in zip(self.shape,self.steps)))*STORAGE_BYTES[self.root.dtype]

    def address(self,index):
        index=tuple(index)
        if len(index)!=len(self.shape) or any(type(i) is not int or not 0<=i<n for i,n in zip(index,self.shape)):
            raise ValueError('logical index outside shape')
        return self.first+sum(i*s for i,s in zip(index,self.steps))

    def addresses(self):return [self.address(index) for index in indices(self.shape)]

    def byte_addresses(self):return [a*STORAGE_BYTES[self.root.dtype] for a in self.addresses()]

    def values(self,root_values):
        if len(root_values)!=self.root.elements:raise ValueError('explicit complete root data required')
        return [root_values[a] for a in self.addresses()]

    def transpose(self,axes):
        axes=tuple(axes)
        if len(axes)!=len(self.shape) or any(type(a) is not int for a in axes) or set(axes)!=set(range(len(self.shape))):
            raise ValueError('axes must be a full permutation')
        return AddressMap(self.root,tuple(self.shape[a] for a in axes),tuple(self.steps[a] for a in axes),self.first)

    def slice(self,bounds):
        bounds=tuple(bounds)
        if len(bounds)!=len(self.shape):raise ValueError('explicit range for every axis required')
        result=[];steps=[];first=self.first
        for limit,stride,items in zip(self.shape,self.steps,bounds):
            if len(items)!=3:raise ValueError('explicit start, stop, positive step required')
            start,stop,step=items
            if any(type(v) is not int for v in items) or step<=0 or not 0<=start<stop<=limit:
                raise ValueError('nonempty positive in-bounds slice required')
            result.append((stop-start+step-1)//step);steps.append(stride*step);first+=start*stride
        return AddressMap(self.root,tuple(result),tuple(steps),first)

    def reshape(self,shape):
        """Derive an affine map from the actual logical address sequence.

        Enumeration is intentional for small independent test cases. A
        noncontiguous input is accepted only when the entire original order
        can be represented by the new positive affine strides.
        """
        shape=dimensions(shape)
        if prod(shape)!=self.elements:raise ValueError('reshape changes element count')
        walk=self.addresses();new_steps=[]
        for axis,size in enumerate(shape):
            jump=prod(shape[axis+1:])
            new_steps.append(walk[jump]-walk[0] if size>1 else jump)
        candidate=AddressMap(self.root,shape,tuple(new_steps),walk[0])
        if candidate.addresses()!=walk:raise ValueError('reshape requires materialization to preserve traversal order')
        return candidate


def root_map(root):return AddressMap(root,root.shape,row_major_steps(root.shape),0,shared=False)


def operand_read_bytes(*operands):
    """Repeated and overlapping operands are real reads, never deduplicated."""
    return sum(item.read_bytes for item in operands)


def write_bytes(*outputs):
    if any(item.shared for item in outputs):raise ValueError('writing a read-only alias is forbidden')
    return sum(item.read_bytes for item in outputs)


def check_placements(pairs):
    selected={}
    for layout,memory in pairs:
        if not isinstance(memory,str) or not memory:raise ValueError('explicit memory required')
        previous=selected.setdefault(layout.root.name,memory)
        if previous!=memory:raise ValueError('same storage root has conflicting memory placement')
    return selected


def rectangular_tiles(layout,tile_shape):
    tile_shape=dimensions(tile_shape)
    if len(tile_shape)!=len(layout.shape):raise ValueError('tile rank differs')
    rows=[]
    for start in product(*(range(0,n,t) for n,t in zip(layout.shape,tile_shape))):
        shape=tuple(min(t,n-i) for i,n,t in zip(start,layout.shape,tile_shape))
        addresses=[layout.address(tuple(a+b for a,b in zip(start,local))) for local in indices(shape)]
        rows.append(dict(start=start,shape=shape,element_addresses=addresses,
                         read_bytes=len(addresses)*STORAGE_BYTES[layout.root.dtype]))
    return rows


def canonical_scalar_bytes(value,dtype):
    """Use independent RNE for floats and exact signed encoding for int32."""
    if dtype=='int32':
        if type(value) is not int or not -2**31<=value<2**31:raise ValueError('exact signed int32 required')
        return struct.pack('<i',value)
    value=store(value,dtype)
    if dtype=='fp32':return struct.pack('<f',value)
    if dtype=='fp16':return struct.pack('<e',value)
    if dtype=='bf16':return struct.pack('<f',value)[2:]
    raise ValueError('unknown storage dtype')


@dataclass(frozen=True)
class Allocation:
    root:Root
    at:float
    ready:float


@dataclass(frozen=True)
class Reader:
    name:str
    operands:tuple
    start:float
    end:float


def lifetimes(allocations,readers,*,returned=(),finish,controls=()):
    """Root identity, actual reader end times and controls determine release."""
    allocated={}
    for item in allocations:
        if item.root.name in allocated:raise ValueError('root allocation counted more than once')
        if not 0<=item.at<=item.ready<=finish:raise ValueError('invalid allocation schedule')
        allocated[item.root.name]=item
    last={name:item.ready for name,item in allocated.items()}
    for reader in readers:
        if not 0<=reader.start<=reader.end<=finish:raise ValueError('invalid reader interval')
        for operand in reader.operands:
            root=operand.root;item=allocated[root.name]
            if item.root!=root or reader.start<item.ready:raise ValueError('reader precedes the actual root producer')
            last[root.name]=max(last[root.name],reader.end)
    for layout,time in controls:
        item=allocated[layout.root.name]
        if not item.at<=time<=finish:raise ValueError('control outside execution')
        last[layout.root.name]=max(last[layout.root.name],time)
    held={layout.root.name for layout in returned}
    if not held<=set(allocated):raise ValueError('returned layout has no root allocation')
    return [dict(name=name,memory=item.root.memory,bytes=item.root.bytes,
                 allocation=item.at,last_reader=last[name],
                 release=finish if item.root.resident or name in held else last[name],
                 held_to_return=item.root.resident or name in held)
            for name,item in allocated.items()]


def interval_peaks(intervals):
    """Half-open interval peak; same-clock event transients are checked separately."""
    peaks={};current={};events=[]
    for row in intervals:
        if row['allocation']<row['release']:
            events.extend([(row['allocation'],1,row['memory'],row['bytes']),
                           (row['release'],0,row['memory'],-row['bytes'])])
    for _,_,memory,change in sorted(events):
        current[memory]=current.get(memory,0)+change
        if current[memory]<0:raise ValueError('invalid intervals')
        peaks[memory]=max(peaks.get(memory,0),current[memory])
    return peaks


def event_peaks(events):
    """Observe allocation/free order, including ties; aliases cannot allocate twice."""
    live={};current={};peaks={}
    for action,name,memory,size in events:
        if action=='allocate':
            if name in live:raise ValueError('duplicate allocation of one root')
            if type(size) is not int or size<=0:raise ValueError('positive physical bytes required')
            live[name]=(memory,size);change=size
        elif action=='free':
            if live.pop(name,None)!=(memory,size):raise ValueError('free does not match a live root')
            change=-size
        else:raise ValueError('unknown memory event')
        current[memory]=current.get(memory,0)+change
        peaks[memory]=max(peaks.get(memory,0),current[memory])
    return peaks


def hand_fork(dtype='fp32',*,retain_slice=False,swap_readers=False,materialize=False):
    """4x6 root, two readers of a six-element holey slice, then a merge.

    This independent schedule deliberately avoids equal-time allocate/free
    ambiguity. Copying the slice adds traffic but shortens the large root's
    lifetime, so zero-copy is not assumed to minimize peak storage.
    """
    root=Root('root',(4,6),dtype)
    selected=root_map(root).slice(((1,4,2),(1,6,2)))
    left=Root('left',(2,3),dtype);right=Root('right',(2,3),dtype);output=Root('output',(2,3),dtype)
    finish=10
    allocations=[Allocation(root,0,0)]
    readers=[];operand=selected
    if materialize:
        copied=Root('copy',(2,3),dtype)
        allocations.append(Allocation(copied,1,2));readers.append(Reader('copy',(selected,),1,2))
        operand=root_map(copied)
    first,last=(7,3) if swap_readers else (3,7)
    # A packing task, when requested, completes before either actual reader.
    start=2.5
    allocations.extend([Allocation(left,start,first),Allocation(right,4 if not swap_readers else start,last),Allocation(output,8,9)])
    readers.extend([Reader('left',(operand,),start,first),
                    Reader('right',(operand,),4 if not swap_readers else start,last),
                    Reader('merge',(root_map(left),root_map(right)),8,9)])
    returned=[root_map(output)]
    if retain_slice:returned.append(selected)
    intervals=lifetimes(allocations,readers,returned=returned,finish=finish)
    write_cost=sum(item.root.bytes for item in allocations if item.root!=root)
    return dict(selected=selected,allocations=allocations,readers=readers,returned=returned,finish=finish,
                intervals=intervals,peaks=interval_peaks(intervals),
                read_bytes=sum(operand_read_bytes(*r.operands) for r in readers),write_bytes=write_cost)
