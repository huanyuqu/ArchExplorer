"""Independent logical array read ledger from SIM53's frozen public contract.

Only small controls: explicitly enumerate source coordinates, preserving repeated
reads. Physical addresses and component timings require actual candidate output.
"""
from collections import Counter
import math
import numpy as np
from onnx_index_expectations import axis,integer_array

WIDTH={'fp32':4,'fp16':2,'bf16':2,'int32':4,'int64':8,'bool':1}


def expected(op,tensors,values):
    lookup={t['name']:t for t in tensors} if isinstance(tensors,list) else tensors
    args=[lookup[n] for n in op['inputs']];out=lookup[op['outputs'][0]];attrs=op['attrs'];kind=op['op']
    shape=tuple(out['shape']);count=math.prod(shape);reads=[]
    if list(shape)!=list(attrs['shape']):raise ValueError('access oracle output shape differs')
    def read(name,at):reads.append((name,tuple(at)))
    if kind in ('gather','gather_elements'):
        data,indices=args;data_shape=tuple(data['shape']);index_shape=tuple(indices['shape']);ax=axis(attrs['axis'],len(data_shape))
        index=integer_array(values[indices['name']])
        if index.shape!=index_shape:raise ValueError('access oracle index shape differs')
        size=data_shape[ax]
        if np.any(index < -size) or np.any(index>=size):raise ValueError('access oracle index outside bounds')
        if kind=='gather':
            scalar=attrs.get('indices_scalar',False)
            if scalar and index_shape!=(1,):raise ValueError('access oracle scalar index physical shape differs')
            logical=data_shape[:ax]+(() if scalar else index_shape)+data_shape[ax+1:]
            if shape!=(logical or (1,)):raise ValueError('access oracle Gather output shape differs')
            for at in np.ndindex(index_shape):read(indices['name'],at)
            for prefix in np.ndindex(data_shape[:ax]):
                for at in np.ndindex(index_shape):
                    selected=int(index[at])%size
                    for suffix in np.ndindex(data_shape[ax+1:]):read(data['name'],prefix+(selected,)+suffix)
            instructions=3*index.size+count
        else:
            if len(index_shape)!=len(data_shape) or shape!=index_shape or any(index_shape[i]>data_shape[i] for i in range(len(shape)) if i!=ax):
                raise ValueError('access oracle GatherElements shape differs')
            for at in np.ndindex(shape):
                read(indices['name'],at);source=list(at);source[ax]=int(index[at])%size;read(data['name'],source)
            instructions=4*count
    elif kind=='argmax_axis':
        data=args[0];ds=tuple(data['shape']);ax=axis(attrs['axis'],len(ds));remaining=ds[:ax]+ds[ax+1:]
        logical=ds[:ax]+(1,)+ds[ax+1:] if attrs['keepdims'] else remaining
        if shape!=(logical or (1,)) or out['dtype']!='int64':raise ValueError('access oracle ArgMax shape/type differs')
        for at in np.ndindex(remaining):
            for selected in range(ds[ax]):read(data['name'],at[:ax]+(selected,)+at[ax:])
        instructions=2*count*(ds[ax]-1)
    elif kind in ('where','integer_add','integer_sub','integer_mul','integer_div','equal','greater','less'):
        if any(tuple(a['shape'])!=shape for a in args):raise ValueError('access oracle eager array shape differs')
        if kind=='where' and args[0]['dtype']!='bool':raise ValueError('access oracle Where requires bool condition')
        for at in np.ndindex(shape):
            for arg in args:read(arg['name'],at)
        instructions=count
    else:raise ValueError('access oracle kind outside prepared contract')
    bytes_by_name=Counter()
    for name,_ in reads:bytes_by_name[name]+=WIDTH[lookup[name]['dtype']]
    return dict(reads=reads,read_by_tensor=dict(bytes_by_name),read_bytes=sum(bytes_by_name.values()),
        write_by_tensor={out['name']:count*WIDTH[out['dtype']]},write_bytes=count*WIDTH[out['dtype']],
        simt_instructions=instructions,sfu_instructions=0,conversion_instructions=0,boundary='component_io')


def compare_counts(actual,expectation):
    for key in ('read_by_tensor','write_by_tensor','read_bytes','write_bytes','simt_instructions',
                'sfu_instructions','conversion_instructions','boundary'):
        if actual[key]!=expectation[key]:raise ValueError('array access count differs: '+key)
    return dict(count_fields_matched=True,scope='logical component IO and abstract issue counts; not hardware cycles')


def compare_indices(actual,expectation):
    observed=[(name,tuple(at)) for name,at in actual]
    if observed!=expectation['reads']:raise ValueError('array logical read coordinates/order differ')
    return dict(logical_coordinates_matched=True,reads=len(observed))
