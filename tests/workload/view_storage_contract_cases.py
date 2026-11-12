"""Frozen SIM47 inputs and consumers of the independent VAL58 expectations."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from view_storage_expectations import AddressMap,Root,root_map

INPUTS=Path(__file__).resolve().parents[1]/'fixtures/view_storage/INPUTS.json'
INPUT_SHA='c00d1a5031af5096f048a0f2bb8b3cc58e248f83728d02812c28cb409769becc'
REJECT=(ValueError,TypeError,KeyError,NotImplementedError)


@lru_cache(maxsize=1)
def frozen_inputs():
    raw=INPUTS.read_bytes();assert hashlib.sha256(raw).hexdigest()==INPUT_SHA
    return json.loads(raw)


@pytest.fixture
def view_api():
    import numpy
    w=importlib.import_module('arch_explorer.workload.api')
    s=importlib.import_module('arch_explorer.simulator.api')
    for name in ('readonly_view','resolve_layout','element_offset','logical_accesses','layout_candidates'):
        assert callable(getattr(w,name)),name
    return w,s


def scene(key,*,index=0,strategy='views'):
    from arch_explorer.adf import from_dict,MappingSpec,WorkloadSpec
    row=deepcopy(frozen_inputs()[key])
    mapping=row['candidates'][index][strategy] if 'candidates' in row else row['mapping']
    return from_dict(row['adf']),WorkloadSpec(**row['workload']),MappingSpec(**mapping)


def hand_graph(w,*,dtype='fp32',resident=False,retain=False):
    """The oracle's 4x6 root and exact [7,9,11,19,21,23] six-element slice."""
    graph=w.Workload('val58-hand-addresses',dtype=dtype)
    graph.tensor('X',(4,6),dtype=dtype,kind='weight' if resident else 'activation')
    graph.meta['inputs']={'X':[[row*6+column-11 for column in range(6)] for row in range(4)]}
    w.readonly_view(graph,'S','X',transform='slice',shape=(2,3),starts=(1,1),stops=(4,6),steps=(2,2))
    w.readonly_view(graph,'T','S',transform='transpose',shape=(3,2),axes=(1,0))
    w.readonly_view(graph,'U','T',transform='slice',shape=(2,2),starts=(1,0),stops=(3,2),steps=(1,1))
    graph.tensor('Y',(2,2),dtype=dtype)
    graph.add('read','scale',['U'],['Y'],shape=[2,2],factor=2.)
    graph.meta['outputs']=['Y','S'] if retain else ['Y']
    return graph


def independent_layouts(program):
    """Interpret only the frozen graph contract, never a production resolver.

    View transformations independently derive the root walk. Declared strides
    and offsets are checked against that derivation, not used as the expected
    answer for their own validation.
    """
    tensors={row['name']:row for row in program['tensors']}
    producers={name:op for op in program['operators'] for name in op['outputs']}
    result={};visiting=set()
    def visit(name):
        if name in result:return result[name]
        if name in visiting:raise ValueError('cycle in declared alias chain')
        visiting.add(name);row=tensors[name]
        if row['alias_of'] is None:
            value=root_map(Root(name,tuple(row['shape']),row['dtype'],
                resident=name not in producers and row['kind'] in ('weight','kv')))
            if row['offset']!=0 or row['strides'] is not None:raise ValueError('independent storage must use its canonical layout')
        else:
            op=producers[name]
            if op['op']!='storage.view' or op['inputs']!=[row['alias_of']]:
                raise ValueError('view storage must be produced from its declared original input')
            source=visit(row['alias_of']);attrs=op['attrs'];transform=attrs['transform']
            if transform=='transpose':value=source.transpose(attrs['axes'])
            elif transform=='slice':value=source.slice(zip(attrs['starts'],attrs['stops'],attrs['steps']))
            elif transform=='reshape':value=source.reshape(attrs['shape'])
            else:raise ValueError('unknown readonly transform')
            if tuple(row['shape'])!=value.shape or row['dtype']!=value.root.dtype:
                raise ValueError('view shape or storage dtype differs')
            if row['offset']!=value.first-source.first:raise ValueError('relative alias offset differs')
            if row['strides'] is not None:
                declared=tuple(row['strides'])
                if len(declared)!=len(value.shape) or any(type(s) is not int or s<=0 for s in declared):
                    raise ValueError('one positive declared element stride per axis required')
                if any(size>1 and actual!=expected for size,actual,expected in zip(value.shape,declared,value.steps)):
                    raise ValueError('declared strides differ from semantic traversal')
                # A singleton axis never varies, so its positive stride is
                # not uniquely determined by an element walk. Preserve that
                # representation while independently checking every address.
                value=AddressMap(value.root,value.shape,tuple(actual if size==1 else expected
                    for size,actual,expected in zip(value.shape,declared,value.steps)),value.first)
        result[name]=value;visiting.remove(name);return value
    for name in tensors:visit(name)
    return result


def assert_layout(w,tensors,name,expected):
    actual=w.resolve_layout(tensors,name)
    assert actual.storage==expected.root.name
    assert tuple(actual.shape)==expected.shape and actual.dtype==expected.root.dtype
    assert tuple(actual.strides)==expected.steps and actual.offset==expected.first
    assert actual.allocation_bytes==expected.root.bytes
    exported=actual.to_dict()
    assert exported['units']=='elements' and exported['readonly'] is True
    for index in __import__('itertools').product(*(range(n) for n in expected.shape)):
        assert w.element_offset(actual,index)==expected.address(index)
    return actual
