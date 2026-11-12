"""SIM47 frozen API consumers; collect on baseline, execute on sole candidate."""
from copy import deepcopy
import json

import pytest

from test_operator_mapping_attention_contract import no_backend
from view_storage_contract_cases import REJECT,assert_layout,hand_graph,independent_layouts,view_api
from view_storage_expectations import Root,root_map,canonical_scalar_bytes


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_public_nested_view_layout_and_CPU_values_match_hand_addresses(view_api,monkeypatch,dtype):
    import numpy as np
    from arch_explorer.adf import WorkloadSpec
    w,_=view_api
    with no_backend(monkeypatch):
        graph=hand_graph(w,dtype=dtype);document=graph.to_dict();before=deepcopy(document)
        assert document['schema']=='arch-explorer.workload-graph.v2'
        expected=independent_layouts(document)
        assert expected['S'].addresses()==[7,9,11,19,21,23]
        assert expected['T'].addresses()==[7,19,9,21,11,23]
        assert expected['U'].addresses()==[9,21,11,23]
        for name in ('X','S','T','U','Y'):assert_layout(w,graph.tensors,name,expected[name])
        restored=w.workload_from_spec(WorkloadSpec('graph',json.loads(json.dumps(document))))
        assert restored.to_dict()==document
        result=w.execute_reference(WorkloadSpec('graph',document))
    assert graph.to_dict()==document==before
    source=result['tensors']['X'];flat=source.reshape(-1).tolist()
    for name in ('S','T','U'):
        value=result['tensors'][name]
        assert np.shares_memory(value,source)
        assert tuple(s//value.itemsize for s in value.strides)==expected[name].steps
        host_offset=(value.__array_interface__['data'][0]-source.__array_interface__['data'][0])//source.itemsize
        assert host_offset==expected[name].first
        actual=[canonical_scalar_bytes(x,dtype) for x in value.reshape(-1).tolist()]
        assert actual==[canonical_scalar_bytes(flat[a],dtype) for a in expected[name].addresses()]
    assert result['outputs']['Y'].tolist()==[[-4.,20.],[0.,24.]]


def test_noncontiguous_explicit_view_reshape_refuses_and_affine_reshape_shares(view_api,monkeypatch):
    w,_=view_api
    with no_backend(monkeypatch):
        graph=w.Workload('reshape-proof',dtype='fp32');graph.tensor('X',(2,3),dtype='fp32')
        w.readonly_view(graph,'T','X',transform='transpose',shape=(3,2),axes=(1,0))
        with pytest.raises(REJECT):w.readonly_view(graph,'invalid','T',transform='reshape',shape=(6,))
        # A separate graph avoids assuming mutating helper rollback semantics.
        other=w.Workload('reshape-shared',dtype='fp32');other.tensor('X',(2,3),dtype='fp32')
        w.readonly_view(other,'R','X',transform='reshape',shape=(1,3,2))
        expected=root_map(Root('X',(2,3))).reshape((1,3,2))
        assert_layout(w,other.tensors,'R',expected)


@pytest.mark.parametrize('mutation',['offset-negative','offset-past-root','zero-stride','negative-stride',
    'dangling','cycle','dtype-reinterpret','wrong-rank'])
def test_layout_resolver_rejects_invalid_alias_chains_before_execution(view_api,monkeypatch,mutation):
    w,_=view_api
    with no_backend(monkeypatch):
        graph=hand_graph(w);valid=graph.to_dict()
        assert w.resolve_layout(graph.tensors,'S').storage=='X'
        rows={row['name']:deepcopy(row) for row in valid['tensors']}
        if mutation=='offset-negative':rows['S']['offset']=-1
        elif mutation=='offset-past-root':rows['S']['offset']=24
        elif mutation=='zero-stride':rows['S']['strides'][1]=0
        elif mutation=='negative-stride':rows['S']['strides'][1]=-2
        elif mutation=='dangling':rows['S']['alias_of']='absent'
        elif mutation=='cycle':rows['X'].update(alias_of='S',strides=[6,1])
        elif mutation=='dtype-reinterpret':rows['S']['dtype']='bf16'
        elif mutation=='wrong-rank':rows['S']['strides']=[1]
        with pytest.raises(REJECT):
            tensors={name:w.Tensor(**row) for name,row in rows.items()}
            w.resolve_layout(tensors,'S')
    assert graph.to_dict()==valid


@pytest.mark.parametrize('mutation',['zero-step','negative-step','empty','past-stop','repeated-axis','bad-shape'])
def test_readonly_operation_rejects_invalid_transform_parameters(view_api,monkeypatch,mutation):
    w,_=view_api
    with no_backend(monkeypatch):
        graph=w.Workload('invalid-transform',dtype='fp32');graph.tensor('X',(4,6),dtype='fp32')
        args=dict(transform='slice',shape=(2,3),starts=[1,1],stops=[4,6],steps=[2,2])
        if mutation=='zero-step':args['steps'][0]=0
        elif mutation=='negative-step':args['steps'][0]=-1
        elif mutation=='empty':args['stops'][0]=1
        elif mutation=='past-stop':args['stops'][0]=5
        elif mutation=='repeated-axis':args=dict(transform='transpose',shape=(6,4),axes=[0,0])
        elif mutation=='bad-shape':args['shape']=(3,3)
        with pytest.raises(REJECT):w.readonly_view(graph,'bad','X',**args)


def test_alias_write_is_rejected_and_input_spec_not_mutated(view_api,monkeypatch):
    from arch_explorer.adf import WorkloadSpec
    w,_=view_api
    with no_backend(monkeypatch):
        graph=hand_graph(w);document=graph.to_dict();original=deepcopy(document)
        document['operators'].append(dict(name='illegal-write',op='scale',inputs=['S'],outputs=['S'],
            attrs=dict(shape=[2,3],factor=2.),label='illegal alias write',type='operator'))
        before=deepcopy(document)
        with pytest.raises(REJECT):w.workload_from_spec(WorkloadSpec('graph',document))
    assert graph.to_dict()==original and document==before


def test_graph_v1_default_offset_migration_and_strict_version_fields(view_api,monkeypatch):
    from arch_explorer.adf import WorkloadSpec
    w,_=view_api
    with no_backend(monkeypatch):
        graph=w.Workload('old-independent',dtype='fp32')
        graph.tensor('X',(2,3),dtype='fp32');graph.tensor('Y',(2,3),dtype='fp32')
        graph.add('scale','scale',['X'],['Y'],shape=[2,3],factor=2.)
        current=graph.to_dict();legacy=deepcopy(current);legacy['schema']='arch-explorer.workload-graph.v1'
        for row in legacy['tensors']:row.pop('offset')
        before=deepcopy(legacy)
        restored=w.workload_from_spec(WorkloadSpec('graph',legacy)).to_dict()
        assert restored==current and legacy==before
        bad=deepcopy(legacy);bad['tensors'][0]['offset']=0
        with pytest.raises(REJECT):w.workload_from_spec(WorkloadSpec('graph',bad))
        bad=deepcopy(current);bad['tensors'][0].pop('offset')
        with pytest.raises(REJECT):w.workload_from_spec(WorkloadSpec('graph',bad))
