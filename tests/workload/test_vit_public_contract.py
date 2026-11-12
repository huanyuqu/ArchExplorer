"""SIM48's public input/structure contract, including old semantic boundaries."""
from copy import deepcopy
import importlib
import sys

import pytest

from precision_transformer_expectations import gelu, store
from test_operator_mapping_attention_contract import no_backend
from vit_contract_cases import KEYS, REJECT, compare_reference, frozen_inputs, independent_case, scene, vit_api
from vit_expectations import Geometry, parameter_shapes


@pytest.mark.parametrize('key',KEYS)
def test_explicit_root_shape_query_builders_and_structure_have_no_precomputed_activations(vit_api,monkeypatch,key):
    w,s = vit_api
    p = deepcopy(frozen_inputs()['cases'][key]['workload']['params'])
    before = deepcopy(p)
    with no_backend(monkeypatch):
        query = {k:p[k] for k in ('batch','image_size','patch','width','layers','heads','mlp_hidden','classes')}
        shapes = w.vit_classifier_input_shapes(**query)
        data = independent_case(p)
        g = Geometry(**data['geometry'])
        outer,block = parameter_shapes(g)
        from vit_contract_cases import OUTER,BLOCK
        wanted = {OUTER[k]:tuple(v) for k,v in outer.items()}
        wanted['images'] = (g.batch,3,g.image_size,g.image_size)
        wanted.update({f'block{i}.{BLOCK[k]}':tuple(v) for i in range(g.layers) for k,v in block.items()})
        assert shapes == wanted
        spec = w.vit_classifier(**p)
        assert spec.kind == 'vit_classifier' and spec.params == p
        structure = w.vit_classifier_structure(**{k:v for k,v in p.items() if k!='inputs'})
        assert structure.kind == 'vit_classifier_structure' and 'inputs' not in structure.params
        compare_reference(w.execute_reference(structure,inputs=p['inputs']),key)
        graph = w.workload_from_spec(spec)
    assert graph.meta['outputs']==['logits'] and p==before
    assert graph.meta['vit']['tensors']==frozen_inputs()['cases'][key]['names']
    assert any(op.op=='gelu_exact' for op in graph.operators)
    assert not any(op.op=='gelu_tanh' for op in graph.operators)
    for op in graph.operators:
        if op.op=='affine_norm': assert op.attrs['epsilon']==1e-12


def test_builder_construction_never_invokes_matmul_or_numeric_reference(vit_api,monkeypatch):
    w,s = vit_api
    p = deepcopy(frozen_inputs()['cases']['regular-2-fp32']['workload']['params'])
    def forbidden(*args,**kwargs):
        raise AssertionError('builder calculated a model activation')
    with no_backend(monkeypatch),monkeypatch.context() as patch:
        for module,name in [('arch_explorer.workload.precision','matmul_fp32'),
                            ('arch_explorer.workload.full_primitives','reference'),
                            ('arch_explorer.workload.reference_api','execute_reference')]:
            patch.setattr(importlib.import_module(module),name,forbidden)
        graph = w.workload_from_spec(w.vit_classifier(**p))
    assert graph.meta['outputs']==['logits']


@pytest.mark.parametrize('failure',['missing','extra','shape','layer-shape','nonfinite','overflow',
    'bool','zero','image-patch','head-width','upper-limit','dtype'])
def test_input_refusals_happen_before_any_execution(vit_api,monkeypatch,failure):
    w,s = vit_api
    p = deepcopy(frozen_inputs()['cases']['regular-2-fp16']['workload']['params'])
    with no_backend(monkeypatch):
        assert w.workload_from_spec(w.vit_classifier(**p)).meta['outputs']==['logits']
        if failure=='missing': del p['inputs']['patch_bias']
        elif failure=='extra': p['inputs']['precomputed_X']=[0.]
        elif failure=='shape': p['inputs']['images']=[0.]
        elif failure=='layer-shape': p['inputs']['block1.Wq']=[[0.]]
        elif failure=='nonfinite': p['inputs']['head_bias'][0]=float('nan')
        elif failure=='overflow': p['inputs']['head_bias'][0]=1e20
        elif failure=='bool': p['batch']=True
        elif failure=='zero': p['classes']=0
        elif failure=='image-patch': p['image_size']=5
        elif failure=='head-width': p['heads']=3
        elif failure=='upper-limit': p['batch']=5
        else: p['dtype']='fp8'
        with pytest.raises(REJECT): w.workload_from_spec(w.vit_classifier(**p))


@pytest.mark.parametrize('failure',['unknown-selector','unknown-rule'])
def test_full_model_rule_refusals_follow_normal_nonexecuting_control(vit_api,monkeypatch,failure):
    from arch_explorer.adf import MappingSpec
    w,s = vit_api
    adf,spec,mapping = scene('regular-2-fp32')
    with no_backend(monkeypatch):
        normal = w.compile_workload(adf,spec,mapping)
        assert normal['plan']['rules']
        bad = deepcopy(mapping.params)
        if failure=='unknown-selector': bad['operator_rules']['not-an-original-operator']={'id':'matmul.tiled','version':'1','options':{'tiles':{'m':2,'k':3,'n':3}}}
        else:
            selector = next(iter(bad['operator_rules']))
            bad['operator_rules'][selector]={'id':'not.registered','version':'1','options':{}}
        with pytest.raises(REJECT): w.compile_workload(adf,spec,MappingSpec('compile_graph_v3',bad))


def test_new_erf_primitive_and_old_tanh_encoder_do_not_share_a_formula(vit_api,monkeypatch):
    from arch_explorer.adf import WorkloadSpec
    w,s = vit_api
    outputs = {}
    for kind in ('gelu_exact','gelu_tanh'):
        graph = w.Workload('independent-'+kind,dtype='fp32')
        graph.tensor('X',(1,3));graph.tensor('Y',(1,3))
        graph.meta['inputs']={'X':[[-2.,0.,2.]]};graph.meta['outputs']=['Y']
        graph.add('activation',kind,['X'],['Y'],shape=[1,3])
        with no_backend(monkeypatch): outputs[kind]=w.execute_reference(WorkloadSpec('graph',graph.to_dict()))['outputs']['Y'].reshape(-1).tolist()
    assert abs(outputs['gelu_exact'][2]-1.9544997361036416)<2e-7
    assert abs(outputs['gelu_exact'][2]-outputs['gelu_tanh'][2])>9e-5
    for kind in ('vit','clip'):
        from arch_explorer.workload.full_models import full_model_workload
        assert full_model_workload(kind,dict(profile='synthetic',dtype='fp32'))
        with pytest.raises(ValueError): full_model_workload(kind,dict(profile='synthetic',dtype='bf16'))


def test_complete_ViT_native_is_refused_before_emitter_or_engine(vit_api,monkeypatch):
    from arch_explorer.adf import MappingSpec
    from adapter_boundary_cases import native_scene
    from arch_explorer.workload.adapters import materialized_native as native
    w,s=vit_api
    adf,_,_,seed=native_scene()
    _,spec,_=scene('regular-2-fp32',policy='materialized')
    mapping=MappingSpec('compile_graph_v3',dict(seed.params,operator_rules={}))
    entered=[];previous=sys.getprofile()
    def observer(frame,event,arg):
        if event=='call' and frame.f_code is native._emitter.__code__:entered.append(frame.f_code.co_name)
    try:
        sys.setprofile(observer)
        with no_backend(monkeypatch),pytest.raises(REJECT):w.compile_workload(adf,spec,mapping)
    finally:sys.setprofile(previous)
    assert entered==[]


def test_shared_board_extraction_preserves_old_encoder_tanh_entry(vit_api,monkeypatch):
    from arch_explorer.archs.multilayer import encoder_inputs,encoder_scene
    from arch_explorer.adf import find
    w,s=vit_api
    inputs=encoder_inputs(layers=2,input_case='boundary');before=deepcopy(inputs)
    with no_backend(monkeypatch):
        adf,spec,mapping=encoder_scene(layers=2,dtype='fp32',inputs=inputs,layout='views')
        graph=w.workload_from_spec(spec)
        paths=set(mapping.params['compute'].values())
        assert len(paths)==2 and len({path.rsplit('/',1)[0] for path in paths})==1
        assert all(find(adf,path).params['precision'].value=='fp32' for path in paths)
        assert spec.params['inputs']==before and spec.params['epsilon']==1e-5
        assert sum(op.op=='gelu_tanh' for op in graph.operators)==2
        assert not any(op.op=='gelu_exact' for op in graph.operators)
        assert w.layout_candidates(spec,mapping)['views'].to_dict()==mapping.to_dict()
    assert inputs==before
