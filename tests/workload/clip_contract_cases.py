"""SIM49 frozen names/data bridge; independent answers remain QA-only."""
from copy import deepcopy
from functools import lru_cache
import hashlib,importlib,json
from pathlib import Path

import pytest
from clip_expectations import Geometry,TOLERANCE,clip_reference,parameter_shapes
from vit_expectations import leaves
from view_storage_expectations import canonical_scalar_bytes

INPUTS=Path(__file__).resolve().parents[1]/'fixtures/clip_compilation/INPUTS.json'
INPUT_SHA='83ce6709411c402bfedaf0cfb5dbd497103a766edb0d97fd979fed6ee2f8f069'
KEYS=tuple(f'{case}-{layers}-{dtype}' for case in ('regular','boundary') for layers in (2,4) for dtype in ('fp32','fp16','bf16'))
REJECT=(ValueError,TypeError,KeyError,NotImplementedError)
GEOMETRY={'image_batch':'image_batch','text_batch':'text_batch','image_size':'image_size','patch':'patch',
    'vision_width':'width','vision_heads':'heads','vision_mlp':'mlp_hidden','vision_layers':'layers',
    'context':'context','vocab':'vocab','text_width':'text_width','text_heads':'text_heads','text_layers':'text_layers','projection':'projection'}
BLOCK={'q_weight':'Wq','k_weight':'Wk','v_weight':'Wv','o_weight':'Wo','q_bias':'bq','k_bias':'bk','v_bias':'bv','o_bias':'bo',
    'norm1_gamma':'norm1_gamma','norm1_beta':'norm1_beta','norm2_gamma':'norm2_gamma','norm2_beta':'norm2_beta',
    'up_weight':'W1','up_bias':'b1','down_weight':'W2','down_bias':'b2'}
VISION={'patch_weight':'patch_weight','class_embedding':'cls_weight','positions':'position_weight',
    'pre_gamma':'pre_gamma','pre_beta':'pre_beta','post_gamma':'post_gamma','post_beta':'post_beta','projection':'projection_weight'}
TEXT={'token_weight':'token_weight','positions':'position_weight','final_gamma':'norm_gamma','final_beta':'norm_beta','projection':'projection_weight'}

@lru_cache(maxsize=1)
def frozen_inputs():
    raw=INPUTS.read_bytes();assert hashlib.sha256(raw).hexdigest()==INPUT_SHA
    return json.loads(raw)

def independent_case(params):
    g=Geometry(**{qa:params[public] for qa,public in GEOMETRY.items()});data=params['inputs']
    expected={'images','token_ids','logit_scale'}
    roots={n:deepcopy(data[n]) for n in expected}
    for tower,names,count in [('vision',VISION,g.vision_layers),('text',TEXT,g.text_layers)]:
        expected.update(tower+'.'+n for n in names.values())
        roots[tower]={n:deepcopy(data[tower+'.'+p]) for n,p in names.items()}
        roots[tower]['blocks']=[]
        for i in range(count):
            expected.update(f'{tower}.block{i}.'+n for n in BLOCK.values())
            roots[tower]['blocks'].append({n:deepcopy(data[f'{tower}.block{i}.'+p]) for n,p in BLOCK.items()})
    assert set(data)==expected
    from dataclasses import asdict
    return dict(geometry=asdict(g),dtype=params['dtype'],case='SIM49-frozen-roots',roots=roots)

def shapes(params):
    g=Geometry(**{qa:params[public] for qa,public in GEOMETRY.items()});v,t,vb,tb=parameter_shapes(g)
    result={'images':(g.image_batch,3,g.image_size,g.image_size),'token_ids':(g.text_batch,g.context),'logit_scale':(1,)}
    for tower,names,base,block,count in [('vision',VISION,v,vb,g.vision_layers),('text',TEXT,t,tb,g.text_layers)]:
        result.update({tower+'.'+names[n]:shape for n,shape in base.items()})
        result.update({f'{tower}.block{i}.'+BLOCK[n]:shape for i in range(count) for n,shape in block.items()})
    return result

@lru_cache(maxsize=12)
def expected(key):return clip_reference(independent_case(frozen_inputs()['cases'][key]['workload']['params']))

@pytest.fixture
def clip_api():
    w=importlib.import_module('arch_explorer.workload.api');s=importlib.import_module('arch_explorer.simulator.api')
    for name in ('clip_input_shapes','clip_similarity','clip_structure','qualified_name','logical_accesses','layout_candidates','validate_compilation'):
        assert callable(getattr(w,name)),name
    return w,s

def scene(key,mode='events',policy='views',index=0):
    from arch_explorer.adf import WorkloadSpec,MappingSpec,from_dict
    d=frozen_inputs();r=d['cases'][key]
    return from_dict(deepcopy(d['hardware'][mode])),WorkloadSpec(**deepcopy(r['workload'])),MappingSpec(**deepcopy(r['candidates'][index][policy]))

def canonical(values,dtype):
    return b''.join(canonical_scalar_bytes(v,dtype) for v in leaves(values))

def close(actual,wanted,dtype,*,exact=False):
    a=actual.reshape(-1).tolist() if hasattr(actual,'reshape') else list(leaves(actual));b=list(leaves(wanted))
    assert len(a)==len(b)
    if exact or dtype=='int32':assert canonical(a,dtype)==canonical(b,dtype)
    else:
        tolerance=TOLERANCE[dtype]
        for x,y in zip(a,b):assert abs(x-y)<=tolerance['absolute']+tolerance['relative']*abs(y),(dtype,x,y)

def compare_reference(ref,key):
    row=frozen_inputs()['cases'][key];p=row['workload']['params'];wanted=expected(key)
    g=ref['graph'];names=g.meta['clip']['tensors'];values=ref['tensors'];dtype=p['dtype']
    assert names==row['names'] and set(g.meta['outputs'])=={'logits_per_image','logits_per_text'}
    stages={'patches':'vision.patches','vision_position':'vision.position','vision_pre_norm':'vision.pre_norm',
        'vision_select_cls':'vision.CLS_select','vision_post_norm':'vision.post_norm','vision_projection':'vision.projection',
        'lookup':'text.lookup','text_position':'text.position','text_final_norm':'text.final_norm','eot_index':'text.eot_index',
        'select_eot':'text.EOT_select','text_projection':'text.projection','image_features':'image_features',
        'text_features':'text_features','logit_scale_exp':'logit_scale.exp','image_scaled':'image_scaled',
        'logits_per_image':'logits_per_image','logits_per_text':'logits_per_text'}
    for label,stage in stages.items():
        kind='int32' if label=='eot_index' else 'fp32' if label=='logit_scale_exp' else dtype
        close(values[names[label]],wanted['stages'][stage],kind,exact=label in ('patches','lookup'))
    mask=[[int(c>r) for c in range(p['context'])] for r in range(p['context'])]
    close(values[names['causal_mask']],mask,'int32')
    instances={i['id']:i for i in g.meta['composition']['instances']}
    assert instances.keys()=={i['id'] for i in row['composition']['instances']}
    arithmetic={'ln1':'norm1','q.mm':'q.mm','q.bias':'q.bias','k.mm':'k.mm','k.bias':'k.bias','v.mm':'v.mm','v.bias':'v.bias',
        'qk':'QK','scale':'scale','softmax':'softmax','av':'AV','merge':'merge','out.mm':'o.mm','out.bias':'o.bias',
        'residual1':'residual1','ln2':'norm2','up.mm':'up.mm','up.bias':'up.bias','quick_gelu':'quick_gelu',
        'down.mm':'down.mm','down.bias':'down.bias','Y':'output'}
    for tower,count in [('vision',p['layers']),('text',p['text_layers'])]:
        for i in range(count):
            instance=instances[f'{tower}_block{i}']
            for local,qa in arithmetic.items():
                close(values[instance['tensor_names'][local]],wanted['stages'][f'{tower}.layer{i}.{qa}'],dtype)
            if tower=='text':
                score=wanted['stages'][f'text.layer{i}.scale']
                payload=[[[0. if c>r else v for c,v in enumerate(line)] for r,line in enumerate(head)] for head in score]
                close(values[names['text_mask_payloads'][i]],payload,dtype)
    close(values[names['text_transposed']],[list(v) for v in zip(*wanted['stages']['text_features'])],dtype)
    assert canonical(values[names['logits_per_text']].reshape(-1).tolist(),dtype)==canonical(values[names['logits_per_image']].T.reshape(-1).tolist(),dtype)

def execution_rows():
    for key in KEYS:
        family,layers,dtype=key.split('-')
        for index in (0,1):
            execute=family=='regular' or layers=='2' and index==1 or layers=='4' and index==0
            mode=('events' if index==0 else 'roofline') if family=='regular' else ('events' if layers=='2' else 'roofline')
            for policy in ('materialized','views'):yield key,mode,policy,index,execute
