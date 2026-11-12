"""Complete small CLIP from two explicit root inputs and independent parameters."""
from .graph import finite_json,name as valid_name
from .operators import Tensor
from .structural import Graph

_DEFAULTS=dict(image_batch=2,text_batch=3,image_size=4,patch=2,width=4,layers=2,heads=2,
    mlp_hidden=7,context=4,vocab=13,text_width=4,text_heads=2,text_layers=2,projection=3,
    dtype='fp32',name='clip_similarity')
_DIMS=tuple(k for k in _DEFAULTS if k not in ('dtype','name'))


def _block_shapes(batch,sequence,width,heads,mlp):
    from .transformer_block import transformer_block_input_shapes
    from .extensions import _record_source
    _record_source(transformer_block_input_shapes)
    return transformer_block_input_shapes(batch=batch,sequence=sequence,hidden=width,heads=heads,mlp_hidden=mlp)


def clip_input_shapes(*,image_batch=2,text_batch=3,image_size=4,patch=2,width=4,layers=2,heads=2,
                      mlp_hidden=7,context=4,vocab=13,text_width=4,text_heads=2,text_layers=2,projection=3):
    dims=(image_batch,text_batch,image_size,patch,width,layers,heads,mlp_hidden,context,vocab,text_width,text_heads,text_layers,projection)
    if any(type(v) is not int or v<1 for v in dims):raise ValueError('CLIP dimensions must be positive integers')
    if (max(image_batch,text_batch)>4 or image_size>16 or width>64 or text_width>32 or mlp_hidden>128
            or max(layers,text_layers)>4 or context>16 or vocab>1024 or projection>64
            or image_size%patch or width%heads or text_width%text_heads or (image_size//patch)**2+1>32):
        raise ValueError('small CLIP bounds/divisibility violated')
    tokens=(image_size//patch)**2+1
    roots=dict(images=(image_batch,3,image_size,image_size),token_ids=(text_batch,context))
    roots.update({'vision.patch_weight':(3*patch*patch,width),'vision.cls_weight':(1,1,width),
        'vision.position_weight':(tokens,width),'vision.pre_gamma':(width,),'vision.pre_beta':(width,),
        'vision.post_gamma':(width,),'vision.post_beta':(width,),'vision.projection_weight':(width,projection),
        'text.token_weight':(vocab,text_width),'text.position_weight':(context,text_width),
        'text.norm_gamma':(text_width,),'text.norm_beta':(text_width,),'text.projection_weight':(text_width,projection),
        'logit_scale':(1,)})
    for tower,count,b,s,h,nh,f in (('vision',layers,image_batch,tokens,width,heads,mlp_hidden),
                                  ('text',text_layers,text_batch,context,text_width,text_heads,4*text_width)):
        shapes=_block_shapes(b,s,h,nh,f)
        for i in range(count):roots.update({f'{tower}.block{i}.'+k:v for k,v in shapes.items() if k!='X'})
    return roots


def _dtype(name,dtype):return 'int32' if name=='token_ids' else 'fp32' if name=='logit_scale' else dtype


def _configuration(params,*,data):
    if type(params) is not dict or set(params)-{*_DEFAULTS,*({'inputs'} if data else set())} or (data and 'inputs' not in params):
        raise ValueError('CLIP requires supported configuration fields and complete explicit roots')
    p={**_DEFAULTS,**finite_json(params)};shapes=clip_input_shapes(**{k:p[k] for k in _DIMS})
    if p['dtype'] not in ('fp32','fp16','bf16'):raise ValueError('CLIP storage must be fp32/fp16/bf16')
    valid_name(p['name'],'CLIP name')
    if data:
        if type(p['inputs']) is not dict or set(p['inputs'])!=set(shapes):raise ValueError('CLIP roots must bind exactly both inputs and all parameters')
        from .reference_api import _input
        from .extensions import _record_source
        _record_source(_input)
        for key,shape in shapes.items():_input(p['inputs'][key],Tensor(key,shape,_dtype(key,p['dtype'])))
        import numpy as np
        ids=np.asarray(p['inputs']['token_ids'])
        if ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=p['vocab']):
            raise ValueError('CLIP token_ids require integer JSON values in [0,vocab), no float/bool/string coercion')
    return p,shapes


def clip_similarity(*,inputs,image_batch=2,text_batch=3,image_size=4,patch=2,width=4,layers=2,heads=2,
                    mlp_hidden=7,context=4,vocab=13,text_width=4,text_heads=2,text_layers=2,projection=3,
                    dtype='fp32',name='clip_similarity'):
    from arch_explorer.adf.evaluation import WorkloadSpec
    p=dict(inputs=inputs,image_batch=image_batch,text_batch=text_batch,image_size=image_size,patch=patch,width=width,layers=layers,
        heads=heads,mlp_hidden=mlp_hidden,context=context,vocab=vocab,text_width=text_width,text_heads=text_heads,
        text_layers=text_layers,projection=projection,dtype=dtype,name=name)
    _configuration(p,data=True)
    return WorkloadSpec('clip_similarity',finite_json(p))


def clip_structure(*,image_batch=2,text_batch=3,image_size=4,patch=2,width=4,layers=2,heads=2,
                   mlp_hidden=7,context=4,vocab=13,text_width=4,text_heads=2,text_layers=2,projection=3,
                   dtype='fp32',name='clip_similarity'):
    from arch_explorer.adf.evaluation import WorkloadSpec
    p=dict(image_batch=image_batch,text_batch=text_batch,image_size=image_size,patch=patch,width=width,layers=layers,
        heads=heads,mlp_hidden=mlp_hidden,context=context,vocab=vocab,text_width=text_width,text_heads=text_heads,
        text_layers=text_layers,projection=projection,dtype=dtype,name=name)
    _configuration(p,data=False)
    return WorkloadSpec('clip_structure',finite_json(p))


def build_part(params,*,tp=1):
    if type(tp) is not int or tp!=1:raise ValueError('CLIP parts support tp=1 only')
    if type(params) is not dict or 'section' not in params:raise ValueError('CLIP part requires section')
    p,shapes=_configuration({k:v for k,v in params.items() if k!='section'},data=False);section=params['section'];dtype=p['dtype']
    b,t,r,patch,h,heads,s,th,tnh,proj=(p[k] for k in ('image_batch','text_batch','image_size','patch','width','heads','context','text_width','text_heads','projection'))
    v=(r//patch)**2+1
    def roots(prefix,keys):return {key:shapes[prefix+key] for key in keys}
    def activation(name,shape,storage=dtype):return Tensor(name,shape,storage,'activation')
    if section=='vision_stem':
        g=Graph(section,dtype,dict(images=activation('images',shapes['images']),**roots('vision.',('patch_weight','cls_weight','position_weight','pre_gamma','pre_beta'))))
        x=g.op('patches','patchify',['images'],(b,v-1,3*patch*patch),patch=patch)
        x=g.linear('projection',x,'patch_weight')
        cls=g.op('cls_broadcast','broadcast',['cls_weight'],(b,1,h));x=g.op('cls','concat',[cls,x],(b,v,h),axis=1)
        pos=g.op('position_broadcast','broadcast',['position_weight'],(b,v,h));x=g.op('position','add',[x,pos],(b,v,h))
        y=g.norm('pre_norm',x,'pre_gamma','pre_beta',1e-5);outputs=[y]
    elif section in ('vision_block','text_block'):
        text=section=='text_block';batch,seq,width,nh,f=(t,s,th,tnh,4*th) if text else (b,v,h,heads,p['mlp_hidden'])
        rt=_block_shapes(batch,seq,width,nh,f)
        if text:rt['M']=activation('M',(s,s),'int32')
        g=Graph(section,dtype,rt);g.encoder(batch=batch,sequence=seq,width=width,heads=nh,mlp_hidden=f,
            epsilon=1e-5,activation='quick_gelu',mask='M' if text else None);outputs=['Y']
    elif section=='vision_head':
        g=Graph(section,dtype,dict(X=activation('X',(b,v,h)),**roots('vision.',('post_gamma','post_beta','projection_weight'))))
        x=g.op('cls_slice','slice',['X'],(b,1,h),starts=[0,0,0],stops=[b,1,h],steps=[1,1,1]);x=g.op('select_cls','reshape',[x],(b,h))
        x=g.norm('post_norm',x,'post_gamma','post_beta',1e-5);outputs=[g.linear('projection',x,'projection_weight')]
    elif section=='text_stem':
        g=Graph(section,dtype,dict(token_ids=activation('token_ids',(t,s),'int32'),**roots('text.',('token_weight','position_weight'))))
        x=g.op('lookup','lookup',['token_ids','token_weight'],(t,s,th));pos=g.op('position_broadcast','broadcast',['position_weight'],(t,s,th))
        g.op('Y','add',[x,pos],(t,s,th));g.op('M','causal_mask',[],(s,s),storage='int32');outputs=['Y','M']
    elif section=='text_head':
        g=Graph(section,dtype,dict(X=activation('X',(t,s,th)),token_ids=activation('token_ids',(t,s),'int32'),
            **roots('text.',('norm_gamma','norm_beta','projection_weight'))))
        x=g.norm('final_norm','X','norm_gamma','norm_beta',1e-5)
        ids=g.op('eot_index','argmax',['token_ids'],(t,),storage='int32');x=g.op('select_eot','select',[x,ids],(t,th))
        outputs=[g.linear('projection',x,'projection_weight')]
    elif section=='similarity':
        g=Graph(section,dtype,dict(image=activation('image',(b,proj)),text=activation('text',(t,proj)),logit_scale=Tensor('logit_scale',(1,),'fp32','weight')))
        image=g.op('image_features','l2_norm',['image'],(b,proj));text=g.op('text_features','l2_norm',['text'],(t,proj))
        scale=g.op('logit_scale_exp','exp',['logit_scale'],(1,),storage='fp32')
        image=g.op('image_scaled','scale_tensor',[image,scale],(b,proj));text=g.op('text_transposed','transpose',[text],(proj,t),axes=[1,0])
        logits=g.mm('similarity',image,text,1,b,proj,t)
        y=g.op('logits_per_image','reshape',[logits],(b,t));g.op('logits_per_text','transpose',[y],(t,b),axes=[1,0])
        outputs=['logits_per_image','logits_per_text']
    else:raise ValueError('unknown CLIP structural section')
    g.w.meta['outputs']=outputs
    return g.w


def _build(p,shapes,*,data,tp):
    if type(tp) is not int or tp!=1:raise ValueError('CLIP supports tp=1 only')
    from .api import workload_from_spec
    from .composition import compose_workloads
    from arch_explorer.adf.evaluation import WorkloadSpec
    config={k:p[k] for k in _DEFAULTS if k!='name'}
    roots={key:dict(shape=list(shape),dtype=_dtype(key,p['dtype']),kind='activation' if key in ('images','token_ids') else 'weight',
                   **({'value':p['inputs'][key]} if data else {})) for key,shape in shapes.items()}
    instances=[];bindings=[];part_outputs={}
    sections=[('vision_stem','vision_stem'),*(('vision_block'+str(i),'vision_block') for i in range(p['layers'])),('vision_head','vision_head'),
        ('text_stem','text_stem'),*(('text_block'+str(i),'text_block') for i in range(p['text_layers'])),('text_head','text_head'),('similarity','similarity')]
    for ident,section in sections:
        spec=WorkloadSpec('clip_part',dict(section=section,**config));g=workload_from_spec(spec)
        produced={n for o in g.operators for n in o.outputs};external=[n for n in g.tensors if n not in produced]
        outputs=({'Y':g.meta['outputs'][0],'M':'M'} if section=='text_stem' else
                 {'I':'logits_per_image','T':'logits_per_text'} if section=='similarity' else {'Y':g.meta['outputs'][0]})
        part_outputs[ident]=outputs
        instances.append(dict(id=ident,workload=spec,inputs={n:n for n in external},outputs=outputs))
        for local in external:
            if section.endswith('_block') and local=='X':
                tower=section.split('_')[0];index=int(ident[len(tower+'_block'):]);parent=tower+'_stem' if not index else tower+'_block'+str(index-1)
                source=dict(instance=parent,port='Y')
            elif local=='M':source=dict(instance='text_stem',port='M')
            elif section in ('vision_head','text_head') and local=='X':
                tower=section.split('_')[0];count=p['layers' if tower=='vision' else 'text_layers'];source=dict(instance=tower+'_block'+str(count-1),port='Y')
            elif section=='similarity' and local in ('image','text'):source=dict(instance='vision_head' if local=='image' else 'text_head',port='Y')
            else:
                root=(local if local in ('images','token_ids','logit_scale') else ident.replace('_','.')+'.'+local if section.endswith('_block')
                      else section.split('_')[0]+'.'+local)
                source={'input':root}
            bindings.append({'to':dict(instance=ident,port=local),'from':source})
    graph=workload_from_spec(compose_workloads(name=p['name'],instances=instances,inputs=roots,bindings=bindings,
        outputs={'logits_per_image':dict(instance='similarity',port='I'),'logits_per_text':dict(instance='similarity',port='T')}))
    names={i['id']:i['tensor_names'] for i in graph.meta['composition']['instances']}
    graph.meta['clip']=dict(config={k:p[k] for k in _DIMS},storage_dtype=p['dtype'],accumulation='fp32',epsilon=1e-5,gelu='quick_gelu',
        integer_storage='int32 token IDs, EOT index and binary mask flags; no float promotion charge',logit_scale='FP32 parameter and exp; scale image features before similarity matmul',
        mask='0 keep/1 blocked int32; finite apply_mask payload then private -inf in masked_softmax, no nonfinite graph tensors',
        policy='uniform-storage inference; BF16 new variant, not legacy BF16 or official mixed precision',
        service='abstract shared SIMT/SFU service with local scratch; no old global-scratch timing or hardware-accuracy equivalence',
        tensors=dict(patches=names['vision_stem']['patches'],vision_position=names['vision_stem']['position'],vision_pre_norm=names['vision_stem']['pre_norm'],
            vision_blocks=[names['vision_block'+str(i)]['Y'] for i in range(p['layers'])],vision_select_cls=names['vision_head']['select_cls'],
            vision_post_norm=names['vision_head']['post_norm'],vision_projection=names['vision_head'][part_outputs['vision_head']['Y']],
            lookup=names['text_stem']['lookup'],text_position=names['text_stem']['Y'],causal_mask=names['text_stem']['M'],
            text_blocks=[names['text_block'+str(i)]['Y'] for i in range(p['text_layers'])],
            text_mask_payloads=[names['text_block'+str(i)]['masked_payload'] for i in range(p['text_layers'])],
            text_probabilities=[names['text_block'+str(i)]['softmax'] for i in range(p['text_layers'])],
            text_final_norm=names['text_head']['final_norm'],eot_index=names['text_head']['eot_index'],select_eot=names['text_head']['select_eot'],
            text_projection=names['text_head'][part_outputs['text_head']['Y']],
            **{key:names['similarity'][key] for key in ('image_features','text_features','logit_scale_exp','image_scaled','text_transposed','logits_per_image','logits_per_text')}))
    return graph


def build_similarity(params,*,tp=1):
    p,shapes=_configuration(params,data=True);return _build(p,shapes,data=True,tp=tp)


def build_structure(params,*,tp=1):
    p,shapes=_configuration(params,data=False);return _build(p,shapes,data=False,tp=tp)
