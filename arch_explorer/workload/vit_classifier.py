"""Small complete ViT, composed from structural graphs and explicit root data.

The legacy ViT's primitive registry owns all numerical definitions. In
particular this model uses stored scaled logits, fused stable softmax and erf
GELU, rather than the public tanh encoder's different quantization boundaries.
"""
from .graph import finite_json, name as valid_name
from .operators import Workload, Tensor
from .structural import Graph as _Graph

_DEFAULTS = dict(batch=1,image_size=8,patch=4,width=8,layers=2,heads=2,
                 mlp_hidden=16,classes=3,dtype='fp32',name='vit_classifier')
_DIMENSIONS = ('batch','image_size','patch','width','layers','heads','mlp_hidden','classes')


def vit_classifier_input_shapes(*, batch=1,image_size=8,patch=4,width=8,layers=2,
                                heads=2,mlp_hidden=16,classes=3):
    dims=(batch,image_size,patch,width,layers,heads,mlp_hidden,classes)
    if any(type(n) is not int or n < 1 for n in dims):
        raise ValueError('ViT dimensions must be positive integers')
    if (batch>4 or image_size>32 or width>64 or layers>4 or mlp_hidden>128 or classes>64
            or image_size%patch or width%heads or (image_size//patch)**2+1>64):
        raise ValueError('small ViT bounds B<=4/image<=32/width<=64/layers<=4/MLP<=128/classes<=64/tokens<=64; image/patch and width/heads must divide')
    from .transformer_block import transformer_block_input_shapes
    from .extensions import _record_source
    _record_source(transformer_block_input_shapes)
    block=transformer_block_input_shapes(batch=batch,sequence=(image_size//patch)**2+1,
        hidden=width,heads=heads,mlp_hidden=mlp_hidden)
    shapes=dict(images=(batch,3,image_size,image_size),patch_weight=(3*patch*patch,width),
        patch_bias=(width,),cls_weight=(1,1,width),position_weight=((image_size//patch)**2+1,width))
    for i in range(layers):
        shapes.update({'block'+str(i)+'.'+k:v for k,v in block.items() if k!='X'})
    shapes.update(norm_gamma=(width,),norm_beta=(width,),head_weight=(width,classes),head_bias=(classes,))
    return shapes


def _configuration(params, *, data):
    if type(params) is not dict or set(params)-{*_DEFAULTS,*({'inputs'} if data else set())} or (data and 'inputs' not in params):
        raise ValueError('ViT requires supported configuration fields and explicit root inputs')
    p={**_DEFAULTS,**finite_json(params)}
    shapes=vit_classifier_input_shapes(**{k:p[k] for k in _DIMENSIONS})
    if p['dtype'] not in ('fp32','fp16','bf16'):raise ValueError('ViT dtype must be fp32/fp16/bf16')
    valid_name(p['name'],'ViT name')
    if data:
        if type(p['inputs']) is not dict or set(p['inputs'])!=set(shapes):
            raise ValueError('ViT inputs must bind exactly images and every independent root parameter')
        from .reference_api import _input
        from .extensions import _record_source
        _record_source(_input)
        for key,shape in shapes.items():_input(p['inputs'][key],Tensor(key,shape,p['dtype']))
    return p,shapes


def vit_classifier(*, inputs,batch=1,image_size=8,patch=4,width=8,layers=2,heads=2,
                   mlp_hidden=16,classes=3,dtype='fp32',name='vit_classifier'):
    from arch_explorer.adf.evaluation import WorkloadSpec
    p=dict(inputs=inputs,batch=batch,image_size=image_size,patch=patch,width=width,layers=layers,
           heads=heads,mlp_hidden=mlp_hidden,classes=classes,dtype=dtype,name=name)
    _configuration(p,data=True)
    return WorkloadSpec('vit_classifier',finite_json(p))


def vit_classifier_structure(*, batch=1,image_size=8,patch=4,width=8,layers=2,heads=2,
                             mlp_hidden=16,classes=3,dtype='fp32',name='vit_classifier'):
    from arch_explorer.adf.evaluation import WorkloadSpec
    p=dict(batch=batch,image_size=image_size,patch=patch,width=width,layers=layers,
           heads=heads,mlp_hidden=mlp_hidden,classes=classes,dtype=dtype,name=name)
    _configuration(p,data=False)
    return WorkloadSpec('vit_classifier_structure',finite_json(p))


def build_part(params, *, tp=1):
    """Structural subgraph builder; section is stem/block/head, data bound above it."""
    if type(tp) is not int or tp!=1:raise ValueError('ViT parts support tp=1 only')
    if type(params) is not dict or 'section' not in params:raise ValueError('ViT part requires a section')
    p,shapes=_configuration({k:v for k,v in params.items() if k!='section'},data=False)
    section=params['section']
    b,r,patch,h,heads,f,classes=(p[k] for k in ('batch','image_size','patch','width','heads','mlp_hidden','classes'))
    count=(r//patch)**2;s=count+1;d=h//heads
    if section=='stem':
        g=_Graph('vit_stem',p['dtype'],{k:v for k,v in shapes.items() if k in ('images','patch_weight','patch_bias','cls_weight','position_weight')})
        x=g.op('patches','patchify',['images'],(b,count,3*patch*patch),patch=patch)
        x=g.linear('projection',x,'patch_weight','patch_bias')
        cls=g.op('cls_broadcast','broadcast',['cls_weight'],(b,1,h))
        x=g.op('cls','concat',[cls,x],(b,s,h),axis=1)
        pos=g.op('position_broadcast','broadcast',['position_weight'],(b,s,h))
        g.op('Y','add',[x,pos],(b,s,h))
    elif section=='block':
        roots={'X':(b,s,h),**{k.removeprefix('block0.'):v for k,v in shapes.items() if k.startswith('block0.')}}
        g=_Graph('vit_block',p['dtype'],roots)
        g.encoder(batch=b,sequence=s,width=h,heads=heads,mlp_hidden=f,epsilon=1e-12,activation='gelu_exact')
    elif section=='head':
        g=_Graph('vit_head',p['dtype'],{'X':(b,s,h),**{k:shapes[k] for k in ('norm_gamma','norm_beta','head_weight','head_bias')}})
        x=g.norm('final_norm','X','norm_gamma','norm_beta')
        x=g.op('cls_slice','slice',[x],(b,1,h),starts=[0,0,0],stops=[b,1,h],steps=[1,1,1])
        x=g.op('select_cls','reshape',[x],(b,h))
        x=g.linear('classifier',x,'head_weight','head_bias')
        # A stable output name without another copy or storage conversion.
        g.w.meta['outputs']=[x]
    else:raise ValueError('ViT part section must be stem/block/head')
    g.w.meta.setdefault('outputs',['Y'])
    return g.w


def _build(p,shapes,*,data,tp):
    if type(tp) is not int or tp!=1:raise ValueError('ViT classifier supports tp=1 only')
    from .api import workload_from_spec
    from .composition import compose_workloads
    from arch_explorer.adf.evaluation import WorkloadSpec
    config={k:p[k] for k in _DEFAULTS if k!='name'}
    roots={key:dict(shape=list(shape),dtype=p['dtype'],kind='activation' if key=='images' else 'weight',
                   **({'value':p['inputs'][key]} if data else {})) for key,shape in shapes.items()}
    instances=[];bindings=[]
    for section,index in [('stem',None),*(('block',i) for i in range(p['layers'])),('head',None)]:
        ident='block'+str(index) if section=='block' else section
        spec=WorkloadSpec('vit_part',dict(section=section,**config))
        graph=workload_from_spec(spec)
        produced={n for op in graph.operators for n in op.outputs}
        external=[n for n in graph.tensors if n not in produced]
        output=graph.meta['outputs'][0]
        instances.append(dict(id=ident,workload=spec,inputs={n:n for n in external},outputs={'Y':output}))
        for local in external:
            source=({'instance':'stem' if index==0 else 'block'+str(index-1),'port':'Y'} if section=='block' and local=='X'
                    else {'instance':'block'+str(p['layers']-1),'port':'Y'} if section=='head' and local=='X'
                    else {'input':ident+'.'+local if section=='block' else local})
            bindings.append({'to':dict(instance=ident,port=local),'from':source})
    graph=workload_from_spec(compose_workloads(name=p['name'],instances=instances,inputs=roots,
        bindings=bindings,outputs={'logits':dict(instance='head',port='Y')}))
    names={i['id']:i['tensor_names'] for i in graph.meta['composition']['instances']}
    graph.meta['vit']=dict(config={k:p[k] for k in _DIMENSIONS},storage_dtype=p['dtype'],accumulation='fp32',
        epsilon=1e-12,gelu='erf',causal=False,dropout=0,
        boundary='processed NCHW RGB images and explicit parameters resident in shared storage; no preprocessing',
        policy='uniform-storage variant; BF16 newly supported, not official mixed-precision kernel equivalence',
        service='abstract fused function service; FP32 internal reductions use component-local scratch, zero global scratch bytes; no calibrated erf instruction claim',
        tensors=dict(patches=names['stem']['patches'],projection=names['stem']['projection.bias'],cls=names['stem']['cls'],
            position=names['stem']['Y'],blocks=[names['block'+str(i)]['Y'] for i in range(p['layers'])],
            final_norm=names['head']['final_norm'],select_cls=names['head']['select_cls'],logits='logits'))
    return graph


def build_classifier(params, *, tp=1):
    p,shapes=_configuration(params,data=True)
    return _build(p,shapes,data=True,tp=tp)


def build_structure(params, *, tp=1):
    p,shapes=_configuration(params,data=False)
    return _build(p,shapes,data=False,tp=tp)
