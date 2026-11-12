"""VAL60 stdlib-only CLIP equations and hand-address expectations.

QA root names deliberately do not claim to be the pending SIM49 public API.
No ArchExplorer, NumPy, reference, compiler or layout resolver is imported.
Only existing independent QA rounding, patch and interval arithmetic is reused.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from precision_transformer_expectations import FORMATS, dot32, f32, layer_norm, store, sum32
from vit_expectations import Geometry as ImageGeometry
from vit_expectations import leaves, patchify, rounded, shaped, stable_softmax, tensor_shape

EPSILON = 1e-5
# Frozen before observing any new CLIP production output. Analytic address,
# index, copy and RNE checks are exact. These are full-chain scalar bounds.
TOLERANCE = {
    'fp32': {'absolute':1e-5, 'relative':1e-5},
    'fp16': {'absolute':2**-10, 'relative':2**-10},
    'bf16': {'absolute':2**-7, 'relative':2**-7},
}


@dataclass(frozen=True)
class Geometry:
    image_batch: int = 2
    text_batch: int = 3
    image_size: int = 4
    patch: int = 2
    vision_width: int = 4
    vision_heads: int = 2
    vision_mlp: int = 7
    vision_layers: int = 2
    context: int = 4
    vocab: int = 11
    text_width: int = 4
    text_heads: int = 2
    text_layers: int = 2
    projection: int = 3

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('positive integer geometry required')
        if self.image_size % self.patch or self.vision_width % self.vision_heads or self.text_width % self.text_heads:
            raise ValueError('patch/head dimensions must divide')
        if self.vocab > 2**31:
            raise ValueError('int32 vocabulary bound exceeded')

    @property
    def image_tokens(self): return (self.image_size//self.patch)**2+1


def block_shapes(width, mlp):
    result = {}
    for name in ('q','k','v','o','up','down'):
        inside,outside = (width,mlp) if name=='up' else (mlp,width) if name=='down' else (width,width)
        result[name+'_weight']=(inside,outside);result[name+'_bias']=(outside,)
    result.update({n:(width,) for n in ('norm1_gamma','norm1_beta','norm2_gamma','norm2_beta')})
    return result


def parameter_shapes(g):
    vision = {'patch_weight':(3*g.patch*g.patch,g.vision_width),
        'class_embedding':(1,1,g.vision_width),'positions':(g.image_tokens,g.vision_width),
        'pre_gamma':(g.vision_width,),'pre_beta':(g.vision_width,),
        'post_gamma':(g.vision_width,),'post_beta':(g.vision_width,),
        'projection':(g.vision_width,g.projection)}
    text = {'token_weight':(g.vocab,g.text_width),'positions':(g.context,g.text_width),
        'final_gamma':(g.text_width,),'final_beta':(g.text_width,),
        'projection':(g.text_width,g.projection)}
    return vision,text,block_shapes(g.vision_width,g.vision_mlp),block_shapes(g.text_width,4*g.text_width)


def token_ids(value, *, batch, context, vocab):
    if not isinstance(value,list) or len(value)!=batch or any(not isinstance(r,list) or len(r)!=context for r in value):
        raise ValueError('token shape differs')
    if any(type(i) is not int or i<0 or i>=vocab or i>2**31-1 for row in value for i in row):
        raise ValueError('tokens must be int32 vocabulary indices, never cast from float/bool/string')
    return deepcopy(value)


def first_argmax(ids):
    if not ids or any(type(v) is not int or v<0 or v>2**31-1 for v in ids):
        raise ValueError('nonempty nonnegative int32 tokens required')
    return ids.index(max(ids))


def lookup_addresses(ids, width):
    """Selected row addresses, with every repeated access retained."""
    return [[i*width+c for c in range(width)] for row in ids for i in row]


def lookup_traffic(ids, *, vocab, width, dtype):
    if dtype not in FORMATS:raise ValueError('explicit floating storage dtype required')
    unit=4 if dtype=='fp32' else 2
    count=sum(len(row) for row in ids)
    return dict(index_bytes=count*4,selected_weight_bytes=count*width*unit,
        output_bytes=count*width*unit,root_weight_bytes=vocab*width*unit,
        integer_conversion_instructions=0,
        index_policy='semantic one int32 read per selected row; bind service multiplicity only after the public access contract freezes')


def eot_addresses(ids, width):
    context=len(ids[0])
    return [[(b*context+first_argmax(row))*width+c for c in range(width)] for b,row in enumerate(ids)]


def eot_traffic(ids, *, width, dtype):
    if dtype not in FORMATS:raise ValueError('explicit floating storage dtype required')
    batch,context=len(ids),len(ids[0]);unit=4 if dtype=='fp32' else 2
    return dict(argmax_reads=batch*context*4,argmax_writes=batch*4,
        selection_indices=batch*4,selection_reads=batch*width*unit,
        selection_writes=batch*width*unit,feature_root_bytes=batch*context*width*unit)


def causal_mask(context):
    """Private semantic mask; no claim about SIM49's pending storage encoding."""
    return [[column<=row for column in range(context)] for row in range(context)]


def mask_traffic(context,planes,*,dtype,mask_item_bytes):
    """Hand pointwise access table with an explicit, externally bound mask unit.

    This preparation does not assign a public mask dtype or storage encoding.
    The formal adapter/access contract must confirm per-cell issue multiplicity.
    """
    if dtype not in FORMATS or type(mask_item_bytes) is not int or mask_item_bytes<1:
        raise ValueError('explicit floating score dtype and positive mask width required')
    cells=context*context;unit=4 if dtype=='fp32' else 2
    return dict(constructed_mask_writes=cells*mask_item_bytes,
        applied_mask_reads=planes*cells*mask_item_bytes,
        score_reads=planes*cells*unit,score_writes=planes*cells*unit)


def similarity_addresses(images,texts):
    return dict(image=list(range(images*texts)),
        text=[i*texts+t for t in range(texts) for i in range(images)],
        root_elements=images*texts)


def quick_gelu(x):
    # FP32 internal arithmetic and exactly one final storage boundary are
    # distinct from materializing sigmoid/multiply as separate low-dtype ops.
    argument=f32(-f32(1.702)*x)
    try: exponential=f32(math.exp(argument))
    except OverflowError: exponential=math.inf
    return f32(x/f32(1+exponential))


def l2_norm(row,dtype):
    try:length=f32(math.sqrt(dot32(row,row)))
    except (OverflowError,ValueError) as error:
        raise ValueError('L2 nonfinite FP32 squared norm') from error
    if not math.isfinite(length) or length==0:
        raise ValueError('L2 zero/nonfinite norm; no implicit epsilon')
    result=[store(f32(x/length),dtype) for x in row]
    if any(not math.isfinite(x) for x in result):raise ValueError('nonfinite normalized feature')
    return result


def prefix_softmax(row, end, dtype):
    if type(end) is not int or end<0 or end>=len(row):raise ValueError('invalid causal diagonal')
    # A semantic upper-triangle -inf mask is evaluated through its allowed
    # prefix. No nonfinite external JSON root or production sentinel is used.
    return stable_softmax(row[:end+1],dtype)+[0.]*(len(row)-end-1)


def _block(x,p,heads,dtype,causal,prefix,record):
    batch,sequence,width=len(x),len(x[0]),len(x[0][0]);dimension=width//heads
    rows=lambda a:[row for item in a for row in item]
    def linear(name,a,weight,bias):
        mm=record(prefix+'.'+name+'.mm',[[store(dot32(row,c,fused=True),dtype) for c in zip(*weight)] for row in a])
        return record(prefix+'.'+name+'.bias',[[store(f32(v+b),dtype) for v,b in zip(row,bias)] for row in mm])
    def norm(name,a,n):return record(prefix+'.'+name,layer_norm(a,p[n+'_gamma'],p[n+'_beta'],EPSILON,dtype))
    def add(name,a,b):return record(prefix+'.'+name,[[store(f32(v+w),dtype) for v,w in zip(r,s)] for r,s in zip(a,b)])
    incoming=rows(x);z=norm('norm1',incoming,'norm1')
    qkv={k:linear(k,z,p[k+'_weight'],p[k+'_bias']) for k in ('q','k','v')}
    context=[[[0.]*width for _ in range(sequence)] for _ in range(batch)]
    qk=[];scaled=[];probabilities=[];attended=[]
    factor=f32(dimension**-.5)
    for b in range(batch):
        for h in range(heads):
            matrices={k:[r[h*dimension:(h+1)*dimension] for r in a[b*sequence:(b+1)*sequence]] for k,a in qkv.items()}
            scores=[[store(dot32(r,k,fused=True),dtype) for k in matrices['k']] for r in matrices['q']]
            qk.append(scores)
            score=[[store(f32(v*factor),dtype) for v in row] for row in scores];scaled.append(score)
            probability=[prefix_softmax(r,i,dtype) if causal else stable_softmax(r,dtype) for i,r in enumerate(score)]
            probabilities.append(probability);av=[]
            for i,row in enumerate(probability):
                values=[store(dot32(row,c,fused=True),dtype) for c in zip(*matrices['v'])]
                av.append(values);context[b][i][h*dimension:(h+1)*dimension]=values
            attended.append(av)
    record(prefix+'.QK',qk);record(prefix+'.scale',scaled)
    record(prefix+'.softmax',probabilities);record(prefix+'.AV',attended);record(prefix+'.merge',context)
    o=linear('o',rows(context),p['o_weight'],p['o_bias'])
    residual=add('residual1',incoming,o)
    up=linear('up',norm('norm2',residual,'norm2'),p['up_weight'],p['up_bias'])
    activation=record(prefix+'.quick_gelu',[[store(quick_gelu(v),dtype) for v in row] for row in up])
    down=linear('down',activation,p['down_weight'],p['down_bias'])
    return record(prefix+'.output',shaped(leaves(add('residual2',residual,down)),(batch,sequence,width)))


def synthetic_case(g=Geometry(),dtype='fp32',case='regular'):
    if dtype not in FORMATS or case not in ('regular','boundary','constant'):raise ValueError('unknown QA case')
    salt=0
    def data(shape,name):
        nonlocal salt
        salt+=1
        values=[((i*13+salt*17)%43-21)/128 for i in range(math.prod(shape))]
        if 'gamma' in name:values=[1+v for v in values]
        if name=='images':
            values=[((i*7+salt*11)%53-26)/32 for i in range(len(values))]
            if case=='constant':values=[1.]*len(values)
            if case=='boundary':
                ulp=2.**(1-FORMATS[dtype][0])
                values=[(-1 if i%3==0 else 1)*(1+ulp*(.5,.625,.375,1.5)[i%4]) for i in range(len(values))]
        return shaped(values,shape)
    v,t,vb,tb=parameter_shapes(g)
    vision={k:data(s,k) for k,s in v.items()};text={k:data(s,k) for k,s in t.items()}
    vision['blocks']=[{k:data(s,k) for k,s in vb.items()} for _ in range(g.vision_layers)]
    text['blocks']=[{k:data(s,k) for k,s in tb.items()} for _ in range(g.text_layers)]
    ids=[[((b*3+i*2)%g.vocab) for i in range(g.context)] for b in range(g.text_batch)]
    for b,row in enumerate(ids):
        row[0]=0;row[(b+1)%g.context]=g.vocab-1
        if g.context>2:row[(b+2)%g.context]=g.vocab-1
    return dict(geometry=asdict(g),dtype=dtype,case=case,roots=dict(images=data((g.image_batch,3,g.image_size,g.image_size),'images'),
        token_ids=ids,vision=vision,text=text,logit_scale=[.25]))


def clip_reference(data):
    g=Geometry(**data['geometry']);dtype=data['dtype'];roots=data['roots']
    if dtype not in FORMATS or set(roots)!={'images','token_ids','vision','text','logit_scale'}:
        raise ValueError('complete explicit QA roots required')
    ids=token_ids(roots['token_ids'],batch=g.text_batch,context=g.context,vocab=g.vocab)
    vision,text,vb,tb=parameter_shapes(g)
    for tower,shapes,blocks,count in [('vision',vision,vb,g.vision_layers),('text',text,tb,g.text_layers)]:
        values=roots[tower]
        if set(values)!=set(shapes)|{'blocks'} or len(values['blocks'])!=count:raise ValueError('complete per-layer parameters required')
        if any(tensor_shape(values[n])!=shape for n,shape in shapes.items()):raise ValueError('parameter shape differs')
        for block in values['blocks']:
            if set(block)!=set(blocks) or any(tensor_shape(block[n])!=s for n,s in blocks.items()):raise ValueError('layer shape differs')
    if tensor_shape(roots['images'])!=(g.image_batch,3,g.image_size,g.image_size) or tensor_shape(roots['logit_scale'])!=(1,):
        raise ValueError('image/FP32 scale shape differs')
    v={k:rounded(x,dtype) for k,x in roots['vision'].items() if k!='blocks'}
    t={k:rounded(x,dtype) for k,x in roots['text'].items() if k!='blocks'}
    vblocks=[{k:rounded(x,dtype) for k,x in p.items()} for p in roots['vision']['blocks']]
    tblocks=[{k:rounded(x,dtype) for k,x in p.items()} for p in roots['text']['blocks']]
    images=rounded(roots['images'],dtype);scale=rounded(roots['logit_scale'],'fp32')[0]
    stages={}
    def record(name,value):
        if any(not math.isfinite(x) for x in leaves(value)):raise ValueError('nonfinite semantic output '+name)
        stages[name]=deepcopy(value);return value
    def linear(name,rows,weight):return record(name,[[store(dot32(r,c,fused=True),dtype) for c in zip(*weight)] for r in rows])
    rows=lambda x:[row for batch in x for row in batch]
    def norm(name,x,p,which):return record(name,layer_norm(x,p[which+'_gamma'],p[which+'_beta'],EPSILON,dtype))
    ig=ImageGeometry(batch=g.image_batch,image_size=g.image_size,patch=g.patch,width=g.vision_width,heads=g.vision_heads)
    patches=record('vision.patches',patchify(images,ig))
    x=linear('vision.patch_projection',rows(patches),v['patch_weight'])
    x=shaped(leaves(x),(g.image_batch,g.image_tokens-1,g.vision_width))
    x=record('vision.CLS_concat',[[deepcopy(v['class_embedding'][0][0])]+item for item in x])
    x=record('vision.position',[[[store(f32(a+b),dtype) for a,b in zip(row,pos)] for row,pos in zip(item,v['positions'])] for item in x])
    x=shaped(leaves(norm('vision.pre_norm',rows(x),v,'pre')),(g.image_batch,g.image_tokens,g.vision_width))
    vision_layers=[]
    for i,p in enumerate(vblocks):
        x=_block(x,p,g.vision_heads,dtype,False,'vision.layer'+str(i),record);vision_layers.append(deepcopy(x))
    selected=record('vision.CLS_select',[item[0] for item in x])
    vf=linear('vision.projection',norm('vision.post_norm',selected,v,'post'),v['projection'])
    x=record('text.lookup',[[deepcopy(t['token_weight'][i]) for i in row] for row in ids])
    record('text.causal_allowed',causal_mask(g.context))
    x=record('text.position',[[[store(f32(a+b),dtype) for a,b in zip(row,pos)] for row,pos in zip(item,t['positions'])] for item in x])
    text_layers=[]
    for i,p in enumerate(tblocks):
        x=_block(x,p,g.text_heads,dtype,True,'text.layer'+str(i),record);text_layers.append(deepcopy(x))
    x=shaped(leaves(norm('text.final_norm',rows(x),t,'final')),(g.text_batch,g.context,g.text_width))
    indices=record('text.eot_index',[first_argmax(row) for row in ids])
    selected=record('text.EOT_select',[item[index] for item,index in zip(x,indices)])
    tf=linear('text.projection',selected,t['projection'])
    vf=record('image_features',[l2_norm(row,dtype) for row in vf])
    tf=record('text_features',[l2_norm(row,dtype) for row in tf])
    try:exponential=record('logit_scale.exp',[f32(math.exp(scale))])[0]
    except OverflowError as error:raise ValueError('FP32 exp(scale) overflow') from error
    scaled=record('image_scaled',[[store(f32(v*exponential),dtype) for v in row] for row in vf])
    logits=record('logits_per_image',[[store(dot32(image,text,fused=True),dtype) for text in tf] for image in scaled])
    reverse=record('logits_per_text',[list(row) for row in zip(*logits)])
    return dict(stages=stages,vision_layers=vision_layers,text_layers=text_layers,eot_indices=indices,
                logits_per_image=logits,logits_per_text=reverse)
