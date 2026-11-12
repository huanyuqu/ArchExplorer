"""Source-sized openpi graph. No checkpoint, numerical actions or hardware fit.

Weights retain FP32 initializer storage; explicit converted copies are shared.
The static DAG is a non-fused implementation of the source, including dead
prefix final norm, but not duplicate diagnostic vision heads.
"""
from dataclasses import replace
import math
from arch_explorer.model_specs.pi05 import expand
from arch_explorer.workload.full_models import Graph
from arch_explorer.workload.full_primitives import FullOperator
from arch_explorer.workload.operators import Operator


def mask_positions(params):
    s = expand(params); p = s['params']; valid=[]; positions=[]
    for b in range(p['batch']):
        v = [p['image_masks'][c][b] for c in range(3) for _ in range(s['vision']['tokens'])] + p['token_mask'][b]
        total=0; pos=[]
        for x in v:
            total += int(x); pos.append(total-1)
        valid.append(v); positions.append(pos)
    return valid, positions, [[sum(v)+j for j in range(p['action_horizon'])] for v in valid]


class PiGraph(Graph):
    def __init__(self, spec):
        super().__init__('pi05', 'bf16' if spec['params']['dtype']=='bfloat16' else 'fp32')
        self.spec=spec; self.act=self.w.dtype; self.weight_casts={}

    def weight(self, name, shape, dtype=None):
        dtype=dtype or self.act
        base='weights.'+name
        if base not in self.w.tensors:
            self.w.tensor(base, shape, 'weight', 'fp32')
        assert self.w.tensors[base].shape==tuple(shape)
        if dtype=='fp32': return base
        key=(base,dtype)
        if key not in self.weight_casts:
            out=self.cast(base+'.'+dtype,base,dtype)
            self.w.tensors[out]=replace(self.w.tensors[out],kind='weight')
            self.weight_casts[key]=out
        return self.weight_casts[key]

    def cast(self, name, x, dtype):
        if self.w.tensors[x].dtype==dtype: return x
        return self.primitive(name,'cast',[x],self.w.tensors[x].shape,dtype)

    def primitive(self,name,op,inputs,shape,dtype=None,**attrs):
        # Non-broadcast concatenation/layout and fused source formulas use
        # explicit semantic attributes, never shape-based guesses.
        out=super().primitive(name,op,inputs,shape,dtype,**attrs)
        a=self.w.operators[-1].attrs
        a['result_dtype']=self.w.tensors[out].dtype
        if op in ('rms','rope','ada_affine'):
            a['scratch_bytes']=4*(2*math.prod(shape)+math.prod(shape[:-1]))
        if op in ('mul','ada_affine','gated_add','rms_scale'):
            a['read_bytes']=sum(math.prod(shape)*self.w.tensors[n].bytes/math.prod(self.w.tensors[n].shape) for n in inputs)
        if op == 'prefix_mask':
            a['read_bytes']=2*math.prod(shape)  # query and key bool at every output cell
        if op == 'suffix_mask':
            a['read_bytes']=self.w.tensors[inputs[0]].bytes*shape[1]
        if op == 'suffix_positions':
            a['extra_row_simt']=self.w.tensors[inputs[0]].shape[-1]  # sum valid prefix once per batch
        if op == 'attention_mask':
            a['read_bytes']=self.w.tensors[inputs[0]].bytes+self.w.tensors[inputs[1]].bytes*attrs['heads']
        if op not in ('copy','cast','concat','split_heads','merge_heads','patchify','transpose','lookup','select'):
            a['casts_per_element']=sum(self.w.tensors[n].dtype=='bf16' for n in inputs)+(self.w.tensors[out].dtype=='bf16')
        return out

    def mm(self,name,x,y,b,m,k,n,shared=False,output_dtype=None,group=1):
        dtype=self.w.tensors[x].dtype
        assert dtype==self.w.tensors[y].dtype
        assert math.prod(self.w.tensors[x].shape)==b*m*k
        assert math.prod(self.w.tensors[y].shape)==(1 if shared else b//group)*k*n
        out=self.w.tensor(name,(b,m,n),dtype=output_dtype or dtype)
        self.w.operators.append(Operator(name,'matmul',(x,y),(out,),
            dict(batch=b,m=m,k=k,n=n,shared_b=shared,kv_group=group,dtype=dtype,output_dtype=output_dtype or dtype),name))
        return out

    def linear(self,name,x,n,bias=True,weight=None,dtype=None):
        dtype=dtype or self.w.tensors[x].dtype
        x=self.cast(name+'.input_cast',x,dtype)
        shape=self.w.tensors[x].shape; k=shape[-1]; m=math.prod(shape[:-1]); key=weight or name
        y=self.mm(name+'.mm',x,self.weight(key+'.weight',(k,n),dtype),1,m,k,n,True)
        y=self.view(name+'.shape',y,(*shape[:-1],n))
        if bias: y=self.primitive(name+'.bias','bias',[y,self.weight(key+'.bias',(n,),dtype)],(*shape[:-1],n),dtype)
        return y

    def ln(self,name,x,key):
        sh=self.w.tensors[x].shape; dtype=self.w.tensors[x].dtype
        return self.primitive(name,'affine_norm',[x,self.weight(key+'.gamma',(sh[-1],),'fp32'),self.weight(key+'.beta',(sh[-1],),'fp32')],sh,dtype,epsilon=1e-6)

    def rms(self,name,x,key,cond=None,use_gate=True):
        sh=self.w.tensors[x].shape; dtype=self.w.tensors[x].dtype
        z=self.primitive(name+'.rms','rms',[x],sh,'fp32',epsilon=1e-6)
        if cond is None:
            z=self.primitive(name+'.scale','rms_scale',[z,self.weight(key+'.scale',(sh[-1],),'fp32')],sh,'fp32')
            return self.cast(name+'.cast',z,dtype),None
        mod=self.linear(name+'.mod',cond,3*sh[-1],weight=key+'.mod',dtype=dtype)
        parts=[]
        for i,label in enumerate(('scale','shift','gate') if use_gate else ('scale','shift')):
            parts.append(self.primitive(name+'.'+label,'select',[mod],(sh[0],1,sh[-1]),dtype,segment=i))
        z=self.primitive(name+'.affine','ada_affine',[z,*parts[:2]],sh,'fp32')
        return self.cast(name+'.cast',z,dtype),(parts[2] if use_gate else None)

    def attention(self,name,x,key,heads,kv,d,positions=None,mask=None,cache=None,vision=False):
        b,s,_=self.w.tensors[x].shape; dtype=self.w.tensors[x].dtype
        qkv=[]
        for label,count in [('q',heads),('k',kv),('v',kv)]:
            y=self.linear(name+'.'+label,x,count*d,bias=vision,weight=key+'.'+label)
            y=self.view(name+'.'+label+'.reshape',y,(b,s,count,d))
            if positions is not None and label!='v':
                y=self.primitive(name+'.'+label+'.rope','rope',[y,positions],(b,s,count,d),dtype,max_wavelength=10000)
            if label=='q':
                # Flax attention divides by sqrt(head_dim) converted to the
                # query dtype; Gemma multiplies by the FP32 scalar d**-.5.
                scaling = dict(divisor=math.sqrt(d), divisor_dtype=dtype) if vision else dict(factor=d**-.5)
                y=self.primitive(name+'.q.scale','scale',[y],(b,s,count,d),dtype,**scaling)
            qkv.append(y)
        cache_out=tuple(qkv[1:])
        if cache:
            for i in (1,2):
                old=cache[i-1]; old_s=self.w.tensors[old].shape[1]
                qkv[i]=self.primitive(name+('.K' if i==1 else '.V')+'.concat','concat',[old,qkv[i]],(b,old_s+s,kv,d),dtype,axis=1)
        length=self.w.tensors[qkv[1]].shape[1]
        q=self.primitive(name+'.q.layout','copy',[qkv[0]],(b*heads,s,d),dtype,layout='BTHD->BHTD')
        k=self.primitive(name+'.k.layout','copy',[qkv[1]],(b*kv,d,length),dtype,layout='BTHD->BHDT')
        v=self.primitive(name+'.v.layout','copy',[qkv[2]],(b*kv,length,d),dtype,layout='BTHD->BHTD')
        # SigLIP's Flax attention does not enable force_fp32_for_softmax.
        # Gemma explicitly requests FP32 QK. These are distinct source paths.
        logits_dtype = dtype if vision else 'fp32'
        logits=self.mm(name+'.qk',q,k,b*heads,s,d,length,output_dtype=logits_dtype,group=heads//kv)
        if mask: logits=self.primitive(name+'.mask','attention_mask',[logits,mask],(b*heads,s,length),'fp32',big_neg=-2.3819763e38,heads=heads)
        probs=self.primitive(name+'.softmax','softmax',[logits],(b*heads,s,length),logits_dtype,
                             intermediate_dtype=logits_dtype, reduction_dtype='fp32')
        if vision and dtype=='bf16':
            # FP32 vector lowering with BF16 round trips after subtraction,
            # exp and row sum. Baseline input/output conversions are already
            # charged by primitive(); max selects an existing BF16 value.
            attrs=self.w.operators[-1].attrs
            attrs['casts_per_element'] += 4
            attrs['extra_row_simt'] = 2
        probs=self.cast(name+'.prob_cast',probs,dtype)
        av=self.mm(name+'.av',probs,v,b*heads,s,length,d,group=heads//kv)
        z=self.primitive(name+'.merge','merge_heads',[av],(b,s,heads*d),dtype,heads=heads)
        return self.linear(name+'.out',z,self.w.tensors[x].shape[-1],bias=vision,weight=key+'.out'),cache_out

    def mlp(self,name,x,key,hidden,vision=False):
        up=self.linear(name+'.up',x,hidden,bias=vision,weight=key+'.up')
        z=self.primitive(name+'.gelu','gelu_tanh',[up],self.w.tensors[up].shape,self.w.tensors[up].dtype)
        if not vision:
            gate=self.linear(name+'.gate',x,hidden,bias=False,weight=key+'.gate')
            z=self.primitive(name+'.multiply','mul',[z,gate],self.w.tensors[z].shape,self.w.tensors[z].dtype)
        return self.linear(name+'.down',z,self.w.tensors[x].shape[-1],bias=vision,weight=key+'.down')

    def vision(self,name,image):
        v=self.spec['vision']; b=self.spec['params']['batch']; s=v['tokens']; h=v['width']; p=v['patch']
        x=self.primitive(name+'.patchify','copy',[image],(b,s,p*p*3),'fp32',layout='NHWC patches stride=patch VALID')
        x=self.linear(name+'.stem',x,h,weight='vision.stem')
        x=self.primitive(name+'.position','add',[x,self.weight('vision.position',(s,h),'fp32')],(b,s,h),'fp32')
        x=self.cast(name+'.cast',x,self.act)
        for i in range(v['layers']):
            n=f'{name}.block{i}'; key=f'vision.block{i}'
            y=self.ln(n+'.ln1',x,key+'.ln1'); y,_=self.attention(n+'.attn',y,key+'.attn',v['heads'],v['heads'],v['head_dim'],vision=True)
            x=self.primitive(n+'.add1','add',[x,y],(b,s,h),self.act)
            y=self.mlp(n+'.mlp',self.ln(n+'.ln2',x,key+'.ln2'),key+'.mlp',v['mlp_dim'],True)
            x=self.primitive(n+'.add2','add',[x,y],(b,s,h),self.act)
        x=self.ln(name+'.final',x,'vision.final')
        return self.linear(name+'.projection',x,self.spec['language']['prefix']['width'],weight='vision.projection')

    def language(self,name,x,kind,positions,mask,cache=None,cond=None):
        l=self.spec['language']; x=self.cast(name+'.input_cast',x,self.act); caches=[]
        for i in range(l['layers']):
            n=f'{name}.layer{i}'; key=f'{kind}.layer{i}'
            y,gate=self.rms(n+'.norm1',x,key+'.norm1',cond)
            y,kv=self.attention(n+'.attn',y,key+'.attn',l['heads'],l['kv_heads'],l['head_dim'],positions,mask,cache[i] if cache else None)
            caches.append(kv)
            if not cache:
                for tensor in kv: self.w.tensors[tensor]=replace(self.w.tensors[tensor],kind='kv')
            x=self.primitive(n+'.residual1','gated_add' if gate else 'add',[x,y]+([gate] if gate else []),self.w.tensors[x].shape,self.act)
            y,gate=self.rms(n+'.norm2',x,key+'.norm2',cond)
            y=self.mlp(n+'.mlp',y,key+'.mlp',l[kind]['mlp_dim'])
            x=self.primitive(n+'.residual2','gated_add' if gate else 'add',[x,y]+([gate] if gate else []),self.w.tensors[x].shape,self.act)
        x,_=self.rms(name+'.final',x,kind+'.final',cond,use_gate=False)
        return x,caches


def pi05_workload(params):
    spec=expand(params); p=spec['params']; g=PiGraph(spec); b,h,a,t=(p[k] for k in ('batch','action_horizon','action_dim','max_token_len'))
    P=spec['prefix_tokens']; dp=spec['language']['prefix']['width']; de=spec['language']['expert']['width']
    valid,pos,spos=mask_positions(p); outputs=[]; cache_ids=[]
    for iteration in range(p['iterations']):
        root=f'request{iteration}'
        images=[g.w.tensor(f'{root}.image{cam}',spec['shapes']['images.each'],dtype='fp32') for cam in range(3)]
        vision=[g.vision(f'{root}.vision{cam}',x) for cam,x in enumerate(images)]
        ids=g.w.tensor(root+'.token_ids',(b,t),dtype='int32')
        text=g.primitive(root+'.embed','lookup',[ids,g.weight('prefix.embedding',(spec['vocabulary'],dp),'fp32')],(b,t,dp),'fp32')
        text=g.primitive(root+'.embed_scale','scale',[text],(b,t,dp),'fp32',factor=dp**.5)
        text=g.cast(root+'.embed_cast',text,g.act)
        prefix=g.primitive(root+'.prefix.concat','concat',[*vision,text],(b,P,dp),g.act,axis=1)
        masks=[]
        for cam in range(3):
            raw=g.w.tensor(f'{root}.camera_mask{cam}',(b,),dtype='int8')
            masks.append(g.primitive(f'{root}.camera_mask{cam}.expand','copy',[raw],
                                    (b,spec['vision']['tokens']),'int8',layout='camera mask broadcast'))
        masks.append(g.w.tensor(root+'.token_mask',(b,t),dtype='int8'))
        rawmask=g.primitive(root+'.valid','concat',masks,(b,P),'int8',axis=1)
        positions=g.primitive(root+'.positions','positions',[rawmask],(b,P),'int32',values=pos)
        pmask=g.primitive(root+'.prefix.mask','prefix_mask',[rawmask],(b,P,P),'int8')
        _,cache=g.language(root+'.prefix',prefix,'prefix',positions,pmask)
        cache_ids.append(cache)
        suffixpos=g.primitive(root+'.suffix.positions','suffix_positions',[rawmask],(b,h),'int32',values=spos)
        smask=g.primitive(root+'.suffix.mask','suffix_mask',[rawmask],(b,h,P+h),'int8')
        x=g.w.tensor(root+'.noise',(b,h,a),dtype='fp32')
        clock=g.w.tensor(root+'.time0',(b,),dtype='fp32')
        for step in range(p['num_steps']):
            n=f'{root}.step{step}'
            act=g.linear(n+'.action_in',x,de,weight='action_in',dtype='fp32')
            te=g.primitive(n+'.time.sin','sincos',[clock],(b,de),'fp32',min_period=.004,max_period=4.)
            for j in range(2):
                te=g.linear(n+f'.time.fc{j}',te,de,weight=f'time.fc{j}',dtype='fp32')
                te=g.primitive(n+f'.time.swish{j}','silu',[te],(b,de),'fp32')
            z,_=g.language(n+'.expert',act,'expert',suffixpos,smask,cache,te)
            velocity=g.linear(n+'.action_out',z,a,weight='action_out',dtype='fp32')
            x=g.primitive(n+'.flow','flow',[x,velocity],(b,h,a),'fp32',dt=-1/p['num_steps'])
            clock=g.primitive(n+'.time.next','scale_add',[clock],(b,),'fp32',value=-1/p['num_steps'])
        outputs.append(x)
    g.w.meta['full_model']=dict(kind='pi05',source_spec=spec,outputs=outputs,prefix_cache_ids=cache_ids,
        valid_prefix=valid,prefix_positions=pos,suffix_positions=spos,iterations=p['iterations'],
        weight_storage_policy='FP32 source initializer + one shared converted copy per operand dtype; checkpoint bytes unknown',
        precision_policy='explicit casts; SigLIP QK/softmax activation dtype with FP32 sum; Gemma QK/RMS/RoPE FP32; source NNX action/time FP32',
        compilation_policy='non-fused source operations; prefix final norm retained, unused diagnostic head duplicated output omitted; final adaNorm full 3D modulation charged, unused gate not consumed',
        timing_gaps=['SFU formulas analytical, not ISA counts','dot accumulation/fusion compiler dependent','no numerical action values or hardware latency reference'])
    return g.w
