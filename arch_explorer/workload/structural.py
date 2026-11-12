"""Shared structure-only projection and stored-attention encoder composition."""
from .operators import Workload,Tensor,Operator


class Graph:
    def __init__(self,name,dtype,roots):
        from .extensions import _record_source
        _record_source(Graph)
        self.w=Workload(name,dtype=dtype)
        self._names=set();self._produced=set()
        for key,value in roots.items():
            if type(value) is Tensor:self.w.tensors[key]=value
            else:self.w.tensor(key,value,dtype=dtype,kind='activation' if key in ('images','X') else 'weight')

    def op(self,name,kind,inputs,shape,*,storage=None,**attrs):
        y=self.w.tensor(name,tuple(shape),dtype=storage or self.w.dtype)
        self._append(name,kind,inputs,[y],shape=list(shape),**attrs)
        return y

    def _append(self,name,kind,inputs,outputs,**attrs):
        operator=Operator(name,kind,tuple(inputs),tuple(outputs),attrs,kind)
        return self.w._append_operator(operator,names=self._names,produced=self._produced)

    def mm(self,name,x,y,b,m,k,n,shared=False):
        z=self.w.tensor(name,(b,m,n))
        self._append(name,'matmul',[x,y],[z],batch=b,m=m,k=k,n=n,shared_b=shared,
                   dtype=self.w.dtype,output_dtype=self.w.dtype)
        return z

    def linear(self,name,x,weight,bias=None):
        shape=self.w.tensors[x].shape
        b,m,k=(1,*shape) if len(shape)==2 else shape
        n=self.w.tensors[weight].shape[-1]
        z=self.mm(name+'.mm',x,weight,b,m,k,n,True)
        if len(shape)==2:z=self.op(name+'.reshape','reshape',[z],(m,n))
        return self.op(name+'.bias','bias',[z,bias],self.w.tensors[z].shape) if bias is not None else z

    def norm(self,name,x,gamma,beta,epsilon=1e-12):
        return self.op(name,'affine_norm',[x,gamma,beta],self.w.tensors[x].shape,epsilon=epsilon)

    def encoder(self,*,batch,sequence,width,heads,mlp_hidden,epsilon,activation,mask=None):
        if activation not in ('gelu_exact','quick_gelu'):raise ValueError('stored encoder activation unsupported')
        b,s,h,f=batch,sequence,width,mlp_hidden;d=h//heads
        z=self.norm('ln1','X','norm1_gamma','norm1_beta',epsilon)
        qkv=[]
        for key in ('q','k','v'):
            x=self.linear(key,z,'W'+key,'b'+key)
            x=self.op(key+'.split','reshape',[x],(b,s,heads,d))
            axes=[0,2,3,1] if key=='k' else [0,2,1,3]
            shape=(b,heads,d,s) if key=='k' else (b,heads,s,d)
            x=self.op(key+'.transpose','transpose',[x],shape,axes=axes)
            qkv.append(self.op(key+'.heads','reshape',[x],(b*heads,d,s) if key=='k' else (b*heads,s,d)))
        score=self.mm('qk',qkv[0],qkv[1],b*heads,s,d,s)
        score=self.op('scale','scale',[score],(b*heads,s,s),factor=d**-.5)
        if mask is not None:
            score=self.op('masked_payload','apply_mask',[score,mask],(b*heads,s,s))
            prob=self.op('softmax','masked_softmax',[score,mask],(b*heads,s,s))
        else:prob=self.op('softmax','softmax',[score],(b*heads,s,s))
        av=self.mm('av',prob,qkv[2],b*heads,s,s,d)
        x=self.op('unbatch','reshape',[av],(b,heads,s,d))
        x=self.op('merge_transpose','transpose',[x],(b,s,heads,d),axes=[0,2,1,3])
        x=self.op('merge','reshape',[x],(b,s,h))
        x=self.linear('out',x,'Wo','bo')
        x=self.op('residual1','add',['X',x],(b,s,h))
        norm=self.norm('ln2',x,'norm2_gamma','norm2_beta',epsilon)
        up=self.linear('up',norm,'W1','b1')
        act=self.op(activation,activation,[up],(b,s,f))
        down=self.linear('down',act,'W2','b2')
        return self.op('Y','add',[x,down],(b,s,h))
