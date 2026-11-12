"""Integer selection and finite-mask encoding; independent of any model/backend."""
from dataclasses import replace
import itertools
import math

KINDS=('lookup','argmax','select','causal_mask','apply_mask','masked_softmax')


def validate(op,tensors):
    from .graph import shape
    if op.op not in KINDS or set(op.attrs)!={'shape'} or len(op.outputs)!=1:
        raise ValueError('indexed/mask operation requires exactly shape and one output')
    output=tensors[op.outputs[0]];inputs=[tensors[n] for n in op.inputs]
    if shape(op.attrs['shape'])!=output.shape:raise ValueError('indexed output shape differs')
    for t in (*inputs,output):
        if (t.dtype not in ('fp32','fp16','bf16','int32') or t.kind=='kv' or not 1<=len(t.shape)<=4
                or (t is output and (t.alias_of is not None or t.strides is not None))):
            raise ValueError('indexed/mask operations require independent output and readonly rank1..4 typed inputs')
    if op.op=='causal_mask':
        if inputs or output.dtype!='int32' or len(output.shape)!=2 or output.shape[0]!=output.shape[1]:
            raise ValueError('causal_mask creates a square int32 flag tensor, 0=keep/1=blocked')
        return
    count=1 if op.op=='argmax' else 2
    if len(inputs)!=count:raise ValueError('indexed/mask input count differs')
    x=inputs[0]
    if op.op=='lookup':
        table=inputs[1]
        if (x.dtype!='int32' or len(x.shape)!=2 or len(table.shape)!=2 or table.dtype=='int32'
                or output.dtype!=table.dtype or output.shape!=(*x.shape,table.shape[1])):
            raise ValueError('lookup requires int32 [B,S], floating [V,H] table and [B,S,H] output')
    elif op.op=='argmax':
        if x.dtype!='int32' or len(x.shape)!=2 or output.dtype!='int32' or output.shape!=x.shape[:-1]:
            raise ValueError('argmax requires int32 [B,S] and first-maximum int32 [B] output')
    elif op.op=='select':
        ids=inputs[1]
        if (len(x.shape)!=3 or x.dtype=='int32' or ids.dtype!='int32' or ids.shape!=(x.shape[0],)
                or output.dtype!=x.dtype or output.shape!=(x.shape[0],x.shape[2])):
            raise ValueError('select requires floating [B,S,H], int32 [B], and [B,H] output')
    else:
        mask=inputs[1]
        if (x.dtype=='int32' or len(x.shape)<2 or output.shape!=x.shape or output.dtype!=x.dtype
                or mask.dtype!='int32' or mask.shape!=x.shape[-2:]):
            raise ValueError('mask operations require floating scores and an int32 last-two-axis flag mask')


def index_values(op,values):
    """Validate actual dynamic indices without truncation or negative wrapping."""
    import numpy as np
    ids=np.asarray(values[0] if op.op=='lookup' else values[1])
    if ids.dtype!=np.dtype('int32'):raise ValueError(op.op+' indices must have actual int32 storage')
    bound=values[1].shape[0] if op.op=='lookup' else values[0].shape[1]
    if np.any(ids<0) or np.any(ids>=bound):raise ValueError(op.op+' index outside [0,'+str(bound)+')')
    return ids


def mask_values(mask):
    import numpy as np
    mask=np.asarray(mask)
    if mask.dtype!=np.dtype('int32') or np.any((mask!=0)&(mask!=1)):
        raise ValueError('mask flags require actual int32 0/1 values')
    return mask.astype(bool)


def reference(op,values):
    import numpy as np
    from .full_primitives import reference as full_reference
    from .extensions import _record_source
    _record_source(full_reference)
    # The legacy rich FullOperator retains its -inf masks and reference policy.
    if 'rows' in op.attrs:return full_reference(op,values)
    if op.op in ('lookup','select'):
        _record_source(index_values);index_values(op,values)
        return full_reference(op,values)
    if op.op=='argmax':
        if values[0].dtype!=np.dtype('int32'):raise ValueError('argmax requires actual int32 input')
        return full_reference(op,values)
    if op.op=='causal_mask':return np.triu(np.ones(op.attrs['shape'],dtype=np.int32),1)
    _record_source(mask_values);blocked=mask_values(values[1])
    x=np.asarray(values[0],dtype=np.float32)
    if op.op=='apply_mask':
        # +0 preserves the legacy mask-add behavior on signed zero; blocked
        # scores are a finite zero payload, interpreted only together with mask.
        return np.where(blocked,np.float32(0),np.add(x,np.float32(0)))
    if np.any(np.all(blocked,axis=-1)):raise ValueError('masked_softmax requires at least one allowed element in every row')
    internal=np.where(blocked,np.float32(-np.inf),x)
    return full_reference(replace(op,op='softmax'),[internal])


def access_indices(op,tensors,values):
    """Logical semantic reads; caller resolves them through the shared layout API."""
    if op.op=='causal_mask':return
    if op.op=='argmax':
        for index in itertools.product(*(range(n) for n in tensors[op.inputs[0]].shape)):
            yield op.inputs[0],index
        return
    if op.op in ('lookup','select'):
        from .reference_api import _input
        from .extensions import _record_source
        _record_source(_input);_record_source(index_values)
        name=op.inputs[0 if op.op=='lookup' else 1]
        if name not in values:raise ValueError('dynamic address audit requires values for '+name)
        ids=_input(values[name],tensors[name])
        bound=tensors[op.inputs[1]].shape[0] if op.op=='lookup' else tensors[op.inputs[0]].shape[1]
        if (ids<0).any() or (ids>=bound).any():raise ValueError('dynamic address audit index outside declared storage')
        for index in itertools.product(*(range(n) for n in ids.shape)):
            yield name,index
            for h in range(tensors[op.outputs[0]].shape[-1]):
                yield (op.inputs[1],(int(ids[index]),h)) if op.op=='lookup' else (op.inputs[0],(index[0],int(ids[index]),h))
        return
    for index in itertools.product(*(range(n) for n in tensors[op.outputs[0]].shape)):
        yield op.inputs[0],index
        yield op.inputs[1],index[-2:]
