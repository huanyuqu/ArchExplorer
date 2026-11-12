"""Explicit FP32 RMS/gamma, complex-pair RoPE and immutable storage copies.

The decode boundary supplies already normalized compressed KV and already
rotated positional keys. Cosines/sines are inputs, not free transcendental
instructions. These definitions do not select placement or network service.
"""
import math

KINDS = ('rms_gamma', 'rope_pairs', 'storage.copy')
_DEFINITIONS = None


def validate(op, tensors):
    inputs = [tensors[n] for n in op.inputs]
    outputs = [tensors[n] for n in op.outputs]
    if not inputs or not outputs or any(t.dtype != 'fp32' or t.kind == 'kv' for t in (*inputs,*outputs)):
        raise ValueError('decode primitives require explicit FP32, non-KV-append storage')
    if any(t.alias_of is not None or t.strides is not None for t in outputs):
        raise ValueError('decode primitive outputs must be materialized independent storage')
    if op.op == 'storage.copy':
        if len(inputs)!=1 or op.attrs or any(t.shape!=inputs[0].shape for t in outputs):
            raise ValueError('storage.copy requires one input, identical outputs and no attributes')
        return
    if len(outputs)!=1 or op.attrs.get('shape')!=list(outputs[0].shape) or inputs[0].shape!=outputs[0].shape:
        raise ValueError('decode primitive shape differs')
    width=inputs[0].shape[-1]
    if op.op=='rms_gamma':
        if (len(inputs)!=2 or set(op.attrs)!={'shape','epsilon'} or inputs[1].shape!=(width,)
                or type(op.attrs['epsilon']) not in (int,float) or not math.isfinite(op.attrs['epsilon'])
                or not 1e-38<=op.attrs['epsilon']<=1):
            raise ValueError('rms_gamma requires last-axis gamma and explicit positive epsilon in [1e-38,1]')
    elif op.op=='rope_pairs':
        if len(inputs)!=3 or set(op.attrs)!={'shape'} or width%2 or any(t.shape!=(width//2,) for t in inputs[1:]):
            raise ValueError('rope_pairs requires even last dimension and explicit cos/sin half-width vectors')
    else:raise ValueError('unknown decode primitive')


def reference(op, inputs):
    import numpy as np
    if op.op=='storage.copy':return {n:inputs[0].copy() for n in op.outputs}
    if op.op=='rms_gamma':
        from dataclasses import replace
        from .full_primitives import reference as full_reference
        from .extensions import _record_source
        _record_source(full_reference)
        normalized=full_reference(replace(op,op='rms'),inputs[:1])
        return np.multiply(normalized,inputs[1],dtype=np.float32)
    if op.op=='rope_pairs':
        x,c,s=inputs
        y=np.empty_like(x)
        y[...,0::2]=np.subtract(np.multiply(x[...,0::2],c,dtype=np.float32),
                              np.multiply(x[...,1::2],s,dtype=np.float32),dtype=np.float32)
        y[...,1::2]=np.add(np.multiply(x[...,0::2],s,dtype=np.float32),
                         np.multiply(x[...,1::2],c,dtype=np.float32),dtype=np.float32)
        return y
    raise ValueError('unknown decode reference')


def install():
    global _DEFINITIONS
    from .extensions import OperatorDefinition,register_operator
    if _DEFINITIONS is None:_DEFINITIONS=tuple(OperatorDefinition(n,validate,reference=reference) for n in KINDS)
    for definition in _DEFINITIONS:register_operator(definition)
