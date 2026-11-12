"""Contiguous floating-storage tensor contracts and missing references.

Existing add/mul/scale/exp/transpose numerical definitions remain authoritative.
These shape contracts are also used by their registered validators in the new
minimal-attrs form; legacy rich primitive descriptors keep their old behavior.
"""
import math
from .vision_ops import KINDS as VISION_KINDS

KINDS = ('add', 'sub', 'mul', 'div', 'scale', 'exp', 'transpose',
         'broadcast', 'reduce_max', 'reduce_sum', 'reshape', 'cast', 'affine_norm', 'gelu_tanh','slice',
         'quick_gelu','l2_norm','scale_tensor','relu','erf','rsqrt','sqrt', *VISION_KINDS)
NEW_KINDS = ('sub', 'div', 'broadcast', 'reduce_max', 'reduce_sum','slice','scale_tensor','relu','erf','rsqrt','sqrt')


def validate(op, tensors):
    from .resource_budget import limit
    from .graph import shape
    if op.op not in KINDS or len(op.outputs) != 1:
        raise ValueError('unsupported materialized tensor operation')
    a = op.attrs
    from .vision_ops import attributes
    from .extensions import _record_source
    if op.op in VISION_KINDS:_record_source(attributes)
    extra = attributes(op.op) if op.op in VISION_KINDS else {'starts','stops','steps'} if op.op=='slice' else {'factor'} if op.op == 'scale' else {'axes'} if op.op == 'transpose' else {'epsilon'} if op.op == 'affine_norm' else (
        {'axis', 'keepdims'} if op.op in ('reduce_max', 'reduce_sum') else set())
    if op.op=='cast' and 'to' in a:extra={'to'}
    if set(a) != {'shape'} | extra:
        raise ValueError(op.name+': materialized attributes differ')
    output = tensors[op.outputs[0]]
    inputs = [tensors[n] for n in op.inputs]
    count = 2 if op.op in ('add', 'sub', 'mul', 'div','scale_tensor') else 3 if op.op == 'affine_norm' else 1
    if op.op not in VISION_KINDS and len(inputs) != count:
        raise ValueError(op.name+': materialized input count differs')
    for tensor in (*inputs, output):
        if (tensor.dtype not in (('fp32','fp16','bf16','int32','int64','bool') if op.op in ('reshape','transpose','slice','broadcast','concat','cast') else ('fp32','fp16','bf16')) or tensor.kind == 'kv'
                or (tensor is output and (tensor.alias_of is not None or tensor.strides is not None))
                or not 1 <= len(tensor.shape) <= limit('max_rank',4)):
            raise ValueError('materialized operations require independent outputs and readonly floating inputs, rank 1..4')
    if op.op not in ('cast','scale_tensor') and any(t.dtype != output.dtype for t in inputs):
        raise ValueError('mixed storage dtype requires explicit cast')
    if shape(a['shape']) != output.shape:
        raise ValueError(op.name+': declared shape differs from output')
    if op.op=='cast':
        if 'to' in a and a['to']!=output.dtype:raise ValueError('Cast to must match declared output storage')
        if (inputs[0].dtype in ('int32','int64','bool') or output.dtype in ('int32','int64','bool')) and 'to' not in a:
            raise ValueError('integer/bool Cast requires explicit to storage')
    if op.op in VISION_KINDS:
        from .vision_ops import validate as validate_vision
        _record_source(validate_vision)
        validate_vision(op,inputs,output)
        return
    source = inputs[0].shape
    if op.op=='scale_tensor':
        if inputs[0].shape!=output.shape or inputs[0].dtype!=output.dtype or inputs[1].dtype!='fp32' or inputs[1].shape!=(1,):
            raise ValueError('scale_tensor requires matching floating data/output and an explicit FP32 [1] scalar')
    elif op.op == 'slice':
        from .storage import view_tensor
        view_tensor(tensors,op.inputs[0],op.outputs[0],'slice',a)
    elif op.op == 'reshape':
        if math.prod(source) != math.prod(output.shape):
            raise ValueError('materialized reshape must preserve element count')
    elif op.op == 'affine_norm':
        if (inputs[0].shape != output.shape or any(t.shape != (source[-1],) for t in inputs[1:])
                or type(a['epsilon']) not in (int, float) or not math.isfinite(a['epsilon'])
                or not 0 < a['epsilon'] <= 3.4028234663852886e38):
            raise ValueError('affine_norm requires matching x, H-shaped gamma/beta and positive FP32 epsilon')
        from .precision import round_storage
        if float(round_storage(a['epsilon'], 'fp32')) <= 0:
            raise ValueError('affine_norm epsilon must remain positive after FP32 decoding')
    elif op.op == 'transpose':
        axes = a['axes']
        if (not isinstance(axes, (list, tuple)) or any(type(n) is not int for n in axes)
                or sorted(axes) != list(range(len(source)))
                or tuple(source[n] for n in axes) != output.shape):
            raise ValueError('materialized transpose axes/shape differ')
    elif op.op == 'broadcast':
        aligned = (1,) * (len(output.shape)-len(source)) + source
        if len(source) > len(output.shape) or any(x != y and x != 1 for x, y in zip(aligned, output.shape)):
            raise ValueError('explicit broadcast target cannot expand its input')
    elif op.op in ('reduce_max', 'reduce_sum'):
        if type(a['axis']) is not int or a['axis'] not in (-1, len(source)-1) or type(a['keepdims']) is not bool:
            raise ValueError('reduction requires the last axis and explicit bool keepdims')
        expected = source[:-1] + ((1,) if a['keepdims'] else ())
        if not expected or output.shape != expected:
            raise ValueError('reduction output shape differs; scalar results unsupported')
    elif any(t.shape != output.shape for t in inputs):
        raise ValueError('elementwise operands require equal shapes; use explicit broadcast')
    if op.op == 'scale' and (type(a['factor']) not in (int, float) or not math.isfinite(a['factor'])
                             or abs(a['factor']) > 3.4028234663852886e38):
        raise ValueError('scale must be finite and representable in FP32')


def reference(op, inputs):
    """Only the semantics absent from the existing primitive registry."""
    import numpy as np
    if op.op=='cast':
        if set(op.attrs)!={'shape','to'}:return inputs[0]
        from .integer import cast_storage
        return cast_storage(inputs[0],op.attrs['to'])
    if op.op=='erf':
        return np.vectorize(math.erf,otypes=[np.float32])(np.asarray(inputs[0],np.float32))
    if op.op=='rsqrt':
        return np.reciprocal(np.sqrt(np.asarray(inputs[0],np.float32)),dtype=np.float32)
    if op.op=='sqrt':
        if np.any(np.asarray(inputs[0])<0):raise ValueError('sqrt requires nonnegative finite values')
        return np.sqrt(np.asarray(inputs[0],np.float32),dtype=np.float32)
    if op.op == 'relu':
        return np.maximum(inputs[0], np.float32(0))
    if op.op in ('scale_tensor','l2_norm'):
        from dataclasses import replace
        from .full_primitives import reference as full_reference
        from .extensions import _record_source
        _record_source(full_reference)
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                result=full_reference(replace(op,op='scale') if op.op=='scale_tensor' else op,inputs)
        except FloatingPointError as error:
            raise ValueError(op.op+': nonfinite/zero FP32 norm or overflow') from error
        if not np.isfinite(result).all():raise ValueError(op.op+': nonfinite output')
        return result
    if op.op == 'slice':
        a=op.attrs
        return inputs[0][tuple(slice(a,b,s) for a,b,s in zip(a['starts'],a['stops'],a['steps']))].copy()
    if op.op == 'sub':
        return np.subtract(inputs[0], inputs[1], dtype=np.float32)
    if op.op == 'div':
        return np.divide(inputs[0], inputs[1], dtype=np.float32)
    if op.op == 'broadcast':
        return np.broadcast_to(inputs[0], tuple(op.attrs['shape'])).copy()
    if op.op == 'reduce_max':
        return np.max(inputs[0], axis=-1, keepdims=op.attrs['keepdims'])
    if op.op == 'reduce_sum':
        return np.sum(inputs[0], axis=-1, keepdims=op.attrs['keepdims'], dtype=np.float32)
    raise NotImplementedError('reference owned by the existing semantic definition: '+op.op)
