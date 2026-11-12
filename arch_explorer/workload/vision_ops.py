"""Minimal shape contracts for generic packing, concatenation and fused vectors.

Numerical definitions and abstract function costs stay in full_primitives.
No model or backend is selected here.
"""
KINDS = ('patchify', 'concat', 'bias', 'softmax', 'gelu_exact')
COPIES = ('patchify', 'concat')


def attributes(kind):
    return {'patch'} if kind == 'patchify' else {'axis'} if kind == 'concat' else set()


def validate(op, inputs, output):
    a = op.attrs
    if op.op == 'patchify':
        p = a['patch']
        if len(inputs) != 1 or len(inputs[0].shape) != 4 or type(p) is not int or p < 1:
            raise ValueError('patchify requires one NCHW tensor and positive integer patch')
        b, c, h, w = inputs[0].shape
        if h % p or w % p or output.shape != (b, (h//p)*(w//p), c*p*p):
            raise ValueError('patchify requires divisible H/W and [B,patches,C*patch*patch] output')
    elif op.op == 'concat':
        axis = a['axis']
        if len(inputs) < 2 or type(axis) is not int or not -len(output.shape) <= axis < len(output.shape):
            raise ValueError('concat requires at least two tensors and a valid integer axis')
        axis %= len(output.shape)
        if any(len(t.shape) != len(output.shape) or any(
                d != output.shape[i] for i, d in enumerate(t.shape) if i != axis) for t in inputs):
            raise ValueError('concat non-axis dimensions must match')
        if sum(t.shape[axis] for t in inputs) != output.shape[axis]:
            raise ValueError('concat axis sizes must sum to the output')
    elif op.op == 'bias':
        if len(inputs) != 2 or inputs[0].shape != output.shape or inputs[1].shape != (output.shape[-1],):
            raise ValueError('bias requires matching activation and one last-dimension vector')
    elif len(inputs) != 1 or inputs[0].shape != output.shape:
        raise ValueError(op.op+' requires one matching input/output')
