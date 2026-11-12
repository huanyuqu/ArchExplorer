"""VAL59 independent, stdlib-only ViT equations and hand-address arithmetic.

These are QA data names, separate from the SIM48 public schema. No production
builder, reference, mapper or resolver is imported. The preparation convention
preserves the legacy ViT semantic boundaries: separate linear matmul/bias,
stored QK and scale, one stable FP32 softmax computation then stored output,
affine population norm with epsilon=1e-12, and erf GELU. Binding to the frozen
public API is supplied by the separate pure binding module.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
from fractions import Fraction
import math

from precision_transformer_expectations import (
    FORMATS, dot32, f32, gelu, layer_norm, store, sum32,
)
from view_storage_expectations import AddressMap, Root, STORAGE_BYTES


EPSILON = 1e-12
# Frozen before observing any SIM48 production output. Analytic permutation,
# copy and rounding discriminators remain bit-exact; these are full-chain bounds.
TOLERANCE = {
    'fp32': {'absolute': 8e-6, 'relative': 8e-6},
    'fp16': {'absolute': 2**-11, 'relative': 2**-10},
    'bf16': {'absolute': 2**-8, 'relative': 2**-7},
}


@dataclass(frozen=True)
class Geometry:
    batch: int = 2
    image_size: int = 4
    patch: int = 2
    width: int = 4
    heads: int = 2
    mlp: int = 7
    layers: int = 2
    classes: int = 3

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in asdict(self).values()):
            raise ValueError('positive integer geometry required')
        if self.image_size % self.patch or self.width % self.heads:
            raise ValueError('image/patch and width/heads must divide exactly')

    @property
    def patches(self):
        return (self.image_size // self.patch)**2

    @property
    def tokens(self):
        return self.patches + 1


def leaves(value):
    if isinstance(value, list):
        for child in value:
            yield from leaves(child)
    else:
        yield value


def tensor_shape(value):
    if isinstance(value, list):
        if not value:
            raise ValueError('empty tensor')
        shapes = [tensor_shape(child) for child in value]
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError('ragged tensor')
        return (len(value),) + shapes[0]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite numeric tensor required')
    return ()


def shaped(values, shape):
    values = iter(values)
    def build(dims):
        return [build(dims[1:]) for _ in range(dims[0])] if dims else next(values)
    result = build(tuple(shape))
    if next(values, None) is not None:
        raise ValueError('too many tensor values')
    return result


def rounded(value, dtype):
    if dtype not in FORMATS:
        raise ValueError('explicit supported storage dtype required')
    if isinstance(value, list):
        return [rounded(child, dtype) for child in value]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite numeric root required')
    result = store(value, dtype)
    if not math.isfinite(result):
        raise ValueError('root or semantic result overflows storage dtype')
    return result


def patch_addresses(g):
    """NCHW -> patch row/col -> channel/local row/local col, in elements."""
    r, p, count = g.image_size, g.patch, g.image_size // g.patch
    return [[[((b * 3 + c) * r + pr * p + dr) * r + pc * p + dc
              for c in range(3) for dr in range(p) for dc in range(p)]
             for pr in range(count) for pc in range(count)] for b in range(g.batch)]


def patchify(images, g):
    if tensor_shape(images) != (g.batch, 3, g.image_size, g.image_size):
        raise ValueError('NCHW image shape differs')
    source = list(leaves(images))
    return [[[source[a] for a in token] for token in batch] for batch in patch_addresses(g)]


def cls_addresses(g, dtype):
    root = Root('final_norm', (g.batch, g.tokens, g.width), dtype)
    return AddressMap(root, (g.batch, g.width), (g.tokens * g.width, 1))


def prefix_traffic(g, dtype):
    """Semantic read/write lower bounds, not predicted actual model peak."""
    unit = STORAGE_BYTES[dtype]
    image = g.batch * 3 * g.image_size**2 * unit
    projection = g.batch * g.patches * g.width * unit
    tokens = g.batch * g.tokens * g.width * unit
    return {
        'packing': {'reads': image, 'writes': image},
        'CLS_concat': {'reads': projection + g.batch * g.width * unit, 'writes': tokens},
        'CLS_select': {'logical_reads': g.batch * g.width * unit,
                       'retained_root_bytes_if_exported': tokens},
    }


def parameter_shapes(g):
    outer = {'patch_weight': (3*g.patch*g.patch, g.width), 'patch_bias': (g.width,),
             'class_embedding': (1, 1, g.width), 'positions': (g.tokens, g.width),
             'final_gamma': (g.width,), 'final_beta': (g.width,),
             'head_weight': (g.width, g.classes), 'head_bias': (g.classes,)}
    block = {}
    for key in ('q', 'k', 'v', 'o', 'up', 'down'):
        inside, outside = ((g.width, g.mlp) if key == 'up' else
                           (g.mlp, g.width) if key == 'down' else (g.width, g.width))
        block[key + '_weight'] = (inside, outside)
        block[key + '_bias'] = (outside,)
    block.update({key: (g.width,) for key in ('norm1_gamma', 'norm1_beta', 'norm2_gamma', 'norm2_beta')})
    return outer, block


def synthetic_case(g=Geometry(), dtype='fp32', case='regular'):
    """Only roots/parameters: no computed patch, layer activation or logits."""
    if dtype not in FORMATS or case not in ('regular', 'boundary', 'constant'):
        raise ValueError('unknown synthetic case')
    salt = 0
    def data(shape, kind):
        nonlocal salt
        salt += 1
        values = [((i * 13 + salt * 17) % 43 - 21) / (32 if kind == 'images' else 128)
                  for i in range(math.prod(shape))]
        if 'gamma' in kind:
            values = [1 + v for v in values]
        if case == 'constant' and kind == 'images':
            values = [1.0] * len(values)
        if case == 'boundary' and kind == 'images':
            precision = FORMATS[dtype][0]
            unit = 2.**(1-precision)
            offsets = (unit/2, unit*5/8, unit*3/8, unit*3/2)
            values = [(-1 if i % 3 == 0 else 1) * (1 + offsets[i % 4]) for i in range(len(values))]
        return shaped(values, shape)
    outer, block = parameter_shapes(g)
    roots = {name: data(shape, name) for name, shape in outer.items()}
    roots['images'] = data((g.batch, 3, g.image_size, g.image_size), 'images')
    roots['blocks'] = [{name: data(shape, name) for name, shape in block.items()} for _ in range(g.layers)]
    return {'geometry': asdict(g), 'dtype': dtype, 'case': case, 'roots': roots}


def stable_softmax(row, dtype):
    """One semantic op: FP32 internals, one result storage boundary."""
    maximum = max(row)
    exp = [f32(math.exp(f32(x - maximum))) for x in row]
    total = sum32(exp)
    return [store(f32(x / total), dtype) for x in exp]


def vit_reference(data):
    g, dtype = Geometry(**data['geometry']), data['dtype']
    outer, block = parameter_shapes(g)
    roots = data['roots']
    if set(roots) != set(outer) | {'images', 'blocks'} or len(roots['blocks']) != g.layers:
        raise ValueError('complete roots and distinct per-layer parameters required')
    for name, shape in dict(outer, images=(g.batch, 3, g.image_size, g.image_size)).items():
        if tensor_shape(roots[name]) != shape:
            raise ValueError('root shape differs: ' + name)
    for instance in roots['blocks']:
        if set(instance) != set(block) or any(tensor_shape(instance[k]) != s for k, s in block.items()):
            raise ValueError('complete layer parameter names and shapes required')
    v = {k: rounded(x, dtype) for k, x in roots.items() if k != 'blocks'}
    weights = [{k: rounded(x, dtype) for k, x in layer.items()} for layer in roots['blocks']]
    stages = {}
    def record(name, value):
        if any(not math.isfinite(x) for x in leaves(value)):
            raise ValueError('nonfinite semantic result: ' + name)
        stages[name] = deepcopy(value)
        return value
    def rows(value):
        return [row for batch in value for row in batch]
    def linear(name, x, weight, bias):
        out = [[store(dot32(row, column, fused=True), dtype) for column in zip(*weight)] for row in x]
        record(name + '.mm', out)
        return record(name + '.bias', [[store(f32(x + b), dtype) for x, b in zip(row, bias)] for row in out])
    def add(name, x, y):
        return record(name, [[store(f32(a+b), dtype) for a, b in zip(left, right)] for left, right in zip(x, y)])
    def norm(name, x, gamma, beta):
        return record(name, layer_norm(x, gamma, beta, EPSILON, dtype))

    patches = record('patches', patchify(v['images'], g))
    projected = linear('patch_projection', rows(patches), v['patch_weight'], v['patch_bias'])
    projected = shaped(leaves(projected), (g.batch, g.patches, g.width))
    tokens = record('CLS_concat', [[deepcopy(v['class_embedding'][0][0])] + item for item in projected])
    tokens = [add('position.batch' + str(b), item, v['positions']) for b, item in enumerate(tokens)]
    record('position', tokens)
    layer_outputs = []
    for index, parameter in enumerate(weights):
        prefix = 'layer' + str(index)
        incoming = rows(tokens)
        z = norm(prefix + '.norm1', incoming, parameter['norm1_gamma'], parameter['norm1_beta'])
        qkv = {key: linear(prefix + '.' + key, z, parameter[key+'_weight'], parameter[key+'_bias'])
               for key in ('q', 'k', 'v')}
        context = [[[0.] * g.width for _ in range(g.tokens)] for _ in range(g.batch)]
        dimension = g.width // g.heads
        qk, scaled, probabilities, attended_heads = [], [], [], []
        scale = f32(dimension**-.5)
        for b in range(g.batch):
            for h in range(g.heads):
                matrices = {key: [row[h*dimension:(h+1)*dimension]
                                 for row in value[b*g.tokens:(b+1)*g.tokens]] for key, value in qkv.items()}
                scores = [[store(dot32(row, key, fused=True), dtype) for key in matrices['k']]
                          for row in matrices['q']]
                qk.append(scores)
                scores = [[store(f32(x * scale), dtype) for x in row] for row in scores]
                scaled.append(scores)
                probability = [stable_softmax(row, dtype) for row in scores]
                probabilities.append(probability)
                attended = []
                for s, row in enumerate(probability):
                    values = [store(dot32(row, column, fused=True), dtype) for column in zip(*matrices['v'])]
                    attended.append(values)
                    context[b][s][h*dimension:(h+1)*dimension] = values
                attended_heads.append(attended)
        record(prefix + '.QK', qk)
        record(prefix + '.scale', scaled)
        record(prefix + '.softmax', probabilities)
        record(prefix + '.AV', attended_heads)
        record(prefix + '.attention_merge', context)
        projected = linear(prefix + '.o', rows(context), parameter['o_weight'], parameter['o_bias'])
        residual = add(prefix + '.residual1', incoming, projected)
        normalized = norm(prefix + '.norm2', residual, parameter['norm2_gamma'], parameter['norm2_beta'])
        up = linear(prefix + '.up', normalized, parameter['up_weight'], parameter['up_bias'])
        activated = record(prefix + '.gelu_exact', [[store(gelu(x, 'erf'), dtype) for x in row] for row in up])
        down = linear(prefix + '.down', activated, parameter['down_weight'], parameter['down_bias'])
        output = add(prefix + '.residual2', residual, down)
        tokens = record(prefix + '.output', shaped(leaves(output), (g.batch, g.tokens, g.width)))
        layer_outputs.append(deepcopy(tokens))
    normalized = norm('final_norm', rows(tokens), v['final_gamma'], v['final_beta'])
    selected = record('CLS_select', [normalized[b*g.tokens] for b in range(g.batch)])
    logits = linear('head', selected, v['head_weight'], v['head_bias'])
    return {'stages': stages, 'layers': layer_outputs, 'logits': logits}
