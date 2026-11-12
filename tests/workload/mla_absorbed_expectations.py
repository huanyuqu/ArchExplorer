"""Independent SIM50 FP32 absorbed decode oracle, derived from frozen contract.

No imports of production graph/reference/lowering/layout resolution or NumPy.
The full upstream projections absorbed into WQK/WVO are ROOT definitions here.
Historical latent/positional caches are already normalized/rotated and include
the current token. No cache append or full decoder block is silently added.
"""
from fractions import Fraction
import math

from mla_independent_expectations import fp32_region_softmax
from precision_transformer_expectations import dot32, f32, sum32


GEOMETRY = ('batch','sequence','width','query_rank','latent_rank','heads',
            'content_dim','positional_dim','output_width')


def root_shapes(params):
    g = {k:params[k] for k in GEOMETRY}
    if any(type(v) is not int or v <= 0 for v in g.values()):
        raise ValueError('positive integer geometry required')
    b,s,d,q,r,h,c,p,o = (g[k] for k in GEOMETRY)
    if p % 2:
        raise ValueError('adjacent-pair rotary width must be even')
    shapes = dict(X=(b,d),W_DQ=(d,q),q_gamma=(q,),latent_kv=(b,s,r),
                  pos_key=(b,s,p),rope_cos=(p//2,),rope_sin=(p//2,))
    for i in range(h):
        shapes.update({f'head{i}.WQK':(q,r),f'head{i}.WQR':(q,p),f'head{i}.WVO':(r,o)})
    return shapes


def _rows(values, width):
    return [values[i:i+width] for i in range(0,len(values),width)]


def _matmul(left, right):
    return [[dot32(row,col,fused=True) for col in zip(*right)] for row in left]


def flat_root(raw,shape):
    """Flat dictionary roots may contain shaped arrays, not flat vectors.

    Accept explicit C-order vectors for independent hand cases too. Nested
    arrays must match every declared extent, so ragged input cannot disappear
    during flattening merely because its scalar count happens to match.
    """
    if not isinstance(raw,(list,tuple)):
        raise ValueError('array root required')
    if len(raw) == math.prod(shape) and all(not isinstance(v,(list,tuple)) for v in raw):
        return list(raw)
    if not shape or len(raw) != shape[0]:
        raise ValueError('nested root shape mismatch')
    if len(shape) == 1:
        if any(isinstance(v,(list,tuple)) for v in raw):
            raise ValueError('nested root rank mismatch')
        return list(raw)
    return [scalar for child in raw for scalar in flat_root(child,shape[1:])]


def absorbed_decode(params):
    if params.get('dtype','fp32') != 'fp32':
        raise ValueError('SIM50 frozen production convention only supports FP32')
    epsilon = params.get('epsilon',1e-6)
    if isinstance(epsilon,bool) or not isinstance(epsilon,(int,float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('finite positive epsilon required')
    shapes = root_shapes(params)
    inputs = params['inputs']
    if set(inputs) != set(shapes):
        raise ValueError('exact external root set required')
    roots = {}
    for name,shape in shapes.items():
        raw = flat_root(inputs[name],shape)
        if any(
                isinstance(x,bool) or not isinstance(x,(int,float,Fraction))
                or not math.isfinite(x) for x in raw):
            raise ValueError('complete finite flat root required: '+name)
        roots[name] = [f32(x) for x in raw]
        if any(not math.isfinite(x) for x in roots[name]):
            raise ValueError('root exceeds FP32 range')
    b,s,d,q,r,h,c,p,o = (params[k] for k in GEOMETRY)
    down = _matmul(_rows(roots['X'],d),_rows(roots['W_DQ'],q))
    norm = []
    for row in down:
        squares = [f32(x*x) for x in row]
        mean = f32(sum32(squares)/q)
        denom = f32(math.sqrt(f32(mean+f32(epsilon))))
        norm.append([f32(f32(x/denom)*gamma) for x,gamma in zip(row,roots['q_gamma'])])
    stages = {'query.down':down,'query.norm':norm}
    outputs = []
    scale = f32(1/math.sqrt(c+p))
    for head in range(h):
        prefix = f'head{head}'
        qa = _matmul(norm,_rows(roots[prefix+'.WQK'],r))
        raw_qr = _matmul(norm,_rows(roots[prefix+'.WQR'],p))
        qr = []
        for row in raw_qr:
            rotated = []
            for pair in range(p//2):
                even,odd = row[2*pair:2*pair+2]
                cos,sin = roots['rope_cos'][pair],roots['rope_sin'][pair]
                rotated.extend([f32(f32(even*cos)-f32(odd*sin)),
                                f32(f32(even*sin)+f32(odd*cos))])
            qr.append(rotated)
        content_scores,position_scores,scores,probabilities,context,partial = [],[],[],[],[],[]
        lat_transposes, pos_transposes = [],[]
        for batch in range(b):
            cache = _rows(roots['latent_kv'][batch*s*r:(batch+1)*s*r],r)
            position = _rows(roots['pos_key'][batch*s*p:(batch+1)*s*p],p)
            dc = [dot32(qa[batch],row,fused=True) for row in cache]
            dp = [dot32(qr[batch],row,fused=True) for row in position]
            added = [f32(x+y) for x,y in zip(dc,dp)]
            scaled = [f32(x*scale) for x in added]
            prob = fp32_region_softmax(scaled,'fp32')
            ctx = [dot32(prob,col,fused=True) for col in zip(*cache)]
            result = [dot32(ctx,col,fused=True) for col in zip(*_rows(roots[prefix+'.WVO'],o))]
            content_scores.append(dc);position_scores.append(dp);scores.append(added)
            probabilities.append(prob);context.append(ctx);partial.append(result)
            lat_transposes.extend(list(map(list,zip(*cache))))
            pos_transposes.extend(list(map(list,zip(*position))))
        stages.update({prefix+'.query':qa,prefix+'.pos_query':raw_qr,prefix+'.rope':qr,
                       prefix+'.latent.transpose':lat_transposes,prefix+'.pos.transpose':pos_transposes,
                       prefix+'.content_scores':content_scores,prefix+'.pos_scores':position_scores,
                       prefix+'.scores':scores,prefix+'.scale':[[f32(x*scale) for x in row] for row in scores],
                       prefix+'.softmax':probabilities,prefix+'.context':context,prefix+'.output':partial})
        outputs.append(partial)
    merged = outputs[0]
    for head in range(1,h):
        merged = [[f32(x+y) for x,y in zip(a,b)] for a,b in zip(merged,outputs[head])]
        stages[f'merge.{head}'] = merged
    stages['Y'] = merged
    return dict(roots=roots,stages=stages,output=merged,policy={
        'storage':'fp32','matmul':'ordered FP32 FMA','RMS':'rounded squared products and FP32 sum',
        'softmax':'FP32 region with host-libm exp rounded FP32','merge':'ascending head order'})
