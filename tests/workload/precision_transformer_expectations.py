"""Independent VAL56 numerical and storage expectations, using only stdlib.

The binary quantizer uses rational distance and integer ties-to-even, rather
than the production NumPy/bit-mask implementation. The scalar block convention
uses FP32 FMA accumulation, explicit materialized output rounding, separate
projection bias and a FP32 softmax region. The frozen binding selects tanh GELU;
erf remains a separate mathematical discriminator. The immutable preparation
archive retains the pre-interface convention and its original results.
This module does not import ArchExplorer, NumPy, device engines or SDKs.
"""
from fractions import Fraction
import math


FORMATS = {'fp32': (24, -126, 127), 'fp16': (11, -14, 15),
           'bf16': (8, -126, 127)}


def power2(exponent):
    return Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**-exponent)


def ieee_rne(value, dtype):
    """Round exact rational/finite host input to the chosen IEEE binary grid."""
    precision, emin, emax = FORMATS[dtype]
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError('binary rounding requires a real numeric scalar')
    if isinstance(value, float) and not math.isfinite(value):
        return value
    negative = (value < 0 or isinstance(value, float) and
                math.copysign(1.0, value) < 0)
    magnitude = abs(Fraction(value))
    if magnitude == 0:
        return math.copysign(0.0, -1 if negative else 1)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < power2(exponent):
        exponent -= 1
    step = power2(max(exponent, emin) - precision + 1)
    units = magnitude / step
    integer, remainder = divmod(units.numerator, units.denominator)
    twice = 2 * remainder
    if twice > units.denominator or twice == units.denominator and integer % 2:
        integer += 1
    rounded = integer * step
    result = math.inf if rounded >= power2(emax + 1) else float(rounded)
    return math.copysign(result, -1 if negative else 1)


def f32(value):
    return ieee_rne(value, 'fp32')


def store(value, dtype):
    """Explicit FP32 computation result -> target storage boundary."""
    return ieee_rne(f32(value), dtype)


def sum32(values):
    accumulator = 0.0
    for value in values:
        accumulator = f32(Fraction(accumulator) + Fraction(value))
    return accumulator


def dot32(left, right, *, fused=False):
    if len(left) != len(right) or not left:
        raise ValueError('dot operands must have the same positive length')
    accumulator = 0.0
    for a, b in zip(left, right):
        product = Fraction(a) * Fraction(b)
        if not fused:
            product = Fraction(f32(product))
        accumulator = f32(Fraction(accumulator) + product)
    return accumulator


def linear(rows, weights, dtype):
    if not rows or not weights or len(rows[0]) != len(weights):
        raise ValueError('linear input/weight dimensions differ')
    columns = list(zip(*weights))
    return [[store(dot32(row, column, fused=True), dtype) for column in columns] for row in rows]


def bias(rows, values, dtype):
    return [[store(f32(x + b), dtype) for x, b in zip(row, values)] for row in rows]


def add(left, right, dtype):
    return [[store(f32(x + y), dtype) for x, y in zip(a, b)] for a, b in zip(left, right)]


def layer_norm(rows, gamma, beta, epsilon, dtype):
    """Population variance (divide by H), affine after FP32 normalization."""
    result = []
    for row in rows:
        mean = f32(sum32(row) / len(row))
        centered = [f32(x - mean) for x in row]
        variance = f32(dot32(centered, centered) / len(row))
        denominator = f32(math.sqrt(f32(variance + f32(epsilon))))
        result.append([store(f32(f32(f32(x / denominator) * g) + b), dtype)
                       for x, g, b in zip(centered, gamma, beta)])
    return result


def gelu(value, definition):
    if definition == 'erf':
        return f32(value * .5 * (1 + math.erf(value / math.sqrt(2))))
    if definition == 'tanh':
        return f32(value * .5 * (1 + math.tanh(math.sqrt(2/math.pi) *
                                             (value + .044715 * value**3))))
    raise ValueError('GELU definition must explicitly be erf or tanh')


def softmax(row, dtype):
    maximum = max(row)
    shifted = [store(f32(x - maximum), dtype) for x in row]
    exponentials = [store(f32(math.exp(x)), dtype) for x in shifted]
    denominator = store(sum32(exponentials), dtype)
    return [store(f32(x / denominator), dtype) for x in exponentials]


def tiny_inputs():
    """Complete deterministic untrained inputs, with distinct dense weights."""
    config = dict(batch=2, seq=3, hidden=8, heads=2, mlp=16, epsilon=1e-5)

    def vector(count, salt, divisor):
        return [((i * 17 + salt * 11) % 37 - 18) / divisor for i in range(count)]

    inputs = {'x': [[vector(8, b * 3 + s, 16) for s in range(3)] for b in range(2)]}
    for index, name in enumerate(('q', 'k', 'v', 'o', 'fc1', 'fc2')):
        inner, outer = (8, 16) if name == 'fc1' else (16, 8) if name == 'fc2' else (8, 8)
        data = vector(inner * outer, index + 3, 128)
        inputs['w_' + name] = [data[i*outer:(i+1)*outer] for i in range(inner)]
        inputs['b_' + name] = vector(outer, index + 19, 128)
    for index in (1, 2):
        inputs[f'gamma{index}'] = [1 + v for v in vector(8, index + 41, 128)]
        inputs[f'beta{index}'] = vector(8, index + 51, 128)
    return config, inputs


def encoder_block(config, values, *, dtype, gelu_definition):
    """Pre-LN noncausal eval block; explicit output stages, no dropout/cache."""
    if dtype not in FORMATS or gelu_definition not in ('erf', 'tanh'):
        raise ValueError('precision and GELU definition must be explicit')
    if set(config)!={'batch','seq','hidden','heads','mlp','epsilon'}:
        raise ValueError('only the declared noncausal eval block is supported')
    batch, seq, hidden, heads, mlp = (config[k] for k in ('batch','seq','hidden','heads','mlp'))
    if (any(type(n) is not int or n<1 for n in (batch,seq,hidden,heads,mlp))
            or hidden % heads or not math.isfinite(config['epsilon']) or config['epsilon']<=0):
        raise ValueError('invalid block dimensions')

    def shape(value):
        if isinstance(value,list):
            if not value:
                raise ValueError('empty block tensor')
            children=[shape(v) for v in value]
            if any(s!=children[0] for s in children):
                raise ValueError('ragged block tensor')
            return (len(value),)+children[0]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('block inputs must be finite numeric scalars')
        return ()

    shapes={'x':(batch,seq,hidden)}
    for name in ('q','k','v','o','fc1','fc2'):
        inner,outer=(hidden,mlp) if name=='fc1' else (mlp,hidden) if name=='fc2' else (hidden,hidden)
        shapes['w_'+name]=(inner,outer);shapes['b_'+name]=(outer,)
    shapes.update({name:(hidden,) for name in ('gamma1','beta1','gamma2','beta2')})
    if set(values)!=set(shapes) or any(shape(values[k])!=expected for k,expected in shapes.items()):
        raise ValueError('complete block input names and shapes must match')

    def rounded(value):
        if isinstance(value,list):
            return [rounded(x) for x in value]
        result=store(value,dtype)
        if not math.isfinite(result):
            raise ValueError('input overflows declared block storage')
        return result

    v = {name: rounded(value) for name, value in values.items()}
    x = [row for item in v['x'] for row in item]
    norm1 = layer_norm(x, v['gamma1'], v['beta1'], config['epsilon'], dtype)
    projection = {name: bias(linear(norm1, v['w_' + name], dtype), v['b_' + name], dtype)
                  for name in ('q', 'k', 'v')}
    width = hidden // heads
    context = [[[0.0] * hidden for _ in range(seq)] for _ in range(batch)]
    probabilities = []
    scale = f32(1 / math.sqrt(width))
    for b in range(batch):
        for h in range(heads):
            matrices = {name: [row[h*width:(h+1)*width]
                              for row in projection[name][b*seq:(b+1)*seq]]
                        for name in ('q', 'k', 'v')}
            logits = linear(matrices['q'], list(map(list, zip(*matrices['k']))), dtype)
            # Frozen SIM45: QK first stores its result at tensor precision,
            # followed by an explicit FP32 softmax region and cast back.
            scores = [[store(f32(z * scale), 'fp32') for z in row] for row in logits]
            probability32 = [softmax(row, 'fp32') for row in scores]
            probability = [[store(z,dtype) for z in row] for row in probability32]
            probabilities.append(probability32)
            attended = linear(probability, matrices['v'], dtype)
            for s, row in enumerate(attended):
                context[b][s][h*width:(h+1)*width] = row
    merged = [row for item in context for row in item]
    attention = bias(linear(merged, v['w_o'], dtype), v['b_o'], dtype)
    residual1 = add(x, attention, dtype)
    norm2 = layer_norm(residual1, v['gamma2'], v['beta2'], config['epsilon'], dtype)
    expanded = bias(linear(norm2, v['w_fc1'], dtype), v['b_fc1'], dtype)
    activated = [[store(gelu(z, gelu_definition), dtype) for z in row] for row in expanded]
    contracted = bias(linear(activated, v['w_fc2'], dtype), v['b_fc2'], dtype)
    output = add(residual1, contracted, dtype)
    stages = dict(input=x, norm1=norm1, **projection, attention=attention,
                  probabilities=probabilities, residual1=residual1, norm2=norm2,
                  expanded=expanded, activated=activated, contracted=contracted,
                  output=[output[b*seq:(b+1)*seq] for b in range(batch)])
    return stages


def frozen_block_reference(params):
    """Bind the independent equations to SIM45's explicit public input names."""
    names={'X':'x','Wq':'w_q','Wk':'w_k','Wv':'w_v','Wo':'w_o',
           'W1':'w_fc1','W2':'w_fc2','bq':'b_q','bk':'b_k','bv':'b_v','bo':'b_o',
           'b1':'b_fc1','b2':'b_fc2','norm1_gamma':'gamma1','norm1_beta':'beta1',
           'norm2_gamma':'gamma2','norm2_beta':'beta2'}
    config=dict(batch=params['batch'],seq=params['sequence'],hidden=params['hidden'],
                heads=params['heads'],mlp=params['mlp_hidden'],epsilon=params['epsilon'])
    if set(params['inputs'])!=set(names):
        raise ValueError('frozen block input contract changed')
    return encoder_block(config,{names[k]:v for k,v in params['inputs'].items()},
                         dtype=params['dtype'],gelu_definition='tanh')


def tensor_bytes(shape, dtype):
    if dtype not in (*FORMATS, 'int32') or not shape or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('storage requires positive integer shape and explicit dtype')
    return math.prod(shape) * (4 if dtype in ('fp32', 'int32') else 2)


def materialized_movement(shape, source_dtype, target_dtype):
    return dict(read_bytes=tensor_bytes(shape, source_dtype),
                write_bytes=tensor_bytes(shape, target_dtype),
                allocation_bytes=tensor_bytes(shape, target_dtype))


def lifetime_peak(tensors, tasks, finish):
    """Independent declared shape/dtype bytes and actual task-end sweep.

    tensors: name/shape/dtype/allocation/readers/persistent/returned.
    tasks: task-name -> start/end. Allocation times come from real observations.
    Last readers, persistent inputs and returned storage are kept distinct.
    """
    actions, intervals = [], []
    for tensor in tensors:
        readers = tensor['readers']
        last = max(tasks[name]['end'] for name in readers) if readers else finish
        release = finish if tensor.get('persistent') or tensor.get('returned') else last
        start = tensor['allocation']
        if not 0 <= start <= release <= finish:
            raise ValueError('invalid observed lifetime')
        size = tensor_bytes(tensor['shape'], tensor['dtype'])
        intervals.append(dict(name=tensor['name'], bytes=size, allocation=start,
                              last_reader=last, release=release))
        if start < release:
            actions.extend([(start, 1, size), (release, 0, -size)])
    current = peak = 0
    for _, _, delta in sorted(actions):
        current += delta
        if current < 0:
            raise ValueError('storage released before allocation')
        peak = max(peak, current)
    assert current == 0
    return peak, intervals


def audit_materialized_storage(compilation, result, *, allocation_prefix='board:'):
    """Audit actual compilation/trace against independent shape/dtype storage.

    This consumes the existing public v3 JSON contract, not engine internals.
    It does not use scheduler free counters or trust plan.bytes as an oracle.
    """
    declared={t['name']:t for t in compilation['program']['tensors']}
    evaluation=result['evaluation']
    trace=evaluation['details']['trace']
    tasks={r['operator']:dict(start=r['start_ns'],end=r['end_ns'])
           for r in trace if r['kind']=='compute'}
    allocations={r['name']:r['start_ns'] for r in trace if r['kind']=='storage'}
    tensors=[]
    for entry in compilation['plan']['storage']:
        tensor=declared[entry['tensor']]
        expected=tensor_bytes(tuple(tensor['shape']),tensor['dtype'])
        if entry['bytes']!=expected:
            raise AssertionError('declared storage bytes disagree with independent shape/dtype: '+tensor['name'])
        producer=entry['producer']
        if producer is not None:
            if any(tasks[r]['start']<tasks[producer]['end'] for r in entry['consumers']):
                raise AssertionError('consumer started before producer completed')
        tensors.append(dict(name=tensor['name'],shape=tensor['shape'],dtype=tensor['dtype'],
                            allocation=allocations[allocation_prefix+tensor['name']],
                            readers=entry['consumers'],
                            persistent=producer is None and tensor['kind'] in ('weight','kv'),
                            returned=not entry['consumers']))
    return lifetime_peak(tensors,tasks,evaluation['latency_ns'])
