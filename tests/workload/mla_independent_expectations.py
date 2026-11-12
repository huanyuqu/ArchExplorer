"""VAL61 mathematical preparation; no production imports or derived answers.

Public mathematical source: DeepSeek-V3 Technical Report, arXiv:2412.19437v2,
§2.1.1, equations 2, 4, 5, 9--11. Inputs start AFTER query projection and
RoPE; they are not a full DeepSeek decoder or a calibrated device model.
Matrices below use row-vector convention. Absorption is a separate program:
its intermediate storage boundaries differ, including at FP32 precision.
"""
from dataclasses import dataclass
from fractions import Fraction
import math

from precision_transformer_expectations import dot32, f32, store, sum32


DIMENSIONS = ('batch', 'heads', 'sequence', 'latent', 'content', 'position',
              'value', 'output')


def validate_dimensions(shape):
    if set(shape) != set(DIMENSIONS):
        raise ValueError('explicit complete MLA dimensions required')
    if any(type(v) is not int or v <= 0 for v in shape.values()):
        raise ValueError('MLA dimensions must be positive integers')


def input_shapes(shape):
    validate_dimensions(shape)
    b, h, s, l, c, r, v, o = (shape[k] for k in DIMENSIONS)
    return {'query_content': (b, h, c), 'query_position': (b, h, r),
            'cache_latent': (b, s, l), 'cache_position': (b, s, r),
            'key_up': (h, l, c), 'value_up': (h, l, v),
            'output_weight': (h * v, o)}


def _unflatten(values, shape):
    if len(shape) == 1:
        return values
    stride = math.prod(shape[1:])
    return [_unflatten(values[i*stride:(i+1)*stride], shape[1:])
            for i in range(shape[0])]


def synthetic_roots(shape, *, salt=0):
    """Small deterministic roots only; no activations, weights or paper results."""
    return {name: [(((i*7 + j*3 + salt) % 19) - 9) / 16
                   for i in range(math.prod(dims))]
            for j, (name, dims) in enumerate(input_shapes(shape).items())}


def fp32_region_softmax(row, dtype):
    """Named preparation convention: FP32 region, one final storage boundary.

    Exp uses host libm then FP32 rounding; this is not an exact-transcendental
    or native-instruction accuracy claim. A formal consumer must select its
    documented convention explicitly rather than silently reuse this policy.
    """
    maximum = max(row)
    exps = [f32(math.exp(f32(x - maximum))) for x in row]
    denominator = sum32(exps)
    return [store(f32(x / denominator), dtype) for x in exps]


def attention_core(shape, roots, *, dtype, program, softmax_policy):
    """Nonabsorbed equations or absorbed-key/absorbed-value attention core.

    Raw roots -> dtype, ordered FP32 FMA, each named intermediate -> dtype;
    each score dot is rounded separately before content+position addition.
    Scale is FP32 1/sqrt(content+position), then stored. Return all stages.
    No norm, down projection, cache append, mask, residual or bias is inferred.
    """
    if dtype not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('explicit supported storage dtype required')
    if program not in ('nonabsorbed', 'absorbed'):
        raise ValueError('select nonabsorbed or absorbed program explicitly')
    shapes = input_shapes(shape)
    if set(roots) != set(shapes):
        raise ValueError('roots do not match the independent core boundary')
    data = {}
    for name, dims in shapes.items():
        values = roots[name]
        if len(values) != math.prod(dims) or any(
                isinstance(x, bool) or not isinstance(x, (int, float, Fraction))
                or not math.isfinite(x) for x in values):
            raise ValueError('finite, flat, complete root required: ' + name)
        data[name] = _unflatten([store(x, dtype) for x in values], dims)
    b, h, s, l, c, r, v, o = (shape[k] for k in DIMENSIONS)
    key_up, value_up = data['key_up'], data['value_up']
    stages = {'expanded_keys': [], 'expanded_values': [], 'absorbed_queries': [],
              'scores': [], 'probabilities': [], 'latent_context': [],
              'head_output': [], 'output': []}
    scale = f32(1 / math.sqrt(c + r))
    for batch in range(b):
        heads = []
        for head in range(h):
            latent = data['cache_latent'][batch]
            positions = data['cache_position'][batch]
            qc, qr = data['query_content'][batch][head], data['query_position'][batch][head]
            if program == 'nonabsorbed':
                keys = [[store(dot32(row, col, fused=True), dtype)
                         for col in zip(*key_up[head])] for row in latent]
                values = [[store(dot32(row, col, fused=True), dtype)
                           for col in zip(*value_up[head])] for row in latent]
                stages['expanded_keys'].append(keys)
                stages['expanded_values'].append(values)
                dots = [store(dot32(qc, row, fused=True), dtype) for row in keys]
            else:
                qa = [store(dot32(qc, row, fused=True), dtype) for row in key_up[head]]
                stages['absorbed_queries'].append(qa)
                dots = [store(dot32(qa, row, fused=True), dtype) for row in latent]
            pos_dots = [store(dot32(qr, row, fused=True), dtype) for row in positions]
            scores = [store(f32(store(f32(x+y), dtype)*scale), dtype)
                      for x, y in zip(dots, pos_dots)]
            probabilities = softmax_policy(scores, dtype)
            if len(probabilities) != s or any(not math.isfinite(x) or x < 0
                                             for x in probabilities):
                raise ValueError('invalid softmax policy result')
            stages['scores'].append(scores)
            stages['probabilities'].append(probabilities)
            if program == 'nonabsorbed':
                result = [store(dot32(probabilities, col, fused=True), dtype)
                          for col in zip(*values)]
            else:
                context = [store(dot32(probabilities, col, fused=True), dtype)
                           for col in zip(*latent)]
                stages['latent_context'].append(context)
                result = [store(dot32(context, col, fused=True), dtype)
                          for col in zip(*value_up[head])]
            heads.extend(result)
            stages['head_output'].append(result)
        stages['output'].append([store(dot32(heads, col, fused=True), dtype)
                                 for col in zip(*data['output_weight'])])
    return stages


def exact_absorbed_score(query, key_up, latent):
    """Two exact-rational associations, before storage/softmax/scale."""
    q = list(map(Fraction, query))
    w = [[Fraction(x) for x in row] for row in key_up]
    cache = list(map(Fraction, latent))
    expanded = [sum(cache[j]*w[j][i] for j in range(len(cache)))
                for i in range(len(q))]
    absorbed = [sum(a*b for a, b in zip(q, row)) for row in w]
    return (sum(a*b for a, b in zip(q, expanded)),
            sum(a*b for a, b in zip(absorbed, cache)))


def cache_offsets(shape, *, batch, head, token):
    """Independent contiguous element offsets: head does not duplicate caches."""
    validate_dimensions(shape)
    b, h, s, l, _, r, _, _ = (shape[k] for k in DIMENSIONS)
    if not all(type(x) is int for x in (batch, head, token)) or not (
            0 <= batch < b and 0 <= head < h and 0 <= token < s):
        raise ValueError('cache index outside independent boundary')
    return {'cache_latent': tuple((batch*s+token)*l+i for i in range(l)),
            'cache_position': tuple((batch*s+token)*r+i for i in range(r))}


def grouped_cache_bytes(shape, groups, *, unit_bytes):
    """One full cache fetch per explicitly declared group of consuming tiles.

    Group members are tile identities, not head counts. More heads on a tile
    do not imply more DRAM copies. This declares a cold whole-cache scenario,
    not an inference of the paper's traffic scope or a warm-cache model.
    """
    validate_dimensions(shape)
    members = [tile for group in groups for tile in group]
    if (not groups or any(not group for group in groups)
            or len(set(members)) != len(members)
            or type(unit_bytes) is not int or unit_bytes <= 0):
        raise ValueError('nonempty disjoint groups and positive byte unit required')
    payload = (shape['batch'] * shape['sequence'] *
               (shape['latent'] + shape['position']) * unit_bytes)
    return {'payload': payload, 'private': len(members)*payload,
            'grouped': len(groups)*payload,
            'group_copies': {i: payload for i in range(len(groups))}}


def multicast_edge_bytes(routes, payload_bytes):
    """One immutable payload, one epoch, explicit rooted tree, store-and-forward.

    Charge every used tree edge once; callers invoke separately for distinct
    queries, root spans, payload versions, groups and epochs. Reject reconvergent
    graphs so that accidental free fan-in cannot hide duplicate edge service.
    """
    if type(payload_bytes) is not int or payload_bytes <= 0 or not routes:
        raise ValueError('positive payload and at least one route required')
    edges, parents, root = set(), {}, None
    for path in routes:
        if len(path) < 2 or len(set(path)) != len(path):
            raise ValueError('route must be a nonempty acyclic path')
        root = path[0] if root is None else root
        if path[0] != root:
            raise ValueError('a multicast transaction has one root')
        for source, dest in zip(path, path[1:]):
            if dest == root or dest in parents and parents[dest] != source:
                raise ValueError('multicast topology must be a rooted tree')
            parents[dest] = source
            edges.add((source, dest))
    return {edge: payload_bytes for edge in sorted(edges)}


def port_service_lower_bounds(transactions, links):
    """Conservation lower bounds, not scheduler latency predictions.

    Each transaction is an edge->bytes mapping. Each link explicitly supplies
    bytes/ns and per-message ns. Shared edges serialize their total service;
    independent links may overlap. No global sum is claimed as total latency.
    """
    busy = {}
    for transaction in transactions:
        for edge, size in transaction.items():
            rate, latency = links[edge]
            if not math.isfinite(rate) or not math.isfinite(latency) or rate <= 0 or latency < 0:
                raise ValueError('explicit finite service parameters required')
            busy[edge] = busy.get(edge, Fraction(0)) + Fraction(size)/Fraction(rate) + Fraction(latency)
    return busy


@dataclass(frozen=True)
class RootLifetime:
    name: str
    size: int
    allocated: Fraction
    freed: Fraction
    readers: tuple = ()  # (consumer, read_start, read_finish)
    returned_until: object = None


def physical_peak(roots, *, capacity=None):
    """Actual allocation/free intervals; free precedes alloc at the same tick.

    Pass one entry per physical root; aliases belong in its readers. Reject
    premature release or duplicate ownership. A storage-task end is never
    substituted for an actual free time. Returned roots require explicit pins.
    """
    events, seen = [], set()
    for root in roots:
        if root.name in seen or type(root.size) is not int or root.size <= 0:
            raise ValueError('one positive allocation per physical root required')
        seen.add(root.name)
        if not (math.isfinite(root.allocated) and math.isfinite(root.freed)
                and root.allocated < root.freed):
            raise ValueError('finite positive root lifetime required')
        for _, start, end in root.readers:
            if not (root.allocated <= start <= end <= root.freed):
                raise ValueError('free before last consumer or read before allocation')
        if root.returned_until is not None and not (
                root.allocated <= root.returned_until <= root.freed):
            raise ValueError('returned root lost before retention boundary')
        events.extend([(root.allocated, 1, root.size, root.name),
                       (root.freed, 0, -root.size, root.name)])
    resident = peak = 0
    samples = []
    for tick, _, delta, name in sorted(events):
        resident += delta
        samples.append((tick, name, resident))
        peak = max(peak, resident)
    if capacity is not None:
        if type(capacity) is not int or capacity < 0:
            raise ValueError('capacity is a nonnegative integer number of bytes')
        if peak > capacity:
            raise MemoryError('physical roots require %d bytes, capacity %d' % (peak, capacity))
    return {'peak_bytes': peak, 'samples': samples}
