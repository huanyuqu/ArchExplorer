"""Independent bounded scalar FP32 expectations for VAL55, with no production imports.

Arithmetic rounds each multiply/add and tensor output to IEEE binary32. This is
a software reference with explicit order, not a GPU instruction/FMA guarantee.
Logical counts describe mathematics; a backend must declare any extra work.
"""
from math import exp, isfinite, sqrt
from struct import pack, unpack


def f32(value):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError('finite scalar required')
    try:
        result = unpack('<f', pack('<f', value))[0]
    except (OverflowError, ValueError) as error:
        raise ValueError('outside finite FP32') from error
    if not isfinite(result):
        raise ValueError('outside finite FP32')
    return result


def shape2(value):
    if (not isinstance(value, (tuple, list)) or not value
            or any(not isinstance(row, (tuple, list)) or not row for row in value)):
        raise ValueError('nonempty rank-2 matrix required')
    cols = len(value[0])
    if any(len(row) != cols for row in value):
        raise ValueError('ragged matrix')
    for row in value:
        for number in row:
            f32(number)
    return len(value), cols


def transpose(value):
    rows, cols = shape2(value)
    return [[f32(value[i][j]) for i in range(rows)] for j in range(cols)]


def broadcast(value, shape):
    rows, cols = shape2(value)
    if (len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape)
            or rows not in (1, shape[0]) or cols not in (1, shape[1])):
        raise ValueError('incompatible materialized broadcast')
    return [[f32(value[0 if rows == 1 else i][0 if cols == 1 else j])
             for j in range(shape[1])] for i in range(shape[0])]


def matmul(left, right):
    m, k = shape2(left)
    k2, n = shape2(right)
    if k != k2:
        raise ValueError('matmul reduction extents differ')
    result = []
    for i in range(m):
        row = []
        for j in range(n):
            total = f32(0)
            for p in range(k):
                total = f32(total + f32(f32(left[i][p])*f32(right[p][j])))
            row.append(total)
        result.append(row)
    return result


def reduce2(value, operation, axis, keepdims=True):
    rows, cols = shape2(value)
    if type(axis) is not int or axis not in (0, 1, -1, -2) or type(keepdims) is not bool:
        raise ValueError('explicit rank-2 axis/keepdims required')
    if operation not in ('sum', 'max'):
        raise ValueError('unknown reduction')
    vectors = value if axis % 2 else transpose(value)
    result = []
    for vector in vectors:
        total = f32(vector[0])
        for item in vector[1:]:
            total = f32(total+f32(item)) if operation == 'sum' else max(total,f32(item))
        result.append(total)
    if not keepdims:
        return result
    return [[v] for v in result] if axis % 2 else [result]


def stable_softmax(value):
    rows, cols = shape2(value)
    maxima = reduce2(value, 'max', 1)
    shifted = [[f32(f32(value[i][j])-maxima[i][0]) for j in range(cols)] for i in range(rows)]
    exponentials = [[f32(exp(v)) for v in row] for row in shifted]
    sums = reduce2(exponentials, 'sum', 1)
    normalized = [[f32(v/sums[i][0]) for v in row] for i,row in enumerate(exponentials)]
    return dict(maxima=maxima, shifted=shifted, exponentials=exponentials,
                sums=sums, normalized=normalized)


def attention(q, k, v):
    nq, depth = shape2(q)
    nk, depth2 = shape2(k)
    nv, width = shape2(v)
    if depth != depth2 or nk != nv:
        raise ValueError('attention input extents differ')
    kt = transpose(k)
    scores = matmul(q, kt)
    scale = f32(1/sqrt(depth))
    scaled = [[f32(number*scale) for number in row] for row in scores]
    softmax = stable_softmax(scaled)
    output = matmul(softmax['normalized'], v)
    return dict(kt=kt, scores=scores, scale=scale, scaled=scaled,
                **softmax, output=output)


def non_square_case():
    return dict(
        q=[[.25,-.5,.75],[-.125,.5,-.25],[.5,.5,.5]],
        k=[[.5,.25,-.125],[-.25,.375,.5],[1,0,.25],[0,-.5,.75],[.125,.125,-.25]],
        v=[[-1,2],[0,1],[.5,-.5],[2,0],[-.25,.75]],
        tiles=dict(m=2,k=2,n=3))


def rectangles(rows, cols, row_tile, col_tile):
    if any(type(n) is not int or n < 1 for n in (rows,cols,row_tile,col_tile)):
        raise ValueError('positive integral extents required')
    return [dict(row=i,col=j,rows=min(row_tile,rows-i),cols=min(col_tile,cols-j))
            for i in range(0,rows,row_tile) for j in range(0,cols,col_tile)]


def declared_attention_lifetimes():
    """A declared serial logical schedule in ticks, not a device timing claim.

    Allocate at writer start, release after the last real reader; output retained
    through makespan. The actual mapped backend schedule is audited separately.
    """
    tasks = [
        dict(name='transpose',start=0,end=2,reads=['K'],writes=['KT']),
        dict(name='qk',start=2,end=5,reads=['Q','KT'],writes=['S']),
        dict(name='scale',start=5,end=6,reads=['S'],writes=['Scaled']),
        dict(name='max',start=6,end=8,reads=['Scaled'],writes=['Max']),
        dict(name='subtract',start=8,end=9,reads=['Scaled','Max'],writes=['Shifted']),
        dict(name='exp',start=9,end=11,reads=['Shifted'],writes=['E']),
        dict(name='sum',start=11,end=13,reads=['E'],writes=['Denom']),
        dict(name='normalize',start=13,end=15,reads=['E','Denom'],writes=['P']),
        dict(name='pv',start=15,end=18,reads=['P','V'],writes=['Y'])]
    sizes = dict(Q=36,K=60,V=40,KT=60,S=60,Scaled=60,Max=12,
                 Shifted=60,E=60,Denom=12,P=60,Y=24)
    intervals = []
    for name, size in sizes.items():
        writers = [t for t in tasks if name in t['writes']]
        readers = [t for t in tasks if name in t['reads']]
        assert len(writers) <= 1
        start = writers[0]['start'] if writers else 0
        end = max(t['end'] for t in readers) if readers else max(t['end'] for t in tasks)
        intervals.append(dict(name=name,start=start,end=end,bytes=size,
                              readers=[t['name'] for t in readers]))
    return tasks, intervals
