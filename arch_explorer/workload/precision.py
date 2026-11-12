"""One storage-rounding authority and the existing ordered FP32 FFMA oracle."""
import numpy as np

FLOAT_DTYPES = ('fp32', 'fp16', 'bf16')


def _record(function):
    from .extensions import _record_source
    _record_source(function)


def bf16(value):
    """Existing ties-to-even bit algorithm, with an FP32 host container."""
    _record(bf16)
    x = np.asarray(value, dtype=np.float32)
    bits = x.view(np.uint32)
    # Explicit uint32 operands preserve 0d width under NumPy 1.x promotion.
    bias = np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (np.add(bits, bias, dtype=np.uint32) & np.uint32(0xffff0000)).view(np.float32)
    return np.where(np.isfinite(x), rounded, x)


def round_storage(value, dtype):
    """Decode finite inputs as FP32, then round to the declared storage dtype."""
    _record(round_storage)
    if dtype not in FLOAT_DTYPES:
        raise ValueError('floating storage must be fp32/fp16/bf16')
    try:
        with np.errstate(over='raise', invalid='raise'):
            x = np.asarray(value, dtype=np.float32)
            result = bf16(x) if dtype == 'bf16' else x.astype(np.float16 if dtype == 'fp16' else np.float32)
    except FloatingPointError as error:
        raise ValueError('storage conversion overflowed '+dtype) from error
    if not np.isfinite(result).all():
        raise ValueError('storage conversion requires finite '+dtype+' values')
    return np.array(result, order='C', copy=True)


def storage_bytes(value, dtype):
    """Canonical little-endian payload, independent of the BF16 container."""
    _record(storage_bytes)
    if dtype == 'bf16':
        return np.ascontiguousarray((np.asarray(value, dtype=np.float32).view(np.uint32) >> 16).astype('<u2')).tobytes()
    return np.ascontiguousarray(value, dtype={'fp32':'<f4', 'fp16':'<f2'}[dtype]).tobytes()


def matmul_fp32(left, right, blocks=None):
    """K-ordered CPU FFMA interpretation reused by ordinary and tiled matmul."""
    _record(matmul_fp32)
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if left.ndim != 3 or right.ndim != 3 or left.shape[-1] != right.shape[-2]:
        raise ValueError('ordered matmul requires explicit batched matrices')
    batch = left.shape[0]
    if right.shape[0] not in (1, batch):
        raise ValueError('ordered matmul batch differs')
    result = np.zeros((batch, left.shape[1], right.shape[2]), dtype=np.float32)
    spans = blocks if blocks is not None else [dict(start=0, stop=left.shape[2])]
    with np.errstate(over='raise', invalid='raise'):
        for span in spans:
            for j in range(span['start'], span['stop']):
                result = np.float32(left[:, :, j, None].astype(np.float64) *
                                    right[:, None, j, :].astype(np.float64) + result.astype(np.float64))
    return result
