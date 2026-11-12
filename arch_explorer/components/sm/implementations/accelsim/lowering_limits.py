"""Bounded reference-generated SM90 program domain, independent of operators."""
import hashlib
import json

CAPABILITIES = ('scalar-fp32', 'global-memory', 'sm90')
LIMITS = {'tensor_elements':8192,'total_tensor_elements':131072,
          'matrix_fmas':250000,'kernels':256,'trace_records':1000000,
          'global_bytes':64*2**20}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

