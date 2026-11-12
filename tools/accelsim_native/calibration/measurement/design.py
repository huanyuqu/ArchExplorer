"""Finite pre-device design and independent analytic unsigned CPU oracle."""
import hashlib
import json
from pathlib import Path
import struct

WORK_ITEM, FUNCTION_PREFIX = 'WP-SIM-34', 'sim34'
FAMILIES = {'dep': (2880, 3360, 3744, 3936),
            'chase': (1501, 1717, 1831, 2003)}
HELDOUT = {'dep': 3744, 'chase': 1831}
MUL, ADD = 1664525, 1013904223
INDEX_MUL, INDEX_ADD, INDEX_MODULUS = 45, 91, 256
START_MUL, START_ADD = 73, 19
INPUT_SEED, INPUT_STRIDE = 0x3451c72d, 0x103117
SAMPLES_PER_CASE = 3


def budget(count):
    """Actual process/launch denominator, inside the fixed task ceilings."""
    if type(count) is not int or not 1 <= count <= 12:
        raise ValueError('finite candidate denominator outside authorized ceiling')
    return {'cases': count, 'host_processes': count+1, 'host_timeout_s': 120,
            'total_window_s': 1800, 'capture_launches': count,
            'timing_launches': count*SAMPLES_PER_CASE,
            'samples_per_case': SAMPLES_PER_CASE, 'warmup_launches': 0,
            'helper_launches': 0, 'app_bytes': 2**20,
            'capture_bytes': 16*2**20, 'host_raw_bytes': 64*2**20,
            'warp_records_per_case': 4096}


BUDGET = budget(sum(map(len, FAMILIES.values())))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def affine(n, mul, add, modulus):
    """Exponentiate affine composition, separate from C++ iterative oracle."""
    a,b=1,0
    while n:
        if n&1: a,b=(mul*a)%modulus,(mul*b+add)%modulus
        mul,add=(mul*mul)%modulus,(mul*add+add)%modulus
        n//=2
    return a,b


def words(values):
    return struct.pack('<'+'I'*len(values),*values)


def values(family,n):
    # Inputs remain identical across lengths within each registered family.
    if family=='chase':
        main=[(INDEX_MUL*i+INDEX_ADD)%INDEX_MODULUS for i in range(INDEX_MODULUS)]
        aux=[(START_MUL*i+START_ADD)%INDEX_MODULUS for i in range(32)]
        a,b=affine(n,INDEX_MUL,INDEX_ADD,INDEX_MODULUS)
        expected=[(a*x+b)%INDEX_MODULUS for x in aux]
    elif family=='dep':
        main=[(INPUT_SEED+INPUT_STRIDE*i)&0xffffffff for i in range(32)]
        aux=[MUL,ADD]
        a,b=affine(n,MUL,ADD,2**32)
        expected=[(a*x+b)&0xffffffff for x in main]
    else:
        raise ValueError('family outside the frozen finite candidate set')
    return words(main),words(aux),words(expected)


def cases():
    return [{'id':f'{family}-{n}','family':family,'steps':n,
             'partition':'heldout' if n==HELDOUT[family] else 'fit',
             'function':f'{FUNCTION_PREFIX}_{family}_{n}','grid':[1,1,1], 'block':[32,1,1],
             'oracle':({'kind':'affine-index-u32','mul':INDEX_MUL,'add':INDEX_ADD,
                        'modulus':INDEX_MODULUS} if family=='chase' else
                       {'kind':'affine-u32','mul':MUL,'add':ADD,'modulus':2**32})}
            for family,lengths in FAMILIES.items() for n in lengths]
