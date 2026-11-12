"""Exact SIM48 fixture binding; expected values stay in the pure QA oracle."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import struct

import pytest

from vit_expectations import Geometry, TOLERANCE, leaves, parameter_shapes, store, tensor_shape, vit_reference

INPUTS = Path(__file__).resolve().parents[1] / 'fixtures/vit_compilation/INPUTS.json'
INPUT_SHA = '107e1b2a633ebb16500d3e03b9eb39fbb83addfaff6ab932838b9a5beecac97a'
REJECT = (ValueError, TypeError, KeyError, NotImplementedError)
OUTER = {'patch_weight':'patch_weight', 'patch_bias':'patch_bias',
         'class_embedding':'cls_weight', 'positions':'position_weight',
         'final_gamma':'norm_gamma', 'final_beta':'norm_beta',
         'head_weight':'head_weight', 'head_bias':'head_bias'}
BLOCK = {'q_weight':'Wq', 'k_weight':'Wk', 'v_weight':'Wv', 'o_weight':'Wo',
         'q_bias':'bq', 'k_bias':'bk', 'v_bias':'bv', 'o_bias':'bo',
         'norm1_gamma':'norm1_gamma', 'norm1_beta':'norm1_beta',
         'norm2_gamma':'norm2_gamma', 'norm2_beta':'norm2_beta',
         'up_weight':'W1', 'up_bias':'b1', 'down_weight':'W2', 'down_bias':'b2'}
KEYS = tuple(f'{case}-{layers}-{dtype}' for case in ('regular','boundary')
             for layers in (2,4) for dtype in ('fp32','fp16','bf16'))


@lru_cache(maxsize=1)
def frozen_inputs():
    raw = INPUTS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == INPUT_SHA
    return json.loads(raw)


def independent_case(params):
    """Rename supplied roots only; never call the public shape/helper/reference."""
    g = Geometry(**{k:params[k] for k in ('batch','image_size','patch','width','layers','heads','classes')},
                 mlp=params['mlp_hidden'])
    values = params['inputs']
    outer, block = parameter_shapes(g)
    expected_names = {'images'} | set(OUTER.values()) | {
        f'block{i}.{public}' for i in range(g.layers) for public in BLOCK.values()}
    assert set(values) == expected_names
    roots = {name:deepcopy(values[public]) for name,public in OUTER.items()}
    roots['images'] = deepcopy(values['images'])
    roots['blocks'] = [{name:deepcopy(values[f'block{i}.{public}']) for name,public in BLOCK.items()}
                       for i in range(g.layers)]
    from dataclasses import asdict
    return {'geometry':asdict(g),'dtype':params['dtype'],'case':'frozen-SIM48-roots','roots':roots}


@lru_cache(maxsize=12)
def expected(key):
    return vit_reference(independent_case(frozen_inputs()['cases'][key]['workload']['params']))


@pytest.fixture
def vit_api():
    w = importlib.import_module('arch_explorer.workload.api')
    s = importlib.import_module('arch_explorer.simulator.api')
    for name in ('vit_classifier_input_shapes','vit_classifier','vit_classifier_structure',
                 'qualified_name','layout_candidates','logical_accesses','validate_compilation'):
        assert callable(getattr(w,name)), name
    return w,s


def scene(key, mode='events', policy='views', index=0):
    from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
    doc = frozen_inputs()
    row = doc['cases'][key]
    return (from_dict(deepcopy(doc['hardware'][mode])), WorkloadSpec(**deepcopy(row['workload'])),
            MappingSpec(**deepcopy(row['candidates'][index][policy])))


def canonical(values, dtype):
    result = bytearray()
    for value in values:
        assert store(float(value),dtype) == float(value), (dtype,value)
        if dtype == 'bf16':
            result.extend(struct.pack('<H',struct.unpack('<I',struct.pack('<f',float(value)))[0] >> 16))
        else:
            result.extend(struct.pack('<f' if dtype=='fp32' else '<e',float(value)))
    return bytes(result)


def close(actual, wanted, dtype, *, exact=False):
    # NumPy is used only to read actual production outputs. No production or
    # NumPy operation generates the independent expected values.
    actual = actual.reshape(-1).tolist() if hasattr(actual,'reshape') else list(leaves(actual))
    wanted = list(leaves(wanted))
    assert len(actual) == len(wanted)
    assert canonical(actual,dtype)
    if exact or dtype != 'fp32':
        assert canonical(actual,dtype) == canonical(wanted,dtype)
    else:
        tol = TOLERANCE[dtype]
        for a,b in zip(actual,wanted):
            assert abs(a-b) <= tol['absolute'] + tol['relative']*abs(b), (a,b)


def compare_reference(reference, key):
    params = frozen_inputs()['cases'][key]['workload']['params']
    wanted = expected(key)
    meta = reference['graph'].meta
    names = meta['vit']['tensors']
    assert names == frozen_inputs()['cases'][key]['names']
    tensors = reference['tensors']
    dtype = params['dtype']
    checks = {'patches':wanted['stages']['patches'], 'projection':wanted['stages']['patch_projection.bias'],
              'cls':wanted['stages']['CLS_concat'], 'position':wanted['stages']['position'],
              'final_norm':wanted['stages']['final_norm'], 'select_cls':wanted['stages']['CLS_select'],
              'logits':wanted['logits']}
    for label,value in checks.items():
        close(tensors[names[label]],value,dtype,exact=label=='patches')
    assert len(names['blocks']) == params['layers']
    for name,values in zip(names['blocks'],wanted['layers']):
        close(tensors[name],values,dtype)
    instances = {row['id']:row for row in meta['composition']['instances']}
    assert set(instances) == {'stem','head'} | {f'block{i}' for i in range(params['layers'])}
    arithmetic = {'ln1':'norm1','q.mm':'q.mm','q.bias':'q.bias','k.mm':'k.mm','k.bias':'k.bias',
                  'v.mm':'v.mm','v.bias':'v.bias','qk':'QK','scale':'scale','softmax':'softmax','av':'AV',
                  'merge':'attention_merge','out.mm':'o.mm','out.bias':'o.bias','residual1':'residual1',
                  'ln2':'norm2','up.mm':'up.mm','up.bias':'up.bias','gelu_exact':'gelu_exact',
                  'down.mm':'down.mm','down.bias':'down.bias','Y':'output'}
    for i in range(params['layers']):
        for local,qa_name in arithmetic.items():
            # Tiling replaces original matmul operators. Their named output
            # tensors remain the semantic materialization boundary in both
            # original and mapped graphs.
            name = instances[f'block{i}']['tensor_names'][local]
            close(tensors[name],wanted['stages'][f'layer{i}.{qa_name}'],dtype)
    assert meta['outputs'] == ['logits']


def execution_rows():
    """Exactly the predeclared bounded coverage, no cartesian backend rerun."""
    rows = []
    for key in KEYS:
        case,layers,dtype = key.split('-')
        for index in (0,1):
            execute = case=='regular' or int(layers)==2 and index==1 or int(layers)==4 and index==0
            mode = ('events' if index==0 else 'roofline') if case=='regular' else ('events' if layers=='2' else 'roofline')
            for policy in ('materialized','views'):
                rows.append((key,mode,policy,index,execute))
    return rows
