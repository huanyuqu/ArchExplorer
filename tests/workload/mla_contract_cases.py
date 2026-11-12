"""Bind frozen SIM50 input identities; independent answers remain QA-only."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from mla_absorbed_expectations import absorbed_decode,root_shapes

INPUTS = Path(__file__).resolve().parents[1]/'fixtures/workload_scenarios/INPUTS.json'
INPUT_SHA = 'e32af5b84faf48159c92eda84ff36ef3566a58a0dc2818030b0aeda5226ac052'
SCENARIOS = ('grouped_multicast','grouped_unicast','private_multicast','private_unicast')
REJECT = (ValueError,TypeError,KeyError,NotImplementedError)


def independent_params(*, batch=2,heads=3,sequence=3):
    """Additional unequal-dimension synthetic roots, no production generator."""
    import math
    params = dict(batch=batch,heads=heads,sequence=sequence,width=5,query_rank=4,
                  latent_rank=2,content_dim=3,positional_dim=4,output_width=3,
                  dtype='fp32',epsilon=1e-6)
    shapes = root_shapes(params)
    params['inputs'] = {name:[((i*7+j*3)%23-11)/32 for i in range(math.prod(shape))]
                        for j,(name,shape) in enumerate(shapes.items())}
    params['inputs']['q_gamma'] = [1,.75,1.25,.5]
    params['inputs']['rope_cos'] = [.6,.8]
    params['inputs']['rope_sin'] = [.8,-.6]
    def shaped(values,dims):
        if len(dims)==1:return values
        stride = math.prod(dims[1:])
        return [shaped(values[i*stride:(i+1)*stride],dims[1:]) for i in range(dims[0])]
    params['inputs'] = {name:shaped(values,shapes[name]) for name,values in params['inputs'].items()}
    return params


@lru_cache(maxsize=1)
def frozen_inputs():
    raw = INPUTS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == INPUT_SHA
    return json.loads(raw)


@pytest.fixture
def mla_api():
    # Resolve registered component definitions before forbidding backend work.
    # This matches the public package initialization used by other consumers.
    from arch_explorer.components import Kernel  # noqa: F401
    from arch_explorer.components.extensions import get_provider
    get_provider('events')
    w = importlib.import_module('arch_explorer.workload.api')
    s = importlib.import_module('arch_explorer.simulator.api')
    for name in ('workload_from_spec','compile_workload','layout_candidates'):
        assert callable(getattr(w,name)),name
    return w,s


def scene(key):
    from arch_explorer.adf import MappingSpec,WorkloadSpec,from_dict
    data = frozen_inputs();row = data['scenarios'][key]
    return (from_dict(deepcopy(row['adf'])),WorkloadSpec(**deepcopy(data['workload'])),
            MappingSpec(**deepcopy(row['mapping'])))


def flattened(value):
    if hasattr(value,'tolist'):
        value = value.tolist()
    if isinstance(value,(list,tuple)):
        return [item for child in value for item in flattened(child)]
    return [float(value)]


def compare_stages(reference, params):
    """Compare each public original intermediate against separate scalar code."""
    expected = absorbed_decode(params)
    graph,values = reference['graph'],reference['tensors']
    assert graph.meta['outputs'] == ['Y']
    instances = graph.meta['mla']['instances']
    assert set(instances) == {'query','merge'} | {f'head{i}' for i in range(params['heads'])}
    listed = [name for record in instances.values() for name in record['operators']]
    assert len(listed) == len(set(listed))
    assert set(listed) == {op.name for op in graph.operators}
    compared = []
    for op in graph.operators:
        key = op.name
        for suffix in ('.mm','.reshape','.rows'):
            if key.endswith(suffix) and key[:-len(suffix)] in expected['stages']:
                key = key[:-len(suffix)]
                break
        assert key in expected['stages'],('new stable stage needs independent binding',op.name)
        wanted = flattened(expected['stages'][key])
        got = flattened(values[op.name])
        assert len(got) == len(wanted),(op.name,len(got),len(wanted))
        # FP32 ordered FMA is exact; host libm SFU and normalization admit a
        # small predeclared absolute+relative bound, never fitted to the result.
        assert got == pytest.approx(wanted,abs=2e-6,rel=2e-6),op.name
        compared.append(op.name)
    return dict(compared_stages=compared,policy=expected['policy'],
                output=expected['output'],CPU_oracle=True,component_timing=False)
