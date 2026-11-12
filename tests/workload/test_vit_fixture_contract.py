"""Source-stage checks of exact inputs and pure binding, not product success."""
from copy import deepcopy
import math

import pytest

from vit_contract_cases import BLOCK, KEYS, OUTER, execution_rows, expected, frozen_inputs, independent_case
from vit_expectations import Geometry, leaves, parameter_shapes, tensor_shape


@pytest.mark.parametrize('key',KEYS)
def test_frozen_root_names_shapes_and_independent_oracle_are_complete(key):
    row = deepcopy(frozen_inputs()['cases'][key])
    p = row['workload']['params']
    before = deepcopy(p)
    data = independent_case(p)
    g = Geometry(**data['geometry'])
    outer,block = parameter_shapes(g)
    assert tensor_shape(data['roots']['images']) == (g.batch,3,g.image_size,g.image_size)
    assert all(tensor_shape(data['roots'][name])==shape for name,shape in outer.items())
    assert all(tensor_shape(layer[name])==shape for layer in data['roots']['blocks'] for name,shape in block.items())
    assert len(data['roots']['blocks'])==g.layers and len(p['inputs'])==9+16*g.layers
    assert all(data['roots']['blocks'][i]!=data['roots']['blocks'][j] for i in range(g.layers) for j in range(i))
    answer = expected(key)
    assert all(math.isfinite(x) for x in leaves(answer['logits']))
    assert p == before
    assert 'X' not in p['inputs'] and 'logits' not in p['inputs']


def test_execution_coverage_matches_predeclared_bounded_scope():
    rows = execution_rows()
    assert len(rows)==48 and len(set(rows))==48
    assert sum(row[-1] for row in rows)==36
    for key in KEYS:
        selected = [row for row in rows if row[0]==key]
        assert {(row[2],row[3]) for row in selected} == {(p,i) for p in ('views','materialized') for i in (0,1)}
    assert any(frozen_inputs()['cases'][key]['workload']['params']['batch']>1 for key in KEYS)
