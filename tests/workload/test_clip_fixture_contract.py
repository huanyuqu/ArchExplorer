"""Pure formal-input/oracle binding; no production behavior is asserted."""
from copy import deepcopy
import pytest
from clip_contract_cases import KEYS,expected,frozen_inputs,independent_case,shapes
from clip_expectations import clip_reference,first_argmax
from vit_expectations import tensor_shape

@pytest.mark.parametrize('key',KEYS)
def test_complete_frozen_roots_and_independent_answers_are_separate(key):
    row=frozen_inputs()['cases'][key];p=row['workload']['params'];before=deepcopy(p)
    data=independent_case(p);answer=expected(key)
    assert p==before
    assert set(p['inputs'])==set(shapes(p))==set(row['roots'])
    assert all(tensor_shape(p['inputs'][n])==shape for n,shape in shapes(p).items())
    assert tensor_shape(answer['logits_per_image'])==(p['image_batch'],p['text_batch'])
    assert tensor_shape(answer['logits_per_text'])==(p['text_batch'],p['image_batch'])
    assert answer['eot_indices']==[first_argmax(r) for r in p['inputs']['token_ids']]
    assert row['roots']['token_ids']['dtype']=='int32' and row['roots']['logit_scale']['dtype']=='fp32'
    assert set(row['candidates'][0])=={'materialized','views'} and len(row['candidates'])==2
    assert not any(n.startswith('precomputed') for n in p['inputs'])

def test_explicit_bridge_rejects_extra_or_missing_parameter_instead_of_guessing():
    p=deepcopy(frozen_inputs()['cases']['regular-2-fp32']['workload']['params'])
    bad=deepcopy(p);bad['inputs']['precomputed_features']=[0.]
    with pytest.raises(AssertionError):independent_case(bad)
    del p['inputs']['vision.block1.Wq']
    with pytest.raises(KeyError):independent_case(p)
