"""QA-only no-answer inputs, deliberately separate from future production API.

No model weights, measured timings, private paper data or calibrated rates.
Unknown fields at the SIM public boundary must be bound only after its freeze.
"""
from mla_independent_expectations import synthetic_roots


CONVENTION = {
    'boundary': 'query content/position and shared latent/position cache after projections/RoPE',
    'matrices': 'row-vector; explicit flat C-order roots',
    'accumulation': 'ordered FP32 FMA, named intermediate storage boundaries',
    'programs': ['nonabsorbed', 'absorbed'],
    'dtypes': ['fp32', 'fp16', 'bf16'],
    'softmax': 'explicit fp32_region_softmax policy; host libm exp then FP32',
    'absorption_equivalence': 'exact rational before rounding only',
    'dynamic_cache_append': 'not included; immutable preexisting decode cache',
    'mask': 'all supplied historical tokens visible to one decode query',
    'capacity': 'actual physical alloc/free intervals with returned-root retention',
    'network': 'conservation per explicit edge, no fitted latency or traffic saving factor',
}

CASES = [
    dict(name='rectangular', salt=0,
         shape=dict(batch=2, heads=3, sequence=5, latent=2, content=4,
                    position=2, value=3, output=5)),
    dict(name='one_token', salt=5,
         shape=dict(batch=1, heads=1, sequence=1, latent=3, content=2,
                    position=2, value=4, output=3)),
    dict(name='short_cache', salt=11,
         shape=dict(batch=1, heads=2, sequence=3, latent=3, content=2,
                    position=4, value=2, output=3)),
]


def frozen_inputs():
    return dict(schema='arch-explorer.val61.qa-independent-inputs.v1',
                convention=CONVENTION,
                cases=[dict(case, roots=synthetic_roots(case['shape'], salt=case['salt']))
                       for case in CASES])
