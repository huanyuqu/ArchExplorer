"""Entry contract checks; these do not claim SIM-12 execution coverage."""
import json
from pathlib import Path
import random

import pytest

from arch_explorer.adf import WorkloadSpec, COMPONENTS, fingerprint
from arch_explorer.agent_env import ExplorationEnv
from arch_explorer.agent_env.sampling import parameter_variant
from non_gpu_fixture import shared_adf
from arch_explorer.archs.adf_seeds import memory

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('kind', ['vit', 'clip'])
def test_config_and_spec_roundtrip(kind, tmp_path):
    adf = shared_adf()
    adf.elements[2].name = 'hbm'
    params = {'profile': 'vit_b16_224' if kind == 'vit' else 'clip_vit_b32', 'dtype': 'fp16'}
    if kind == 'clip':
        params.update(image_batch=2, text_batch=3)
    config = {'seed_adf': adf.to_dict(), 'workloads': [{'kind': kind, 'params': params, 'name': '', 'weight': 1.0}],
        'backend': 'events', 'max_evaluations': 12, 'parameter_samples': 2, 'mapping_samples': 2,
        'mapping_templates': ['gsm_tiled'],
        'parameter_domains': [{'path': 'chip/core/vector', 'param': 'clock_ghz',
            'values': [1.0, 1.2], 'source': 'explicit non-GPU entry test'}],
        'search': {'proposals': [[{'op': 'create', 'container': 'chip/core',
            'spec': memory('aux', capacity=1024).to_dict()}]]}}
    assert len(config['workloads']) == 1
    spec = WorkloadSpec(**config['workloads'][0])
    assert WorkloadSpec(**json.loads(json.dumps(spec.to_dict()))) == spec
    config['trace_path'] = str(tmp_path / 'trace.jsonl')
    env = ExplorationEnv(config)
    assert env.config['workloads'] == config['workloads']
    assert env.used == 0
    assert env.budget == 12
    assert env.parameter_samples == env.mapping_samples == 2
    if kind == 'clip':
        assert spec.params['image_batch'] != spec.params['text_batch']
    # Hardware proposals are exercised without pretending to execute the graph.
    variants = [parameter_variant(env.adf, COMPONENTS, random.Random(seed), env.radius,
                                 domains=env.parameter_domains) for seed in range(8)]
    assert any(fingerprint(v) != fingerprint(env.adf) for v in variants)
    for proposal in config['search']['proposals']:
        result = env.apply_edits(edits=proposal)
        assert 'error' not in result, result
        assert env.used == 0
        # Every intermediate replacement retains storage and compute.
        assert env.adf.child('hbm') and env.adf.child('l2') and env.adf.child('core')


@pytest.mark.parametrize('kind', ['vit', 'clip'])
@pytest.mark.parametrize('params', [None, [], 'profile'])
def test_params_must_be_object(kind, params):
    with pytest.raises(ValueError, match='object'):
        WorkloadSpec(kind, params)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match='kind'):
        WorkloadSpec('arbitrary_network', {})
