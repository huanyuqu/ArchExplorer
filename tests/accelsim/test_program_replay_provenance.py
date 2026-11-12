"""A producer label is provenance; reference values and source bindings aren't."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from arch_explorer.adf import MappingSpec, WorkloadSpec, from_dict
from arch_explorer.simulator.accelsim import program


VALIDATOR_SOURCE = 'components/sm/implementations/accelsim/program.py'
LOWERING_SOURCE = 'components/sm/implementations/accelsim/lowering.py'


@pytest.fixture(scope='module')
def base_request(tmp_path_factory):
    specs = program.demo_specs('vit')
    return program.prepare_request(from_dict(specs['adf']), WorkloadSpec(**specs['workload']),
        MappingSpec(**specs['mappings'][0]), output_dir=tmp_path_factory.mktemp('provenance'))


def altered(base_request, tmp_path, change):
    request = deepcopy(base_request)
    original = Path(request['program']['path'])
    document = json.loads(original.read_text())
    change(document)
    shutil.copytree(original.parent / 'program', tmp_path / 'program')
    target = tmp_path / 'PROGRAM.json'
    target.write_bytes(program.canonical(document))
    request['program'] = {'path': str(target), 'sha256': program.sha(target.read_bytes())}
    return request


def test_valid_version_label_preserves_producer_and_reports_consumer(base_request, tmp_path):
    label = '2.5.4' if np.__version__ != '2.5.4' else '2.2.6'
    request = altered(base_request, tmp_path, lambda d: d['reference'].update(numpy_version=label))
    before = Path(request['program']['path']).read_bytes()
    assessment = program.assess_request(request)
    assert assessment['gaps'] and all(g['kind'] == 'engine' for g in assessment['gaps'])
    provenance = assessment['validation_provenance']
    assert provenance['producer']['numpy_version'] == label
    assert provenance['consumer']['numpy_version'] == np.__version__
    assert Path(request['program']['path']).read_bytes() == before


@pytest.mark.parametrize('label', (None, '', 'a' * 129, '2.5.4\n', 254, {}))
def test_malformed_provenance_is_not_accepted(base_request, tmp_path, label):
    request = altered(base_request, tmp_path, lambda d: d['reference'].update(numpy_version=label))
    result = program.assess_request(request)
    assert result['status'] == 'unsupported'
    assert any('bounded nonempty version label' in g['reason'] for g in result['gaps'])


def damage(document, case):
    if case == 'tensor':
        document['reference']['tensors']['images']['sha256'] = 'f' * 64
    elif case == 'initialization':
        # Initializer-specific facts are metadata of the graph-reference
        # contract, separate from the initializer callable's source identity.
        initialization = document['reference']['metadata']['initialization']
        assert len(initialization['scope_sha256']) == 64
        initialization['scope_sha256'] = 'f' * 64
    elif case == 'coverage':
        document['reference']['operator_coverage']['matmul'] += 1
    elif case == 'steps':
        document['reference']['steps'].pop()
    elif case == 'complete':
        document['reference']['complete'] = False
    elif case == 'numeric':
        document['numeric_checks'][0]['max_abs_error'] = 1.0
    elif case == 'outputs':
        first = next(iter(document['outputs'].values()))
        first['values'][0][0] += 0.25
    elif case in ('validator-source', 'lowering-source'):
        name = VALIDATOR_SOURCE if case == 'validator-source' else LOWERING_SOURCE
        assert name in document['binding']['sources'], 'must alter an actual bound authority'
        document['binding']['sources'][name] = '0' * 64
    elif case == 'missing-source':
        document['binding']['sources'].pop(LOWERING_SOURCE)
    else:
        document['binding']['sources']['unknown.py'] = '0' * 64


@pytest.mark.parametrize('case', ('tensor', 'initialization', 'coverage', 'steps', 'complete',
    'numeric', 'outputs', 'validator-source', 'lowering-source', 'missing-source', 'extra-source'))
def test_rehashed_semantic_or_source_changes_remain_rejected(base_request, tmp_path, monkeypatch, case):
    request = altered(base_request, tmp_path, lambda d: damage(d, case))
    calls = []
    monkeypatch.setattr(program, 'run_native', lambda *a, **k: calls.append(True))
    result = program.assess_request(request)
    assert result['status'] == 'unsupported' and result['gaps'][0]['kind'] == 'artifact'
    with pytest.raises(ValueError):
        program.evaluate_request(request)
    assert not calls
    if case == 'tensor':
        assert '/reference/tensors/images/sha256' in result['gaps'][0]['reason']
        assert 'producer NumPy' in result['gaps'][0]['reason'] and 'consumer NumPy' in result['gaps'][0]['reason']
