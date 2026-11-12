"""Finite historical migration remains separate from today's open closure.

These are source-vector contract fixtures, not relabelled historical receipts
or a claim that synthetic vectors authenticate actual installed files.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from arch_explorer.components.accelsim import source_compat
from arch_explorer.components.accelsim.program import _check_binding, _sources


EXPECTED_PRODUCER_DIGESTS = {'sim35': '94c8989cc90ff42b79eab4b04995b65cb52ebd8e3cd82c43b71bec7bc16c01ad', 'sim35a': '55033b08d1dbbffded28a3d4e1a4b209128d5d7c5bbd6010fad71dbde410dc00'}


@pytest.fixture(params=tuple(EXPECTED_PRODUCER_DIGESTS))
def frozen_migration(request):
    era = request.param
    producer = deepcopy(source_compat.LEGACY_PROGRAM_SOURCES[era])
    digest = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == EXPECTED_PRODUCER_DIGESTS[era], 'historical producer identity changed'
    # The consumer is today's complete closure, not the old intermediate
    # components/accelsim paths. Values here remain explicitly synthetic:
    # only the unchanged semantic subset is copied from the frozen producer.
    paths = set(_sources())
    consumer = {path: '1' * 64 for path in paths}
    for old, new in source_compat.PROGRAM_PATHS.items():
        if old.startswith('workload/'):
            assert new in consumer
            consumer[new] = producer[old]
    return era, producer, consumer


def test_reader_still_delegates_exact_historical_vector_to_finite_policy(frozen_migration):
    era, producer, consumer = frozen_migration
    assert source_compat.program_source_era(producer, consumer) == era
    # Current reader equality and unequal-source legacy delegation are distinct
    # routes; removing either one must not erase the other's safeguards.
    _check_binding({'sources': producer}, {'sources': consumer}, read_only=True)
    # Historical inputs are permitted only through the explicit read-only
    # reader. New execution must still use the exact current complete vector.
    with pytest.raises(ValueError):
        _check_binding({'sources': producer}, {'sources': consumer})
    _check_binding({'sources': consumer}, {'sources': consumer})


@pytest.mark.parametrize('defect', ['program-hash', 'missing-reference', 'extra-producer-key',
                                   'changed-reference-semantics', 'missing-consumer-key',
                                   'extra-consumer-key'])
def test_finite_migration_cannot_be_generalized_by_rehashing_or_missing_sources(
        frozen_migration, defect):
    _, producer, consumer = frozen_migration
    if defect == 'program-hash':
        producer['simulator/accelsim/program.py'] = '0' * 64
    elif defect == 'missing-reference':
        del producer['workload/reference_execution.py']
    elif defect == 'extra-producer-key':
        producer['external/arbitrary.py'] = '0' * 64
    elif defect == 'changed-reference-semantics':
        consumer['workload/reference_execution.py'] = '2' * 64
    elif defect == 'missing-consumer-key':
        del consumer['components/_support/accelsim/source_compat.py']
    else:
        consumer['external/arbitrary.py'] = '0' * 64
    with pytest.raises(ValueError):
        source_compat.program_source_era(producer, consumer)
    with pytest.raises(ValueError):
        _check_binding({'sources': producer}, {'sources': consumer}, read_only=True)
    with pytest.raises(ValueError):
        _check_binding({'sources': producer}, {'sources': consumer})
